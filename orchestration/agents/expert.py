"""
Expert agent for DelPHEA-irAKI clinical assessment.

Individual expert agents that evaluate cases and participate in consensus building.
"""

import json
import logging
from typing import Dict

from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler

from config.loader import ConfigurationLoader
from orchestration.clients import VLLMClient
from orchestration.messages import (
    DebateComment,
    DebatePrompt,
    ExpertRound1Reply,
    ExpertRound3Reply,
    QuestionnaireMsg,
    TerminateDebate,
)


class irAKIExpertAgent(RoutedAgent):
    """Clinical expert agent for irAKI assessment."""

    def __init__(
        self,
        expert_id: str,
        case_id: str,
        config_loader: ConfigurationLoader,
        vllm_client: VLLMClient = None,
        runtime_config=None,
    ) -> None:
        """Initialize expert agent with specialty configuration.

        Args:
            expert_id: Expert identifier
            case_id: Case identifier
            config_loader: Configuration loader instance
            vllm_client: Optional shared VLLMClient
            runtime_config: Runtime configuration (for creating own client if needed)
        """
        super().__init__(f"irAKI Expert {expert_id}")
        self._expert_id = expert_id
        self._case_id = case_id
        self._config_loader = config_loader

        # handle shared vs owned VLLMClient
        if vllm_client:
            self._vllm_client = vllm_client
            self._owns_vllm_client = False
        else:
            if not runtime_config:
                raise ValueError(
                    "Either vllm_client or runtime_config must be provided"
                )
            self._vllm_client = VLLMClient(runtime_config)
            self._owns_vllm_client = True

        self.logger = logging.getLogger(f"expert.{expert_id}")

        # load expert profile - fail fast if not found
        self._expert_profile = self._config_loader.get_expert_profile(expert_id)
        self.logger.info(
            f"Initialized expert: {self._expert_profile['name']} ({self._expert_profile['specialty']})"
        )

    @message_handler
    async def handle_questionnaire(
        self, message: QuestionnaireMsg, ctx: MessageContext
    ) -> None:
        """Handle Round 1 & 3 irAKI assessment.

        Args:
            message: Questionnaire message with patient data
            ctx: Message context
        """
        try:
            # get prompt template
            prompt_template = self._config_loader.get_prompt_template(
                "iraki_assessment"
            )

            # build prompt with expert profile and case data
            prompt = self._build_assessment_prompt(message, prompt_template)

            llm_response = await self._vllm_client.generate_structured_response(
                prompt=prompt, response_format={"type": "json_object"}
            )

            # clean CI bounds before creating Pydantic model
            if "ci_iraki" in llm_response:
                ci_lower, ci_upper = llm_response["ci_iraki"]
                ci_lower = max(0.0, min(ci_lower, llm_response["p_iraki"]))
                ci_upper = min(1.0, max(ci_upper, llm_response["p_iraki"]))
            else:
                p = llm_response["p_iraki"]
                width = 0.1  # default ±10% uncertainty
                ci_lower, ci_upper = max(0.0, p - width), min(1.0, p + width)

            # create reply object based on round phase
            if message.round_phase == "round1":
                required_fields = [
                    "scores",
                    "evidence",
                    "p_iraki",
                    "confidence",
                    "differential_diagnosis",
                ]
                reply_class = ExpertRound1Reply
            else:  # round3
                required_fields = [
                    "scores",
                    "evidence",
                    "p_iraki",
                    "confidence",
                    "changes_from_round1",
                    "verdict",
                    "final_diagnosis",
                    "recommendations",
                ]
                reply_class = ExpertRound3Reply

            # validate required fields
            missing = [f for f in required_fields if f not in llm_response]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            # create clean dict with corrected CI bounds
            clean_response = {
                **llm_response,
                "ci_iraki": [ci_lower, ci_upper],
                "expert_id": self._expert_id,
                "case_id": message.case_id,
            }

            reply = reply_class(**clean_response)

            # send via RPC to moderator
            ack_response = await self.send_message(
                reply, AgentId("Moderator", message.case_id)
            )

            if not ack_response.ok:
                self.logger.error(
                    f"Moderator rejected response: {ack_response.message}"
                )
            else:
                self.logger.info(
                    f"Successfully submitted {message.round_phase} assessment"
                )

        except Exception as e:
            self.logger.error(f"Failed to generate {message.round_phase} response: {e}")
            # send error response to moderator
            try:
                error_reply = reply_class(
                    case_id=message.case_id,
                    expert_id=self._expert_id,
                    scores={},
                    evidence={},
                    p_iraki=0.5,
                    ci_iraki=[0.4, 0.6],
                    confidence=0.1,
                    **(
                        {"differential_diagnosis": []}
                        if message.round_phase == "round1"
                        else {
                            "changes_from_round1": {},
                            "verdict": False,
                            "final_diagnosis": "Error in assessment",
                            "recommendations": [],
                        }
                    ),
                )
                await self.send_message(
                    error_reply, AgentId("Moderator", message.case_id)
                )
            except:
                pass  # if we can't send error, just log

    @message_handler
    async def handle_debate_prompt(
        self, message: DebatePrompt, ctx: MessageContext
    ) -> None:
        """Handle Round 2 debate.

        Args:
            message: Debate prompt with conflict information
            ctx: Message context
        """
        try:
            prompt_template = self._config_loader.get_prompt_template("debate")
            prompt = self._build_debate_prompt(message, prompt_template)

            llm_response = await self._vllm_client.generate_structured_response(
                prompt=prompt, response_format={"type": "json_object"}
            )

            comment = DebateComment(
                q_id=message.q_id,
                author=self._expert_id,
                text=llm_response.get("text", ""),
                citations=llm_response.get("citations", []),
                satisfied=llm_response.get("satisfied", False),
            )

            self.logger.info(f"Generated debate comment for {message.q_id}")

        except Exception as e:
            self.logger.error(f"Failed to generate debate response: {e}")

    @message_handler
    async def handle_terminate_debate(
        self, message: TerminateDebate, ctx: MessageContext
    ) -> None:
        """Handle debate termination signal.

        Args:
            message: Debate termination message
            ctx: Message context
        """
        self.logger.info(f"Exiting debate for {message.case_id}:{message.q_id}")

    async def aclose(self):
        """Clean up resources."""
        # only close if we own the client (not shared)
        if hasattr(self, "_owns_vllm_client") and self._owns_vllm_client:
            await self._vllm_client.close()

    def _build_assessment_prompt(
        self, message: QuestionnaireMsg, template: Dict
    ) -> str:
        """Build irAKI assessment prompt from template.

        Args:
            message: Questionnaire message
            template: Prompt template

        Returns:
            str: Formatted prompt
        """
        # format questions with their contexts
        formatted_questions = []
        for i, question_obj in enumerate(message.questions):
            q_text = question_obj["question"]
            context = question_obj.get("clinical_context", {})

            q_formatted = f"{i+1}. {q_text}"
            if context:
                if "supportive_evidence" in context:
                    q_formatted += f"\n   Supporting evidence: {', '.join(context['supportive_evidence'][:3])}"
                if "contradictory_evidence" in context:
                    q_formatted += f"\n   Contradictory evidence: {', '.join(context['contradictory_evidence'][:3])}"

            formatted_questions.append(q_formatted)

        # get confidence instructions
        confidence_template = self._config_loader.get_prompt_template(
            "confidence_instructions"
        )

        # build prompt from template
        prompt = template["base_template"].format(
            expert_name=self._expert_profile["name"],
            expert_experience=f"{self._expert_profile['experience_years']} years {self._expert_profile['specialty']}",
            expert_focus=", ".join(self._expert_profile.get("expertise", [])),
            case_id=message.case_id,
            patient_info=json.dumps(message.patient_info, indent=2),
            icu_summary=message.icu_summary,
            medication_history=json.dumps(message.medication_history, indent=2),
            lab_values=json.dumps(message.lab_values, indent=2),
            imaging_reports=message.imaging_reports,
            questions="\n".join(formatted_questions),
            confidence_instructions=confidence_template["ci_instructions"],
            round_phase=message.round_phase,
            specialty=self._expert_profile["specialty"],
        )

        return prompt

    def _build_debate_prompt(self, message: DebatePrompt, template: Dict) -> str:
        """Build debate prompt from template.

        Args:
            message: Debate prompt message
            template: Prompt template

        Returns:
            str: Formatted prompt
        """
        return template["base_template"].format(
            expert_name=self._expert_profile["name"],
            specialty=self._expert_profile["specialty"],
            q_id=message.q_id,
            round_no=message.round_no,
            clinical_context=message.clinical_context or {},
            minority_view=message.minority_view,
        )
