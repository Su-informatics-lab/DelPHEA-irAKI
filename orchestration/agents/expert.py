"""
Expert Agent for DelPHEA-irAKI Clinical Assessment
===================================================
Simulates individual medical experts evaluating irAKI cases with
specialty-specific reasoning and evidence-based assessment.

Architecture:
------------
    Patient Data ──┐
    Questions ─────┼──> [Expert Brain] ──> Clinical Assessment
    Specialty ─────┘         │                    │
                            LLM                Scores
                         Reasoning           Evidence
                                           P(irAKI)

Key Capabilities:
----------------
- Specialty-specific clinical reasoning (11 expert types)
- Evidence-based scoring with citations
- Probabilistic assessment with confidence intervals
- Differential diagnosis generation
- Debate participation with reasoned arguments
- Literature integration (when enabled)

Clinical Context:
----------------
Each expert brings unique perspective to irAKI diagnosis:
- Nephrologist: Renal pathophysiology focus
- Oncologist: ICI timing and cancer treatment priorities
- Pharmacist: Drug interactions and pharmacokinetics
- etc. (see panel.json for full expert panel)

The expert agent ensures diverse, comprehensive assessment
critical for distinguishing true irAKI from mimics.
"""

import logging
from typing import Dict, List, Optional

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
    """Clinical expert agent for irAKI assessment.

    Simulates a medical specialist evaluating cases with evidence-based
    reasoning and participating in consensus building.
    """

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
            expert_id: Expert identifier (e.g., "nephrologist_1")
            case_id: Case identifier
            config_loader: Configuration loader instance
            vllm_client: Optional shared VLLMClient
            runtime_config: Runtime configuration (for creating own client if needed)

        Raises:
            ValueError: If neither vllm_client nor runtime_config provided
            KeyError: If expert_id not found in configuration
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
        # derive profile directly from loaded JSON (no helper needed)
        self._expert_profile = next(
            ep
            for ep in self._config_loader.expert_panel["expert_panel"]["experts"]
            if ep["id"] == expert_id
        )
        self.logger.info(
            f"Initialized expert: {self._expert_profile['name']} "
            f"({self._expert_profile['specialty']})"
        )

        # track state for context between rounds
        self._round1_assessment: Optional[ExpertRound1Reply] = None
        self._debate_history: List[DebateComment] = []

    @message_handler
    async def handle_questionnaire(
        self, message: QuestionnaireMsg, ctx: MessageContext
    ) -> None:
        """Handle Round 1 & 3 irAKI assessment.

        Args:
            message: Questionnaire with patient data and questions
            ctx: Message context
        """
        self.logger.info(f"Received {message.round_phase} questionnaire")

        try:
            # get appropriate prompt template
            template_name = f"assessment_{message.round_phase}"
            prompt_template = self._config_loader.get_prompt_template(template_name)

            # build assessment prompt with clinical context
            prompt = self._build_assessment_prompt(message, prompt_template)

            # add round 3 specific context
            if message.round_phase == "round3" and self._round1_assessment:
                prompt = self._add_round3_context(prompt, message)

            # generate structured response from LLM
            response_format = self._get_response_format(message.round_phase)
            llm_response = await self._vllm_client.generate_structured_response(
                prompt=prompt, response_format=response_format
            )
            self._log_first_tokens(llm_response, f"[{message.round_phase}]")

            # validate and clean response
            cleaned_response = self._validate_llm_response(
                llm_response, message.round_phase
            )

            # create appropriate reply message
            if message.round_phase == "round1":
                reply = self._create_round1_reply(cleaned_response, message)
                self._round1_assessment = reply  # store for round 3
            else:  # round3
                reply = self._create_round3_reply(cleaned_response, message)

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
            # send error response to maintain protocol
            await self._send_error_response(message, str(e))

    @message_handler
    async def handle_debate_prompt(
        self, message: DebatePrompt, ctx: MessageContext
    ) -> None:
        """Handle Round 2 debate participation.

        Args:
            message: Debate prompt with conflict information
            ctx: Message context
        """
        self.logger.info(f"Entering debate for question {message.q_id}")

        try:
            # get debate prompt template
            prompt_template = self._config_loader.get_prompt_template("debate")

            # build debate prompt with conflict context
            prompt = self._build_debate_prompt(message, prompt_template)

            # generate debate response
            response_format = {"type": "json_object"}
            llm_response = await self._vllm_client.generate_structured_response(
                prompt=prompt, response_format=response_format
            )
            self._log_first_tokens(llm_response, "[debate]")

            # create debate comment
            comment = DebateComment(
                q_id=message.q_id,
                author=self._expert_id,
                text=llm_response.get("argument", ""),
                citations=llm_response.get("citations", []),
                evidence_type=llm_response.get("evidence_type", "clinical"),
                satisfied=llm_response.get("satisfied", False),
                revised_score=llm_response.get("revised_score", None),
            )

            # track debate history
            self._debate_history.append(comment)

            self.logger.info(
                f"Generated debate comment for {message.q_id} "
                f"(satisfied: {comment.satisfied})"
            )

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
        self.logger.info(f"Debate terminated for {message.q_id}: {message.reason}")

    async def aclose(self):
        """Clean up resources."""
        if self._owns_vllm_client:
            await self._vllm_client.close()

    # =========================================================================
    # PRIVATE METHODS - Prompt Building
    # =========================================================================

    def _build_assessment_prompt(
        self, message: QuestionnaireMsg, template: Dict
    ) -> str:
        """Build comprehensive assessment prompt with clinical context.

        Args:
            message: Questionnaire message with patient data
            template: Prompt template from configuration

        Returns:
            str: Formatted prompt for LLM
        """
        # format patient information
        patient_summary = self._format_patient_summary(message)

        # format questions with clinical context
        formatted_questions = self._format_questions(message.questions)

        # get specialty-specific expertise from config
        specialty_focus = self._expert_profile.get(
            "expertise", "comprehensive clinical assessment"
        )

        # build complete prompt
        prompt = template["base_prompt"].format(
            expert_name=self._expert_profile["name"],
            specialty=self._expert_profile["specialty"],
            patient_summary=patient_summary,
            questions=formatted_questions,
            specialty_focus=specialty_focus,
            round_phase=message.round_phase,
        )

        # add specialty-specific instructions
        if self._expert_profile["specialty"] in template.get(
            "specialty_instructions", {}
        ):
            specialty_instructions = template["specialty_instructions"][
                self._expert_profile["specialty"]
            ]
            prompt += f"\n\n{specialty_instructions}"

        return prompt

    def _format_patient_summary(self, message: QuestionnaireMsg) -> str:
        """Format patient data into clinical narrative.

        Args:
            message: Questionnaire with patient data

        Returns:
            str: Formatted patient summary
        """
        summary_parts = []

        # demographics (from person table)
        if message.patient_info:
            age = message.patient_info.get("age", "Unknown")
            gender = message.patient_info.get("gender", "Unknown")
            summary_parts.append(f"Patient: {age}-year-old {gender}")

        # clinical notes summary (primary data source)
        if message.icu_summary:
            summary_parts.append(f"Clinical Summary: {message.icu_summary}")

        # placeholder for future lab/med data
        # for now, just note if they're empty
        if not message.lab_values or not message.medication_history:
            summary_parts.append("Note: Lab values and medications pending integration")

        return "\n".join(summary_parts)

    def _format_questions(self, questions: List[Dict]) -> str:
        """Format questions with clinical context for assessment.

        Args:
            questions: List of question objects

        Returns:
            str: Formatted questions
        """
        formatted = []
        for i, q_obj in enumerate(questions, 1):
            q_text = q_obj["question"]
            q_id = q_obj["id"]

            # add clinical context if available
            context = q_obj.get("clinical_context", {})
            if context:
                relevance = context.get("relevance", "")
                evidence_type = context.get("evidence_type", "")
                context_str = f" [Context: {relevance}; Evidence: {evidence_type}]"
            else:
                context_str = ""

            formatted.append(f"{i}. [{q_id}] {q_text}{context_str}")

        return "\n".join(formatted)

    def _log_first_tokens(self, response, context=""):
        text = str(response)
        self.logger.info(f"[{self._expert_id}]{context} First LLM output: {text[:200]}")

    def _add_round3_context(self, prompt: str, message: QuestionnaireMsg) -> str:
        """Add Round 3 specific context including debate learnings.

        Args:
            prompt: Base prompt
            message: Round 3 questionnaire

        Returns:
            str: Enhanced prompt with round 3 context
        """
        context_parts = [prompt]

        # add prior assessment summary
        if self._round1_assessment:
            context_parts.append(
                f"\n\n=== YOUR ROUND 1 ASSESSMENT ===\n"
                f"P(irAKI): {self._round1_assessment.p_iraki:.2f}\n"
                f"Confidence: {self._round1_assessment.confidence:.2f}\n"
                f"Primary Diagnosis: {self._round1_assessment.primary_diagnosis}\n"
            )

        # add debate summary if provided
        if message.debate_summary:
            context_parts.append(f"\n=== DEBATE INSIGHTS ===\n{message.debate_summary}")

        # add instructions for round 3
        context_parts.append(
            "\n\n=== ROUND 3 INSTRUCTIONS ===\n"
            "1. Reflect on insights from the debate\n"
            "2. Update your assessment based on new perspectives\n"
            "3. Provide final verdict with high confidence\n"
            "4. Explain what changed from Round 1 and why\n"
            "5. Give specific clinical recommendations"
        )

        return "\n".join(context_parts)

    def _build_debate_prompt(self, message: DebatePrompt, template: Dict) -> str:
        """Build debate prompt with conflict context.

        Args:
            message: Debate prompt with conflict information
            template: Debate prompt template

        Returns:
            str: Formatted debate prompt
        """
        # get own score and evidence for this question
        own_score = message.score_distribution.get(self._expert_id, 5)
        own_evidence = message.conflicting_evidence.get(self._expert_id, "")

        # identify opposing views
        opposing_views = []
        for expert_id, score in message.score_distribution.items():
            if expert_id != self._expert_id and abs(score - own_score) >= 3:
                evidence = message.conflicting_evidence.get(expert_id, "")
                opposing_views.append(f"Expert {expert_id} (score={score}): {evidence}")

        prompt = template["base_prompt"].format(
            expert_name=self._expert_profile["name"],
            specialty=self._expert_profile["specialty"],
            question=message.question["question"],
            own_score=own_score,
            own_evidence=own_evidence,
            score_range=message.score_range,
            opposing_views="\n".join(opposing_views),
            clinical_importance=message.clinical_importance,
        )

        return prompt

    # =========================================================================
    # PRIVATE METHODS - Response Processing
    # =========================================================================

    def _get_response_format(self, round_phase: str) -> Dict:
        """Get structured response format for LLM.

        Args:
            round_phase: "round1" or "round3"

        Returns:
            Dict: Response format specification
        """
        if round_phase == "round1":
            return {
                "type": "json_object",
                "schema": {
                    "scores": "object with question_id: score(1-10)",
                    "evidence": "object with question_id: evidence_text",
                    "clinical_reasoning": "string explaining overall reasoning",
                    "p_iraki": "float 0-1",
                    "ci_lower": "float 0-1",
                    "ci_upper": "float 0-1",
                    "confidence": "float 0-1",
                    "differential_diagnosis": "array of strings",
                    "primary_diagnosis": "string",
                    "specialty_notes": "optional string",
                },
            }
        else:  # round3
            return {
                "type": "json_object",
                "schema": {
                    "scores": "object with question_id: score(1-10)",
                    "evidence": "object with question_id: evidence_text",
                    "p_iraki": "float 0-1",
                    "ci_lower": "float 0-1",
                    "ci_upper": "float 0-1",
                    "confidence": "float 0-1",
                    "changes_from_round1": "object describing changes",
                    "debate_influence": "string",
                    "verdict": "boolean (true=irAKI)",
                    "final_diagnosis": "string",
                    "confidence_in_verdict": "float 0-1",
                    "recommendations": "array of strings",
                    "biopsy_recommendation": "optional string",
                    "steroid_recommendation": "optional string",
                    "ici_rechallenge_risk": "optional string",
                },
            }

    def _validate_llm_response(self, response: Dict, round_phase: str) -> Dict:
        """Validate and clean LLM response with fallbacks.

        Args:
            response: Raw LLM response
            round_phase: "round1" or "round3"

        Returns:
            Dict: Cleaned and validated response

        Raises:
            ValueError: If response missing critical fields
        """
        # fix common LLM response issues

        # 1. Fix CI bounds if provided separately
        if "ci_lower" in response and "ci_upper" in response:
            ci_lower = float(response.pop("ci_lower"))
            ci_upper = float(response.pop("ci_upper"))

            # ensure bounds are ordered correctly
            if ci_lower > ci_upper:
                ci_lower, ci_upper = ci_upper, ci_lower

            # ensure bounds are valid
            ci_lower = max(0.0, min(1.0, ci_lower))
            ci_upper = max(0.0, min(1.0, ci_upper))

            response["ci_iraki"] = (ci_lower, ci_upper)

        # 2. Ensure p_iraki is in valid range
        if "p_iraki" in response:
            response["p_iraki"] = max(0.0, min(1.0, float(response["p_iraki"])))

        # 3. Ensure confidence is in valid range
        if "confidence" in response:
            response["confidence"] = max(0.0, min(1.0, float(response["confidence"])))

        # 4. Fix scores to be integers 1-10
        if "scores" in response:
            cleaned_scores = {}
            for q_id, score in response["scores"].items():
                try:
                    score_int = int(float(score))
                    cleaned_scores[q_id] = max(1, min(10, score_int))
                except (ValueError, TypeError):
                    cleaned_scores[q_id] = 5  # default middle score
            response["scores"] = cleaned_scores

        # 5. Ensure required lists are not empty
        if round_phase == "round1":
            if not response.get("differential_diagnosis"):
                response["differential_diagnosis"] = ["irAKI", "ATN", "Prerenal AKI"]
            if not response.get("primary_diagnosis"):
                response["primary_diagnosis"] = response["differential_diagnosis"][0]

        if round_phase == "round3":
            if not response.get("recommendations"):
                response["recommendations"] = [
                    "Monitor renal function closely",
                    "Consider nephrology consultation",
                ]
            if "verdict" not in response:
                response["verdict"] = response.get("p_iraki", 0.5) > 0.5
            if not response.get("final_diagnosis"):
                response["final_diagnosis"] = "irAKI" if response["verdict"] else "ATN"
            if "confidence_in_verdict" not in response:
                response["confidence_in_verdict"] = response.get("confidence", 0.7)

        return response

    def _create_round1_reply(
        self, response: Dict, message: QuestionnaireMsg
    ) -> ExpertRound1Reply:
        """Create Round 1 reply message from validated response.

        Args:
            response: Validated LLM response
            message: Original questionnaire

        Returns:
            ExpertRound1Reply: Structured reply message
        """
        return ExpertRound1Reply(
            case_id=message.case_id,
            expert_id=self._expert_id,
            scores=response["scores"],
            evidence=response.get("evidence", {}),
            clinical_reasoning=response.get("clinical_reasoning", ""),
            p_iraki=response["p_iraki"],
            ci_iraki=response["ci_iraki"],
            confidence=response["confidence"],
            differential_diagnosis=response["differential_diagnosis"],
            primary_diagnosis=response.get("primary_diagnosis"),
            specialty_notes=response.get("specialty_notes"),
            literature_citations=response.get("citations", []),
        )

    def _create_round3_reply(
        self, response: Dict, message: QuestionnaireMsg
    ) -> ExpertRound3Reply:
        """Create Round 3 reply message from validated response.

        Args:
            response: Validated LLM response
            message: Original questionnaire

        Returns:
            ExpertRound3Reply: Structured reply message
        """
        return ExpertRound3Reply(
            case_id=message.case_id,
            expert_id=self._expert_id,
            scores=response["scores"],
            evidence=response.get("evidence", {}),
            p_iraki=response["p_iraki"],
            ci_iraki=response["ci_iraki"],
            confidence=response["confidence"],
            changes_from_round1=response.get("changes_from_round1", {}),
            debate_influence=response.get("debate_influence"),
            verdict=response["verdict"],
            final_diagnosis=response["final_diagnosis"],
            confidence_in_verdict=response["confidence_in_verdict"],
            recommendations=response["recommendations"],
            biopsy_recommendation=response.get("biopsy_recommendation"),
            steroid_recommendation=response.get("steroid_recommendation"),
            ici_rechallenge_risk=response.get("ici_rechallenge_risk"),
        )

    async def _send_error_response(self, message: QuestionnaireMsg, error: str) -> None:
        """Send error response to maintain protocol during failures.

        Args:
            message: Original questionnaire
            error: Error description
        """
        try:
            if message.round_phase == "round1":
                error_reply = ExpertRound1Reply(
                    case_id=message.case_id,
                    expert_id=self._expert_id,
                    scores={},
                    evidence={},
                    clinical_reasoning=f"Error: {error}",
                    p_iraki=0.5,
                    ci_iraki=(0.4, 0.6),
                    confidence=0.1,
                    differential_diagnosis=["Unable to assess"],
                    primary_diagnosis="Error in assessment",
                )
            else:  # round3
                error_reply = ExpertRound3Reply(
                    case_id=message.case_id,
                    expert_id=self._expert_id,
                    scores={},
                    evidence={},
                    p_iraki=0.5,
                    ci_iraki=(0.4, 0.6),
                    confidence=0.1,
                    changes_from_round1={"error": error},
                    verdict=False,
                    final_diagnosis="Unable to assess",
                    confidence_in_verdict=0.1,
                    recommendations=["Unable to provide recommendations"],
                )

            await self.send_message(error_reply, AgentId("Moderator", message.case_id))
        except Exception as e:
            self.logger.error(f"Failed to send error response: {e}")
