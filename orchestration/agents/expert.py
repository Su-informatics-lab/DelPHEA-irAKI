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
    """Clinical expert agent for irAKI assessment."""

    def __init__(
        self,
        expert_id: str,
        case_id: str,
        config_loader: ConfigurationLoader,
        vllm_client: VLLMClient = None,
        runtime_config=None,
    ) -> None:
        super().__init__(f"irAKI Expert {expert_id}")
        self._expert_id = expert_id
        self._case_id = case_id
        self._config_loader = config_loader

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

        # Fail fast if expert not found
        self._expert_profile = next(
            ep
            for ep in self._config_loader.expert_panel["expert_panel"]["experts"]
            if ep["id"] == expert_id
        )
        self.logger.info(
            f"Initialized expert: {self._expert_profile['name']} "
            f"({self._expert_profile['specialty']})"
        )

        self._round1_assessment: Optional[ExpertRound1Reply] = None
        self._debate_history: List[DebateComment] = []

    # =========================
    # Message Handlers
    # =========================
    @message_handler
    async def handle_questionnaire(
        self, message: QuestionnaireMsg, ctx: MessageContext
    ) -> None:
        """Handle Round 1 & 3 assessments."""
        self.logger.info(f"Received {message.round_phase} questionnaire")

        # Prompt template (fail fast)
        if message.round_phase not in ("round1", "round3"):
            raise ValueError(f"Unknown round_phase: {message.round_phase!r}")
        tpl_name = "round1" if message.round_phase == "round1" else "round3"
        prompt_template = self._config_loader.get_prompt_template(tpl_name)
        if "base_prompt" not in prompt_template:
            raise KeyError(f"Prompt '{tpl_name}' missing required 'base_prompt'")

        # Build prompt
        prompt = self._build_assessment_prompt(message, prompt_template)
        if message.round_phase == "round3" and self._round1_assessment:
            prompt = self._add_round3_context(prompt, message)

        # Generate structured response
        response_format = self._get_response_format(message.round_phase)
        llm_response = await self._vllm_client.generate_structured_response(
            prompt=prompt, response_format=response_format
        )
        self._log_first_tokens(llm_response, f"[{message.round_phase}]")

        # Validate / normalize LLM output
        cleaned = self._validate_llm_response(llm_response, message.round_phase)

        # Build reply message
        if message.round_phase == "round1":
            reply = self._create_round1_reply(cleaned, message)
            self._round1_assessment = reply
        else:
            reply = self._create_round3_reply(cleaned, message)

        # Send to moderator via RPC
        target = AgentId(type="moderator", key=message.case_id)
        method = "record_round1" if message.round_phase == "round1" else "record_round3"
        ack = await self.call_rpc(ctx, target, method, reply)
        if not ack.ok:
            raise RuntimeError(f"Moderator rejected {method}: {ack.message}")

        self.logger.info(f"Submitted {message.round_phase} assessment")

    @message_handler
    async def handle_debate_prompt(
        self, message: DebatePrompt, ctx: MessageContext
    ) -> None:
        """Handle Round 2 debate participation."""
        self.logger.info(f"Entering debate for question {message.q_id}")

        prompt_template = self._config_loader.get_prompt_template("debate")
        if "base_prompt" not in prompt_template:
            raise KeyError("Prompt 'debate' missing required 'base_prompt'")

        prompt = self._build_debate_prompt(message, prompt_template)

        response_format = {"type": "json_object"}
        llm_response = await self._vllm_client.generate_structured_response(
            prompt=prompt, response_format=response_format
        )
        self._log_first_tokens(llm_response, "[debate]")

        comment = DebateComment(
            q_id=message.q_id,
            author=self._expert_id,
            text=llm_response.get("argument", ""),
            citations=llm_response.get("citations", []),
            evidence_type=llm_response.get("evidence_type", "clinical"),
            satisfied=llm_response.get("satisfied", False),
            revised_score=llm_response.get("revised_score", None),
        )
        self._debate_history.append(comment)
        self.logger.info(
            f"Generated debate comment for {message.q_id} (satisfied: {comment.satisfied})"
        )

    @message_handler
    async def handle_terminate_debate(
        self, message: TerminateDebate, ctx: MessageContext
    ) -> None:
        self.logger.info(f"Debate terminated for {message.q_id}: {message.reason}")

    async def aclose(self):
        if self._owns_vllm_client:
            await self._vllm_client.close()

    # =========================
    # Prompt building helpers
    # =========================
    def _build_assessment_prompt(
        self, message: QuestionnaireMsg, template: Dict
    ) -> str:
        """
        Build assessment prompt using the simplified QuestionnaireMsg:
        - demographics: Dict[str, Any]
        - clinical_notes: str (aggregated notes)
        - questions: List[Dict]
        """
        # format sections
        demographics_str = str(message.demographics or {})
        clinical_notes_str = message.clinical_notes or ""
        questions_str = self._format_questions(message.questions)

        # base prompt
        prompt = template["base_prompt"].format(
            expert_name=self._expert_profile["name"],
            specialty=self._expert_profile["specialty"],
            round_phase=message.round_phase,
            case_id=message.case_id,
            demographics=demographics_str,
            clinical_notes=clinical_notes_str,
            questions=questions_str,
        )

        # round-specific instructions
        round_instructions = (
            template.get("round3_instructions", "")
            if message.round_phase == "round3"
            else template.get("round1_instructions", "")
        )
        prompt = f"{prompt}\n\nINSTRUCTIONS\n{round_instructions}"

        # optional specialty addendum
        spec_instr = template.get("specialty_instructions", {}).get(
            self._expert_profile["specialty"]
        )
        if spec_instr:
            prompt += f"\n\n{spec_instr}"

        return prompt

    def _format_patient_summary(self, message: QuestionnaireMsg) -> str:
        parts = []
        if message.patient_info:
            age = message.patient_info.get("age", "Unknown")
            gender = message.patient_info.get("gender", "Unknown")
            parts.append(f"Patient: {age}-year-old {gender}")
        if message.icu_summary:
            parts.append(f"Clinical Summary: {message.icu_summary}")
        if not message.lab_values or not message.medication_history:
            parts.append("Note: Lab values and medications pending integration")
        return "\n".join(parts)

    def _format_questions(self, questions: List[Dict]) -> str:
        out = []
        for i, q in enumerate(questions, 1):
            q_text = q["question"]
            q_id = q["id"]
            ctx = q.get("clinical_context", {})
            if ctx:
                relevance = ctx.get("relevance", "")
                evidence_type = ctx.get("evidence_type", "")
                extra = f" [Context: {relevance}; Evidence: {evidence_type}]"
            else:
                extra = ""
            out.append(f"{i}. [{q_id}] {q_text}{extra}")
        return "\n".join(out)

    def _log_first_tokens(self, response, context=""):
        text = str(response)
        self.logger.info(f"[{self._expert_id}]{context} First LLM output: {text[:200]}")

    def _add_round3_context(self, prompt: str, message: QuestionnaireMsg) -> str:
        sections = [prompt]
        if self._round1_assessment:
            sections.append(
                f"\n\n=== YOUR ROUND 1 ASSESSMENT ===\n"
                f"P(irAKI): {self._round1_assessment.p_iraki:.2f}\n"
                f"Confidence: {self._round1_assessment.confidence:.2f}\n"
                f"Primary Diagnosis: {self._round1_assessment.primary_diagnosis}\n"
            )
        if message.debate_summary:
            sections.append(f"\n=== DEBATE INSIGHTS ===\n{message.debate_summary}")
        sections.append(
            "\n\n=== ROUND 3 INSTRUCTIONS ===\n"
            "1. Reflect on debate insights\n"
            "2. Update your assessment\n"
            "3. Provide final verdict with confidence\n"
            "4. Explain changes from Round 1\n"
            "5. Provide specific recommendations"
        )
        return "\n".join(sections)

    # =========================
    # Response processing
    # =========================
    def _get_response_format(self, round_phase: str) -> Dict:
        if round_phase == "round1":
            return {
                "type": "json_object",
                "schema": {
                    "scores": "object question_id:int(1-10)",
                    "evidence": "object question_id:str",
                    "clinical_reasoning": "string",
                    "p_iraki": "float 0-1",
                    "ci_lower": "float 0-1",
                    "ci_upper": "float 0-1",
                    "confidence": "float 0-1",
                    "differential_diagnosis": "array string",
                    "primary_diagnosis": "string",
                    "specialty_notes": "optional string",
                },
            }
        return {
            "type": "json_object",
            "schema": {
                "scores": "object question_id:int(1-10)",
                "evidence": "object question_id:str",
                "p_iraki": "float 0-1",
                "ci_lower": "float 0-1",
                "ci_upper": "float 0-1",
                "confidence": "float 0-1",
                "changes_from_round1": "object",
                "debate_influence": "string",
                "verdict": "boolean",
                "final_diagnosis": "string",
                "confidence_in_verdict": "float 0-1",
                "recommendations": "array string",
                "biopsy_recommendation": "optional string",
                "steroid_recommendation": "optional string",
                "ici_rechallenge_risk": "optional string",
            },
        }

    def _validate_llm_response(self, response: Dict, round_phase: str) -> Dict:
        # CI tuple
        if "ci_lower" in response and "ci_upper" in response:
            lo = float(response.pop("ci_lower"))
            hi = float(response.pop("ci_upper"))
            if lo > hi:
                lo, hi = hi, lo
            response["ci_iraki"] = (max(0.0, min(1.0, lo)), max(0.0, min(1.0, hi)))
        # p_iraki / confidence
        if "p_iraki" in response:
            response["p_iraki"] = max(0.0, min(1.0, float(response["p_iraki"])))
        if "confidence" in response:
            response["confidence"] = max(0.0, min(1.0, float(response["confidence"])))
        # scores 1-10
        if "scores" in response:
            cleaned = {}
            for qid, sc in response["scores"].items():
                try:
                    cleaned[qid] = max(1, min(10, int(float(sc))))
                except Exception:
                    cleaned[qid] = 5
            response["scores"] = cleaned
        # required defaults
        if round_phase == "round1":
            if not response.get("differential_diagnosis"):
                response["differential_diagnosis"] = ["irAKI", "ATN", "Prerenal AKI"]
            if not response.get("primary_diagnosis"):
                response["primary_diagnosis"] = response["differential_diagnosis"][0]
        else:
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
