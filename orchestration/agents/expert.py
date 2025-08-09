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
    """Clinical expert agent for irAKI assessment (YAGNI version)."""

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

        # fail fast if expert not found
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
        """Handle Round 1 & 3 assessments with strict schema."""
        self.logger.info(f"Received {message.round_phase} questionnaire")

        # strict inputs (YAGNI)
        if not hasattr(message, "demographics"):
            raise AttributeError("QuestionnaireMsg.demographics is required")
        if not hasattr(message, "clinical_notes_text"):
            raise AttributeError("QuestionnaireMsg.clinical_notes_text is required")

        if message.round_phase not in ("round1", "round3"):
            raise ValueError(f"Unknown round_phase: {message.round_phase!r}")

        tpl_name = "round1" if message.round_phase == "round1" else "round3"
        template = self._config_loader.get_prompt_template(tpl_name)
        base = template.get("base_prompt")
        if not base:
            raise KeyError(f"Prompt '{tpl_name}' missing required 'base_prompt'")

        # build prompt (only the two inputs + questions)
        prompt = base.format(
            expert_name=self._expert_profile["name"],
            specialty=self._expert_profile["specialty"],
            round_phase=message.round_phase,
            case_id=message.case_id,
            demographics=str(message.demographics),
            clinical_notes=message.clinical_notes_text,
            questions=self._format_questions(message.questions),
        )

        # per-round extra instructions
        instructions = (
            template.get("round3_instructions", "")
            if message.round_phase == "round3"
            else template.get("round1_instructions", "")
        )
        if instructions:
            prompt = f"{prompt}\n\nINSTRUCTIONS\n{instructions}"

        # generate structured response
        response_format = self._get_response_format(message.round_phase)
        llm_response = await self._vllm_client.generate_structured_response(
            prompt=prompt, response_format=response_format
        )
        self._log_first_tokens(llm_response, f"[{message.round_phase}]")

        cleaned = self._validate_llm_response(llm_response, message.round_phase)

        # build reply pydantic object
        if message.round_phase == "round1":
            reply = self._create_round1_reply(cleaned, message)
            self._round1_assessment = reply
            method = "record_round1"
        else:
            reply = self._create_round3_reply(cleaned, message)
            method = "record_round3"

        # RPC to moderator (strict)
        target = AgentId(type="moderator", key=message.case_id)
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

        template = self._config_loader.get_prompt_template("debate")
        base = template.get("base_prompt")
        if not base:
            raise KeyError("Prompt 'debate' missing required 'base_prompt'")

        # build debate prompt
        own_score = message.score_distribution.get(self._expert_id, 5)
        own_evidence = message.conflicting_evidence.get(self._expert_id, "")
        opposing = []
        for eid, sc in message.score_distribution.items():
            if eid != self._expert_id and abs(sc - own_score) >= 3:
                opposing.append(
                    f"Expert {eid} (score={sc}): {message.conflicting_evidence.get(eid, '')}"
                )

        prompt = base.format(
            expert_name=self._expert_profile["name"],
            specialty=self._expert_profile["specialty"],
            question=message.question["question"],
            own_score=own_score,
            own_evidence=own_evidence,
            score_range=message.score_range,
            opposing_views="\n".join(opposing),
            clinical_importance=message.clinical_importance,
        )

        llm_response = await self._vllm_client.generate_structured_response(
            prompt=prompt, response_format={"type": "json_object"}
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
    # Helpers
    # =========================
    def _format_questions(self, questions: List[Dict]) -> str:
        lines = []
        for i, q in enumerate(questions, 1):
            qid = q["id"]
            qtext = q["question"]
            lines.append(f"{i}. [{qid}] {qtext}")
        return "\n".join(lines)

    def _log_first_tokens(self, response, context=""):
        text = str(response)
        self.logger.info(f"[{self._expert_id}]{context} First LLM output: {text[:200]}")

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

    def _validate_llm_response(self, resp: Dict, round_phase: str) -> Dict:
        # CI -> tuple
        lo = float(resp.pop("ci_lower")) if "ci_lower" in resp else 0.25
        hi = float(resp.pop("ci_upper")) if "ci_upper" in resp else 0.75
        if lo > hi:
            lo, hi = hi, lo
        resp["ci_iraki"] = (max(0.0, min(1.0, lo)), max(0.0, min(1.0, hi)))

        # clamp probabilities
        if "p_iraki" in resp:
            resp["p_iraki"] = max(0.0, min(1.0, float(resp["p_iraki"])))
        if "confidence" in resp:
            resp["confidence"] = max(0.0, min(1.0, float(resp["confidence"])))

        # normalize scores
        if "scores" in resp:
            cleaned = {}
            for qid, sc in resp["scores"].items():
                cleaned[qid] = max(1, min(10, int(float(sc))))
            resp["scores"] = cleaned

        # minimal required defaults by round
        if round_phase == "round1":
            resp.setdefault("differential_diagnosis", ["irAKI", "ATN", "Prerenal AKI"])
            resp.setdefault("primary_diagnosis", resp["differential_diagnosis"][0])
        else:
            resp.setdefault(
                "recommendations",
                ["Monitor renal function closely", "Consider nephrology consult"],
            )
            resp.setdefault("verdict", resp.get("p_iraki", 0.5) > 0.5)
            resp.setdefault("final_diagnosis", "irAKI" if resp["verdict"] else "ATN")
            resp.setdefault("confidence_in_verdict", resp.get("confidence", 0.7))
        return resp

    def _create_round1_reply(
        self, resp: Dict, message: QuestionnaireMsg
    ) -> ExpertRound1Reply:
        return ExpertRound1Reply(
            case_id=message.case_id,
            expert_id=self._expert_id,
            scores=resp["scores"],
            evidence=resp.get("evidence", {}),
            clinical_reasoning=resp.get("clinical_reasoning", ""),
            p_iraki=resp["p_iraki"],
            ci_iraki=resp["ci_iraki"],
            confidence=resp["confidence"],
            differential_diagnosis=resp["differential_diagnosis"],
            primary_diagnosis=resp.get("primary_diagnosis"),
            specialty_notes=resp.get("specialty_notes"),
            literature_citations=resp.get("citations", []),
        )

    def _create_round3_reply(
        self, resp: Dict, message: QuestionnaireMsg
    ) -> ExpertRound3Reply:
        return ExpertRound3Reply(
            case_id=message.case_id,
            expert_id=self._expert_id,
            scores=resp["scores"],
            evidence=resp.get("evidence", {}),
            p_iraki=resp["p_iraki"],
            ci_iraki=resp["ci_iraki"],
            confidence=resp["confidence"],
            changes_from_round1=resp.get("changes_from_round1", {}),
            debate_influence=resp.get("debate_influence"),
            verdict=resp["verdict"],
            final_diagnosis=resp["final_diagnosis"],
            confidence_in_verdict=resp["confidence_in_verdict"],
            recommendations=resp["recommendations"],
            biopsy_recommendation=resp.get("biopsy_recommendation"),
            steroid_recommendation=resp.get("steroid_recommendation"),
            ici_rechallenge_risk=resp.get("ici_rechallenge_risk"),
        )
