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
"""
expert agent for delphea-iraki clinical assessment

simulates individual medical experts evaluating iraki cases with
specialty-specific reasoning and evidence-based assessment.
"""
"""
Expert Agent for DelPHEA-irAKI Clinical Assessment

Simulates individual medical experts evaluating irAKI cases with
specialty-specific reasoning and evidence-based assessment.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

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

# word-to-probability mapping for loose model outputs
_WORD_CONF: Dict[str, float] = {
    "very low": 0.1,
    "low": 0.25,
    "somewhat low": 0.35,
    "moderate": 0.5,
    "medium": 0.5,
    "somewhat high": 0.65,
    "high": 0.8,
    "very high": 0.9,
    "certain": 0.98,
}

# -------------------------
# coercion + normalization
# -------------------------


def _as_float(x: Any) -> float:
    """coerce common textual numerics/hedges into float in [0,1] when sensible."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in _WORD_CONF:
            return _WORD_CONF[s]
        # "65%" -> 0.65  or "0.65" -> 0.65
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*%?\s*$", s)
        if m:
            v = float(m.group(1))
            return v / 100.0 if "%" in s else v
        # "0.6-0.8" -> midpoint
        if "-" in s:
            try:
                a, b = s.split("-", 1)
                a, b = float(a), float(b)
                return (a + b) / 2.0
            except Exception:
                pass
    raise ValueError(f"cannot coerce to float: {x!r}")


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _as_ci2(x: Any) -> Tuple[float, float]:
    """accept [l,u], (l,u), {'lower':l,'upper':u}, or 'l-u'."""
    if isinstance(x, (list, tuple)) and len(x) == 2:
        l, u = _clip01(_as_float(x[0])), _clip01(_as_float(x[1]))
        return (l, u) if l <= u else (u, l)
    if isinstance(x, dict) and "lower" in x and "upper" in x:
        l, u = _clip01(_as_float(x["lower"])), _clip01(_as_float(x["upper"]))
        return (l, u) if l <= u else (u, l)
    if isinstance(x, str) and "-" in x:
        a, b = x.split("-", 1)
        l, u = _clip01(_as_float(a)), _clip01(_as_float(b))
        return (l, u) if l <= u else (u, l)
    raise ValueError(f"cannot coerce to ci: {x!r}")


def normalize_round1(
    raw: Dict[str, Any],
    question_ids: List[str],
    case_id: str,
    expert_id: str,
) -> Dict[str, Any]:
    """normalize loose llm json to match ExpertRound1Reply schema."""
    out: Dict[str, Any] = {"case_id": case_id, "expert_id": expert_id}

    # scores: list -> dict via question_ids, or passthrough dict
    scores = raw.get("scores")
    if isinstance(scores, list):
        out["scores"] = {
            qid: int(float(scores[i]))
            for i, qid in enumerate(question_ids)
            if i < len(scores)
        }
    elif isinstance(scores, dict):
        out["scores"] = {str(k): int(float(v)) for k, v in scores.items()}
    else:
        raise ValueError("scores must be list or dict")

    # evidence: not used → keep empty
    out["evidence"] = {}

    # reasoning
    out["clinical_reasoning"] = str(
        raw.get("clinical_reasoning") or raw.get("reasoning") or ""
    )

    # p_iraki and ci
    p = raw.get("p_iraki", "0.5")
    try:
        out["p_iraki"] = _clip01(_as_float(p))
    except Exception:
        out["p_iraki"] = 0.5

    if "ci_iraki" in raw:
        try:
            out["ci_iraki"] = _as_ci2(raw["ci_iraki"])
        except Exception:
            out["ci_iraki"] = (
                max(0.0, out["p_iraki"] - 0.2),
                min(1.0, out["p_iraki"] + 0.2),
            )
    else:
        # some prompts may emit separate fields
        lo = raw.get("ci_lower")
        up = raw.get("ci_upper")
        if lo is not None and up is not None:
            try:
                out["ci_iraki"] = _as_ci2([lo, up])
            except Exception:
                out["ci_iraki"] = (
                    max(0.0, out["p_iraki"] - 0.2),
                    min(1.0, out["p_iraki"] + 0.2),
                )
        else:
            out["ci_iraki"] = (
                max(0.0, out["p_iraki"] - 0.2),
                min(1.0, out["p_iraki"] + 0.2),
            )

    # confidence
    c = raw.get("confidence", "moderate")
    try:
        out["confidence"] = _clip01(_as_float(c))
    except Exception:
        out["confidence"] = 0.5

    # differential
    ddx = raw.get("differential_diagnosis") or raw.get("differential") or []
    if isinstance(ddx, str):
        ddx = [ddx]
    out["differential_diagnosis"] = [
        str(x) for x in ddx if isinstance(x, (str, int, float))
    ]
    if not out["differential_diagnosis"]:
        out["differential_diagnosis"] = ["irAKI", "ATN", "Prerenal AKI"]

    out["primary_diagnosis"] = (
        raw.get("primary_diagnosis") or out["differential_diagnosis"][0]
    )

    # optional notes/citations
    if "specialty_notes" in raw:
        out["specialty_notes"] = str(raw["specialty_notes"])
    if isinstance(raw.get("literature_citations"), list):
        out["literature_citations"] = [str(x) for x in raw["literature_citations"]]
    if isinstance(raw.get("citations"), list) and "literature_citations" not in out:
        out["literature_citations"] = [str(x) for x in raw["citations"]]

    return out


def _normalize_scores_like(
    val: Any, question_ids: Optional[List[str]]
) -> Dict[str, int]:
    """normalize scores when model returns a list."""
    if isinstance(val, dict):
        return {str(k): int(float(v)) for k, v in val.items()}
    if isinstance(val, list) and question_ids:
        return {
            qid: int(float(val[i]))
            for i, qid in enumerate(question_ids)
            if i < len(val)
        }
    raise ValueError("scores must be dict or list with question_ids available")


# -------------------------
# expert agent
# -------------------------


class irAKIExpertAgent(RoutedAgent):
    """clinical expert agent for iraki assessment."""

    def __init__(
        self,
        expert_id: str,
        case_id: str,
        config_loader: ConfigurationLoader,
        vllm_client: Optional[VLLMClient] = None,
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
                    "either vllm_client or runtime_config must be provided"
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

    # -------------------------
    # message handlers
    # -------------------------

    @message_handler
    async def handle_questionnaire(
        self, message: QuestionnaireMsg, ctx: MessageContext
    ) -> None:
        """handle round 1 and round 3 assessments with strict structured output."""
        self.logger.info(f"Received {message.round_phase} questionnaire")

        if message.round_phase not in ("round1", "round3"):
            raise ValueError(f"unknown round_phase: {message.round_phase!r}")

        tpl_name = "round1" if message.round_phase == "round1" else "round3"
        template = self._config_loader.get_prompt_template(tpl_name)
        base = template.get("base_prompt")
        if not base:
            raise KeyError(f"prompt '{tpl_name}' missing required 'base_prompt'")

        # extract deterministic inputs
        demographics_str = self._extract_demographics(message)
        notes_str = self._extract_notes(message)
        questions_str = self._format_questions(message.questions)

        # build prompt
        prompt = base.format(
            expert_name=self._expert_profile["name"],
            specialty=self._expert_profile["specialty"],
            round_phase=message.round_phase,
            case_id=message.case_id,
            demographics=demographics_str,
            clinical_notes=notes_str,
            questions=questions_str,
        )

        # per-round instructions (optional)
        instructions = (
            template.get("round3_instructions", "")
            if message.round_phase == "round3"
            else template.get("round1_instructions", "")
        )
        if instructions:
            prompt = f"{prompt}\n\nINSTRUCTIONS\n{instructions}"

        # openai-style json response
        response_format = self._get_response_format(message.round_phase)
        llm_response = await self._vllm_client.generate_structured_response(
            prompt=prompt, response_format=response_format
        )
        self._log_first_tokens(llm_response, f"[{message.round_phase}]")

        q_ids = [q["id"] for q in message.questions]

        # normalize/validate and send to moderator
        target = AgentId(type="moderator", key=message.case_id)

        if message.round_phase == "round1":
            cleaned = normalize_round1(
                raw=llm_response,
                question_ids=q_ids,
                case_id=message.case_id,
                expert_id=self._expert_id,
            )
            reply = self._create_round1_reply(cleaned, message)
            self._round1_assessment = reply
            ack = await self.send_message(reply, target)
        else:
            cleaned = self._validate_llm_response(
                llm_response, round_phase="round3", question_ids=q_ids
            )
            reply = self._create_round3_reply(cleaned, message)
            ack = await self.send_message(reply, target)

        if not ack or not getattr(ack, "ok", False):
            raise RuntimeError(
                f"moderator rejected {message.round_phase}: {getattr(ack, 'message', 'no ack')}"
            )

        self.logger.info(f"Submitted {message.round_phase} assessment")

    @message_handler
    async def handle_debate_prompt(
        self, message: DebatePrompt, ctx: MessageContext
    ) -> None:
        """handle round 2 debate participation."""
        self.logger.info(f"Entering debate for question {message.q_id}")

        template = self._config_loader.get_prompt_template("debate")
        base = template.get("base_prompt")
        if not base:
            raise KeyError("prompt 'debate' missing required 'base_prompt'")

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
        if getattr(self, "_owns_vllm_client", False):
            await self._vllm_client.close()

    # -------------------------
    # extractors (deterministic)
    # -------------------------

    def _extract_demographics(self, message: QuestionnaireMsg) -> str:
        """prefer message.patient_info, else message.demographics, else empty."""
        info = getattr(message, "patient_info", None) or getattr(
            message, "demographics", None
        )
        if not info:
            return ""
        age = info.get("age", "Unknown")
        gender = info.get("gender", "Unknown")
        race = info.get("race", "Unknown")
        eth = info.get("ethnicity", "Unknown")
        return f"Age: {age} | Gender: {gender} | Race: {race} | Ethnicity: {eth}"

    def _extract_notes(self, message: QuestionnaireMsg, max_chars: int = 4000) -> str:
        """prefer clinical_notes_text, else join clinical_notes[].text, else patient_summary."""
        txt = getattr(message, "clinical_notes_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt[:max_chars]

        notes = getattr(message, "clinical_notes", None)
        if isinstance(notes, list) and notes:
            joined = "\n\n".join(
                str(n.get("text", "")) for n in notes if isinstance(n, dict)
            )
            return joined[:max_chars]

        summ = getattr(message, "patient_summary", None)
        if isinstance(summ, str) and summ.strip():
            return summ[:max_chars]

        return ""

    # -------------------------
    # helpers
    # -------------------------

    def _format_questions(self, questions: List[Dict]) -> str:
        lines = []
        for i, q in enumerate(questions, 1):
            qid = q["id"]
            qtext = q["question"]
            lines.append(f"{i}. [{qid}] {qtext}")
        return "\n".join(lines)

    def _log_first_tokens(self, response: Any, context: str = "") -> None:
        text = str(response)
        self.logger.info(f"[{self._expert_id}]{context} First LLM output: {text[:200]}")

    # -------------------------
    # response processing
    # -------------------------

    def _get_response_format(self, round_phase: str) -> Dict[str, Any]:
        """lightweight schema hint; models may still drift, so we normalize."""
        if round_phase == "round1":
            return {
                "type": "json_object",
                "schema": {
                    "scores": "object question_id:int(1-10) or array[int]",
                    "evidence": "object question_id:str or array[str]",
                    "clinical_reasoning": "string",
                    "p_iraki": "float 0-1 or percent string",
                    "ci_iraki": "array[float,float] or 'l-u' or {lower,upper}",
                    "confidence": "float 0-1 or word",
                    "differential_diagnosis": "array string",
                    "primary_diagnosis": "string",
                    "specialty_notes": "optional string",
                    "citations": "optional array string",
                },
            }
        return {
            "type": "json_object",
            "schema": {
                "scores": "object question_id:int(1-10) or array[int]",
                "evidence": "object question_id:str or array[str]",
                "p_iraki": "float 0-1 or percent string",
                "ci_iraki": "array[float,float] or 'l-u' or {lower,upper}",
                "confidence": "float 0-1 or word",
                "changes_from_round1": "object",
                "debate_influence": "string",
                "verdict": "boolean",
                "final_diagnosis": "string",
                "confidence_in_verdict": "float 0-1 or word",
                "recommendations": "array string",
                # optional round-3 extras are omitted in constructor; safe to ignore here
                "biopsy_recommendation": "optional string",
                "steroid_recommendation": "optional string",
                "ici_rechallenge_risk": "optional string",
            },
        }

    def _validate_llm_response(
        self, resp: Dict[str, Any], round_phase: str, question_ids: Optional[List[str]]
    ) -> Dict[str, Any]:
        """coerce common drift back to schema for round3 (round1 uses normalize_round1)."""
        out: Dict[str, Any] = dict(resp)

        # p and ci
        try:
            out["p_iraki"] = _clip01(_as_float(out.get("p_iraki", 0.5)))
        except Exception:
            out["p_iraki"] = 0.5

        if "ci_iraki" in out:
            try:
                out["ci_iraki"] = _as_ci2(out["ci_iraki"])
            except Exception:
                out["ci_iraki"] = (
                    max(0.0, out["p_iraki"] - 0.2),
                    min(1.0, out["p_iraki"] + 0.2),
                )
        else:
            lo = out.pop("ci_lower", None)
            up = out.pop("ci_upper", None)
            if lo is not None and up is not None:
                try:
                    out["ci_iraki"] = _as_ci2([lo, up])
                except Exception:
                    out["ci_iraki"] = (
                        max(0.0, out["p_iraki"] - 0.2),
                        min(1.0, out["p_iraki"] + 0.2),
                    )
            else:
                out["ci_iraki"] = (
                    max(0.0, out["p_iraki"] - 0.2),
                    min(1.0, out["p_iraki"] + 0.2),
                )

        # confidence
        try:
            out["confidence"] = _clip01(_as_float(out.get("confidence", 0.7)))
        except Exception:
            out["confidence"] = 0.7

        # scores
        try:
            out["scores"] = _normalize_scores_like(out.get("scores", {}), question_ids)
        except Exception:
            out["scores"] = {}

        # evidence: not used → keep empty
        out["evidence"] = {}

        # defaults for round3-only fields
        if round_phase == "round3":
            out.setdefault(
                "recommendations", ["Monitor renal function", "Nephrology consult"]
            )
            out.setdefault("verdict", out["p_iraki"] > 0.5)
            out.setdefault("final_diagnosis", "irAKI" if out["verdict"] else "ATN")
            try:
                out["confidence_in_verdict"] = _clip01(
                    _as_float(out.get("confidence_in_verdict", out["confidence"]))
                )
            except Exception:
                out["confidence_in_verdict"] = out["confidence"]

        return out

    def _create_round1_reply(
        self, resp: Dict[str, Any], message: QuestionnaireMsg
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
            literature_citations=resp.get(
                "literature_citations", resp.get("citations", [])
            ),
        )

    def _create_round3_reply(
        self, resp: Dict[str, Any], message: QuestionnaireMsg
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
        )
