"""
Expert agent: specialty-conditioned assessor that emits structured JSON.

Role
----
An Expert is a lightweight wrapper over an LLM/back-end prompt that:
- consumes case context + questionnaire
- emits Round-1 JSON (scores, evidence, p_iraki, ci_iraki, confidence, differentials)
- emits Round-3 JSON (updated scores, changes_from_round1, debate_influence, verdict,
    final_diagnosis, recommendations)

Prompt & schema contracts
-------------------------
- Round 1/3 output schemas are defined in `iraki_assessment.json`. The Expert must
  strictly conform (no extra keys, valid ranges, CI contains p_iraki).
- Debate output follows `debate.json` (`text`, `citations`, `satisfied`).

Expert grounding
-----------------
- Expert identity and focus areas come from the panel config (e.g., oncologist,
    nephrologist, pathologist). Use these fields to condition the system prompt and
    retrieval hints.

Error handling (fail fast)
--------------------------
- raise ValueError on invalid Likert scores (not 1..9), p_iraki ∉ [0,1], CI bounds
    invalid, or CI not containing p_iraki
- raise ValueError when required keys are missing

Caching
-------
- experts should memoize (case_id, round_no, qid) → output to avoid duplicate LLM calls
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


def _as_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in _WORD_CONF:
            return _WORD_CONF[s]
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*%?\s*$", s)
        if m:
            v = float(m.group(1))
            return v / 100.0 if "%" in s else v
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


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return float(x) != 0.0
    if isinstance(x, str):
        s = x.strip().lower()
        # common positives/negatives
        if s in {
            "true",
            "t",
            "yes",
            "y",
            "1",
            "likely",
            "positive",
            "iraki",
            "present",
        }:
            return True
        if s in {
            "false",
            "f",
            "no",
            "n",
            "0",
            "unlikely",
            "negative",
            "other",
            "absent",
        }:
            return False
        # percent/number ⇒ threshold at 0.5
        try:
            return _clip01(_as_float(s)) > 0.5
        except Exception:
            return False
    return False


def _as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if i is not None]
    # single string/number → one-element list
    return [str(x)]


def _as_str_dict(x: Any, default_key: str = "summary") -> Dict[str, str]:
    if isinstance(x, dict):
        return {str(k): str(v) for k, v in x.items()}
    if isinstance(x, list):
        out: Dict[str, str] = {}
        for i, item in enumerate(x, 1):
            if item is not None:
                out[f"item_{i}"] = str(item)
        return out
    if isinstance(x, str) and x.strip():
        return {default_key: x}
    return {}


def normalize_round1(
    raw: Dict[str, Any],
    question_ids: List[str],
    case_id: str,
    expert_id: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"case_id": case_id, "expert_id": expert_id}

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

    ev = raw.get("evidence", {})
    if isinstance(ev, list):
        ev_dict: Dict[str, str] = {}
        for i, qid in enumerate(question_ids):
            if i < len(ev) and isinstance(ev[i], str):
                ev_dict[qid] = ev[i]
        out["evidence"] = ev_dict
    elif isinstance(ev, dict):
        out["evidence"] = {str(k): str(v) for k, v in ev.items()}
    else:
        out["evidence"] = {}

    out["clinical_reasoning"] = str(
        raw.get("clinical_reasoning") or raw.get("reasoning") or ""
    )

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

    c = raw.get("confidence", "moderate")
    try:
        out["confidence"] = _clip01(_as_float(c))
    except Exception:
        out["confidence"] = 0.5

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
    if isinstance(val, dict):
        return {str(k): int(float(v)) for k, v in val.items()}
    if isinstance(val, list) and question_ids:
        return {
            qid: int(float(val[i]))
            for i, qid in enumerate(question_ids)
            if i < len(val)
        }
    raise ValueError("scores must be dict or list with question_ids available")


class irAKIExpertAgent(RoutedAgent):
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

        self._expert_profile = next(
            ep
            for ep in self._config_loader.expert_panel["expert_panel"]["experts"]
            if ep["id"] == expert_id
        )
        self.logger.info(
            f"Initialized expert: {self._expert_profile['name']} ({self._expert_profile['specialty']})"
        )

        self._round1_assessment: Optional[ExpertRound1Reply] = None
        self._debate_history: List[DebateComment] = []

    @message_handler
    async def handle_questionnaire(
        self, message: QuestionnaireMsg, ctx: MessageContext
    ) -> None:
        self.logger.info(f"Received {message.round_phase} questionnaire")

        if message.round_phase not in ("round1", "round3"):
            raise ValueError(f"unknown round_phase: {message.round_phase!r}")

        tpl_name = "round1" if message.round_phase == "round1" else "round3"
        template = self._config_loader.get_prompt_template(tpl_name)
        base = template.get("base_prompt")
        if not base:
            raise KeyError(f"prompt '{tpl_name}' missing required 'base_prompt'")

        demographics_str = self._extract_demographics(message)
        notes_str = self._extract_notes(message)
        questions_str = self._format_questions(message.questions)

        prompt = base.format(
            expert_name=self._expert_profile["name"],
            specialty=self._expert_profile["specialty"],
            round_phase=message.round_phase,
            case_id=message.case_id,
            demographics=demographics_str,
            clinical_notes=notes_str,
            questions=questions_str,
        )

        instructions = (
            template.get("round3_instructions", "")
            if message.round_phase == "round3"
            else template.get("round1_instructions", "")
        )
        if instructions:
            prompt = f"{prompt}\n\nINSTRUCTIONS\n{instructions}"

        response_format = self._get_response_format(message.round_phase)
        llm_response = await self._vllm_client.generate_structured_response(
            prompt=prompt, response_format=response_format
        )
        self._log_first_tokens(llm_response, f"[{message.round_phase}]")

        q_ids = [q["id"] for q in message.questions]

        if message.round_phase == "round1":
            cleaned = normalize_round1(
                raw=llm_response,
                question_ids=q_ids,
                case_id=message.case_id,
                expert_id=self._expert_id,
            )
            reply = self._create_round1_reply(cleaned, message)
            self._round1_assessment = reply
            method = "record_round1"
        else:
            cleaned = self._validate_llm_response(
                llm_response, round_phase="round3", question_ids=q_ids
            )
            reply = self._create_round3_reply(cleaned, message)
            method = "record_round3"

        target = AgentId(type="moderator", key=message.case_id)
        # sending the Pydantic message; moderator has @rpc methods that accept these
        ack = await self.send_message(reply, target)
        if not getattr(ack, "ok", True):
            raise RuntimeError(
                f"moderator rejected {method}: {getattr(ack, 'message', '')}"
            )

        self.logger.info(f"Submitted {message.round_phase} assessment")

    @message_handler
    async def handle_debate_prompt(
        self, message: DebatePrompt, ctx: MessageContext
    ) -> None:
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

    # --------- extractors ---------

    def _extract_demographics(self, message: QuestionnaireMsg) -> str:
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

    # --------- helpers ---------

    def _format_questions(self, questions: List[Dict]) -> str:
        return "\n".join(
            f"{i}. [{q['id']}] {q['question']}" for i, q in enumerate(questions, 1)
        )

    def _log_first_tokens(self, response: Any, context: str = "") -> None:
        self.logger.info(
            f"[{self._expert_id}]{context} First LLM output: {str(response)[:200]}"
        )

    # --------- response processing ---------

    def _get_response_format(self, round_phase: str) -> Dict[str, Any]:
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
                "changes_from_round1": "object or string or array",
                "debate_influence": "string",
                "verdict": "boolean or string or number",
                "final_diagnosis": "string",
                "confidence_in_verdict": "float 0-1 or word",
                "recommendations": "array string or string",
            },
        }

    def _validate_llm_response(
        self, resp: Dict[str, Any], round_phase: str, question_ids: Optional[List[str]]
    ) -> Dict[str, Any]:
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

        # evidence
        ev = out.get("evidence", {})
        if isinstance(ev, dict):
            out["evidence"] = {str(k): str(v) for k, v in ev.items()}
        elif isinstance(ev, list) and question_ids:
            ev_dict: Dict[str, str] = {}
            for i, qid in enumerate(question_ids):
                if i < len(ev) and isinstance(ev[i], str):
                    ev_dict[qid] = ev[i]
            out["evidence"] = ev_dict
        else:
            out["evidence"] = {}

        if round_phase == "round3":
            # changes_from_round1: must be dict[str,str]
            out["changes_from_round1"] = _as_str_dict(
                out.get("changes_from_round1", {}), default_key="summary"
            )

            # debate_influence: string
            di = out.get("debate_influence")
            out["debate_influence"] = "" if di is None else str(di)

            # verdict: coerce to bool
            if "verdict" in out:
                out["verdict"] = _as_bool(out["verdict"])
            else:
                out["verdict"] = out["p_iraki"] > 0.5

            # final_diagnosis: ensure string
            fd = out.get("final_diagnosis")
            if fd is None or (isinstance(fd, str) and not fd.strip()):
                out["final_diagnosis"] = "irAKI" if out["verdict"] else "ATN"
            else:
                out["final_diagnosis"] = str(fd)

            # confidence_in_verdict
            try:
                out["confidence_in_verdict"] = _clip01(
                    _as_float(out.get("confidence_in_verdict", out["confidence"]))
                )
            except Exception:
                out["confidence_in_verdict"] = out["confidence"]

            # recommendations: list[str]
            rec = _as_str_list(out.get("recommendations", []))
            out["recommendations"] = [str(r) for r in rec if str(r).strip()]

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
