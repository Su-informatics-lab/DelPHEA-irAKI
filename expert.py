"""
expert agent: specialty-conditioned assessor that emits strictly validated json.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from llm_backend import LLMBackend
from messages import ExpertRound1Reply, ExpertRound3Reply
from models import AssessmentR1, AssessmentR3, DebateTurn
from prompts.rounds import format_round1_prompt, format_round3_prompt
from schema import load_qids
from validators import call_llm_with_schema


class Expert:
    """specialty-conditioned assessor with strict schema validation."""

    def __init__(
        self,
        expert_id: str,
        specialty: str,
        persona: Dict[str, Any],
        backend: LLMBackend,
        prompts_path: str = "prompts/expert_prompts.json",
    ) -> None:
        """initialize an expert wrapper."""
        if not expert_id:
            raise ValueError("expert_id cannot be empty")
        if backend is None:
            raise ValueError("backend cannot be None")

        self.expert_id = expert_id
        self.specialty = specialty
        self.persona = persona or {}
        self.backend = backend
        self.prompts_path = prompts_path

        self.logger = logging.getLogger(f"expert.{expert_id}")

    # --------- round 1 / round 3 ---------

    def assess_round1(
        self,
        case: Dict[str, Any],
        questionnaire_path: str,
        repair_hint: Optional[str] = None,  # reserved for future use
    ) -> AssessmentR1:
        """produce a round-1 assessment using instructor + pydantic conversion."""
        # validate questionnaire early (fail loud if malformed/missing)
        load_qids(questionnaire_path)

        info = self._extract_case_strings(case)
        prompt_text = format_round1_prompt(
            expert_name=self._expert_name(),
            specialty=self.specialty,
            case_id=info["case_id"],
            demographics=info["demographics"],
            clinical_notes=info["clinical_notes"],
            qpath=questionnaire_path,
        )

        reply = call_llm_with_schema(
            response_model=ExpertRound1Reply,  # messages.* schema for instructor
            prompt_text=prompt_text,
            backend=self.backend,  # validators infer base_url/model
            temperature=0.0,
            max_retries=1,
        )
        payload = reply.model_dump()
        self._log_preview(payload, "[r1 parsed]")
        # return models.* pydantic to satisfy moderator/aggregator contracts
        return AssessmentR1(**payload)

    def assess_round3(
        self,
        case: Dict[str, Any],
        questionnaire_path: str,
        debate_context: Dict[str, Any],
        repair_hint: Optional[str] = None,  # reserved for future use
    ) -> AssessmentR3:
        """produce a round-3 reassessment using instructor + pydantic conversion."""
        load_qids(questionnaire_path)

        info = self._extract_case_strings(case)
        prompt_text = format_round3_prompt(
            expert_name=self._expert_name(),
            specialty=self.specialty,
            case_id=info["case_id"],
            demographics=info["demographics"],
            clinical_notes=info["clinical_notes"],
            qpath=questionnaire_path,
            debate_ctx=debate_context,
        )

        reply = call_llm_with_schema(
            response_model=ExpertRound3Reply,  # messages.* schema for instructor
            prompt_text=prompt_text,
            backend=self.backend,
            temperature=0.0,
            max_retries=1,
        )
        payload = reply.model_dump()
        self._log_preview(payload, "[r3 parsed]")
        return AssessmentR3(**payload)

    # --------- debate (round 2 single turn) ---------

    def debate(
        self,
        qid: str,
        round_no: int,
        clinical_context: Dict[str, Any],
        minority_view: str,
    ) -> DebateTurn:
        """produce a single debate turn for a specific question."""
        if not qid:
            raise ValueError("debate requires a non-empty qid")
        if round_no not in (1, 2, 3):
            raise ValueError(f"unsupported round_no for debate: {round_no}")

        # keep using backend's debate hook (yagni: no instructor schema unless needed)
        payload = {
            "expert_id": self.expert_id,
            "specialty": self.specialty,
            "qid": qid,
            "round_no": round_no,
            "clinical_context": clinical_context,
            "minority_view": minority_view,
        }
        raw = self.backend.debate(payload)
        self._log_preview(raw, f"[debate {qid} raw]")
        return DebateTurn(**raw)

    # --------- helpers ---------

    def _expert_name(self) -> str:
        """derive display name for prompts."""
        name = self.persona.get("name") or self.persona.get("display_name") or ""
        return (
            name.strip() if isinstance(name, str) and name.strip() else self.expert_id
        )

    def _flatten_dict_lines(
        self, d: Dict[str, Any], allow_keys: Optional[List[str]] = None
    ) -> str:
        """turn a small dict into stable 'k: v' lines; optionally filter keys."""
        if not isinstance(d, dict):
            return str(d)
        items = d.items()
        if allow_keys:
            allow = set(allow_keys)
            items = [(k, v) for k, v in d.items() if k in allow]
        return "\n".join(
            f"{k}: {v}" for k, v in items if v is not None and str(v).strip()
        )

    def _coerce_demographics(self, case: Dict[str, Any]) -> str:
        """derive a demographics text block from multiple possible case layouts."""
        if case.get("demographics_text"):
            return str(case["demographics_text"]).strip()

        if isinstance(case.get("demographics"), dict):
            return self._flatten_dict_lines(case["demographics"])
        if "demographics" in case and isinstance(case["demographics"], str):
            return case["demographics"].strip()

        candidates = [
            ("person", None),
            ("patient", None),
            ("patient_info", None),
            ("person_demographics", None),
            ("summary", "demographics"),
            ("meta", "demographics"),
            ("profile", None),
        ]
        for top, sub in candidates:
            if top in case and case[top] is not None:
                node = case[top]
                if sub and isinstance(node, dict) and node.get(sub):
                    val = node[sub]
                    if isinstance(val, dict):
                        return self._flatten_dict_lines(val)
                    return str(val).strip()
                if isinstance(node, dict):
                    txt = self._flatten_dict_lines(
                        node, allow_keys=["age", "sex", "gender", "race", "ethnicity"]
                    )
                    if txt:
                        return txt

        return "demographics: not available in source case"

    def _coerce_notes(self, case: Dict[str, Any]) -> str:
        """normalize aggregated clinical notes into a single bounded string."""
        if case.get("notes_agg") is not None:
            notes_val = case["notes_agg"]
        elif case.get("clinical_notes") is not None:
            notes_val = case["clinical_notes"]
        elif case.get("notes_text") is not None:
            notes_val = case["notes_text"]
        else:
            raise ValueError(
                "notes_agg|clinical_notes|notes_text is required in case payload"
            )

        if isinstance(notes_val, list):
            parts: List[str] = []
            for item in notes_val:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    t = item.get("text") or item.get("note") or ""
                    if t:
                        parts.append(str(t))
            notes = "\n---\n".join(p for p in parts if p.strip())
        else:
            notes = str(notes_val)

        if len(notes) > 16000:
            notes = notes[:16000] + "\n...[truncated]"
        return notes

    def _extract_case_strings(self, case: Dict[str, Any]) -> Dict[str, str]:
        """normalize required text fields from heterogeneous case dicts."""
        if not isinstance(case, dict):
            raise ValueError("case must be a dict")

        cid = (
            case.get("case_id")
            or case.get("id")
            or case.get("patient_id")
            or case.get("person_id")
        )
        if not cid:
            raise ValueError(
                "case_id|id|patient_id|person_id is required in case payload"
            )

        demo = self._coerce_demographics(case)
        notes = self._coerce_notes(case)

        return {"case_id": str(cid), "demographics": demo, "clinical_notes": notes}

    def _log_preview(self, obj: Any, tag: str) -> None:
        """log a short preview of backend output for debugging (best-effort)."""
        try:
            s = str(obj)
            self.logger.debug("%s %s", tag, s[:500])
        except Exception:
            # never let logging break the pipeline
            pass
