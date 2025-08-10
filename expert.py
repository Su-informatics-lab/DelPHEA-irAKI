"""
expert agent: specialty-conditioned assessor that emits strictly validated json.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from llm_backend import LLMBackend
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
        *,
        temperature_r1: float = 0.0,
        temperature_r2: float = 0.3,
        temperature_r3: float = 0.0,
    ) -> None:
        # validate
        if not expert_id:
            raise ValueError("expert_id cannot be empty")
        if backend is None:
            raise ValueError("backend cannot be None")

        self.expert_id = expert_id
        self.specialty = specialty
        self.persona = persona or {}
        self.backend = backend
        self.prompts_path = prompts_path

        # per-round temperatures (cli-controlled via delphea_iraki.py)
        self.temperature_r1 = float(temperature_r1)
        self.temperature_r2 = float(temperature_r2)  # debate
        self.temperature_r3 = float(temperature_r3)

        # logger
        self.logger = logging.getLogger(f"expert.{self.expert_id}")

    # --------- round 1 / round 3 ---------

    def assess_round1(
        self,
        case: Dict[str, Any],
        questionnaire_path: str,
        repair_hint: Optional[str] = None,  # reserved for future use
    ) -> AssessmentR1:
        """produce a round-1 assessment using strict pydantic validation at the boundary."""
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
        # belt-and-suspenders: require identifiers in the JSON output
        prompt_text += (
            f"\n\nIMPORTANT: In your JSON output, you MUST include precisely these keys:\n"
            f'- "case_id": "{info["case_id"]}"\n'
            f'- "expert_id": "{self.expert_id}"\n'
            f"Return ONLY valid JSON."
        )

        reply = call_llm_with_schema(
            response_model=AssessmentR1,
            prompt_text=prompt_text,
            backend=self.backend,
            temperature=self.temperature_r1,
        )

        # harden: inject identifiers if the model omitted them
        if hasattr(reply, "model_dump"):
            d = reply.model_dump()
            d.setdefault("case_id", info["case_id"])
            d.setdefault("expert_id", self.expert_id)
            reply = AssessmentR1(**d)

        self._log_preview(reply, "r1-validated")
        return reply  # already a validated AssessmentR1

    def assess_round3(
        self,
        case: Dict[str, Any],
        questionnaire_path: str,
        debate_context: Dict[str, Any],
        repair_hint: Optional[str] = None,  # reserved for future use
    ) -> AssessmentR3:
        """produce a round-3 reassessment using strict pydantic validation at the boundary."""
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
        # require identifiers again for consistency
        prompt_text += (
            f"\n\nIMPORTANT: In your JSON output, you MUST include precisely these keys:\n"
            f'- "case_id": "{info["case_id"]}"\n'
            f'- "expert_id": "{self.expert_id}"\n'
            f"Return ONLY valid JSON."
        )

        reply = call_llm_with_schema(
            response_model=AssessmentR3,
            prompt_text=prompt_text,
            backend=self.backend,
            temperature=self.temperature_r3,
        )

        # harden: inject identifiers if the model omitted them
        if hasattr(reply, "model_dump"):
            d = reply.model_dump()
            d.setdefault("case_id", info["case_id"])
            d.setdefault("expert_id", self.expert_id)
            reply = AssessmentR3(**d)

        self._log_preview(reply, "r3-validated")
        return reply  # already a validated AssessmentR3

    # --------- debate (round 2 single turn) ---------

    def debate(
        self,
        qid: str,
        round_no: int,
        clinical_context: Dict[str, Any],
        minority_view: str,
    ) -> DebateTurn:
        """produce a single debate turn for a specific question.

        note: if your backend supports passing temperature for debate, it can
        honor 'temperature' in the payload below (r2 temperature).
        """
        if not qid:
            raise ValueError("debate requires a non-empty qid")
        if round_no not in (1, 2, 3):
            raise ValueError(f"unsupported round_no for debate: {round_no}")

        payload = {
            "expert_id": self.expert_id,
            "specialty": self.specialty,
            "qid": qid,
            "round_no": round_no,
            "clinical_context": clinical_context,
            "minority_view": minority_view,
            "temperature": self.temperature_r2,  # r2 temperature
        }
        raw = self.backend.debate(payload)
        self._log_preview(raw, f"[debate {qid} raw]")

        # minimal hardening: accept dict, pydantic model, or json string
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        elif isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception as e:
                raise ValueError(f"backend.debate returned non-json string: {e}") from e
        if not isinstance(raw, dict):
            raise TypeError("backend.debate must return a dict-like (or json string)")

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
            items = ((k, v) for k, v in d.items() if k in allow)
        # ensure stable ordering
        return "\n".join(
            f"{k}: {v}"
            for k, v in sorted(items, key=lambda kv: str(kv[0]))
            if v is not None and str(v).strip()
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
        """normalize aggregated clinical notes into a single string (no truncation here).

        input-size control is handled upstream via budgeting and centrally
        in validators via token budgeting. this function only normalizes.
        """
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
                    t = (
                        item.get("text")
                        or item.get("note")
                        or item.get("REPORT_TEXT")
                        or item.get("report_text")
                        or ""
                    )
                    if t:
                        parts.append(str(t))
            notes = "\n---\n".join(p for p in parts if p.strip())
        else:
            notes = str(notes_val)

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
