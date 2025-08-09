"""
expert agent: specialty‑conditioned assessor that emits strictly validated json.

role
----
an expert is a lightweight wrapper over an llm/back‑end prompt that:
- consumes case context + questionnaire
- emits round‑1 json (scores, evidence, p_iraki, ci_iraki, confidence, differentials)
- emits round‑3 json (updated scores, changes_from_round1, debate_influence, verdict,
  final_diagnosis, recommendations)
- emits debate turns (text, citations, satisfied)

contracts (fail‑fast)
---------------------
- round 1/3 outputs must conform to pydantic models in `models.py`
  (AssessmentR1 / AssessmentR3), with:
  - scores: dict[qid] -> int in 1..9
  - evidence: dict[qid] -> str (non‑empty)
  - p_iraki in [0,1], ci_iraki = [lo, hi] with 0<=lo<=p<=hi<=1
  - confidence in [0,1]
  - r3 adds: changes_from_round1, debate_influence, verdict, final_diagnosis,
    confidence_in_verdict, recommendations
- debate output follows `DebateTurn` (text, citations, satisfied)
- strict schema echo: qids must exactly match those in the questionnaire; moderator enforces this

implementation notes
--------------------
- prompt text blocks live in json files under prompts/.
- this class uses a guarded llm→json→validate loop via validators.call_llm_with_schema.
- transport is abstracted behind `LLMBackend`; it must expose .generate(str)->str and .debate(payload)->dict.
- errors fail fast; the moderator may retry with a repair hint or auto‑repair as a last resort.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from llm_backend import LLMBackend
from models import AssessmentR1, AssessmentR3, DebateTurn
from prompts.rounds import format_round1_prompt, format_round3_prompt
from schema import load_qids
from validators import call_llm_with_schema


class Expert:
    """specialty‑conditioned assessor with strict schema validation."""

    def __init__(
        self,
        expert_id: str,
        specialty: str,
        persona: Dict[str, Any],
        backend: LLMBackend,
        prompts_path: str = "prompts/expert_prompts.json",
    ) -> None:
        """initialize an expert wrapper.

        args:
            expert_id: stable identifier for the expert.
            specialty: clinical specialty (e.g., nephrology, oncology).
            persona: dict carrying persona details from panel.json.
            backend: llm transport with .generate() and .debate().
            prompts_path: path to json prompt templates (kept for compatibility).
        """
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
        repair_hint: Optional[
            str
        ] = None,  # kept for api stability; unused by formatter
    ) -> AssessmentR1:
        """produce a round‑1 assessment using json prompts + guarded validation."""
        qids = load_qids(questionnaire_path)
        info = self._extract_case_strings(case)
        prompt = format_round1_prompt(
            expert_name=self._expert_name(),
            specialty=self.specialty,
            case_id=info["case_id"],
            demographics=info["demographics"],
            clinical_notes=info["clinical_notes"],
            qpath=questionnaire_path,
        )
        payload = call_llm_with_schema(
            backend=self.backend,
            prompt=prompt,
            round_no=1,
            max_retries=2,
            qids=qids,
        )
        self._log_preview(payload, "[r1 parsed]")
        return AssessmentR1(**payload)

    def assess_round3(
        self,
        case: Dict[str, Any],
        questionnaire_path: str,
        debate_context: Dict[str, Any],
        repair_hint: Optional[
            str
        ] = None,  # kept for api stability; unused by formatter
    ) -> AssessmentR3:
        """produce a round‑3 reassessment using json prompts + guarded validation."""
        qids = load_qids(questionnaire_path)
        info = self._extract_case_strings(case)
        prompt = format_round3_prompt(
            expert_name=self._expert_name(),
            specialty=self.specialty,
            case_id=info["case_id"],
            demographics=info["demographics"],
            clinical_notes=info["clinical_notes"],
            qpath=questionnaire_path,
            debate_ctx=debate_context,
        )
        payload = call_llm_with_schema(
            backend=self.backend,
            prompt=prompt,
            round_no=3,
            max_retries=2,
            qids=qids,
        )
        self._log_preview(payload, "[r3 parsed]")
        return AssessmentR3(**payload)

    # --------- debate ---------

    def debate(
        self,
        qid: str,
        round_no: int,
        clinical_context: Dict[str, Any],
        minority_view: str,
    ) -> DebateTurn:
        """produce a single debate turn for a specific question."""
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
        return name if isinstance(name, str) and name.strip() else self.expert_id

    def _extract_case_strings(self, case: Dict[str, Any]) -> Dict[str, str]:
        """normalize required text fields from heterogeneous case dicts; fail fast."""
        if not isinstance(case, dict):
            raise ValueError("case must be a dict")

        # id
        cid = case.get("case_id") or case.get("id")
        if not cid:
            raise ValueError("case_id is required in case payload")

        # demographics: accept preformatted text or dict
        if "demographics_text" in case and case["demographics_text"]:
            demo = str(case["demographics_text"])
        elif "demographics" in case and case["demographics"] is not None:
            demo_val = case["demographics"]
            if isinstance(demo_val, dict):
                demo = "\n".join(f"{k}: {v}" for k, v in demo_val.items())
            else:
                demo = str(demo_val)
        else:
            raise ValueError(
                "demographics_text|demographics is required in case payload"
            )

        # clinical notes: prefer pre‑aggregated
        notes: Optional[str]
        if case.get("notes_agg"):
            notes = str(case["notes_agg"])
        elif case.get("clinical_notes"):
            notes = str(case["clinical_notes"])
        elif case.get("notes_text"):
            notes = str(case["notes_text"])
        else:
            raise ValueError(
                "notes_agg|clinical_notes|notes_text is required in case payload"
            )

        # clamp very long inputs to protect token budget while preserving signal
        if isinstance(notes, str) and len(notes) > 16000:
            notes = notes[:16000] + "\n...[truncated]"

        return {"case_id": str(cid), "demographics": demo, "clinical_notes": notes}

    def _log_preview(self, obj: Any, tag: str) -> None:
        """log a short preview of backend output for debugging (best‑effort)."""
        try:
            s = str(obj)
            self.logger.debug("%s %s", tag, s[:500])
        except Exception:
            # never let logging break the pipeline
            pass
