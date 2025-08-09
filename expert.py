"""
expert agent: specialty-conditioned assessor that emits strictly validated json.

role
----
an expert is a lightweight wrapper over an llm/back-end prompt that:
- consumes case context + questionnaire
- emits round-1 json (scores, evidence, p_iraki, ci_iraki, confidence, differentials)
- emits round-3 json (updated scores, changes_from_round1, debate_influence, verdict,
  final_diagnosis, recommendations)
- emits debate turns (text, citations, satisfied)

contracts (fail-fast)
---------------------
- round 1/3 outputs must conform to pydantic models in `models.py`
  (AssessmentR1 / AssessmentR3), with:
  - scores: dict[qid]->int in 1..9
  - evidence: dict[qid]->str
  - p_iraki in [0,1], ci_iraki = (lo,hi) with 0<=lo<=p<=hi<=1
  - confidence in [0,1]
  - r3 adds: changes_from_round1, debate_influence, verdict, final_diagnosis,
    confidence_in_verdict, recommendations
- debate output follows `DebateTurn` (text, citations, satisfied)
- strict schema echo: qids must exactly match those in questionnaire_full.json

implementation notes
--------------------
- this version uses strict pydantic validation; malformed llm json raises a ValidationError.
- transport is abstracted behind `LLMBackend`; swap implementations without touching this class.
- no autogen handlers are used; the moderator calls these methods directly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from expert_validation import PayloadValidationError, call_llm_with_schema
from llm_backend import LLMBackend
from models import AssessmentR1, AssessmentR3, DebateTurn
from schema import load_qids


class Expert:
    """specialty-conditioned assessor with strict schema validation."""

    def __init__(
        self,
        expert_id: str,
        specialty: str,
        persona: Dict[str, Any],
        backend: LLMBackend,
    ) -> None:
        """initialize an expert wrapper.

        args:
            expert_id: stable identifier for the expert.
            specialty: clinical specialty (e.g., nephrology, oncology).
            persona: dict carrying persona details from panel.json.
            backend: llm transport implementing assess_round1/assess_round3/debate.

        raises:
            ValueError: if expert_id is empty or backend is None.
        """
        if not expert_id:
            raise ValueError("expert_id cannot be empty")
        if backend is None:
            raise ValueError("backend cannot be None")

        self.expert_id = expert_id
        self.specialty = specialty
        self.persona = persona
        self.backend = backend

        self.logger = logging.getLogger(f"expert.{expert_id}")

    def assess_round1(
        self, case: Dict[str, Any], questionnaire_path: str
    ) -> AssessmentR1:
        """produce a round-1 assessment validated against expected qids."""
        expected_qids = load_qids(questionnaire_path)
        payload = {
            "expert_id": self.expert_id,
            "specialty": self.specialty,
            "persona": self.persona,
            "case": case,
            "questionnaire_path": questionnaire_path,
        }
        raw = self.backend.assess_round1(payload)
        self._log_preview(raw, "[r1 raw]")
        assessed = AssessmentR1.model_validate(
            raw, context={"expected_qids": expected_qids}
        )
        return assessed

    def assess_round3(
        self,
        case: Dict[str, Any],
        questionnaire_path: str,
        debate_context: Dict[str, Any],
    ) -> AssessmentR3:
        """produce a round-3 reassessment validated against expected qids."""
        expected_qids = load_qids(questionnaire_path)
        payload = {
            "expert_id": self.expert_id,
            "specialty": self.specialty,
            "persona": self.persona,
            "case": case,
            "questionnaire_path": questionnaire_path,
            "debate_context": debate_context,
        }
        raw = self.backend.assess_round3(payload)
        self._log_preview(raw, "[r3 raw]")
        reassessed = AssessmentR3.model_validate(
            raw, context={"expected_qids": expected_qids}
        )
        return reassessed

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
        return DebateTurn.model_validate(raw)

    # --------- helpers ---------

    def _log_preview(self, obj: Any, tag: str) -> None:
        """log a short preview of backend output for debugging."""
        try:
            self.logger.debug(f"{tag} {str(obj)[:200]}")
        except Exception:
            # never let logging break the pipeline
            pass

    def assess(self, case, round_no: int) -> dict:
        prompt = self._render_prompt(case, round_no)  # your existing prompt builder
        try:
            result = call_llm_with_schema(self.backend, prompt, round_no, max_retries=2)
            result["_status"] = "ok"
            return result
        except PayloadValidationError as e:
            # fail loud, let moderator decide to re-solicit or drop
            return {"_status": "invalid_assessment", "_error": str(e)}
