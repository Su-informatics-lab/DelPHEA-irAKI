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
  - scores: dict[qid] -> int in 1..9
  - evidence: dict[qid] -> str (non-empty)
  - p_iraki in [0,1], ci_iraki = [lo, hi] with 0<=lo<=p<=hi<=1
  - confidence in [0,1]
  - r3 adds: changes_from_round1, debate_influence, verdict, final_diagnosis,
    confidence_in_verdict, recommendations
- debate output follows `DebateTurn` (text, citations, satisfied)
- strict schema echo: qids must exactly match those in the questionnaire; moderator enforces this

implementation notes
--------------------
- prompts are defined in json and formatted at runtime (see prompts/expert_prompts.json).
- this class uses a guarded llm→json→validate loop via validators.call_llm_with_schema.
- transport is abstracted behind `LLMBackend`; it must expose .generate(str)->str and .debate(payload)->dict.
- errors fail fast; the moderator may retry with a repair hint or auto-repair as a last resort.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from llm_backend import LLMBackend
from models import AssessmentR1, AssessmentR3, DebateTurn
from prompts import format_round1_prompt, format_round3_prompt
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
        """initialize an expert wrapper.

        args:
            expert_id: stable identifier for the expert.
            specialty: clinical specialty (e.g., nephrology, oncology).
            persona: dict carrying persona details from panel.json.
            backend: llm transport with .generate() and .debate().
            prompts_path: path to json prompt templates.
        """
        if not expert_id:
            raise ValueError("expert_id cannot be empty")
        if backend is None:
            raise ValueError("backend cannot be None")

        self.expert_id = expert_id
        self.specialty = specialty
        self.persona = persona
        self.backend = backend
        self.prompts_path = prompts_path

        self.logger = logging.getLogger(f"expert.{expert_id}")

    # --------- round 1 / round 3 ---------

    def assess_round1(
        self,
        case: Dict[str, Any],
        questionnaire_path: str,
        repair_hint: Optional[str] = None,
    ) -> AssessmentR1:
        """produce a round-1 assessment using json prompts + guarded validation."""
        qids = load_qids(questionnaire_path)
        prompt = format_round1_prompt(
            prompts_path=self.prompts_path,
            questionnaire_path=questionnaire_path,
            case=case,
            repair_hint=repair_hint,
        )
        # use guarded generator (json parse + content validation + repair retries)
        payload = call_llm_with_schema(
            backend=self.backend,
            prompt=prompt,
            round_no=1,
            max_retries=2,
            qids=qids,
        )
        self._log_preview(payload, "[r1 parsed]")
        # pydantic validation; moderator will also enforce exact qid echo
        return AssessmentR1(**payload)

    def assess_round3(
        self,
        case: Dict[str, Any],
        questionnaire_path: str,
        debate_context: Dict[str, Any],
        repair_hint: Optional[str] = None,
    ) -> AssessmentR3:
        """produce a round-3 reassessment using json prompts + guarded validation."""
        qids = load_qids(questionnaire_path)
        prompt = format_round3_prompt(
            prompts_path=self.prompts_path,
            questionnaire_path=questionnaire_path,
            case=case,
            debate_ctx=debate_context,
            repair_hint=repair_hint,
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
        # keep debate flow as-is; prompts may be added later if needed
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

    def _log_preview(self, obj: Any, tag: str) -> None:
        """log a short preview of backend output for debugging (best-effort)."""
        try:
            s = str(obj)
            self.logger.debug("%s %s", tag, s[:500])
        except Exception:
            # never let logging break the pipeline
            pass
