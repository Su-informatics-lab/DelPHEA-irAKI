# models.py
# pydantic models for strict io contracts between moderator ↔ experts

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,
)


class AssessmentBase(BaseModel):
    """shared fields and validators for round assessments.

    Per-question fields are four parallel dicts keyed by qid (plus 'rationale'):
      - scores: 1..9 Likert for THIS question
      - rationale: concise argument/explanation for THIS question
      - evidence: supporting snippet/paragraph (quote or note-anchored)
      - q_confidence: 0..1 certainty for THIS question
      - importance: non-negative integers; TOTAL across qids = 100 (contribution weight)
    """

    # per-question dicts
    scores: Dict[str, int] = Field(...)
    rationale: Dict[str, str] = Field(...)
    evidence: Dict[str, str] = Field(...)
    q_confidence: Dict[str, float] = Field(...)
    importance: Dict[str, int] = Field(...)

    # overall probability triplet
    p_iraki: float = Field(..., ge=0.0, le=1.0, description="probability of irAKI")
    ci_iraki: Tuple[float, float] = Field(..., description="[lower, upper] in [0,1]")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="overall self-reported confidence 0..1"
    )

    @field_validator("scores")
    @classmethod
    def _likert_range(cls, v: Dict[str, int]):
        for qid, s in v.items():
            if not isinstance(s, int) or s < 1 or s > 9:
                raise ValueError(f"invalid likert score for {qid}: {s} (must be 1..9)")
        return v

    @model_validator(mode="after")
    def _ci_contains_point_and_is_ordered(self):
        lo, hi = self.ci_iraki
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"invalid ci_iraki bounds: {self.ci_iraki} (must satisfy 0<=lo<=hi<=1)"
            )
        if not (lo <= self.p_iraki <= hi):
            raise ValueError(
                f"p_iraki {self.p_iraki} must lie within ci_iraki {self.ci_iraki}"
            )
        return self

    @model_validator(mode="after")
    def _parallel_qid_consistency(self, info: ValidationInfo):
        # 5 dicts must share identical keys, and optionally match questionnaire order
        dicts = [
            self.scores,
            self.rationale,
            self.evidence,
            self.q_confidence,
            self.importance,
        ]
        keys = [list(d.keys()) for d in dicts]
        if not (keys[0] == keys[1] == keys[2] == keys[3] == keys[4]):
            raise ValueError(
                "scores/rationale/evidence/q_confidence/importance must have IDENTICAL qid lists and order"
            )

        expected = list(info.context.get("expected_qids", [])) if info.context else []
        if expected and keys[0] != expected:
            missing = sorted(set(expected) - set(keys[0]))
            extra = sorted(set(keys[0]) - set(expected))
            raise ValueError(
                f"qid order must match questionnaire. missing={missing} extra={extra}"
            )
        return self

    @model_validator(mode="after")
    def _importance_sum_to_100(self):
        total = sum(self.importance.values())
        if total != 100:
            raise ValueError(f"importance must sum to 100 (got {total})")
        for qid, c in self.q_confidence.items():
            try:
                cf = float(c)
            except Exception:
                cf = -1.0
            if not (0.0 <= cf <= 1.0):
                raise ValueError(f"q_confidence[{qid}] must be in [0,1]")
        for qid, txt in {**self.rationale, **self.evidence}.items():
            if not isinstance(txt, str) or not txt.strip():
                raise ValueError(f"text for {qid} must be a non-empty string")
        return self


class AssessmentR1(AssessmentBase):
    """round-1 expert assessment payload."""

    clinical_reasoning: str
    differential_diagnosis: List[str]
    primary_diagnosis: Optional[str] = None


class AssessmentR3(AssessmentBase):
    """round-3 expert reassessment payload."""

    changes_from_round1: Dict[str, str] = Field(
        ..., description="qid or 'overall' -> change note"
    )
    verdict: bool
    final_diagnosis: str
    confidence_in_verdict: float = Field(..., ge=0.0, le=1.0)
    recommendations: List[str]


class DebateTurn(BaseModel):
    """single debate turn from an expert."""

    text: str
    citations: List[str] = Field(default_factory=list)
    satisfied: bool


class Consensus(BaseModel):
    """final aggregated consensus."""

    iraki_probability: float = Field(..., ge=0.0, le=1.0)
    verdict: bool
    consensus_confidence: float = Field(..., ge=0.0, le=1.0)
    ci_iraki: Tuple[float, float] = Field(..., description="[lower, upper] in [0,1]")
    expert_count: int = Field(..., ge=1)

    @computed_field  # type: ignore[misc]
    @property
    def p_iraki(self) -> float:
        return self.iraki_probability
