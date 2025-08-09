# models.py
# pydantic models for strict io contracts between moderator ↔ experts

from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class AssessmentBase(BaseModel):
    """shared fields and validators for round assessments."""

    scores: Dict[str, int] = Field(..., description="qid -> 1..9 likert score")
    evidence: Dict[str, str] = Field(..., description="qid -> brief evidence snippet")
    p_iraki: float = Field(..., ge=0.0, le=1.0, description="probability of irAKI")
    ci_iraki: Tuple[float, float] = Field(..., description="[lower, upper] in [0,1]")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="self-reported confidence 0..1"
    )

    @field_validator("scores")
    @classmethod
    def _likert_range(cls, v: Dict[str, int]):
        # ensure all scores are integers 1..9
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
    def _strict_schema_echo(self, info: ValidationInfo):
        # ensure qids in scores and evidence exactly match questionnaire expected_qids, if provided
        expected = set(info.context.get("expected_qids", []))
        if expected:
            s_keys, e_keys = set(self.scores.keys()), set(self.evidence.keys())
            if s_keys != expected:
                missing = expected - s_keys
                extra = s_keys - expected
                raise ValueError(
                    f"scores qids must match questionnaire. missing={sorted(missing)} extra={sorted(extra)}"
                )
            if e_keys != expected:
                missing = expected - e_keys
                extra = e_keys - expected
                raise ValueError(
                    f"evidence qids must match questionnaire. missing={sorted(missing)} extra={sorted(extra)}"
                )
        return self


class AssessmentR1(AssessmentBase):
    """round-1 expert assessment payload."""

    clinical_reasoning: str
    differential_diagnosis: List[str]
    primary_diagnosis: str


class AssessmentR3(AssessmentBase):
    """round-3 expert reassessment payload."""

    changes_from_round1: Dict[str, str] = Field(
        ..., description="qid or 'overall' -> change note"
    )
    debate_influence: str
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
