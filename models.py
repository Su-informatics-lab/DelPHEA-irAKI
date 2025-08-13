# models.py
# pydantic models for strict io contracts between moderator ↔ experts

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,
)

# ----------------------- per-question tuple (new) -----------------------


class PerQuestion(BaseModel):
    """single question tuple emitted by an expert in r1/r3."""

    model_config = ConfigDict(strict=True, extra="forbid")

    qid: str = Field(..., description="question id (must match questionnaire)")
    score: int = Field(..., ge=1, le=9, description="1..9 likert")
    reason: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="brief rationale grounded in case facts",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="subjective certainty for THIS question"
    )
    importance: int = Field(
        ..., ge=0, le=100, description="integer points; all questions must sum to 100"
    )

    @field_validator("qid")
    @classmethod
    def _qid_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("qid cannot be empty")
        return v


# ----------------------------- round payloads -----------------------------


class AssessmentBase(BaseModel):
    """shared fields and validators for round assessments.

    new: answers[] tuples (score→reason→confidence→importance), constant-sum 100.
    retains backward-compat via computed fields .scores and .evidence.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    # per-question tuples (replaces scores/evidence dicts)
    answers: List[PerQuestion] = Field(..., min_length=1)

    # overall probability triplet (unchanged)
    p_iraki: float = Field(..., ge=0.0, le=1.0, description="probability of irAKI")
    ci_iraki: Tuple[float, float] = Field(..., description="[lower, upper] in [0,1]")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="overall self-reported confidence 0..1"
    )

    # audit fields (auto-filled)
    importance_total: int = 100
    importance_normalized: bool = False

    # ---- computed back-compat for legacy code that expects dicts ----
    @computed_field  # type: ignore[misc]
    @property
    def scores(self) -> Dict[str, int]:
        return {a.qid: a.score for a in self.answers}

    @computed_field  # type: ignore[misc]
    @property
    def evidence(self) -> Dict[str, str]:
        return {a.qid: a.reason for a in self.answers}

    # ---- structural checks ----
    @model_validator(mode="after")
    def _ci_contains_point_and_is_ordered(self) -> "AssessmentBase":
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
    def _enforce_unique_qids(self) -> "AssessmentBase":
        qids = [a.qid for a in self.answers]
        if len(set(qids)) != len(qids):
            raise ValueError("duplicate qids in answers")
        return self

    @model_validator(mode="after")
    def _check_expected_qids(self, info: ValidationInfo) -> "AssessmentBase":
        # optional strict check against questionnaire
        expected = list(info.context.get("expected_qids", [])) if info.context else []
        if expected:
            got = [a.qid for a in self.answers]
            if got != expected:
                missing = set(expected) - set(got)
                extra = set(got) - set(expected)
                raise ValueError(
                    f"answers.qids must match questionnaire order. missing={sorted(missing)} extra={sorted(extra)}"
                )
        return self

    @model_validator(mode="after")
    def _normalize_importance_once(self) -> "AssessmentBase":
        total = sum(a.importance for a in self.answers)
        self.importance_total = total
        if total == 100:
            return self

        # largest remainder (hamilton) normalization to exactly 100
        if total < 0:
            raise ValueError("sum of importance cannot be negative")

        if total == 0:
            base = 100 // len(self.answers)
            rem = 100 - base * len(self.answers)
            for i, a in enumerate(self.answers):
                a.importance = base + (1 if i < rem else 0)
            self.importance_total = 100
            self.importance_normalized = True
            return self

        scaled = [a.importance * 100.0 / total for a in self.answers]
        floors = [int(x) for x in scaled]
        remainder = 100 - sum(floors)
        # distribute by largest fractional parts
        frac = [(i, scaled[i] - floors[i]) for i in range(len(self.answers))]
        frac.sort(key=lambda t: t[1], reverse=True)
        for j in range(remainder):
            floors[frac[j][0]] += 1
        for a, new_imp in zip(self.answers, floors):
            a.importance = new_imp

        self.importance_total = 100
        self.importance_normalized = True
        return self


class AssessmentR1(AssessmentBase):
    """round-1 expert assessment payload."""

    clinical_reasoning: str
    differential_diagnosis: List[str]
    primary_diagnosis: Optional[str] = None

    @field_validator("differential_diagnosis")
    @classmethod
    def _require_ddx(cls, v: List[str]) -> List[str]:
        if not v or len([x for x in v if isinstance(x, str) and x.strip()]) < 2:
            raise ValueError("provide ≥2 differential diagnoses")
        return v


class AssessmentR3(AssessmentBase):
    """round-3 expert reassessment payload."""

    changes_from_round1: Dict[str, str] = Field(
        ..., description="qid or 'overall' -> change note"
    )
    verdict: bool
    final_diagnosis: str
    confidence_in_verdict: float = Field(..., ge=0.0, le=1.0)
    recommendations: List[str]

    @field_validator("recommendations")
    @classmethod
    def _require_recs(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("must provide at least one clinical recommendation")
        return v


# ------------------------------- consensus --------------------------------


class Consensus(BaseModel):
    """final aggregated consensus."""

    model_config = ConfigDict(strict=True, extra="forbid")

    iraki_probability: float = Field(..., ge=0.0, le=1.0)
    verdict: bool
    consensus_confidence: float = Field(..., ge=0.0, le=1.0)
    ci_iraki: Tuple[float, float] = Field(..., description="[lower, upper] in [0,1]")

    # backward-compat alias so older code can access .p_iraki; also shows up in model_dump()
    @computed_field  # type: ignore[misc]
    @property
    def p_iraki(self) -> float:
        return self.iraki_probability
