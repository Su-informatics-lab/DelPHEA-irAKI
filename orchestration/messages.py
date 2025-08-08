"""
Message Types for DelPHEA-irAKI Agent Communication
====================================================
Defines all message contracts for the multi-agent Delphi consensus process.
These messages ensure type-safe, validated communication between agents.

Message Flow Architecture:
-------------------------
    Moderator                   Experts
        │                          │
        ├──> QuestionnaireMsg ─────┤ (Round 1 & 3)
        │<── ExpertRound1Reply ────┤
        │                          │
        ├──> DebatePrompt ─────────┤ (Round 2)
        │<── DebateComment ────────┤
        │                          │
        └──> TerminateDebate ──────┘

Key Message Types:
-----------------
- QuestionnaireMsg: Patient case data + assessment questions
- ExpertRound1Reply: Initial independent assessment
- DebatePrompt: Conflict resolution trigger
- DebateComment: Expert arguments during debate
- ExpertRound3Reply: Final consensus assessment
- StartCase: Bootstrap signal
- AckMsg: RPC acknowledgments

Clinical Context:
----------------
These messages capture the complete clinical reasoning process for
distinguishing immune-related AKI from alternative causes, including
differential diagnoses, evidence chains, and confidence metrics.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuestionnaireMsg(BaseModel):
    """Patient case data with assessment questions for expert evaluation.

    Contains complete clinical context needed for irAKI assessment including
    demographics, lab values, medications, and structured questions.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    # case identification
    case_id: str
    round_phase: str  # "round1" or "round3"

    # clinical data
    patient_info: Dict[str, Any]
    icu_summary: str
    medication_history: Dict[str, Any]
    lab_values: Dict[str, Any]
    imaging_reports: str

    # assessment framework
    questions: List[Dict[str, Any]]

    # optional fields for round 3
    prior_assessments: Optional[Dict[str, Any]] = None
    debate_summary: Optional[str] = None

    @field_validator("round_phase")
    @classmethod
    def validate_round_phase(cls, v: str) -> str:
        """Ensure round phase is valid."""
        if v not in ["round1", "round3"]:
            raise ValueError(f"Invalid round phase: {v}. Must be 'round1' or 'round3'")
        return v

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure questions have required structure."""
        if not v:
            raise ValueError("Questions list cannot be empty")
        for q in v:
            if "id" not in q or "question" not in q:
                raise ValueError("Each question must have 'id' and 'question' fields")
        return v


class ExpertRound1Reply(BaseModel):
    """Expert's initial independent assessment of irAKI likelihood.

    Captures complete clinical reasoning including scored responses,
    evidence citations, probability estimates, and differential diagnosis.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    # identification
    case_id: str
    expert_id: str

    # scored assessments (1-10 scale per question)
    scores: Dict[str, int]

    # evidence and reasoning
    evidence: Dict[str, str]  # question_id -> evidence text
    clinical_reasoning: str  # overall reasoning narrative

    # probability assessment
    p_iraki: float = Field(ge=0.0, le=1.0)  # P(irAKI)
    ci_iraki: Tuple[float, float]  # 95% CI
    confidence: float = Field(ge=0.0, le=1.0)  # self-assessed confidence

    # differential diagnosis
    differential_diagnosis: List[str]  # alternative diagnoses considered
    primary_diagnosis: Optional[str] = None  # most likely diagnosis

    # optional specialty-specific insights
    specialty_notes: Optional[str] = None
    literature_citations: List[str] = Field(default_factory=list)

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Ensure scores are in valid range."""
        for q_id, score in v.items():
            if not 1 <= score <= 10:
                raise ValueError(f"Score for {q_id} must be 1-10, got {score}")
        return v

    @field_validator("ci_iraki")
    @classmethod
    def validate_ci_bounds(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Ensure CI bounds are valid and ordered."""
        lower, upper = v
        if not (0.0 <= lower <= 1.0 and 0.0 <= upper <= 1.0):
            raise ValueError(f"CI bounds must be in [0,1], got [{lower}, {upper}]")
        if lower > upper:
            # auto-correct inverted bounds
            return (upper, lower)
        return v

    @field_validator("differential_diagnosis")
    @classmethod
    def validate_differential(cls, v: List[str]) -> List[str]:
        """Ensure differential diagnosis is provided."""
        if not v:
            raise ValueError("Must provide at least one differential diagnosis")
        return v


class DebatePrompt(BaseModel):
    """Triggers expert debate on conflicting assessments.

    Identifies questions with significant disagreement and provides
    context for structured debate among experts.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    case_id: str
    q_id: str  # question identifier
    question: Dict[str, Any]  # full question object with context

    # conflict information
    score_distribution: Dict[str, int]  # expert_id -> score
    score_range: str  # e.g., "2-9"
    conflict_severity: str  # "moderate" or "severe"

    # debate context
    conflicting_evidence: Dict[str, str]  # expert_id -> evidence
    clinical_importance: str  # why this conflict matters

    @field_validator("conflict_severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Ensure severity level is valid."""
        if v not in ["moderate", "severe"]:
            raise ValueError(f"Invalid severity: {v}")
        return v


class DebateComment(BaseModel):
    """Expert's contribution to debate on a specific question.

    Structured argument with evidence and satisfaction indicator.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    q_id: str
    author: str  # expert_id
    text: str  # argument text

    # supporting information
    citations: List[str] = Field(default_factory=list)
    evidence_type: Optional[str] = None  # "clinical", "literature", "guidelines"

    # debate progress
    satisfied: bool = False  # ready to end debate
    revised_score: Optional[int] = None  # if expert changes their assessment

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure comment has content."""
        if not v or len(v.strip()) < 10:
            raise ValueError("Debate comment must be substantive (>10 chars)")
        return v


class ExpertRound3Reply(BaseModel):
    """Expert's final consensus assessment after debate.

    Incorporates learnings from debate with updated reasoning and
    final irAKI classification verdict.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    # identification
    case_id: str
    expert_id: str

    # updated assessments
    scores: Dict[str, int]
    evidence: Dict[str, str]

    # final probability assessment
    p_iraki: float = Field(ge=0.0, le=1.0)
    ci_iraki: Tuple[float, float]
    confidence: float = Field(ge=0.0, le=1.0)

    # changes from round 1
    changes_from_round1: Dict[str, str]  # what changed and why
    debate_influence: Optional[str] = None  # how debate affected assessment

    # final clinical judgment
    verdict: bool  # True = irAKI, False = other cause
    final_diagnosis: str  # specific diagnosis
    confidence_in_verdict: float = Field(ge=0.0, le=1.0)

    # clinical recommendations
    recommendations: List[str]  # treatment/monitoring recommendations
    biopsy_recommendation: Optional[str] = None  # if biopsy indicated
    steroid_recommendation: Optional[str] = None  # steroid treatment guidance
    ici_rechallenge_risk: Optional[str] = None  # risk assessment for ICI restart

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Ensure scores are in valid range."""
        for q_id, score in v.items():
            if not 1 <= score <= 10:
                raise ValueError(f"Score for {q_id} must be 1-10, got {score}")
        return v

    @field_validator("ci_iraki")
    @classmethod
    def validate_ci_bounds(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """Ensure CI bounds are valid and ordered."""
        lower, upper = v
        if not (0.0 <= lower <= 1.0 and 0.0 <= upper <= 1.0):
            raise ValueError(f"CI bounds must be in [0,1], got [{lower}, {upper}]")
        if lower > upper:
            # auto-correct inverted bounds
            return (upper, lower)
        return v

    @field_validator("recommendations")
    @classmethod
    def validate_recommendations(cls, v: List[str]) -> List[str]:
        """Ensure clinical recommendations are provided."""
        if not v:
            raise ValueError("Must provide at least one clinical recommendation")
        return v


class TerminateDebate(BaseModel):
    """Signal to end debate on a specific question.

    Sent by moderator when consensus reached or timeout.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    case_id: str
    q_id: str
    reason: str  # "consensus", "timeout", "sufficient_discussion"

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        """Ensure termination reason is valid."""
        valid_reasons = ["consensus", "timeout", "sufficient_discussion", "no_progress"]
        if v not in valid_reasons:
            raise ValueError(f"Invalid termination reason: {v}")
        return v


class StartCase(BaseModel):
    """Bootstrap signal that kicks off the Delphi workflow.

    Besides the case identifier, the moderator may receive a
    lightweight snapshot of patient data and the roster of expert IDs
    (useful when a distributed service launches the run).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    # required
    case_id: str

    # optional metadata
    priority: str = "normal"  # "urgent", "normal", "research"
    patient_data: Optional[Dict[str, Any]] = None  # minimal EHR bundle
    expert_ids: Optional[List[str]] = None  # full list or subset

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        if v not in {"urgent", "normal", "research"}:
            raise ValueError(f"Invalid priority: {v}")
        return v


class AckMsg(BaseModel):
    """Acknowledgment for RPC calls between agents.

    Provides success/failure status with optional error details.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    ok: bool
    message: Optional[str] = None
    error_code: Optional[str] = None  # for structured error handling
    retry_after: Optional[int] = None  # seconds to wait before retry


class HumanReviewExport(BaseModel):
    """Complete case export for human expert validation.

    Contains all assessments, debates, and consensus results for
    clinical validation and quality assurance.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    # case identification
    case_id: str
    export_timestamp: str
    delphea_version: str = "1.0.0"

    # consensus results
    final_consensus: Dict[str, Any]  # beta pooling results
    majority_verdict: bool  # simple majority vote
    expert_agreement_level: float  # 0-1 agreement score

    # detailed assessments
    expert_assessments: List[Dict[str, Any]]  # all expert replies
    debate_transcripts: List[Dict[str, Any]]  # all debate comments

    # clinical summary
    reasoning_summary: str  # synthesized reasoning
    clinical_timeline: Dict[str, Any]  # key events timeline
    differential_summary: List[str]  # consolidated differential

    # recommendations
    consensus_recommendations: List[str]
    dissenting_opinions: Optional[List[Dict[str, str]]] = None

    # quality metrics
    confidence_metrics: Dict[str, float]
    debate_quality_score: Optional[float] = None
    evidence_completeness: Optional[float] = None
