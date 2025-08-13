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

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MsgType(str, Enum):
    ASSESS_R1 = "assess_r1"
    DEBATE = "debate"
    ASSESS_R3 = "assess_r3"


class QuestionnaireMsg(BaseModel):
    """Patient case data with assessment questions for expert evaluation.

    Contains complete clinical context needed for irAKI assessment including
    demographics, lab values, medications, and structured questions.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    # case identification
    case_id: str
    round_phase: str  # "round1" or "round3"

    # minimal clinical payload
    patient_info: Dict[str, Any]  # demographics and small structured bits
    documents: Dict[str, List[str]]  # buckets of raw text, grouped by type

    # assessment framework
    questions: List[Dict[str, Any]]

    # optional for round 3
    prior_assessments: Optional[Dict[str, Any]] = None
    debate_summary: Optional[str] = None

    @field_validator("round_phase")
    @classmethod
    def validate_round_phase(cls, v: str) -> str:
        if v not in ["round1", "round3"]:
            raise ValueError(f"Invalid round phase: {v}. Must be 'round1' or 'round3'")
        return v

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not v:
            raise ValueError("Questions list cannot be empty")
        for q in v:
            if "id" not in q or "question" not in q:
                raise ValueError("Each question must have 'id' and 'question' fields")
        return v

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        # enforce lists of strings, but allow empty
        for k, arr in v.items():
            if not isinstance(arr, list) or any(not isinstance(x, str) for x in arr):
                raise ValueError(f"documents['{k}'] must be a list of strings")
        return v


class ExpertRound1Reply(BaseModel):
    """Expert's initial independent assessment of irAKI likelihood.

    Four parallel per-question dicts keyed by qid:
      - scores: 1..9 Likert signal for THIS question
      - rationale: concise argument/explanation for THIS question
      - evidence: supporting snippet/paragraph (verbatim or note-anchored)
      - q_confidence: 0..1 certainty for THIS question
      - importance: non-negative int; TOTAL importance across qids = 100 (contribution weight)
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    # identification
    case_id: str
    expert_id: str

    # per-question assessments
    scores: Dict[str, int]
    rationale: Dict[str, str]
    evidence: Dict[str, str]
    q_confidence: Dict[str, float]
    importance: Dict[str, int]

    # overall probability assessment
    p_iraki: float = Field(ge=0.0, le=1.0)
    ci_iraki: Tuple[float, float]
    confidence: float = Field(ge=0.0, le=1.0)

    # narrative & differential
    clinical_reasoning: str
    differential_diagnosis: List[str]
    primary_diagnosis: Optional[str] = None

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, v: Dict[str, int]) -> Dict[str, int]:
        for q_id, score in v.items():
            if not 1 <= score <= 9:
                raise ValueError(f"Score for {q_id} must be 1-9, got {score}")
        return v

    @field_validator("ci_iraki", mode="before")
    @classmethod
    def _coerce_ci(cls, v: Any) -> Tuple[float, float]:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (float(v[0]), float(v[1]))
        return v


class ExpertRound3Reply(BaseModel):
    """Expert's final consensus assessment after debate."""

    model_config = ConfigDict(strict=True, extra="forbid")

    # identification
    case_id: str
    expert_id: str

    # per-question assessments
    scores: Dict[str, int]
    rationale: Dict[str, str]
    evidence: Dict[str, str]
    q_confidence: Dict[str, float]
    importance: Dict[str, int]

    # probability assessment
    p_iraki: float = Field(ge=0.0, le=1.0)
    ci_iraki: Tuple[float, float]
    confidence: float = Field(ge=0.0, le=1.0)

    # deltas & verdict
    changes_from_round1: Dict[str, str]
    verdict: bool
    final_diagnosis: str
    confidence_in_verdict: float = Field(ge=0.0, le=1.0)

    # recommendations
    recommendations: List[str]

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, v: Dict[str, int]) -> Dict[str, int]:
        for q_id, score in v.items():
            if not 1 <= score <= 9:
                raise ValueError(f"Score for {q_id} must be 1-9, got {score}")
        return v

    @field_validator("ci_iraki", mode="before")
    @classmethod
    def _coerce_ci_r3(cls, v: Any) -> Tuple[float, float]:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (float(v[0]), float(v[1]))
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
