"""
Message schemas for DelPHEA-irAKI agent communication.

Pydantic models defining all inter-agent messages for the Delphi consensus process.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuestionnaireMsg(BaseModel):
    """Round 1 and Round 3 questionnaire message."""

    model_config = ConfigDict(strict=True, extra="forbid")
    case_id: str
    patient_info: dict
    icu_summary: str
    medication_history: dict
    lab_values: dict
    imaging_reports: str
    questions: List[dict]  # Full question objects with contexts
    round_phase: str = "round1"


class ExpertRound1Reply(BaseModel):
    """Expert's Round 1 assessment reply."""

    model_config = ConfigDict(strict=True, extra="forbid")
    case_id: str
    expert_id: str
    scores: Dict[str, int]
    evidence: Dict[str, str]
    p_iraki: float = Field(ge=0.0, le=1.0)
    ci_iraki: List[float]
    confidence: float = Field(ge=0.0, le=1.0)
    differential_diagnosis: List[str]

    @field_validator("ci_iraki")
    def validate_ci_length(cls, v):
        if len(v) != 2:
            raise ValueError("ci_iraki must have exactly 2 values [lower, upper]")
        return v


class ExpertRound3Reply(BaseModel):
    """Expert's Round 3 final consensus reply."""

    model_config = ConfigDict(strict=True, extra="forbid")
    case_id: str
    expert_id: str
    scores: Dict[str, int]
    evidence: Dict[str, str]
    p_iraki: float = Field(ge=0.0, le=1.0)
    ci_iraki: List[float]
    confidence: float = Field(ge=0.0, le=1.0)
    changes_from_round1: Dict[str, str]
    verdict: bool
    final_diagnosis: str
    recommendations: List[str]

    @field_validator("ci_iraki")
    def validate_ci_length(cls, v):
        if len(v) != 2:
            raise ValueError("ci_iraki must have exactly 2 values [lower, upper]")
        return v


class DebatePrompt(BaseModel):
    """Round 2 debate initiation message."""

    model_config = ConfigDict(strict=True, extra="forbid")
    case_id: str
    q_id: str
    minority_view: str
    round_no: int
    participating_experts: List[str]
    clinical_context: Optional[Dict] = None


class DebateComment(BaseModel):
    """Expert's debate comment."""

    model_config = ConfigDict(strict=True, extra="forbid")
    q_id: str
    author: str
    text: str
    citations: List[str] = Field(default_factory=list)
    satisfied: bool = False


class TerminateDebate(BaseModel):
    """Signal to terminate a debate."""

    model_config = ConfigDict(strict=True, extra="forbid")
    case_id: str
    q_id: str
    reason: str


class StartCase(BaseModel):
    """Signal to start processing a case."""

    model_config = ConfigDict(strict=True, extra="forbid")
    case_id: str


class AckMsg(BaseModel):
    """Acknowledgment message for RPC calls."""

    model_config = ConfigDict(strict=True, extra="forbid")
    ok: bool
    message: Optional[str] = None


class HumanReviewExport(BaseModel):
    """Export format for human expert review."""

    model_config = ConfigDict(strict=True, extra="forbid")
    case_id: str
    final_consensus: Dict[str, Any]
    expert_assessments: List[Dict[str, Any]]
    debate_transcripts: List[Dict[str, Any]]
    reasoning_summary: str
    clinical_timeline: Dict[str, Any]
