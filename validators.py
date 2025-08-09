"""
validators.py
=============
production-ready validators for delphea-iraki expert i/o.

goals (yagni):
- validate only what downstream consensus uses
- fail fast with explicit messages
- provide a lightweight llm→json→validate→repair loop
- keep pydantic v1/v2 compatibility (prefer v1 shim in v2)

public api
----------
from validators import (
    # pydantic-based content validators
    validate_round1_payload,
    validate_round3_payload,
    log_debate_status,
    ValidationError,           # pydantic validation error

    # lightweight schema+content guard + llm retry
    PayloadValidationError,
    validate_round_payload,
    build_repair_prompt,
    call_llm_with_schema,
)
"""

from __future__ import annotations

# pydantic v2 ships as pydantic, but we keep a v1-compatible import path if available
try:  # pragma: no cover
    from pydantic.v1 import BaseModel, ValidationError, conlist, constr, root_validator
except Exception:  # pragma: no cover
    from pydantic import BaseModel, ValidationError, root_validator, conlist, constr  # type: ignore

import json
import logging
from typing import Any, Dict, List, Optional

__all__ = [
    # pydantic validators
    "validate_round1_payload",
    "validate_round3_payload",
    "log_debate_status",
    "ValidationError",
    # non-pydantic validation + llm repair
    "PayloadValidationError",
    "validate_round_payload",
    "build_repair_prompt",
    "call_llm_with_schema",
]


# -------------------- pydantic-based content validators --------------------

NonEmptyStr = constr(min_length=1, strip_whitespace=True)  # noqa: N816


class Round1Model(BaseModel):
    """schema for expert round 1 payload (minimal but sufficient)."""

    clinical_reasoning: constr(min_length=200, strip_whitespace=True)  # type: ignore
    primary_diagnosis: NonEmptyStr  # type: ignore
    differential_diagnosis: conlist(NonEmptyStr, min_items=2)  # type: ignore
    evidence: Dict[str, NonEmptyStr]  # type: ignore

    @root_validator
    def _check_evidence_completeness(cls, values):
        """require a minimum number of non-empty evidence items (count stored)."""
        ev = values.get("evidence") or {}
        filled = sum(1 for v in ev.values() if isinstance(v, str) and v.strip())
        values["_filled_evidence"] = filled
        return values


class ChangesFromRound1(BaseModel):
    summary: NonEmptyStr  # type: ignore
    debate_influence: NonEmptyStr  # type: ignore


class Round3Model(BaseModel):
    """schema for expert round 3 payload (minimal)."""

    changes_from_round1: ChangesFromRound1
    final_diagnosis: NonEmptyStr  # type: ignore
    recommendations: conlist(NonEmptyStr, min_items=1)  # type: ignore


def validate_round1_payload(
    payload: Dict[str, Any], required_evidence: int = 12
) -> Round1Model:
    """validate round 1 payload; raises pydantic ValidationError on failure."""
    model = Round1Model(**payload)
    filled = getattr(model, "_filled_evidence", 0)
    if filled < required_evidence:
        # raise a structured error consistent with pydantic ValidationError
        raise ValidationError.from_exception_data(  # type: ignore[attr-defined]
            "Round1Model",
            [
                {
                    "loc": ("evidence",),
                    "msg": f"need at least {required_evidence} non-empty evidence items, got {filled}",
                    "type": "value_error.evidence_insufficient",
                }
            ],
        )
    return model


def validate_round3_payload(payload: Dict[str, Any]) -> Round3Model:
    """validate round 3 payload; raises pydantic ValidationError on failure."""
    return Round3Model(**payload)


def log_debate_status(
    disagreement_present: bool, logger: Optional[logging.Logger] = None
) -> None:
    """log whether debate was skipped or executed (operator-facing clarity)."""
    lg = logger or logging.getLogger("delphea_iraki")
    if disagreement_present:
        lg.info(
            "debate executed: moderator detected material disagreement among experts"
        )
    else:
        lg.info(
            "debate skipped: no material disagreement detected; proceeding without debate"
        )


# -------------------- lightweight schema/content guard + llm retry --------------------


class PayloadValidationError(ValueError):
    """raised when llm outputs cannot be repaired into a valid payload."""


# default qids; allow override from caller to avoid hard-coding
_DEFAULT_QIDS = [f"Q{i}" for i in range(1, 16)]


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def validate_round_payload(
    payload: Dict[str, Any],
    round_no: int,
    qids: Optional[List[str]] = None,
) -> List[str]:
    """return a list of human-readable errors; empty list means valid.

    this is a fast, dependency-light check used before/around pydantic models.
    """
    errs: List[str] = []
    qids = qids or _DEFAULT_QIDS

    # common required keys
    required_keys = {"scores", "evidence", "p_iraki", "ci_iraki", "confidence"}
    missing = [k for k in required_keys if k not in payload]
    if missing:
        errs.append(f"missing keys: {missing}")
        return errs  # fail fast

    # scores: all qids present and 1–9 ints
    scores = payload["scores"]
    if not isinstance(scores, dict):
        errs.append("scores must be object {qid:int}")
    else:
        for q in qids:
            v = scores.get(q)
            if not isinstance(v, int) or not (1 <= v <= 9):
                errs.append(f"scores[{q}] must be int in 1..9")

    # evidence: all qids non-empty strings
    ev = payload["evidence"]
    if not isinstance(ev, dict):
        errs.append("evidence must be object {qid:str}")
    else:
        for q in qids:
            v = ev.get(q, "")
            if not _is_nonempty_str(v):
                errs.append(f"evidence[{q}] must be non-empty string")

    # p_iraki
    p = payload["p_iraki"]
    if not (isinstance(p, (int, float)) and 0.0 <= p <= 1.0):
        errs.append("p_iraki must be float in [0,1]")

    # ci_iraki
    ci = payload["ci_iraki"]
    if not (
        isinstance(ci, list)
        and len(ci) == 2
        and all(isinstance(x, (int, float)) for x in ci)
        and 0.0 <= ci[0] <= p <= ci[1] <= 1.0
    ):
        errs.append("ci_iraki must be [lower, upper] with 0<=lower<=p_iraki<=upper<=1")

    # confidence
    conf = payload["confidence"]
    if not (isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0):
        errs.append("confidence must be float in [0,1]")

    # round-specific minimal requirements
    if round_no == 1:
        if not _is_nonempty_str(payload.get("clinical_reasoning", "")):
            errs.append("clinical_reasoning must be non-empty string")
        if not isinstance(payload.get("differential_diagnosis", []), list):
            errs.append("differential_diagnosis must be a list")
        if not _is_nonempty_str(payload.get("primary_diagnosis", "")):
            errs.append("primary_diagnosis must be non-empty string")
    elif round_no == 3:
        for k in (
            "changes_from_round1",
            "debate_influence",
            "verdict",
            "final_diagnosis",
            "confidence_in_verdict",
            "recommendations",
        ):
            if k not in payload:
                errs.append(f"missing key in round3: {k}")

    return errs


def build_repair_prompt(last_prompt: str, last_output: str, errors: List[str]) -> str:
    """append a small fixer block to the original prompt; avoids empty fields."""
    err_bullets = "\n".join(f"- {e}" for e in errors)
    return (
        f"{last_prompt}\n\n"
        "JSON FIX NEEDED:\n"
        "Your previous JSON had the following problems:\n"
        f"{err_bullets}\n\n"
        "Please RETURN ONLY corrected JSON that fully satisfies the schema.\n"
        "Do not leave any field blank. If unknown, write a brief rationale such as "
        '"unknown due to missing urinalysis in notes" rather than an empty string.\n'
    )


def call_llm_with_schema(
    backend,
    prompt: str,
    round_no: int,
    max_retries: int = 2,
    qids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """generate → parse → validate → repair-prompt retry; else raise.

    args:
        backend: object exposing backend.generate(str) -> str
        prompt: initial prompt including schema instructions
        round_no: 1 or 3
        max_retries: number of repair attempts after the first try
        qids: optional explicit qid list to validate against

    returns:
        validated payload dict

    raises:
        PayloadValidationError: if json malformed or validation fails after retries
    """
    last_prompt = prompt
    last_text = ""
    for attempt in range(max_retries + 1):
        last_text = backend.generate(last_prompt)
        try:
            payload = json.loads(last_text)
        except json.JSONDecodeError as e:
            if attempt == max_retries:
                raise PayloadValidationError(
                    f"invalid json after {attempt} retries: {e}"
                ) from e
            last_prompt = build_repair_prompt(
                last_prompt, last_text, [f"invalid json: {e}"]
            )
            continue

        errs = validate_round_payload(payload, round_no, qids=qids)
        if not errs:
            return payload

        if attempt == max_retries:
            raise PayloadValidationError(
                f"schema/content errors after {attempt} retries: {errs}"
            )
        last_prompt = build_repair_prompt(last_prompt, last_text, errs)

    # unreachable in normal flow
    raise PayloadValidationError("unreachable retry loop exit")


# -------------------- tiny self-check --------------------

if __name__ == "__main__":
    # minimal smoke tests (will raise on failure)
    demo_round1 = {
        "clinical_reasoning": "x" * 210,
        "primary_diagnosis": "immune-related acute interstitial nephritis (suspected)",
        "differential_diagnosis": ["prerenal azotemia", "contrast-associated aki"],
        "evidence": {f"Q{i}": "present" for i in range(1, 13)},
    }
    demo_round3 = {
        "changes_from_round1": {
            "summary": "refined diagnosis after considering round 1 feedback",
            "debate_influence": "no debate occurred; no changes from peer arguments",
        },
        "final_diagnosis": "probable irAKI",
        "recommendations": ["hold ppi", "consider renal consult"],
    }
    try:
        validate_round1_payload(demo_round1)
        validate_round3_payload(demo_round3)
        log_debate_status(disagreement_present=False)
        print("demo validation passed")
    except ValidationError as e:
        print("validation failed:", e)
        raise
