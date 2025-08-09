"""
validators.py
=============
minimal pydantic-based validators for delphea-iraki expert outputs.

yagni principles:
- validate only what we truly use for downstream consensus
- fail fast, with explicit messages so the moderator can re-prompt a single expert
- keep imports light and compatible with pydantic v1/v2

usage
-----
from validators import (
    validate_round1_payload,
    validate_round3_payload,
    log_debate_status,
    ValidationError,
)

try:
    validate_round1_payload(payload, required_evidence=12)
except ValidationError as e:
    logger.error("round1 validation failed: %s", e)
    # re-prompt the expert with e.errors()

try:
    validate_round3_payload(payload)
except ValidationError as e:
    logger.error("round3 validation failed: %s", e)

"""

# pydantic v2 ships as pydantic, but we keep a v1-compatible import path if available
try:  # pragma: no cover
    from pydantic.v1 import (
        BaseModel,
        Field,
        ValidationError,
        conlist,
        constr,
        root_validator,
    )
except Exception:  # pragma: no cover
    from pydantic import BaseModel, Field, ValidationError, root_validator, conlist, constr  # type: ignore

import logging
from typing import Dict, Optional

NonEmptyStr = constr(min_length=1, strip_whitespace=True)  # noqa: N816


class Round1Model(BaseModel):
    """schema for expert round 1 payload.

    fields are intentionally minimal and reflect what downstream code uses.
    """

    clinical_reasoning: constr(min_length=200, strip_whitespace=True)  # type: ignore
    primary_diagnosis: NonEmptyStr  # type: ignore
    differential_diagnosis: conlist(NonEmptyStr, min_items=2)  # type: ignore
    evidence: Dict[str, NonEmptyStr]  # type: ignore

    @root_validator
    def _check_evidence_completeness(cls, values):
        """require a minimum number of non-empty evidence items (default 12)."""
        ev = values.get("evidence") or {}
        filled = sum(1 for v in ev.values() if isinstance(v, str) and v.strip())
        # store count for external message clarity
        values["_filled_evidence"] = filled
        return values


class ChangesFromRound1(BaseModel):
    summary: NonEmptyStr  # type: ignore
    debate_influence: NonEmptyStr  # type: ignore


class Round3Model(BaseModel):
    """schema for expert round 3 payload."""

    changes_from_round1: ChangesFromRound1
    final_diagnosis: NonEmptyStr  # type: ignore
    recommendations: conlist(NonEmptyStr, min_items=1)  # type: ignore


def validate_round1_payload(
    payload: dict,
    required_evidence: int = 12,
) -> Round1Model:
    """validate round 1 payload; raises ValidationError on failure.

    args:
        payload: raw dict parsed from expert json.
        required_evidence: minimum number of non-empty evidence items.

    returns:
        validated Round1Model (can be used downstream).

    raises:
        ValidationError: when any field fails constraints.
    """
    model = Round1Model(**payload)
    filled = getattr(model, "_filled_evidence", 0)
    if filled < required_evidence:
        # raise a structured error consistent with pydantic ValidationError
        raise ValidationError.from_exception_data(
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


def validate_round3_payload(payload: dict) -> Round3Model:
    """validate round 3 payload; raises ValidationError on failure.

    args:
        payload: raw dict parsed from expert json.

    returns:
        validated Round3Model.
    """
    return Round3Model(**payload)


def log_debate_status(
    disagreement_present: bool,
    logger: Optional[logging.Logger] = None,
) -> None:
    """log a concise message explaining whether debate was skipped or executed.

    args:
        disagreement_present: true if experts materially disagree and a debate occurred.
        logger: optional logger; defaults to module logger.
    """
    lg = logger or logging.getLogger("delphea_iraki")
    if disagreement_present:
        lg.info(
            "debate executed: moderator detected material disagreement among experts"
        )
    else:
        lg.info(
            "debate skipped: no material disagreement detected; proceeding without debate"
        )


if __name__ == "__main__":
    # tiny self-check demo (will raise ValidationError if constraints are not met)
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
    import sys

    try:
        validate_round1_payload(demo_round1)
        validate_round3_payload(demo_round3)
        log_debate_status(disagreement_present=False)
        print("demo validation passed")
    except ValidationError as e:
        print("validation failed:", e, file=sys.stderr)
        raise
