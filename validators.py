# validators.py
# structured output via Instructor (pydantic-powered) with robust retries,
# plus lightweight validators and logging utilities expected by moderator.py.

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from instructor import Mode, patch
from openai import OpenAI


class PayloadValidationError(RuntimeError):
    """raised when model output cannot be coerced into the expected structured payload."""


# backward‑compat shim for older modules importing `ValidationError` from validators
class ValidationError(PayloadValidationError):
    """alias kept for compatibility with older import sites."""


__all__ = [
    "PayloadValidationError",
    "ValidationError",
    "call_llm_with_schema",
    "validate_round1_payload",
    "validate_round3_payload",
    "log_debate_status",
]


def _infer_endpoint_and_model(
    backend: Any, endpoint_url: Optional[str], model: Optional[str]
) -> tuple[str, str]:
    """infer endpoint and model from args or backend."""
    ep = (endpoint_url or "").strip()
    mdl = (model or "").strip()

    if not ep:
        ep = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    if not mdl:
        mdl = os.getenv("OPENAI_MODEL", "").strip()

    if not ep:
        ep = getattr(backend, "_base", "") or getattr(backend, "endpoint_url", "")
    if not mdl:
        mdl = (
            getattr(backend, "_model", "")
            or getattr(backend, "model", "")
            or getattr(backend, "model_name", "")
        )

    if not ep:
        raise ValueError(
            "validators: cannot infer endpoint_url; pass explicitly or set OPENAI_BASE_URL"
        )
    if not mdl:
        raise ValueError(
            "validators: cannot infer model name; pass explicitly or set OPENAI_MODEL"
        )

    ep = ep.rstrip("/")
    if not ep.endswith("/v1"):
        ep = ep + "/v1"
    return ep, mdl


def call_llm_with_schema(
    backend: Any,
    prompt_text: str,
    *,
    endpoint_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """generate and validate a structured payload using Instructor."""
    base_url, model_name = _infer_endpoint_and_model(backend, endpoint_url, model)
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    client = patch(OpenAI(base_url=base_url, api_key=api_key))
    try:
        result = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            response_model=dict[str, Any],
            mode=Mode.JSON,
            max_retries=max_retries,
        )
    except Exception as e:  # noqa: BLE001
        raise PayloadValidationError(f"instructor call failed: {e}") from e

    if isinstance(result, dict):
        return result
    # handle potential pydantic model returns
    dump = getattr(result, "model_dump", None)
    if callable(dump):
        return dump()
    raise PayloadValidationError(f"unexpected instructor return type: {type(result)}")


# ------------------------- light validators -------------------------


def validate_round1_payload(
    payload: Dict[str, Any], *, required_evidence: int = 0
) -> None:
    """lightweight structural sanity check for round 1 payload.

    we keep this permissive to avoid blocking valid outputs across models/versions.
    raise ValidationError for obviously bad shapes so the moderator can trigger a retry.
    """
    if not isinstance(payload, dict) or not payload:
        raise ValidationError("round1 payload must be a non-empty dict")

    # common fields we expect (optional)
    # if present, do minimal checks; do not enforce strict schema here
    ev = payload.get("evidence") or payload.get("evidences")
    if ev is not None and not isinstance(ev, list):
        raise ValidationError("round1 payload 'evidence' must be a list if present")
    if isinstance(ev, list) and required_evidence and len(ev) < required_evidence:
        # allow progress even if not enough evidence, but signal as a validation error to trigger retry flow
        raise ValidationError(
            f"round1 payload has only {len(ev)} evidence items; require >= {required_evidence}"
        )

    # probabilities if present must be within [0,1]
    for k in ("p_iraki", "probability", "prob_iraki"):
        if k in payload:
            p = payload[k]
            try:
                p = float(p)
            except Exception as _:
                raise ValidationError(f"round1 payload '{k}' must be numeric")
            if not (0.0 <= p <= 1.0):
                raise ValidationError(f"round1 payload '{k}' out of range [0,1]")


def validate_round3_payload(payload: Dict[str, Any]) -> None:
    """lightweight structural sanity check for round 3 payload."""
    if not isinstance(payload, dict) or not payload:
        raise ValidationError("round3 payload must be a non-empty dict")
    # allow very permissive checks; round 3 typically mirrors round 1 with revisions
    # ensure probability (if present) is sane
    for k in ("p_iraki", "probability", "prob_iraki"):
        if k in payload:
            p = payload[k]
            try:
                p = float(p)
            except Exception as _:
                raise ValidationError(f"round3 payload '{k}' must be numeric")
            if not (0.0 <= p <= 1.0):
                raise ValidationError(f"round3 payload '{k}' out of range [0,1]")


# ------------------------- logging helpers -------------------------


def log_debate_status(*, disagreement_present: bool, logger) -> None:
    """emit a single operator-facing log line about whether a debate will run."""
    try:
        if disagreement_present:
            logger.info("round 2 (debate) scheduled: disagreements detected")
        else:
            logger.info("round 2 (debate) skipped: no disagreements detected")
    except Exception:
        # logger may be None in some tests; silently ignore
        pass
