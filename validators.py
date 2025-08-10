# validators.py
# instructor-based structured output with robust retries.
# stable api: backend is OPTIONAL; accept prompt or prompt_text; provide helper utilities.

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from instructor import Mode, patch
from openai import OpenAI


class PayloadValidationError(RuntimeError):
    """raised when model output cannot be coerced into the expected structured payload."""


# backward-compat alias for older imports
class ValidationError(PayloadValidationError):
    pass


__all__ = [
    "PayloadValidationError",
    "ValidationError",
    "call_llm_with_schema",
    "validate_round1_payload",
    "validate_round3_payload",
    "log_debate_status",
]


def _infer_endpoint_and_model(
    backend: Any | None, endpoint_url: Optional[str], model: Optional[str]
) -> tuple[str, str]:
    """infer endpoint and model from args/env/backend.

    priority:
      1) explicit args
      2) env OPENAI_BASE_URL/OPENAI_MODEL
      3) backend attributes (_base/endpoint_url, _model/model/model_name)
    """
    # explicit
    ep = (endpoint_url or "").strip()
    mdl = (model or "").strip()

    # env
    if not ep:
        ep = os.getenv("OPENAI_BASE_URL", "").strip()
    if not mdl:
        mdl = os.getenv("OPENAI_MODEL", "").strip()

    # backend
    if backend is not None:
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
            "validators: cannot infer endpoint_url; pass endpoint_url or set OPENAI_BASE_URL"
        )
    if not mdl:
        raise ValueError(
            "validators: cannot infer model name; pass model or set OPENAI_MODEL"
        )

    ep = ep.rstrip("/")
    if not ep.endswith("/v1"):
        ep = ep + "/v1"
    return ep, mdl


def call_llm_with_schema(
    backend: Any | None = None,
    *,
    prompt: Optional[str] = None,
    prompt_text: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None,
    max_retries: int = 5,
    **_: Any,  # ignore unknown kwargs from older call sites
) -> Dict[str, Any]:
    """generate and validate a structured payload using Instructor.

    Args:
        backend: optional backend object; used to discover endpoint/model if not given.
        prompt/prompt_text: prompt string (either key works).
        endpoint_url, model: override discovery.
        temperature, max_tokens, stop, seed: usual generation settings.
        max_retries: structured retries (default 5).

    Returns:
        dict[str, Any]: parsed payload guaranteed to be valid json.
    """
    text = (prompt_text or prompt or "").strip()
    if not text:
        raise PayloadValidationError("call_llm_with_schema: empty prompt")

    base_url, model_name = _infer_endpoint_and_model(backend, endpoint_url, model)
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

    # set default mode=JSON at patch time; do NOT pass `mode` again in create()
    client = patch(OpenAI(base_url=base_url, api_key=api_key), mode=Mode.JSON)

    try:
        result = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": text}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            response_model=dict[str, Any],
            # no 'mode' here to avoid double-passing in some instructor versions
            # retries handled by instructor using the patched client
            max_retries=max_retries,
        )
    except Exception as e:  # noqa: BLE001
        raise PayloadValidationError(f"instructor call failed: {e}") from e

    if isinstance(result, dict):
        return result
    dump = getattr(result, "model_dump", None)
    if callable(dump):
        return dump()
    raise PayloadValidationError(f"unexpected instructor return type: {type(result)}")


# --- lightweight validators used by moderator.py ---


def validate_round1_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict) or not payload:
        raise ValidationError("round1: payload must be a non-empty object")
    if "probability" in payload:
        p = payload["probability"]
        try:
            pf = float(p)
        except Exception as e:  # noqa: BLE001
            raise ValidationError(f"round1: probability not a number: {p}") from e
        if not (0.0 <= pf <= 1.0):
            raise ValidationError(f"round1: probability out of range: {p}")
    if "evidence" in payload and not isinstance(payload["evidence"], (list, str)):
        raise ValidationError("round1: evidence must be a list or string")


def validate_round3_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict) or not payload:
        raise ValidationError("round3: payload must be a non-empty object")
    if "final_probability" in payload:
        p = payload["final_probability"]
        try:
            pf = float(p)
        except Exception as e:  # noqa: BLE001
            raise ValidationError(f"round3: final_probability not a number: {p}") from e
        if not (0.0 <= pf <= 1.0):
            raise ValidationError(f"round3: final_probability out of range: {p}")


def log_debate_status(*, disagreement_present: bool, logger) -> None:
    try:
        logger.info(
            "debate status: %s", "disagreement" if disagreement_present else "converged"
        )
    except Exception:
        pass
