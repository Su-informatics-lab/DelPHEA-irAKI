# validators.py
# structured output via Instructor (pydantic-powered) with robust retries.
# replaces ad-hoc json parsing/repair loops with a single call that returns a dict.
#
# usage
# -----
# from validators import call_llm_with_schema, PayloadValidationError
# payload = call_llm_with_schema(backend, prompt_text)
#
# design notes
# ------------
# - instructor + OpenAI client pointed at your vLLM server (openai-compatible)
# - Mode.JSON avoids requiring server-side json_schema support
# - returns a python dict on success; raises PayloadValidationError on failure
# - fails fast with clear, actionable errors
#
# dependencies
# ------------
# pip install "openai>=1.40.0" "instructor>=1.4.0" "pydantic>=2.7"
#
# env
# ---
# export OPENAI_API_KEY=EMPTY            # vllm usually doesn't check it, but sdk requires it
# export OPENAI_BASE_URL=http://localhost:8000 # optional, validator reads from backend by default

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from instructor import Mode, patch
from openai import OpenAI


class PayloadValidationError(RuntimeError):
    """raised when model output cannot be coerced into the expected structured payload."""


def _infer_endpoint_and_model(
    backend: Any, endpoint_url: Optional[str], model: Optional[str]
) -> tuple[str, str]:
    """infer endpoint and model from args or backend.

    prefers explicit args; falls back to known attributes on LLMBackend.
    """
    # explicit wins
    ep = (endpoint_url or "").strip()
    mdl = (model or "").strip()

    # try environment override if not provided
    if not ep:
        ep = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    if not mdl:
        mdl = os.getenv("OPENAI_MODEL", "").strip()

    # try backend introspection (supports our llm_backend)
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

    # openai sdk expects a /v1 base url
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
    """generate and validate a structured payload using Instructor.

    Args:
        backend: existing backend object (used only to discover endpoint/model).
        prompt_text: fully formatted prompt to send.
        endpoint_url: override openai base url (e.g., http://localhost:8000).
        model: override model id (e.g., openai/gpt-oss-120b).
        temperature: sampling temperature.
        max_tokens: max new tokens.
        stop: optional stop strings.
        seed: optional deterministic seed if supported by server.
        max_retries: number of structured retries (default 5).

    Returns:
        dict[str, Any]: parsed payload guaranteed to be valid json.

    Raises:
        PayloadValidationError: on network errors or if retries exhausted.
        ValueError: when endpoint/model cannot be inferred.
    """
    base_url, model_name = _infer_endpoint_and_model(backend, endpoint_url, model)

    # openai python sdk requires an api key string; vllm typically ignores it
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

    client = patch(OpenAI(base_url=base_url, api_key=api_key))

    try:
        # response_model=dict[str, Any] directs instructor to return a plain dict
        result = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            response_model=dict[str, Any],
            mode=Mode.JSON,  # do not rely on server-side json schema
            max_retries=max_retries,  # bump retries from 2 → 5
        )
    except Exception as e:  # noqa: BLE001
        raise PayloadValidationError(f"instructor call failed: {e}") from e

    if not isinstance(result, dict):
        # instructor may return a pydantic model; coerce to dict if so
        try:
            # pydantic v2 compatible
            dump = getattr(result, "model_dump", None)
            if callable(dump):
                return dump()
        except Exception:
            pass
        # last resort: this should not happen with response_model=dict[str, Any]
        raise PayloadValidationError(
            f"unexpected instructor return type: {type(result)}"
        )

    return result
