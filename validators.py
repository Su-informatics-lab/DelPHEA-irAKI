# validators.py
# instructor-based structured output with robust retries.
# stable api: backend is OPTIONAL; accept prompt or prompt_text; provide helper utilities.

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple, Type

from instructor import Mode, patch
from openai import OpenAI
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError


class PayloadValidationError(RuntimeError):
    """raised when model output cannot be coerced into the expected structured payload."""


class ValidationError(ValueError):
    """raised when a round payload is syntactically invalid."""


def _require_base_model(response_model: Any) -> Type[BaseModel]:
    """ensure response_model is a pydantic BaseModel subclass; fail loud."""
    if not isinstance(response_model, type) or not issubclass(
        response_model, BaseModel
    ):
        typename = getattr(response_model, "__name__", str(type(response_model)))
        raise PayloadValidationError(
            f"response_model must be a subclass of pydantic.BaseModel; got {typename}. "
            "use the reply models from messages.py, e.g.:\n"
            "  from messages import ExpertRound1Reply, ExpertRound3Reply\n"
            "  ... response_model=ExpertRound1Reply\n"
        )
    return response_model


def _infer_endpoint_and_model(
    *, endpoint_url: Optional[str], model: Optional[str], backend: Optional[Any]
) -> Tuple[str, str]:
    """derive endpoint base url and model name, fail if missing."""
    ep = (endpoint_url or "").strip() or os.getenv("OPENAI_BASE_URL", "").strip()
    mdl = (model or "").strip() or os.getenv("OPENAI_MODEL", "").strip()

    # allow backend to provide hints, but do not silently guess
    if backend is not None:
        if not ep:
            ep = (
                getattr(backend, "_base", "")
                or getattr(backend, "endpoint_url", "")
                or getattr(backend, "base_url", "")
            )
        if not mdl:
            mdl = (
                getattr(backend, "model", "")
                or getattr(backend, "model_name", "")
                or getattr(backend, "engine", "")
            )

    if not ep:
        raise ValueError(
            "validators: cannot infer endpoint_url; pass endpoint_url explicitly or set OPENAI_BASE_URL"
        )
    if not mdl:
        raise ValueError(
            "validators: cannot infer model name; pass model explicitly or set OPENAI_MODEL"
        )

    return ep.rstrip("/"), mdl


def _build_messages(
    system_prompt: Optional[str],
    prompt_text: Optional[str],
    prompt: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """normalize to OpenAI chat messages; prompt_text wins if both given."""
    if prompt_text:
        user = prompt_text
    elif prompt and isinstance(prompt, dict) and "text" in prompt:
        user = str(prompt["text"])
    else:
        raise ValueError(
            "validators: need prompt_text (str) or prompt (dict with 'text')"
        )

    system = system_prompt or (
        "you are a meticulous subspecialist clinician. "
        "follow instructions exactly, return only the requested structured object."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm_with_schema(
    *,
    response_model: Any,
    system_prompt: Optional[str] = None,
    prompt_text: Optional[str] = None,
    prompt: Optional[Dict[str, Any]] = None,
    backend: Optional[Any] = None,
    endpoint_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 1,
    extra_create_kwargs: Optional[Dict[str, Any]] = None,
) -> BaseModel:
    """call an OpenAI-compatible server via instructor with strict schema parsing.

    args:
        response_model: pydantic.BaseModel subclass to parse into (REQUIRED).
        system_prompt: optional system priming.
        prompt_text / prompt: pass exactly one user prompt source.
        backend: optional object with .endpoint_url/.model_name hints.
        endpoint_url/model: explicit overrides; must resolve or we fail fast.
        temperature: decoding temperature (default 0.0).
        max_retries: number of parse-retry attempts (default 1).
        extra_create_kwargs: forwarded to the OpenAI create call.

    returns:
        An instance of the given response_model.

    raises:
        PayloadValidationError for schema/parse issues.
        ValueError for missing configuration or invalid inputs.
    """
    # validate response model type up front
    model_cls = _require_base_model(response_model)

    # resolve endpoint + model (fail if not set)
    base_url, model_name = _infer_endpoint_and_model(
        endpoint_url=endpoint_url, model=model, backend=backend
    )

    # compose messages
    messages = _build_messages(system_prompt, prompt_text, prompt)

    # build patched client
    api_key = os.getenv("OPENAI_API_KEY", "no-key-required")
    client = patch(OpenAI(base_url=base_url, api_key=api_key), mode=Mode.JSON_SCHEMA)

    # perform call with strict parsing
    last_err: Optional[Exception] = None
    attempts = max(1, int(max_retries) + 1)
    for attempt in range(attempts):
        try:
            kwargs = dict(
                model=model_name,
                messages=messages,
                temperature=temperature,
                response_model=model_cls,
            )
            if extra_create_kwargs:
                kwargs.update(extra_create_kwargs)

            result = client.chat.completions.create(**kwargs)
            # instructor returns an instance of model_cls
            return result
        except (PydanticValidationError, Exception) as e:  # noqa: BLE001
            last_err = e

    # if we reach here, all attempts failed
    details = textwrap.dedent(
        f"""
        instructor call failed after {attempts} attempt(s)
        endpoint: {base_url}
        model:    {model_name}
        schema:   {model_cls.__name__}
        error:    {last_err!r}
        """
    ).strip()
    raise PayloadValidationError(details) from last_err


# -------------------- optional round-level validators (strict, fail loud) --------------------


def validate_round1_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict) or not payload:
        raise ValidationError("round1: payload must be a non-empty object")
    if "p_iraki" in payload:
        try:
            pf = float(payload["p_iraki"])
        except Exception as e:  # noqa: BLE001
            raise ValidationError(
                f"round1: p_iraki not a number: {payload['p_iraki']}"
            ) from e
        if not (0.0 <= pf <= 1.0):
            raise ValidationError(f"round1: p_iraki out of range: {payload['p_iraki']}")
    if "confidence" in payload:
        try:
            cf = float(payload["confidence"])
        except Exception as e:  # noqa: BLE001
            raise ValidationError(
                f"round1: confidence not a number: {payload['confidence']}"
            ) from e
        if not (0.0 <= cf <= 1.0):
            raise ValidationError(
                f"round1: confidence out of range: {payload['confidence']}"
            )


def validate_round3_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict) or not payload:
        raise ValidationError("round3: payload must be a non-empty object")
    if "p_iraki" in payload:
        try:
            pf = float(payload["p_iraki"])
        except Exception as e:  # noqa: BLE001
            raise ValidationError(
                f"round3: p_iraki not a number: {payload['p_iraki']}"
            ) from e
        if not (0.0 <= pf <= 1.0):
            raise ValidationError(f"round3: p_iraki out of range: {payload['p_iraki']}")
    if "confidence_in_verdict" in payload:
        try:
            cf = float(payload["confidence_in_verdict"])
        except Exception as e:  # noqa: BLE001
            raise ValidationError(
                f"round3: confidence_in_verdict not a number: {payload['confidence_in_verdict']}"
            ) from e
        if not (0.0 <= cf <= 1.0):
            raise ValidationError(
                f"round3: confidence_in_verdict out of range: {payload['confidence_in_verdict']}"
            )
