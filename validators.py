"""
validators.py
-------------
validation & llm→schema glue for delphea-iraki (pydantic v2 idioms).

surface
-------
- call_llm_with_schema: ask backend for json and validate with a pydantic model
- validate_round1_payload / validate_round3_payload: quick shape checks
- log_debate_status: structured single-line debate lifecycle logging

notes
-----
- do not synthesize v1-style error codes; use PydanticCustomError + InitErrorDetails
  and raise once via ValidationError.from_exception_data. see pydantic v2 guidance.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError

# -------------------------- helpers: error builders ---------------------------


def _ve(
    title: str,
    *,
    etype: str,
    msg: str,
    loc: Tuple[Any, ...],
    inp: Any,
    ctx: Optional[Dict[str, Any]] = None,
) -> ValidationError:
    """build a single-line ValidationError using v2 primitives."""
    err = InitErrorDetails(
        type=PydanticCustomError(etype, msg, ctx or {}),
        loc=loc,
        input=inp,
        ctx=ctx or {},
    )
    return ValidationError.from_exception_data(title=title, line_errors=[err])


def _raise_ve(
    title: str,
    *,
    etype: str,
    msg: str,
    loc: Tuple[Any, ...],
    inp: Any,
    ctx: Optional[Dict[str, Any]] = None,
) -> None:
    raise _ve(title, etype=etype, msg=msg, loc=loc, inp=inp, ctx=ctx)


# --------------------- helpers: json extraction/normalization -----------------


def _extract_first_json_object(text: str) -> str:
    """return the first balanced json object found in text; fail loud if none."""
    title = "llm_response"
    if not isinstance(text, str) or not text.strip():
        _raise_ve(
            title,
            etype="empty_response",
            msg="empty model response",
            loc=("content",),
            inp=text,
        )

    # fenced ```json blocks
    fence = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if fence:
        return fence.group(1)

    # find first balanced {...}
    start = text.find("{")
    if start == -1:
        _raise_ve(
            title,
            etype="json_missing_object",
            msg="no json object found",
            loc=("content",),
            inp=text[:200],
        )

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    _raise_ve(
        title,
        etype="json_unbalanced",
        msg="unbalanced json braces",
        loc=("content",),
        inp=text[start : min(start + 200, len(text))],
    )


# ----------------------------- public: llm glue -------------------------------

T = TypeVar("T", bound=BaseModel)


def call_llm_with_schema(
    *,
    response_model: Type[T],
    prompt_text: str,
    backend: Any,
    temperature: float = 0.0,
    max_tokens: int = 1200,
    max_retries: int = 1,
) -> T:
    """call the llm backend and coerce output to the given pydantic model.

    args:
        response_model: pydantic model class (e.g., messages.ExpertRound1Reply).
        prompt_text: fully-rendered prompt to send to the backend.
        backend: object exposing `.generate(prompt, ...)` or `.get_completions(...)`.
        temperature: decoding temperature.
        max_tokens: upper bound for generation length.
        max_retries: number of additional attempts after a parse/validate failure.

    returns:
        instance of `response_model`.

    raises:
        pydantic.ValidationError on extraction/parse/validation errors.
        RuntimeError if backend call fails.
    """

    def _gen_once(prompt: str) -> str:
        if hasattr(backend, "generate"):
            return backend.generate(prompt, max_tokens=max_tokens, temperature=temperature)  # type: ignore[attr-defined]
        if hasattr(backend, "get_completions"):
            return backend.get_completions(prompt, temperature=temperature, max_tokens=max_tokens)  # type: ignore[attr-defined]
        raise RuntimeError("llm backend does not expose a known generation method")

    last_err: Optional[ValidationError] = None
    for attempt in range(max_retries + 1):
        prompt = (
            f"{prompt_text}\n\n"
            "IMPORTANT: return ONLY a valid JSON object that strictly matches the expected schema. "
            "Do not include any prose, markdown, or explanations."
            if attempt > 0
            else prompt_text
        )

        raw = _gen_once(prompt)
        try:
            json_text = _extract_first_json_object(raw)
            data = json.loads(json_text)
        except ValidationError as ve:
            last_err = ve
            continue
        except json.JSONDecodeError as je:
            last_err = _ve(
                "llm_response",
                etype="json_decode_error",
                msg="json decode error: {error}",
                loc=("content",),
                inp=raw[:200],
                ctx={"error": str(je)},
            )
            continue
        except Exception as e:
            last_err = _ve(
                "llm_response",
                etype="unexpected_error",
                msg="{error}",
                loc=("content",),
                inp=str(raw)[:200],
                ctx={"error": f"{type(e).__name__}: {e}"},
            )
            continue

        # pydantic does the real validation; this will raise ValidationError on failure
        return response_model.model_validate(data)

    assert last_err is not None
    raise last_err


# ------------------------- public: payload validators -------------------------


def _require_keys(obj: Dict[str, Any], keys: Iterable[str], title: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        _raise_ve(
            title,
            etype="missing_keys",
            msg=f"missing required keys: {missing}",
            loc=tuple(missing),
            inp=obj,
        )


def _validate_prob_triplet(
    p: float, ci: Tuple[float, float], conf: float, *, title: str
) -> None:
    errs: List[InitErrorDetails] = []

    def _add(etype: str, msg: str, loc: Tuple[Any, ...], inp: Any) -> None:
        errs.append(
            InitErrorDetails(
                type=PydanticCustomError(etype, msg, {}), loc=loc, input=inp, ctx={}
            )
        )

    if not isinstance(p, (int, float)) or not (0.0 <= float(p) <= 1.0):
        _add("range_error", "p_iraki must be in [0,1]", ("p_iraki",), p)
    if not (isinstance(ci, (list, tuple)) and len(ci) == 2):
        _add("type_error", "ci_iraki must be a pair [lower, upper]", ("ci_iraki",), ci)
    else:
        lo, hi = float(ci[0]), float(ci[1])
        if not (0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0):
            _add(
                "range_error", "ci_iraki bounds must be within [0,1]", ("ci_iraki",), ci
            )
        if lo > hi:
            _add("bound_error", "ci_iraki lower bound > upper bound", ("ci_iraki",), ci)
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        _add("range_error", "confidence must be in [0,1]", ("confidence",), conf)

    if errs:
        raise ValidationError.from_exception_data(title=title, line_errors=errs)


def validate_round1_payload(
    payload: Dict[str, Any], *, required_evidence: int = 8
) -> None:
    """validate minimal r1 shape; raise pydantic.ValidationError on failure."""
    title = "round1"
    _require_keys(
        payload,
        (
            "case_id",
            "expert_id",
            "scores",
            "evidence",
            "clinical_reasoning",
            "p_iraki",
            "ci_iraki",
            "confidence",
            "differential_diagnosis",
        ),
        title,
    )

    scores = payload.get("scores", {})
    if not isinstance(scores, dict) or not scores:
        _raise_ve(
            title,
            etype="type_error",
            msg="scores must be a non-empty dict",
            loc=("scores",),
            inp=scores,
        )
    bad_scores = {
        k: v for k, v in scores.items() if not isinstance(v, int) or not (1 <= v <= 10)
    }
    if bad_scores:
        _raise_ve(
            title,
            etype="range_error",
            msg=f"invalid scores (must be 1..10): {bad_scores}",
            loc=("scores",),
            inp=scores,
        )

    evidence = payload.get("evidence", {})
    if not isinstance(evidence, dict):
        _raise_ve(
            title,
            etype="type_error",
            msg="evidence must be a dict",
            loc=("evidence",),
            inp=evidence,
        )
    non_empty = sum(1 for v in evidence.values() if isinstance(v, str) and v.strip())
    if non_empty < min(required_evidence, len(scores)):
        _raise_ve(
            title,
            etype="insufficient_evidence",
            msg=f"insufficient evidence texts: {non_empty} < {min(required_evidence, len(scores))}",
            loc=("evidence",),
            inp=evidence,
        )

    _validate_prob_triplet(
        payload["p_iraki"], payload["ci_iraki"], payload["confidence"], title=title
    )


def validate_round3_payload(payload: Dict[str, Any]) -> None:
    """validate minimal r3 shape; raise pydantic.ValidationError on failure."""
    title = "round3"
    _require_keys(
        payload,
        (
            "case_id",
            "expert_id",
            "scores",
            "evidence",
            "p_iraki",
            "ci_iraki",
            "confidence",
            "changes_from_round1",
        ),
        title,
    )

    scores = payload.get("scores", {})
    if not isinstance(scores, dict) or not scores:
        _raise_ve(
            title,
            etype="type_error",
            msg="scores must be a non-empty dict",
            loc=("scores",),
            inp=scores,
        )
    bad_scores = {
        k: v for k, v in scores.items() if not isinstance(v, int) or not (1 <= v <= 10)
    }
    if bad_scores:
        _raise_ve(
            title,
            etype="range_error",
            msg=f"invalid scores (must be 1..10): {bad_scores}",
            loc=("scores",),
            inp=scores,
        )

    _validate_prob_triplet(
        payload["p_iraki"], payload["ci_iraki"], payload["confidence"], title=title
    )

    ch = payload.get("changes_from_round1")
    if not isinstance(ch, dict):
        _raise_ve(
            title,
            etype="type_error",
            msg="changes_from_round1 must be a dict",
            loc=("changes_from_round1",),
            inp=ch,
        )


# --------------------- public: structured debate logging ----------------------


def log_debate_status(*args, **kwargs) -> None:
    """structured logger for debate lifecycle events (compat shim).

    supports both:
        log_debate_status(logger, case_id, question_id, expert_id, status, reason=None, meta=None)
    and:
        log_debate_status(
            logger=..., case_id=..., question_id=..., expert_id=..., status=..., stage="debate",
            reason=None, meta=None
        )
    """
    logger = kwargs.pop("logger", args[0] if args else None)
    if not isinstance(logger, logging.Logger):
        raise TypeError(
            "log_debate_status: first arg or 'logger=' must be a logging.Logger"
        )

    if len(args) >= 5:
        _, case_id, question_id, expert_id, status, *rest = args
        reason = rest[0] if len(rest) >= 1 else kwargs.pop("reason", None)
        meta = rest[1] if len(rest) >= 2 else kwargs.pop("meta", None)
        stage = kwargs.pop("stage", "debate")
    else:
        try:
            case_id = kwargs.pop("case_id")
            question_id = kwargs.pop("question_id")
            expert_id = kwargs.pop("expert_id")
            status = kwargs.pop("status")
        except KeyError as e:
            raise ValueError(
                f"log_debate_status: missing required field {e.args[0]!r}"
            ) from e
        stage = kwargs.pop("stage", "debate")
        reason = kwargs.pop("reason", None)
        meta = kwargs.pop("meta", None)

    allowed_status = {"start", "turn", "repair", "skip", "timeout", "error", "end"}
    allowed_stage = {"r1", "debate", "r3"}

    if stage not in allowed_stage:
        raise ValueError(
            f"log_debate_status: invalid stage={stage!r}; expected one of {sorted(allowed_stage)}"
        )
    if status not in allowed_status:
        raise ValueError(
            f"log_debate_status: invalid status={status!r}; expected one of {sorted(allowed_status)}"
        )

    if meta is not None:
        try:
            json.dumps(meta)
        except Exception as e:
            raise ValueError(
                f"log_debate_status: 'meta' must be JSON-serializable: {e}"
            ) from e

    payload: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "case_id": case_id,
        "question_id": question_id,
        "expert_id": expert_id,
        "status": status,
    }
    if reason:
        payload["reason"] = reason
    if meta is not None:
        payload["meta"] = meta

    logger.info("debate_event %s", json.dumps(payload, sort_keys=True))
