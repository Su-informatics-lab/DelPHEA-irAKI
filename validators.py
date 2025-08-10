# validators.py
# -------------
# validation & llm→schema glue for delphea-iraki (pydantic v2 idioms).
#
# surface
# -------
# - call_llm_with_schema: ask backend for json and validate with a pydantic model
# - validate_round1_payload / validate_round3_payload: quick shape checks
# - log_debate_status: structured single-line debate lifecycle logging
#
# notes
# -----
# - avoid package-relative imports; repo is run as scripts, not a package.
# - use PydanticCustomError + InitErrorDetails (pydantic v2) for fail-loud errors.
# - token policy is controlled by cli args passed through delphea_iraki.py into
#   call_llm_with_schema (ctx_window, out_tokens_init, retries, retry_factor, reserve_tokens).
#   no env knobs here.

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError

T = TypeVar("T", bound=BaseModel)

# sensible defaults; overridden by delphea_iraki.py via function args
DEFAULT_CTX_WINDOW = 32768
DEFAULT_OUT_TOKENS_INIT = 3200
DEFAULT_RETRIES = 2
DEFAULT_RETRY_FACTOR = 1.5
DEFAULT_RESERVE_TOKENS = 512


# ------------------------------ error helpers ------------------------------


def _raise_ve(
    title: str,
    *,
    etype: str,
    msg: str,
    loc: Tuple[Any, ...],
    inp: Any,
    ctx: Optional[Dict[str, Any]] = None,
) -> None:
    """raise a pydantic.ValidationError with one line error."""
    err = InitErrorDetails(
        type=PydanticCustomError(etype, msg, (ctx or {})),
        loc=loc,
        input=inp,
        ctx=(ctx or {}),
    )
    raise ValidationError.from_exception_data(title=title, line_errors=[err])


# ------------------------------- json parser -------------------------------


def _extract_first_json_object(text: str) -> str:
    """extract first balanced json object from raw model text.

    behavior
    --------
    - finds the first '{' and scans with a brace counter
    - tracks string literals and escape characters to ignore braces inside strings
    - returns the substring containing the balanced object
    - if no '{' found → json_missing_object
      if unbalanced at end → json_unbalanced
    """
    if not isinstance(text, str) or not text:
        _raise_ve(
            "llm_response",
            etype="json_missing_object",
            msg="no content to scan for json object",
            loc=("content",),
            inp=text,
        )

    # fast path: if we see a fenced ```json block, prefer that region
    fence_start = text.find("```json")
    if fence_start == -1:
        fence_start = text.find("```JSON")
    if fence_start != -1:
        fence_end = text.find("```", fence_start + 7)
        if fence_end != -1:
            fenced = text[fence_start + 7 : fence_end].strip()
            # recurse on fenced region (may or may not include braces)
            return _extract_first_json_object(fenced)

    # general scan for first balanced object
    start = text.find("{")
    if start == -1:
        _raise_ve(
            "llm_response",
            etype="json_missing_object",
            msg="no json object found in model output",
            loc=("content",),
            inp=text[:200],
        )

    brace = 0
    i = start
    in_str = False
    esc = False

    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                brace += 1
            elif ch == "}":
                brace -= 1
                if brace == 0:
                    return text[start : i + 1]
        i += 1

    # reached end without closing
    _raise_ve(
        "llm_response",
        etype="json_unbalanced",
        msg="unbalanced json braces",
        loc=("content",),
        inp=text[start : min(start + 200, len(text))],
    )


# ------------------------------ token helpers ------------------------------


def _count_tokens(backend: Any, text: str) -> int:
    """best-effort token count; prefers backend.count_tokens(); falls back to ~4 chars/token."""
    if hasattr(backend, "count_tokens") and callable(getattr(backend, "count_tokens")):
        try:
            return int(backend.count_tokens(text))  # type: ignore[arg-type]
        except Exception:
            pass
    # crude fallback; fail-loud would be worse here
    return max(1, len(text) // 4)


def _discover_context_window(backend: Any) -> int:
    """discover context window from backend.capabilities(); else default."""
    try:
        if hasattr(backend, "capabilities"):
            caps = (
                backend.capabilities()
            )  # e.g., {"context_window": 200000, "json_mode": True}
            if isinstance(caps, dict) and "context_window" in caps:
                return int(caps["context_window"])
    except Exception:
        pass
    return DEFAULT_CTX_WINDOW


# --------------------------- llm → schema validator ---------------------------


def call_llm_with_schema(
    *,
    response_model: Type[T],
    prompt_text: str,
    backend: Any,
    temperature: float = 0.0,
    # policy (controlled by delphea_iraki.py; leave as None to use defaults above)
    ctx_window: Optional[int] = None,
    out_tokens_init: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_factor: Optional[float] = None,
    reserve_tokens: Optional[int] = None,
    # compatibility alias; if provided, treated as out_tokens_init
    max_tokens: Optional[int] = None,
) -> T:
    """call backend, extract first json, validate with response_model.

    token policy (centralized)
    --------------------------
    the caller (delphea_iraki.py) passes:
    - ctx_window: full model window (e.g., 200000)
    - out_tokens_init: starting output budget per attempt
    - max_retries: number of repair attempts (in addition to first try)
    - retry_factor: geometric escalation factor per retry
    - reserve_tokens: safety buffer not to be used by the model output

    safeguards
    ----------
    - never allocate more output tokens than (context - input - reserve)
    - if input alone exceeds (context - minimal_out), fail loud with 'input_too_long'
    - on json_unbalanced / json_missing_object, include partial and escalate budget
    """
    # resolve policy
    ctx = int(
        ctx_window if ctx_window is not None else _discover_context_window(backend)
    )
    out_init = int(
        out_tokens_init
        if out_tokens_init is not None
        else (int(max_tokens) if max_tokens is not None else DEFAULT_OUT_TOKENS_INIT)
    )
    retries = int(max_retries if max_retries is not None else DEFAULT_RETRIES)
    rfactor = float(retry_factor if retry_factor is not None else DEFAULT_RETRY_FACTOR)
    reserve = int(
        reserve_tokens if reserve_tokens is not None else DEFAULT_RESERVE_TOKENS
    )
    minimal_out = 128  # require at least some room to reply

    # compute input length and available budget
    input_tokens = _count_tokens(backend, prompt_text)
    available_out = ctx - input_tokens - reserve
    if available_out < minimal_out:
        _raise_ve(
            "llm_response",
            etype="input_too_long",
            msg="prompt uses {input_tokens} tokens, leaving {available_out} for output; reduce input or raise context window",
            loc=("content",),
            inp=f"[prompt starts] {prompt_text[:200]}",
            ctx={
                "input_tokens": input_tokens,
                "available_out": max(available_out, 0),
                "ctx_window": ctx,
                "reserve": reserve,
            },
        )

    # helper: backend generation
    def _gen_once(prompt: str, tokens: int) -> str:
        if hasattr(backend, "generate"):
            return backend.generate(prompt, max_tokens=tokens, temperature=temperature)  # type: ignore[attr-defined]
        if hasattr(backend, "get_completions"):
            return backend.get_completions(prompt, temperature=temperature, max_tokens=tokens)  # type: ignore[attr-defined]
        raise RuntimeError("llm backend does not expose a known generation method")

    # build output contract suffix
    try:
        expected_keys = list(response_model.model_fields.keys())  # pydantic v2
    except Exception:
        expected_keys = []
    contract = (
        "\n\nOUTPUT CONTRACT — return ONLY one JSON object matching the schema.\n"
        + (
            f"- top-level keys must be exactly: {expected_keys}\n"
            if expected_keys
            else ""
        )
        + "- no prose, no markdown fences.\n"
        + "- include exact case_id and expert_id fields if present in the schema.\n"
        + "- 'ci_iraki' must be [lower, upper] floats if present.\n"
    )

    last_err: Optional[ValidationError] = None
    last_raw: Optional[str] = None

    for attempt in range(retries + 1):
        # compute escalated budget but never exceed available_out
        target_out = int(out_init * (rfactor**attempt))
        tokens_for_attempt = max(min(target_out, available_out), minimal_out)

        if attempt == 0:
            prompt = f"{prompt_text}{contract}"
        else:
            err_types = {e["type"] for e in (last_err.errors() if last_err else [])}
            if {"json_unbalanced", "json_missing_object"} & err_types and last_raw:
                partial = last_raw.strip()
                # include only the tail to keep prompt bounded and relevant
                if len(partial) > 6000:
                    partial = partial[-6000:]
                prompt = (
                    f"{prompt_text}{contract}\n"
                    "the previous output was a PARTIAL json object. "
                    "complete it to a SINGLE valid json object that matches the contract exactly.\n\n"
                    "partial_json:\n```json\n"
                    f"{partial}\n```"
                )
            else:
                prompt = f"{prompt_text}{contract}"

        raw = _gen_once(prompt, tokens_for_attempt)
        last_raw = raw

        try:
            json_text = _extract_first_json_object(raw)
            data = json.loads(json_text)

            # small coercion for ci field if present
            if isinstance(data, dict) and "ci_iraki" in data:
                ci = data["ci_iraki"]
                if isinstance(ci, list) and len(ci) == 2:
                    data["ci_iraki"] = [float(ci[0]), float(ci[1])]

            return response_model(**data)

        except ValidationError as ve:
            last_err = ve
            continue

        except json.JSONDecodeError as jde:
            line_err = InitErrorDetails(
                type=PydanticCustomError(
                    "json_decode_error",
                    "json decoding failed: {msg}",
                    {"msg": str(jde)},
                ),
                loc=("content",),
                input=raw[:200] if isinstance(raw, str) else str(raw)[:200],
                ctx={"error": f"JSONDecodeError: {jde}"},
            )
            last_err = ValidationError.from_exception_data("llm_response", [line_err])
            continue

        except Exception as e:
            line_err = InitErrorDetails(
                type=PydanticCustomError(
                    "unexpected_error", "{error}", {"error": f"{type(e).__name__}: {e}"}
                ),
                loc=("content",),
                input=raw[:200] if isinstance(raw, str) else str(raw)[:200],
                ctx={"error": f"{type(e).__name__}: {e}"},
            )
            last_err = ValidationError.from_exception_data("llm_response", [line_err])
            continue

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
