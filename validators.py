# validators.py
# -------------
# validation & llm→schema glue for delphea-iraki (pydantic v2 idioms).
# contract: five parallel per-question dicts keyed by qid
#   - scores:       {qid: int in [1,9]}
#   - rationale:    {qid: non-empty str}
#   - evidence:     {qid: non-empty str}
#   - q_confidence: {qid: float in [0,1]}
#   - importance:   {qid: nonnegative int}, TOTAL across all qids = 100

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError

T = TypeVar("T", bound=BaseModel)

DEFAULT_CTX_WINDOW = 102400
DEFAULT_OUT_TOKENS_INIT = 5000
DEFAULT_RETRIES = 3
DEFAULT_RETRY_FACTOR = 1.5
DEFAULT_RESERVE_TOKENS = 512

# --- QID/order & importance normalization helpers ---

_QID_LINE_RE = re.compile(r"^\s*(Q[0-9A-Za-z_]+)\s*:", re.MULTILINE)


def _extract_qids_from_prompt(prompt: str) -> List[str]:
    """Pull QIDs from lines like 'Q1: ...' in the prompt, preserving order."""
    seen = set()
    qids: List[str] = []
    for m in _QID_LINE_RE.finditer(prompt or ""):
        q = m.group(1)
        if q not in seen:
            seen.add(q)
            qids.append(q)
    return qids


def _to_int_nonneg(x: Any, default: int = 0) -> int:
    try:
        v = int(float(x))
        return v if v >= 0 else default
    except Exception:
        return default


def _to_float(x: Any, default: float | None = None) -> float | None:
    try:
        return float(x)
    except Exception:
        return default


def _rebalance_importance(imp: Dict[str, Any]) -> Dict[str, int]:
    """Return non-negative int weights summing to 100, preserving insertion order."""
    keys = list(imp.keys())
    n = len(keys)
    if n == 0:
        return {}

    vals = [_to_int_nonneg(imp.get(k, 0), 0) for k in keys]
    total = sum(vals)

    if total <= 0:
        base = 100 // n
        rem = 100 % n
        scaled = [base + (1 if i < rem else 0) for i in range(n)]
    else:
        scaled = [int(round(v * 100.0 / total)) for v in vals]
        drift = 100 - sum(scaled)
        if drift > 0:
            for i in range(drift):
                scaled[i % n] += 1
        elif drift < 0:
            order = sorted(range(n), key=lambda i: scaled[i], reverse=True)
            j = 0
            for _ in range(-drift):
                while scaled[order[j % n]] == 0:
                    j += 1
                scaled[order[j % n]] -= 1
                j += 1

    return {k: scaled[i] for i, k in enumerate(keys)}


def _normalize_qdicts_to_expected(
    payload: Dict[str, Any], expected_qids: List[str]
) -> Dict[str, Any]:
    """
    Return a copy where scores/rationale/evidence/q_confidence/importance
    contain EXACTLY expected_qids in that insertion order (extras dropped, missing filled).
    """
    d = dict(payload or {})
    five = {
        "scores": d.get("scores") if isinstance(d.get("scores"), dict) else {},
        "rationale": d.get("rationale") if isinstance(d.get("rationale"), dict) else {},
        "evidence": d.get("evidence") if isinstance(d.get("evidence"), dict) else {},
        "q_confidence": d.get("q_confidence")
        if isinstance(d.get("q_confidence"), dict)
        else {},
        "importance": d.get("importance")
        if isinstance(d.get("importance"), dict)
        else {},
    }

    norm_scores: Dict[str, int] = {}
    norm_rat: Dict[str, str] = {}
    norm_evid: Dict[str, str] = {}
    norm_qc: Dict[str, float] = {}
    norm_imp: Dict[str, int] = {}

    for q in expected_qids:
        # scores: safe neutral default = 5
        s = five["scores"].get(q)
        s_ok = int(s) if isinstance(s, int) and 1 <= s <= 9 else 5
        norm_scores[q] = s_ok

        # rationale/evidence: allow empty (moderator auto-repair may fill later)
        rat = five["rationale"].get(q)
        norm_rat[q] = rat if isinstance(rat, str) else ""

        evid = five["evidence"].get(q)
        norm_evid[q] = evid if isinstance(evid, str) else ""

        # q_confidence: clamp to [0,1], default 0.5
        qc = _to_float(five["q_confidence"].get(q), 0.5)
        norm_qc[q] = 0.0 if qc is None else max(0.0, min(1.0, float(qc)))

        # importance: raw now; rebalance later
        w = five["importance"].get(q)
        norm_imp[q] = _to_int_nonneg(w, 0)

    d["scores"] = norm_scores
    d["rationale"] = norm_rat
    d["evidence"] = norm_evid
    d["q_confidence"] = norm_qc
    d["importance"] = norm_imp
    return d


# --- error helpers ---


def _raise_ve(
    title: str,
    *,
    etype: str,
    msg: str,
    loc: Tuple[Any, ...],
    inp: Any,
    ctx: Optional[Dict[str, Any]] = None,
) -> None:
    err = InitErrorDetails(
        type=PydanticCustomError(etype, msg, (ctx or {})),
        loc=loc,
        input=inp,
        ctx=(ctx or {}),
    )
    raise ValidationError.from_exception_data(title=title, line_errors=[err])


# --- json extraction ---


def _extract_first_json_object(text: str) -> str:
    if not isinstance(text, str) or not text:
        _raise_ve(
            "llm_response",
            etype="json_missing_object",
            msg="no content to scan for json object",
            loc=("content",),
            inp=text,
        )

    fence_start = text.find("```json")
    if fence_start == -1:
        fence_start = text.find("```JSON")
    if fence_start != -1:
        fence_end = text.find("```", fence_start + 7)
        if fence_end != -1:
            fenced = text[fence_start + 7 : fence_end].strip()
            return _extract_first_json_object(fenced)

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

    _raise_ve(
        "llm_response",
        etype="json_unbalanced",
        msg="unbalanced json braces",
        loc=("content",),
        inp=text[start : min(start + 200, len(text))],
    )


# --- counting / backend caps ---


def _count_tokens(backend: Any, text: str) -> int:
    if hasattr(backend, "count_tokens") and callable(getattr(backend, "count_tokens")):
        try:
            return int(backend.count_tokens(text))  # type: ignore[arg-type]
        except Exception:
            pass
    return max(1, len(text) // 4)


def _discover_context_window(backend: Any) -> int:
    try:
        if hasattr(backend, "capabilities"):
            caps = backend.capabilities()
            if isinstance(caps, dict) and "context_window" in caps:
                return int(caps["context_window"])
    except Exception:
        pass
    return DEFAULT_CTX_WINDOW


# --- misc coercions ---


def _split_lines_semicolons(s: str) -> List[str]:
    parts: List[str] = []
    for chunk in s.replace("\r", "").split("\n"):
        for seg in chunk.split(";"):
            t = seg.strip(" •- \t")
            if t:
                parts.append(t)
    return parts


def _coerce_bool_from_freeform(v: Any) -> Optional[bool]:
    """
    Try to coerce a free-form value into a boolean.

    Accepts:
      - real booleans
      - numbers (nonzero => True)
      - strings with common clinical phrasing:
        * positive-ish:  "likely", "very likely", "highly likely", "probable",
          "definite", "confirmed", "present", "positive", "consistent with",
          "supports", "supported", "supportive", "favors/favours", "diagnosis made"
        * negative-ish:  "unlikely", "improbable", "absent", "negative",
          "ruled out", "excluded", "not present", "not supported",
          "not consistent", "does not support/fit/favor"
        * fallback: "possible", "suspected", "suspicion" → True (schema requires bool)
    Returns None if it can’t decide.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(round(float(v))))
    if not isinstance(v, str):
        return None

    t = v.strip().lower()
    # normalize dashes to plain hyphen
    t = re.sub(r"[\u2010-\u2015–—]", "-", t)

    pos_patterns = [
        r"\b(very|highly)\s*likely\b",
        r"\blikely\b",
        r"\bprobable\b",
        r"\bdefinite\b",
        r"\bconfirmed\b",
        r"\bpresent\b",
        r"\bpositive\b",
        r"\bconsistent\s+(with|for)\b",
        r"\bsupport(s|ed|ive)?\b",
        r"\bfavou?rs?\b",
        r"\bdiagnosis\b.*\b(made|confirmed|present)\b",
        r"\b(iraki|immune[- ]related)[^\.]*\b(likely|probable|present|suspected)\b",
        r"^\s*yes\b",
        r"^\s*true\b",
    ]
    neg_patterns = [
        r"\b(un|im)likely\b",  # unlikely, unlikelyhood, impossible won't match but okay
        r"\bimprobable\b",
        r"\babsent\b",
        r"\bnegative\b",
        r"\bruled?\s*out\b",
        r"\b(excluded|exclude)\b",
        r"\bnot\s+(present|evident|supported|supportive|consistent)\b",
        r"\bdoes\s+not\s+(support|fit|favor|favour)\b",
        r"^\s*no\b",
        r"^\s*false\b",
    ]

    for pat in pos_patterns:
        if re.search(pat, t):
            return True
    for pat in neg_patterns:
        if re.search(pat, t):
            return False

    # Fallback: weakly positive phrases — schema demands a boolean.
    if re.search(r"\b(possible|suspected|suspicion)\b", t):
        return True

    return None


def _coerce_for_model_like_r1_r3(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce loose LLM outputs into types that match our R1/R3 schemas.

    - p_iraki / confidence / confidence_in_verdict → floats
    - ci_iraki → [float, float]
    - verdict → bool (handles "probable", "likely", etc.)
    - recommendations → list[str] (from semicolon/line separated string)
    - changes_from_round1 → dict (wrap string/list as {"summary": ...} / {"items": [...]})
    """
    d = dict(data) if isinstance(data, dict) else {}

    # numeric coercions
    for key in ("p_iraki", "confidence", "confidence_in_verdict"):
        if key in d and not isinstance(d[key], (int, float)):
            f = _to_float(d[key])
            if f is not None:
                d[key] = f

    # ci_iraki -> [lo, hi] floats
    ci = d.get("ci_iraki")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        lo = _to_float(ci[0])
        hi = _to_float(ci[1])
        if lo is not None and hi is not None:
            d["ci_iraki"] = [lo, hi]

    # verdict: coerce freeform → bool
    if "verdict" in d and not isinstance(d.get("verdict"), bool):
        vb = _coerce_bool_from_freeform(d.get("verdict"))
        if vb is not None:
            d["verdict"] = vb

    # recommendations: split a single string into a list of strings
    recs = d.get("recommendations")
    if isinstance(recs, str):
        items = _split_lines_semicolons(recs)
        d["recommendations"] = items if items else []

    # changes_from_round1 must be a dict; convert common non-dict forms.
    ch = d.get("changes_from_round1")
    if isinstance(ch, str):
        ch = ch.strip()
        if ch:
            d["changes_from_round1"] = {"summary": ch}
        else:
            d["changes_from_round1"] = {"summary": ""}
    elif isinstance(ch, list):
        items = [x for x in ch if isinstance(x, str) and x.strip()]
        d["changes_from_round1"] = {"items": items} if items else {"items": []}

    return d


# --- prompt scraping for ids ---

_CASE_RE = re.compile(r'-\s*"case_id":\s*"([^"]+)"')
_EXP_RE = re.compile(r'-\s*"expert_id":\s*"([^"]+)"')


def _ids_from_prompt(prompt: str) -> tuple[str, str]:
    case_id = "unknown_case"
    expert_id = "unknown_expert"
    m1 = _CASE_RE.search(prompt)
    if m1:
        case_id = m1.group(1)
    m2 = _EXP_RE.search(prompt)
    if m2:
        expert_id = m2.group(1)
    return case_id, expert_id


def _round_key_from_model(response_model: Type[BaseModel]) -> str:
    name = getattr(response_model, "__name__", "").lower()
    if "assessmentr1" in name or name == "assessmentr1":
        return "r1"
    if "assessmentr3" in name or name == "assessmentr3":
        return "r3"
    return "gen"


# --- dump helpers ---


def _dump_write(base: Optional[Path], rel: str, content: str) -> None:
    if not base:
        return
    try:
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    except Exception:
        pass


def _dump_json(base: Optional[Path], rel: str, obj: Any) -> None:
    if not base:
        return
    try:
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    except Exception:
        pass


# --- main glue: call_llm_with_schema ---


def call_llm_with_schema(
    *,
    response_model: Type[T],
    prompt_text: str,
    backend: Any,
    temperature: float = 0.0,
    ctx_window: Optional[int] = None,
    out_tokens_init: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_factor: Optional[float] = None,
    reserve_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    expected_qids: Optional[List[str]] = None,
) -> T:
    """Call backend, extract first JSON, normalize QID dicts & importance, validate."""
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
    minimal_out = 128

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

    dump_root_env = os.getenv("DELPHEA_OUT_DIR")
    base: Optional[Path] = Path(dump_root_env) if dump_root_env else None
    case_id, expert_id = _ids_from_prompt(prompt_text)
    round_key = _round_key_from_model(response_model)

    _dump_json(
        base,
        f"{case_id}/experts/{expert_id}/{round_key}/meta.json",
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model_name": getattr(backend, "model_name", None),
            "context_window": ctx,
            "input_tokens_est": input_tokens,
            "available_output_tokens_est": available_out,
            "temperature": temperature,
            "retries": retries,
            "retry_factor": rfactor,
            "reserve_tokens": reserve,
        },
    )

    def _gen_once(prompt: str, tokens: int) -> str:
        if hasattr(backend, "generate"):
            return backend.generate(prompt, max_tokens=tokens, temperature=temperature)  # type: ignore[attr-defined]
        if hasattr(backend, "get_completions"):
            return backend.get_completions(prompt, temperature=temperature, max_tokens=tokens)  # type: ignore[attr-defined]
        raise RuntimeError("llm backend does not expose a known generation method")

    # expose the expected_qids (if any) to pydantic models
    qids_context: List[str] = list(
        expected_qids or _extract_qids_from_prompt(prompt_text)
    )

    def _try_validate(d: Dict[str, Any]) -> T:
        if hasattr(response_model, "model_validate"):
            return response_model.model_validate(
                d, context={"expected_qids": qids_context}
            )
        return response_model(**d)  # type: ignore[call-arg]

    try:
        model_keys = list(response_model.model_fields.keys())
    except Exception:
        model_keys = []

    contract = (
        "\n\nOUTPUT CONTRACT — return ONLY one JSON object matching the schema.\n"
        + (f"- top-level keys should be: {model_keys}\n" if model_keys else "")
        + "- no prose, no markdown fences.\n"
        + "- include exact case_id and expert_id fields if present in the schema.\n"
        + "- 'ci_iraki' must be [lower, upper] floats if present.\n"
        + "- per-question fields MUST be five parallel dicts keyed by the question IDs, with IDENTICAL keys and order:\n"
        + "  * 'scores': integer 1..9\n"
        + "  * 'rationale': non-empty string (your argument)\n"
        + "  * 'evidence': non-empty string (supporting snippet/paragraph)\n"
        + "  * 'q_confidence': float in [0,1]\n"
        + "  * 'importance': non-negative integer; TOTAL across ALL questions MUST equal 100 (contribution weights; zeros allowed)\n"
    )

    last_err: Optional[ValidationError] = None

    for attempt in range(retries + 1):
        target_out = int(out_init * (rfactor**attempt))
        tokens_for_attempt = max(min(target_out, available_out), minimal_out)

        raw = _gen_once(f"{prompt_text}{contract}", tokens_for_attempt)

        _dump_write(
            base,
            f"{case_id}/experts/{expert_id}/{round_key}/attempt-{attempt}.raw.txt",
            raw if isinstance(raw, str) else str(raw),
        )

        try:
            json_text = _extract_first_json_object(raw)
            data = json.loads(json_text)
            if isinstance(data, dict):
                data = _coerce_for_model_like_r1_r3(data)

            # --- normalize & rebalance BEFORE validation, using canonical order ---
            if isinstance(data, dict) and qids_context:
                data = _normalize_qdicts_to_expected(data, qids_context)
                data["importance"] = _rebalance_importance(data.get("importance", {}))
            elif isinstance(data, dict) and not qids_context:
                # fallback to whatever order the model used, then rebalance
                fallback_qids = list((data.get("scores") or {}).keys())
                if fallback_qids:
                    data = _normalize_qdicts_to_expected(data, fallback_qids)
                    data["importance"] = _rebalance_importance(
                        data.get("importance", {})
                    )

            # First try
            try:
                model = _try_validate(data)
                payload = model.model_dump() if hasattr(model, "model_dump") else dict(model)  # type: ignore[arg-type]
                _dump_json(
                    base,
                    f"{case_id}/experts/{expert_id}/{round_key}/validated.json",
                    payload,
                )
                return model
            except ValidationError as ve1:
                last_err = ve1

                # If we failed and we didn't have qids_context, try deriving from model keys
                if isinstance(data, dict) and not qids_context:
                    qids_context = list((data.get("scores") or {}).keys())
                    if qids_context:
                        patched = _normalize_qdicts_to_expected(data, qids_context)
                        patched["importance"] = _rebalance_importance(
                            patched.get("importance", {})
                        )
                        try:
                            model = _try_validate(patched)
                            payload = model.model_dump() if hasattr(model, "model_dump") else dict(model)  # type: ignore[arg-type]
                            _dump_json(
                                base,
                                f"{case_id}/experts/{expert_id}/{round_key}/validated.json",
                                payload,
                            )
                            _dump_json(
                                base,
                                f"{case_id}/experts/{expert_id}/{round_key}/autopatch.json",
                                {
                                    "note": "qid dicts normalized; importance rebalanced to 100",
                                    "attempt": attempt,
                                },
                            )
                            return model
                        except ValidationError as ve2:
                            last_err = ve2
                continue  # retry loop

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


# --- downstream validators (unchanged) ---


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


def _validate_parallel_qdicts(
    payload: Dict[str, Any], expected_qids: Optional[List[str]], title: str
) -> None:
    """Validate five parallel per-question dicts and importance constant-sum=100."""
    for k in ("scores", "rationale", "evidence", "q_confidence", "importance"):
        v = payload.get(k)
        if not isinstance(v, dict) or not v:
            _raise_ve(
                title,
                etype="type_error",
                msg=f"{k} must be a non-empty dict keyed by qid",
                loc=(k,),
                inp=v,
            )

    scores = payload["scores"]
    rationale = payload["rationale"]
    evidence = payload["evidence"]
    q_conf = payload["q_confidence"]
    imp = payload["importance"]

    def _keys(d: Dict[str, Any]) -> List[str]:
        try:
            return list(d.keys())
        except Exception:
            return sorted(list(d.keys()))

    ks_scores = _keys(scores)
    ks_rationale = _keys(rationale)
    ks_evidence = _keys(evidence)
    ks_qc = _keys(q_conf)
    ks_imp = _keys(imp)

    if not (ks_scores == ks_rationale == ks_evidence == ks_qc == ks_imp):
        _raise_ve(
            title,
            etype="qid_mismatch",
            msg="scores/rationale/evidence/q_confidence/importance must have IDENTICAL qid lists and order",
            loc=("scores", "rationale", "evidence", "q_confidence", "importance"),
            inp={
                "scores": ks_scores,
                "rationale": ks_rationale,
                "evidence": ks_evidence,
                "q_confidence": ks_qc,
                "importance": ks_imp,
            },
        )

    if expected_qids:
        if ks_scores != expected_qids:
            missing = sorted(set(expected_qids) - set(ks_scores))
            extra = sorted(set(ks_scores) - set(expected_qids))
            _raise_ve(
                title,
                etype="qid_order_error",
                msg=f"qid order must match questionnaire. missing={missing} extra={extra}",
                loc=("scores",),
                inp=ks_scores,
            )

    for qid, s in scores.items():
        if not isinstance(s, int) or not (1 <= s <= 9):
            _raise_ve(
                title,
                etype="range_error",
                msg=f"scores[{qid}] must be integer 1..9",
                loc=("scores", qid),
                inp=s,
            )

    for name, dct in (("rationale", rationale), ("evidence", evidence)):
        for qid, txt in dct.items():
            if not isinstance(txt, str) or not txt.strip():
                _raise_ve(
                    title,
                    etype="type_error",
                    msg=f"{name}[{qid}] must be a non-empty string",
                    loc=(name, qid),
                    inp=txt,
                )

    for qid, c in q_conf.items():
        try:
            c_val = float(c)
        except Exception:
            c_val = -1.0
        if not (0.0 <= c_val <= 1.0):
            _raise_ve(
                title,
                etype="range_error",
                msg=f"q_confidence[{qid}] must be in [0,1]",
                loc=("q_confidence", qid),
                inp=c,
            )

    total = 0
    for qid, w in imp.items():
        if not isinstance(w, int) or w < 0:
            _raise_ve(
                title,
                etype="range_error",
                msg=f"importance[{qid}] must be a non-negative integer",
                loc=("importance", qid),
                inp=w,
            )
        total += w

    if total != 100:
        _raise_ve(
            title,
            etype="importance_sum_error",
            msg=f"importance must sum to 100 (got {total})",
            loc=("importance",),
            inp=list(imp.values()),
        )


def validate_round1_payload(
    payload: Dict[str, Any], *, expected_qids: Optional[List[str]] = None
) -> None:
    title = "round1"
    _require_keys(
        payload,
        (
            "case_id",
            "expert_id",
            "scores",
            "rationale",
            "evidence",
            "q_confidence",
            "importance",
            "p_iraki",
            "ci_iraki",
            "confidence",
            "clinical_reasoning",
            "differential_diagnosis",
        ),
        title,
    )
    _validate_parallel_qdicts(payload, expected_qids, title)
    _validate_prob_triplet(
        payload["p_iraki"], payload["ci_iraki"], payload["confidence"], title=title
    )
    ddx = payload.get("differential_diagnosis")
    if (
        not isinstance(ddx, list)
        or len([x for x in ddx if isinstance(x, str) and x.strip()]) < 2
    ):
        _raise_ve(
            title,
            etype="insufficient_ddx",
            msg="provide ≥2 differential diagnoses",
            loc=("differential_diagnosis",),
            inp=ddx,
        )
    cr = payload.get("clinical_reasoning")
    if not isinstance(cr, str) or not cr.strip():
        _raise_ve(
            title,
            etype="type_error",
            msg="clinical_reasoning must be a non-empty string (≥200 chars recommended)",
            loc=("clinical_reasoning",),
            inp=cr,
        )


def validate_round3_payload(
    payload: Dict[str, Any], *, expected_qids: Optional[List[str]] = None
) -> None:
    title = "round3"
    _require_keys(
        payload,
        (
            "case_id",
            "expert_id",
            "scores",
            "rationale",
            "evidence",
            "q_confidence",
            "importance",
            "p_iraki",
            "ci_iraki",
            "confidence",
            "changes_from_round1",
            "verdict",
            "final_diagnosis",
            "confidence_in_verdict",
            "recommendations",
        ),
        title,
    )
    _validate_parallel_qdicts(payload, expected_qids, title)
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
    fd = payload.get("final_diagnosis")
    if not isinstance(fd, str) or not fd.strip():
        _raise_ve(
            title,
            etype="type_error",
            msg="final_diagnosis must be a non-empty string",
            loc=("final_diagnosis",),
            inp=fd,
        )
    civ = payload.get("confidence_in_verdict")
    if not isinstance(civ, (int, float)) or not (0.0 <= float(civ) <= 1.0):
        _raise_ve(
            title,
            etype="range_error",
            msg="confidence_in_verdict must be in [0,1]",
            loc=("confidence_in_verdict",),
            inp=civ,
        )
    recs = payload.get("recommendations")
    if (
        not isinstance(recs, list)
        or not recs
        or any(not isinstance(x, str) or not x.strip() for x in recs)
    ):
        _raise_ve(
            title,
            etype="type_error",
            msg="recommendations must be a non-empty list of strings",
            loc=("recommendations",),
            inp=recs,
        )


# --- logging helper (with early_stop) ---


def log_debate_status(*args, **kwargs) -> None:
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

    allowed_status = {
        "start",
        "turn",
        "repair",
        "skip",
        "timeout",
        "error",
        "end",
        "early_stop",
    }
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
