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


def _to_float(x: Any, default: float | None = None) -> float | None:
    try:
        return float(x)
    except Exception:
        return default


def _split_lines_semicolons(s: str) -> List[str]:
    parts: List[str] = []
    for chunk in s.replace("\r", "").split("\n"):
        for seg in chunk.split(";"):
            t = seg.strip(" •- \t")
            if t:
                parts.append(t)
    return parts


def _coerce_for_model_like_r1_r3(data: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(data)

    for key in ("p_iraki", "confidence", "confidence_in_verdict"):
        if key in d and not isinstance(d[key], (int, float)):
            f = _to_float(d[key])
            if f is not None:
                d[key] = f

    if (
        "ci_iraki" in d
        and isinstance(d["ci_iraki"], (list, tuple))
        and len(d["ci_iraki"]) == 2
    ):
        lo = _to_float(d["ci_iraki"][0])
        hi = _to_float(d["ci_iraki"][1])
        if lo is not None and hi is not None:
            d["ci_iraki"] = [lo, hi]

    if "verdict" in d and not isinstance(d["verdict"], bool):
        if isinstance(d["verdict"], str):
            val = d["verdict"].strip().lower()
            truey = {"true", "yes", "y", "present", "positive", "likely"}
            falsey = {"false", "no", "n", "absent", "negative", "unlikely"}
            if val in truey:
                d["verdict"] = True
            elif val in falsey:
                d["verdict"] = False

    recs = d.get("recommendations")
    if isinstance(recs, str):
        items = _split_lines_semicolons(recs)
        d["recommendations"] = items if items else ([])

    return d


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


# ---------- helpers for auto-repair ----------


def _rebalance_importance(imp: Dict[str, Any]) -> Dict[str, int]:
    """Proportionally rescale nonnegative ints so the total is exactly 100."""
    items: List[Tuple[str, int]] = []
    for k, v in (imp or {}).items():
        try:
            iv = int(v)
        except Exception:
            iv = 0
        if iv < 0:
            iv = 0
        items.append((k, iv))
    if not items:
        return {}
    total = sum(iv for _, iv in items)
    n = len(items)
    if total == 100:
        return {k: iv for k, iv in items}
    if total <= 0:
        base = [100 // n] * n
        for i in range(100 - sum(base)):
            base[i % n] += 1
        return {k: base[i] for i, (k, _) in enumerate(items)}
    scaled: List[Tuple[str, int, float]] = []
    for k, iv in items:
        exact = iv * 100.0 / float(total)
        flo = int(exact)
        frac = exact - flo
        scaled.append((k, flo, frac))
    floor_sum = sum(flo for _, flo, _ in scaled)
    remainder = 100 - floor_sum
    scaled.sort(key=lambda t: t[2], reverse=True)
    out = {k: flo for k, flo, _ in scaled}
    for i in range(max(0, remainder)):
        k, _, _ = scaled[i % n]
        out[k] = out.get(k, 0) + 1
    return out


def _normalize_qdicts_to_expected(
    data: Dict[str, Any], expected_qids: List[str]
) -> Dict[str, Any]:
    """Align the five per-QID dicts to the exact expected_qids (order & membership)."""
    if not isinstance(data, dict):
        return data
    out = dict(data)

    # Pull existing dicts (may be missing/malformed)
    scores = data.get("scores") or {}
    rationale = data.get("rationale") or {}
    evidence = data.get("evidence") or {}
    q_conf = data.get("q_confidence") or {}
    imp = data.get("importance") or {}

    def _norm_score(v: Any) -> int:
        try:
            iv = int(v)
        except Exception:
            iv = 5
        if iv < 1 or iv > 9:
            iv = 5
        return iv

    def _norm_text(v: Any, placeholder: str) -> str:
        s = str(v).strip() if isinstance(v, str) else ""
        return s if s else placeholder

    def _norm_conf(v: Any) -> float:
        try:
            f = float(v)
        except Exception:
            f = 0.5
        if f < 0.0 or f > 1.0:
            f = 0.5
        return f

    def _norm_imp(v: Any) -> int:
        try:
            iv = int(v)
        except Exception:
            iv = 0
        return iv if iv >= 0 else 0

    # Build fresh dicts strictly in expected order; drop extras, fill missing.
    new_scores: Dict[str, int] = {}
    new_rationale: Dict[str, str] = {}
    new_evidence: Dict[str, str] = {}
    new_qc: Dict[str, float] = {}
    new_imp: Dict[str, int] = {}

    for q in expected_qids:
        new_scores[q] = _norm_score(scores.get(q))
        new_rationale[q] = _norm_text(
            rationale.get(q),
            "[auto-repair] rationale missing for this question; please review.",
        )
        new_evidence[q] = _norm_text(
            evidence.get(q),
            "[auto-repair] evidence snippet missing; review source notes.",
        )
        new_qc[q] = _norm_conf(q_conf.get(q))
        new_imp[q] = _norm_imp(imp.get(q))

    # If all importance zeros, distribute evenly; otherwise we'll rebalance later.
    if sum(new_imp.values()) <= 0:
        n = len(expected_qids) or 1
        base = [100 // n] * n
        for i in range(100 - sum(base)):
            base[i % n] += 1
        new_imp = {q: base[i] for i, q in enumerate(expected_qids)}

    out["scores"] = new_scores
    out["rationale"] = new_rationale
    out["evidence"] = new_evidence
    out["q_confidence"] = new_qc
    out["importance"] = new_imp
    return out


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
    """Call backend, extract first JSON, normalize & validate with response_model."""
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

            def _try_validate(d: Dict[str, Any]) -> T:
                if hasattr(response_model, "model_validate"):
                    return response_model.model_validate(
                        d, context={"expected_qids": expected_qids or []}
                    )
                return response_model(**d)  # type: ignore[call-arg]

            # First-pass validate
            try:
                model = _try_validate(data)
                payload = model.model_dump() if hasattr(model, "model_dump") else dict(model)  # type: ignore[arg-type]
                _dump_json(
                    base,
                    f"{case_id}/experts/{expert_id}/{round_key}/validated.json",
                    payload,
                )
                return model
            except ValidationError as ve_first:
                err_text = str(ve_first)

                # Case A: importance sum != 100 → auto-rebalance, revalidate once
                if (
                    "importance must sum to 100" in err_text
                    and isinstance(data, dict)
                    and isinstance(data.get("importance"), dict)
                ):
                    patched = dict(data)
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
                                "note": "importance rebalanced to 100",
                                "attempt": attempt,
                            },
                        )
                        return model
                    except ValidationError as ve_second:
                        last_err = ve_second
                        # fall through to other auto-fixes below

                # Case B: qid mismatch/order issues → normalize to expected_qids and revalidate
                has_qid_mismatch = (
                    "qid lists and order" in err_text
                    or "qid_mismatch" in err_text
                    or "qid order must match questionnaire" in err_text
                )
                if has_qid_mismatch and expected_qids and isinstance(data, dict):
                    patched2 = _normalize_qdicts_to_expected(data, expected_qids)
                    # After normalization, ensure importance sums to 100
                    if isinstance(patched2.get("importance"), dict):
                        patched2["importance"] = _rebalance_importance(
                            patched2["importance"]
                        )
                    try:
                        model = _try_validate(patched2)
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
                                "note": "qid dicts normalized to expected order and importance rebalanced",
                                "attempt": attempt,
                            },
                        )
                        return model
                    except ValidationError as ve_third:
                        last_err = ve_third
                        # fall through to retry

                last_err = ve_first
                continue

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
        "early_stop",  # used by moderator
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
