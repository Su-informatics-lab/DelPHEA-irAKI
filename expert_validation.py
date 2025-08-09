# expert_validation.py
from __future__ import annotations

import json
from typing import Any, Dict, List


class PayloadValidationError(ValueError):
    pass


_QIDS = [f"Q{i}" for i in range(1, 16)]  # adjust if your questionnaire count changes


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def validate_round_payload(payload: Dict[str, Any], round_no: int) -> List[str]:
    """return a list of human-readable errors; empty list means valid."""
    errs: List[str] = []
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
        for q in _QIDS:
            v = scores.get(q)
            if not isinstance(v, int) or not (1 <= v <= 9):
                errs.append(f"scores[{q}] must be int in 1..9")

    # evidence: all qids non-empty strings
    ev = payload["evidence"]
    if not isinstance(ev, dict):
        errs.append("evidence must be object {qid:str}")
    else:
        for q in _QIDS:
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

    # round-specific requirements
    if round_no == 1:
        if not _is_nonempty_str(payload.get("clinical_reasoning", "")):
            errs.append("clinical_reasoning must be non-empty string")
        if not isinstance(payload.get("differential_diagnosis", []), list):
            errs.append("differential_diagnosis must be a list")
        if not _is_nonempty_str(payload.get("primary_diagnosis", "")):
            errs.append("primary_diagnosis must be non-empty string")
    elif round_no == 3:
        # minimal required round-3 fields; keep YAGNI
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
    """append a small fixer block to the original prompt; yagnified."""
    err_bullets = "\n".join(f"- {e}" for e in errors)
    return (
        f"{last_prompt}\n\n"
        "⚠️ JSON FIX NEEDED:\n"
        "Your previous JSON had the following problems:\n"
        f"{err_bullets}\n\n"
        "Please RETURN ONLY corrected JSON that fully satisfies the schema.\n"
        "Do not leave any field blank. If unknown, write a short rationale like "
        '"unknown due to missing UA in notes" rather than an empty string.\n'
    )


def call_llm_with_schema(
    backend, prompt: str, round_no: int, max_retries: int = 2
) -> Dict[str, Any]:
    """generate → parse → validate → repair-prompt retry; else raise."""
    last_prompt = prompt
    last_text = ""
    for attempt in range(max_retries + 1):
        last_text = backend.generate(last_prompt)  # your existing call
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

        errs = validate_round_payload(payload, round_no)
        if not errs:
            return payload

        if attempt == max_retries:
            raise PayloadValidationError(
                f"schema/content errors after {attempt} retries: {errs}"
            )
        last_prompt = build_repair_prompt(last_prompt, last_text, errs)

    # should be unreachable
    raise PayloadValidationError("unreachable retry loop exit")
