# prompts/loader.py
# single-file prompt loader for delphea-iraki.
# expects prompts/assessment_prompts.json and nothing else.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# base dir and filename
_PKG_DIR = Path(__file__).resolve().parent
_BASE_DIR = Path(os.getenv("DELPHEA_PROMPTS_DIR", str(_PKG_DIR)))
_FILE = "assessment_prompts.json"

_cache_single: Optional[Dict[str, Any]] = None


def set_prompts_dir(path: Path) -> None:
    """override the prompts directory (clears cache)."""
    global _BASE_DIR, _cache_single
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"prompts dir not found: {p}")
    _BASE_DIR = p
    _cache_single = None


def _resolve_file() -> Path:
    p = _BASE_DIR / _FILE
    if not p.exists():
        raise FileNotFoundError(
            f"missing required {_FILE} at {_BASE_DIR}. "
            "set DELPHEA_PROMPTS_DIR or call set_prompts_dir(...)."
        )
    return p


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid json in {path}: {e}") from e


def _load_single() -> Dict[str, Any]:
    global _cache_single
    if _cache_single is None:
        _cache_single = _read_json(_resolve_file())
    return _cache_single


def _render_schema_block(
    json_schema: Optional[Dict[str, Any]],
    qids: Iterable[str],
    fallback: Optional[str],
) -> str:
    """render a concise, human-readable schema block (single source of truth)."""
    if isinstance(json_schema, dict):
        qids_list = list(qids)
        scores_items = (
            ", ".join(f'"{q}": <int 1-9>' for q in qids_list)
            if qids_list
            else '"Q*": <int 1-9>'
        )
        evid_items = (
            ", ".join(f'"{q}": "<string>"' for q in qids_list)
            if qids_list
            else '"Q*": "<string>"'
        )
        rest_lines = [
            '"p_iraki": <float 0-1>',
            '"ci_iraki": [<float lower>, <float upper>]',
            '"confidence": <float 0-1>',
            '"clinical_reasoning": "<>=200 characters>"',
            '"differential_diagnosis": ["<item1>", "<item2>", "..."]',
            '"primary_diagnosis": "<string>"',
        ]
        rest_joined = ",\n  ".join(rest_lines)
        part_scores = f'  "scores": {{ {scores_items} }},\n'
        part_evid = f'  "evidence": {{ {evid_items} }},\n'
        part_rest = f"  {rest_joined}\n"
        header = "Required JSON schema (keys abbreviated):\n{\n"
        footer = "}"
        # no backslashes inside f-string expressions (precomputed above)
        return f"{header}{part_scores}{part_evid}{part_rest}{footer}"

    if isinstance(fallback, str) and fallback.strip():
        return fallback

    return (
        "Required JSON schema (summary): keys = scores{Q*}, evidence{Q*}, "
        "p_iraki, ci_iraki[2], confidence, clinical_reasoning, "
        "differential_diagnosis[], primary_diagnosis"
    )


def load_unified_round(
    *,
    round_key: str,  # 'r1' or 'r3'
    qids: Iterable[str],  # questionnaire ids for schema rendering
) -> Dict[str, str]:
    """return a string bundle for the requested round.

    returns keys:
      - preamble
      - base_prompt
      - instructions
      - schema_block
      - checklist
      - repair_heading
      - ci_instructions
    """
    single = _load_single()

    rounds = single.get("rounds") or single
    want = (
        round_key
        if round_key in rounds
        else {"r1": "round1", "r3": "round3"}.get(round_key, round_key)
    )
    if want not in rounds or not isinstance(rounds[want], dict):
        raise KeyError(f"{_FILE} missing rounds['{want}']")

    r = rounds[want]
    for k in ("preamble", "base_prompt", "instructions"):
        if k not in r:
            raise KeyError(f"{_FILE}['{want}'] missing '{k}'")

    checklist = single.get("checklist") or r.get("checklist")
    if not isinstance(checklist, list) or not checklist:
        raise KeyError(f"{_FILE} missing 'checklist' (list)")

    ci = single.get("ci_instructions")
    if ci is None:
        raise KeyError(f"{_FILE} missing 'ci_instructions'")

    repair = (
        single.get("repair_heading")
        or r.get("repair_heading")
        or "Repair instructions:"
    )

    schema_block = _render_schema_block(
        single.get("json_schema") or r.get("json_schema"),
        qids,
        single.get("schema_block") or r.get("schema_block"),
    )

    return {
        "preamble": str(r["preamble"]),
        "base_prompt": str(r["base_prompt"]),
        "instructions": str(r["instructions"]),
        "schema_block": schema_block,
        "checklist": "\n".join(f"- {item}" for item in checklist),
        "repair_heading": str(repair),
        "ci_instructions": str(ci),
    }


__all__ = ["set_prompts_dir", "load_unified_round"]
