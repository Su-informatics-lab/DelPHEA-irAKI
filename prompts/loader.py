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


def set_prompts_dir(path: Path | str) -> None:
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
        return f"{header}{part_scores}{part_evid}{part_rest}{footer}"

    if isinstance(fallback, str) and fallback.strip():
        return fallback

    return (
        "Required JSON schema (summary): keys = scores{Q*}, evidence{Q*}, "
        "p_iraki, ci_iraki[2], confidence, clinical_reasoning, "
        "differential_diagnosis[], primary_diagnosis"
    )


def _round_synonyms(round_key: str) -> list[str]:
    """allow 'r1'/'round1' and 'r3'/'round3'."""
    rk = round_key.lower().strip()
    if rk in {"r1", "round1"}:
        return ["r1", "round1"]
    if rk in {"r3", "round3"}:
        return ["r3", "round3"]
    return [rk]


def _round_node(single: Dict[str, Any], round_key: str) -> Dict[str, Any]:
    """return the per-round dict if present; else empty dict."""
    rounds = single.get("rounds")
    if isinstance(rounds, dict):
        for k in _round_synonyms(round_key):
            node = rounds.get(k)
            if isinstance(node, dict):
                return node
    # some configs put the round blocks at top-level (rare); tolerate it
    for k in _round_synonyms(round_key):
        node = single.get(k)
        if isinstance(node, dict):
            return node
    return {}  # fall back to top-level keys entirely


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
    r = _round_node(single, round_key)

    def _want_text(key: str, *, required: bool, default: str = "") -> str:
        val = r.get(key)
        if isinstance(val, str) and val.strip():
            return val
        val = single.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if required and not default:
            raise KeyError(
                f"{_FILE}['{round_key}'] missing '{key}' and no top-level fallback"
            )
        return default

    # textual sections with fallback to top-level
    preamble = _want_text("preamble", required=False, default="")
    base_prompt = _want_text("base_prompt", required=True)  # formatters expect this
    instructions = _want_text("instructions", required=True)
    ci = _want_text("ci_instructions", required=True)
    repair = _want_text(
        "repair_heading", required=False, default="Repair instructions:"
    )

    # checklist may be in round node or top-level
    checklist_node = r.get("checklist", single.get("checklist", []))
    if not (isinstance(checklist_node, list) and checklist_node):
        raise KeyError(
            f"{_FILE} missing 'checklist' (list) for '{round_key}' and top-level"
        )

    # json_schema precedence:
    # 1) rounds.<rk>.json_schema
    # 2) top-level json_schema as a per-round map: json_schema[rk]
    # 3) top-level json_schema (flat object)
    js = r.get("json_schema")
    if js is None:
        top = single.get("json_schema")
        if isinstance(top, dict) and any(k in top for k in _round_synonyms(round_key)):
            for k in _round_synonyms(round_key):
                if isinstance(top.get(k), dict):
                    js = top[k]
                    break
        if js is None:
            js = top if isinstance(top, dict) else None

    schema_block = _render_schema_block(
        js, qids, single.get("schema_block") or r.get("schema_block")
    )

    return {
        "preamble": str(preamble),
        "base_prompt": str(base_prompt),
        "instructions": str(instructions),
        "schema_block": schema_block,
        "checklist": "\n".join(f"- {item}" for item in checklist_node),
        "repair_heading": str(repair),
        "ci_instructions": str(ci),
    }


__all__ = ["set_prompts_dir", "load_unified_round"]
