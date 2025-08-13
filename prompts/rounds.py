# prompts/rounds.py
# round prompt construction using the single-file loader.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from schema import load_question_texts

from .loader import load_unified_round


def _stringify(x: Any) -> str:
    # pretty json for dict/list; pass-through for strings
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False, indent=2)
    return str(x)


def _safe_format(template: str, mapping: Dict[str, Any]) -> str:
    # leave unknown placeholders intact rather than crashing
    class _Safe(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    return template.format_map(_Safe(mapping))


def _qtexts(qpath: str | Path) -> Dict[str, str]:
    q = load_question_texts(Path(qpath))
    if not q:
        raise ValueError("questionnaire is empty or failed to load")
    return q


def _render_questions(qtexts: Dict[str, str]) -> str:
    # stable "QID: text" lines
    return "\n".join(f"{qid}: {text}" for qid, text in qtexts.items())


def _assemble_prompt(
    *, bundle: Dict[str, str], base: str, context_bits: list[str]
) -> str:
    parts: list[str] = [
        bundle["preamble"],
        base,
        *(context_bits if context_bits else []),
        "INSTRUCTIONS:",
        bundle["instructions"],
        "CONFIDENCE & CI GUIDANCE:",
        bundle["ci_instructions"],
        "SCHEMA:",
        bundle["schema_block"],
        "CHECK BEFORE RETURNING JSON:",
        bundle["checklist"],
        f"{bundle['repair_heading']} follow any moderator 'repair' instructions strictly if provided.",
    ]
    return "\n\n".join(parts)


def format_round1_prompt(
    *,
    expert_name: str,
    specialty: str,
    case_id: str,
    demographics: Any,
    clinical_notes: Any,
    qpath: str | Path,
    debate_status: Optional[str] = None,
    debate_plan: Optional[str] = None,
    **_: Any,  # ignore legacy kwargs gracefully
) -> str:
    qtexts = _qtexts(qpath)
    bundle = load_unified_round(round_key="r1", qids=qtexts.keys())

    base = _safe_format(
        bundle["base_prompt"],
        {
            "expert_name": expert_name,
            "specialty": specialty,
            "case_id": case_id,
            "demographics": _stringify(demographics),
            "clinical_notes": _stringify(clinical_notes),
            "questions": _render_questions(qtexts),
        },
    )

    ctx: list[str] = []
    if debate_status:
        ctx += ["debate status:", str(debate_status)]
    if debate_plan:
        ctx += ["debate plan summary:", str(debate_plan)]

    return _assemble_prompt(bundle=bundle, base=base, context_bits=ctx)


def format_round3_prompt(
    *,
    expert_name: str,
    specialty: str,
    case_id: str,
    demographics: Any,
    clinical_notes: Any,
    qpath: str | Path,
    round1_summary: Optional[str] = None,
    peer_feedback_summary: Optional[str] = None,
    debate_status: Optional[str] = None,
    debate_plan: Optional[str] = None,
    **_: Any,  # ignore legacy kwargs gracefully
) -> str:
    qtexts = _qtexts(qpath)
    bundle = load_unified_round(round_key="r3", qids=qtexts.keys())

    base = _safe_format(
        bundle["base_prompt"],
        {
            "expert_name": expert_name,
            "specialty": specialty,
            "case_id": case_id,
            "demographics": _stringify(demographics),
            "clinical_notes": _stringify(clinical_notes),
            "questions": _render_questions(qtexts),
        },
    )

    ctx: list[str] = []
    if round1_summary:
        ctx += ["round-1 summary (your prior assessment):", str(round1_summary)]
    if peer_feedback_summary:
        ctx += ["peer feedback & moderator highlights:", str(peer_feedback_summary)]
    if debate_status:
        ctx += ["debate status:", str(debate_status)]
    if debate_plan:
        ctx += ["debate plan summary:", str(debate_plan)]

    return _assemble_prompt(bundle=bundle, base=base, context_bits=ctx)
