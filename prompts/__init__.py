# prompts/__init__.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from schema import load_qids, load_question_texts

__all__ = ["format_round1_prompt", "format_round3_prompt"]

# module-level cache for the loaded prompt JSON
_PROMPTS: Dict[str, Any] | None = None


def _ensure_prompts_loaded(path: str | Path) -> Dict[str, Any]:
    global _PROMPTS
    if _PROMPTS is None:
        p = Path(path)
        # fail aloud if missing or unreadable
        _PROMPTS = json.loads(p.read_text(encoding="utf-8"))
    return _PROMPTS


def _qids_blocks(qids: List[str]) -> Dict[str, str]:
    qids_scores = ", ".join([f'"{q}": <int 1-9>' for q in qids])
    qids_evidence = ", ".join([f'"{q}": "<1–2 sentences, non-empty>"' for q in qids])
    return {"qids_scores": qids_scores, "qids_evidence": qids_evidence}


def _questionnaire_lines(qids: List[str], qtext: Dict[str, str]) -> str:
    return "\n".join([f"- {qid}: {qtext.get(qid, '')}" for qid in qids])


def format_round1_prompt(
    prompts_path: str | Path,
    questionnaire_path: str | Path,
    case: Dict[str, Any],
    repair_hint: Optional[str] = None,
) -> str:
    """Build the Round 1 prompt from JSON templates (strict, no fallbacks)."""
    p = _ensure_prompts_loaded(prompts_path)["r1"]
    qids = load_qids(str(questionnaire_path))
    qtext = load_question_texts(str(questionnaire_path))
    qb = _qids_blocks(qids)

    parts = [
        p["preamble"].strip(),
        p["schema_block"].format(**qb),
        "Validation checklist:\n" + "\n".join(f"- {x}" for x in p["checklist"]),
        p["questionnaire_heading"] + "\n" + _questionnaire_lines(qids, qtext),
        p["patient_heading"] + "\n" + (case.get("brief", "") or ""),
    ]
    if repair_hint:
        parts.append(p["repair_heading"] + "\n" + repair_hint)
    return "\n\n".join(parts)


def format_round3_prompt(
    prompts_path: str | Path,
    questionnaire_path: str | Path,
    case: Dict[str, Any],
    debate_ctx: Dict[str, Any],
    repair_hint: Optional[str] = None,
) -> str:
    """Build the Round 3 prompt from JSON templates with explicit debate status."""
    p = _ensure_prompts_loaded(prompts_path)["r3"]
    qids = load_qids(str(questionnaire_path))
    # we still fetch qtext to keep symmetry; not displayed unless you add it to JSON later
    _ = load_question_texts(str(questionnaire_path))
    qb = _qids_blocks(qids)

    debate_status = "skipped" if debate_ctx.get("debate_skipped") else "executed"
    plan = debate_ctx.get("debate_plan") or {}
    plan_summary = ", ".join(list(plan.keys())) if plan else "none"

    parts = [
        p["preamble"].strip(),
        p["schema_block"].format(**qb),
        "Validation checklist:\n" + "\n".join(f"- {x}" for x in p["checklist"]),
        f"{p['debate_status_heading']} {debate_status}",
        f"{p['debate_plan_heading']} {plan_summary}",
        p["patient_heading"] + "\n" + (case.get("brief", "") or ""),
    ]
    if repair_hint:
        parts.append(p["repair_heading"] + "\n" + repair_hint)
    return "\n\n".join(parts)
