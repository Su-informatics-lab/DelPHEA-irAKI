# prompts/rounds.py
# round-specific prompt construction – minimal, explicit, fail-loud.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from schema import load_question_texts

# we only depend on loader + schema. no formatter helpers needed (yagni).
from .loader import (
    get_confidence_instructions,
    get_expert_prompts,
    get_iraki_assessment,
    load_triplet,
)

# -------------------- small utilities (keep local to avoid extra deps) --------------------


def _require_keys(obj: Dict[str, Any], keys: Iterable[str], where: str) -> None:
    """ensure required keys exist; raise with context if not."""
    missing = [k for k in keys if k not in obj]
    if missing:
        raise KeyError(f"missing keys {missing} in {where}")


def _stringify(x: Any) -> str:
    """stable stringify for dict/list blocks; pass through strings."""
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False, indent=2)
    return str(x)


def _questions_block(qtexts: Dict[str, str]) -> str:
    """render ordered 'QID: text' lines."""
    return "\n".join(f"{qid}: {text}" for qid, text in qtexts.items())


def _safe_format(template: str, mapping: Dict[str, Any]) -> str:
    """like str.format_map, but leaves unknown placeholders untouched."""

    class _Safe(dict):
        def __missing__(self, key):  # noqa: D401
            return "{" + key + "}"

    return template.format_map(_Safe(mapping))


def _load_qtexts(qpath: str | Path) -> Dict[str, str]:
    qtexts = load_question_texts(Path(qpath))
    if not qtexts:
        raise ValueError("questionnaire is empty or failed to load")
    return qtexts


def _load_prompt_triplet(
    *,
    expert_prompts_path: Optional[str | Path] = None,
    iraki_assessment_path: Optional[str | Path] = None,
    conf_instructions_path: Optional[str | Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """load (expert_prompts, iraki_assessment, confidence_instructions)."""
    if any([expert_prompts_path, iraki_assessment_path, conf_instructions_path]):
        ep, ia, ci = load_triplet(
            expert_prompts_path=expert_prompts_path,
            iraki_assessment_path=iraki_assessment_path,
            conf_instructions_path=conf_instructions_path,
        )
        # normalize confidence payload to a simple string where possible
        if isinstance(ci, dict) and "ci_instructions" in ci:
            ci = ci["ci_instructions"]
        return ep, ia, ci if isinstance(ci, str) else {"ci_instructions": str(ci)}
    # cached defaults
    ep = get_expert_prompts()
    ia = get_iraki_assessment()
    ci_raw = get_confidence_instructions()
    ci = ci_raw if isinstance(ci_raw, dict) else {"ci_instructions": str(ci_raw)}
    return ep, ia, ci


# -------------------- round 1 --------------------


def format_round1_prompt(
    *,
    expert_name: str,
    specialty: str,
    case_id: str,
    demographics: Any,
    clinical_notes: Any,
    qpath: str | Path,
    # optional context
    debate_status: Optional[str] = None,
    debate_plan: Optional[str] = None,
    # optional overrides for where to load jsons from
    expert_prompts_path: Optional[str | Path] = None,
    iraki_assessment_path: Optional[str | Path] = None,
    conf_instructions_path: Optional[str | Path] = None,
    # swallow any extra named args from upstream without breaking
    **_: Any,
) -> str:
    """build the round-1 prompt string for an expert.

    required json keys:
      - expert_prompts.json: 'preamble', 'schema_block', 'checklist', 'repair_heading'
      - iraki_assessment.json: 'base_prompt', 'instructions'
      - confidence_instructions.json: 'ci_instructions' (string)
    """
    ep, ia, ci_obj = _load_prompt_triplet(
        expert_prompts_path=expert_prompts_path,
        iraki_assessment_path=iraki_assessment_path,
        conf_instructions_path=conf_instructions_path,
    )

    _require_keys(
        ep,
        ["preamble", "schema_block", "checklist", "repair_heading"],
        "expert_prompts.json",
    )
    _require_keys(ia, ["base_prompt", "instructions"], "iraki_assessment.json")
    _require_keys(ci_obj, ["ci_instructions"], "confidence_instructions.json")

    # questionnaire rendering
    qtexts = _load_qtexts(qpath)
    qblock = _questions_block(qtexts)

    # base prompt
    base = _safe_format(
        ia["base_prompt"],
        {
            "expert_name": expert_name,
            "specialty": specialty,
            "case_id": case_id,
            "demographics": _stringify(demographics),
            "clinical_notes": _stringify(clinical_notes),
            "questions": qblock,
        },
    )

    # optional debate context (if provided by moderator)
    debate_bits: list[str] = []
    if debate_status:
        debate_bits += ["debate status:", str(debate_status)]
    if debate_plan:
        debate_bits += ["debate plan summary:", str(debate_plan)]

    # checklist bullets
    checklist = "\n".join(f"- {item}" for item in ep["checklist"])

    # assemble final prompt
    parts: list[str] = [
        ep["preamble"],
        base,
        *(debate_bits if debate_bits else []),
        "INSTRUCTIONS:",
        str(ia["instructions"]),
        "CONFIDENCE & CI GUIDANCE:",
        str(ci_obj["ci_instructions"]),
        "SCHEMA:",
        str(ep["schema_block"]),  # kept literal (no risky formatting)
        "CHECK BEFORE RETURNING JSON:",
        checklist,
        f"{ep['repair_heading']} follow any moderator 'repair' instructions strictly if provided.",
    ]
    return "\n\n".join(parts)


# -------------------- round 3 (refinement/final) --------------------


def format_round3_prompt(
    *,
    expert_name: str,
    specialty: str,
    case_id: str,
    demographics: Any,
    clinical_notes: Any,
    qpath: str | Path,
    # prior context
    round1_summary: Optional[str] = None,
    peer_feedback_summary: Optional[str] = None,
    debate_status: Optional[str] = None,
    debate_plan: Optional[str] = None,
    # optional overrides for where to load jsons from
    expert_prompts_path: Optional[str | Path] = None,
    iraki_assessment_path: Optional[str | Path] = None,
    conf_instructions_path: Optional[str | Path] = None,
    # accept extra kwargs without error
    **_: Any,
) -> str:
    """build the round-3 (refinement/final) prompt string for an expert."""
    ep, ia, ci_obj = _load_prompt_triplet(
        expert_prompts_path=expert_prompts_path,
        iraki_assessment_path=iraki_assessment_path,
        conf_instructions_path=conf_instructions_path,
    )

    _require_keys(
        ep,
        ["preamble", "schema_block", "checklist", "repair_heading"],
        "expert_prompts.json",
    )
    _require_keys(ia, ["base_prompt", "instructions"], "iraki_assessment.json")
    _require_keys(ci_obj, ["ci_instructions"], "confidence_instructions.json")

    qtexts = _load_qtexts(qpath)
    qblock = _questions_block(qtexts)

    base = _safe_format(
        ia["base_prompt"],
        {
            "expert_name": expert_name,
            "specialty": specialty,
            "case_id": case_id,
            "demographics": _stringify(demographics),
            "clinical_notes": _stringify(clinical_notes),
            "questions": qblock,
        },
    )

    context_bits: list[str] = []
    if round1_summary:
        context_bits += [
            "round-1 summary (your prior assessment):",
            str(round1_summary),
        ]
    if peer_feedback_summary:
        context_bits += [
            "peer feedback & moderator highlights:",
            str(peer_feedback_summary),
        ]

    debate_bits: list[str] = []
    if debate_status:
        debate_bits += ["debate status:", str(debate_status)]
    if debate_plan:
        debate_bits += ["debate plan summary:", str(debate_plan)]

    checklist = "\n".join(f"- {item}" for item in ep["checklist"])

    parts: list[str] = [
        ep["preamble"],
        base,
        *(context_bits if context_bits else []),
        *(debate_bits if debate_bits else []),
        "INSTRUCTIONS:",
        str(ia["instructions"]),
        "CONFIDENCE & CI GUIDANCE:",
        str(ci_obj["ci_instructions"]),
        "SCHEMA:",
        str(ep["schema_block"]),
        "CHECK BEFORE RETURNING JSON:",
        checklist,
        f"{ep['repair_heading']} follow any moderator 'repair' instructions strictly if provided.",
    ]
    return "\n\n".join(parts)
