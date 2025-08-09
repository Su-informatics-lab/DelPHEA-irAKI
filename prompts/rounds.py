# prompts/rounds.py
# round-specific prompt construction – minimal, explicit, fail-loud.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from schema import load_question_texts

from .loader import (
    get_confidence_instructions,
    get_expert_prompts,
    get_iraki_assessment,
    load_triplet,
)

# -------------------- small utilities (kept local to avoid extra deps) --------------------


def _require_keys(obj: Dict[str, Any], keys: Iterable[str], where: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise KeyError(f"missing keys {missing} in {where}")


def _stringify(x: Any) -> str:
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False, indent=2)
    return str(x)


def _questions_block(qtexts: Dict[str, str]) -> str:
    return "\n".join(f"{qid}: {text}" for qid, text in qtexts.items())


def _safe_format(template: str, mapping: Dict[str, Any]) -> str:
    class _Safe(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    return template.format_map(_Safe(mapping))


def _load_qtexts(qpath: str | Path) -> Dict[str, str]:
    qtexts = load_question_texts(Path(qpath))
    if not qtexts:
        raise ValueError("questionnaire is empty or failed to load")
    return qtexts


def _select_expert_layer(
    ep_all: Dict[str, Any], layer_key: str, *, where: str
) -> Dict[str, Any]:
    """support both layered (ep_all['r1'/'r3']) and flat expert_prompts layouts."""
    base_keys = ["preamble", "schema_block", "checklist", "repair_heading"]
    if layer_key in ep_all:
        layer = ep_all[layer_key]
        if not isinstance(layer, dict):
            raise TypeError(
                f"{where} expected dict at '{layer_key}', got {type(layer).__name__}"
            )
        _require_keys(layer, base_keys, f"{where}['{layer_key}']")
        return layer
    # backward-compat: flat layout with keys at top level
    if all(k in ep_all for k in base_keys):
        return ep_all
    # explicit failure with guidance
    raise KeyError(
        f"cannot find round layer '{layer_key}' in {where}, and top-level keys "
        f"{base_keys} are also missing; ensure expert_prompts.json is layered by round "
        f"or contains those keys at top level"
    )


def _load_prompt_triplet(
    *,
    expert_prompts_path: Optional[str | Path] = None,
    iraki_assessment_path: Optional[str | Path] = None,
    conf_instructions_path: Optional[str | Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if any([expert_prompts_path, iraki_assessment_path, conf_instructions_path]):
        ep, ia, ci = load_triplet(
            expert_prompts_path=expert_prompts_path,
            iraki_assessment_path=iraki_assessment_path,
            conf_instructions_path=conf_instructions_path,
        )
    else:
        ep = get_expert_prompts()
        ia = get_iraki_assessment()
        ci = get_confidence_instructions()
    # normalize ci into a dict with 'ci_instructions' string
    if isinstance(ci, dict):
        if "ci_instructions" not in ci:
            raise KeyError("confidence_instructions.json missing 'ci_instructions'")
        ci_obj = ci
    elif isinstance(ci, str):
        ci_obj = {"ci_instructions": ci}
    else:
        raise TypeError("confidence_instructions must be dict or str")
    return ep, ia, ci_obj


# -------------------- round 1 --------------------


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
    expert_prompts_path: Optional[str | Path] = None,
    iraki_assessment_path: Optional[str | Path] = None,
    conf_instructions_path: Optional[str | Path] = None,
    **_: Any,
) -> str:
    """build the round-1 prompt string for an expert."""
    ep_all, ia, ci_obj = _load_prompt_triplet(
        expert_prompts_path=expert_prompts_path,
        iraki_assessment_path=iraki_assessment_path,
        conf_instructions_path=conf_instructions_path,
    )
    ep = _select_expert_layer(ep_all, "r1", where="expert_prompts.json")
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

    debate_bits: list[str] = []
    if debate_status:
        debate_bits += ["debate status:", str(debate_status)]
    if debate_plan:
        debate_bits += ["debate plan summary:", str(debate_plan)]

    checklist = "\n".join(f"- {item}" for item in ep["checklist"])

    parts: list[str] = [
        ep["preamble"],
        base,
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


# -------------------- round 3 (refinement/final) --------------------


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
    expert_prompts_path: Optional[str | Path] = None,
    iraki_assessment_path: Optional[str | Path] = None,
    conf_instructions_path: Optional[str | Path] = None,
    **_: Any,
) -> str:
    """build the round-3 (refinement/final) prompt string for an expert."""
    ep_all, ia, ci_obj = _load_prompt_triplet(
        expert_prompts_path=expert_prompts_path,
        iraki_assessment_path=iraki_assessment_path,
        conf_instructions_path=conf_instructions_path,
    )
    ep = _select_expert_layer(ep_all, "r3", where="expert_prompts.json")
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
