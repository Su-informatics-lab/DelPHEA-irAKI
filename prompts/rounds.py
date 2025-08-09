# prompts/rounds.py
# round-specific prompt construction

from __future__ import annotations

from pathlib import Path

from .formatter import (
    load_qtexts,
    mk_qids_placeholders,
    questions_block,
    require_keys,
    safe_schema_format,
)
from .loader import get_conf_instructions, get_expert_prompts, get_iraki_assessment


def format_round1_prompt(
    *,
    expert_name: str,
    specialty: str,
    case_id: str,
    demographics: str,
    clinical_notes: str,
    qpath: str | Path,
) -> str:
    ep = get_expert_prompts()["r1"]
    ia = get_iraki_assessment()["round1"]
    ci = get_conf_instructions()["ci_instructions"].strip()

    require_keys(
        ep,
        ["preamble", "schema_block", "checklist", "repair_heading"],
        "expert_prompts.r1",
    )
    require_keys(ia, ["base_prompt", "instructions"], "iraki_assessment.round1")
    if not ci:
        raise KeyError("missing ci_instructions in confidence_instructions.json")

    qtexts = load_qtexts(qpath)
    qb = mk_qids_placeholders(qtexts.keys())
    schema_block = safe_schema_format(ep["schema_block"], qb)

    base = ia["base_prompt"].format(
        expert_name=expert_name,
        specialty=specialty,
        case_id=case_id,
        demographics=demographics,
        clinical_notes=clinical_notes,
        questions=questions_block(qtexts),
    )

    checklist = "\n".join(f"- {item}" for item in ep["checklist"])

    return "\n\n".join(
        [
            ep["preamble"],
            base,
            "INSTRUCTIONS:",
            ia["instructions"],
            "CONFIDENCE & CI GUIDANCE:",
            ci,
            "SCHEMA:",
            schema_block,
            "CHECK BEFORE RETURNING JSON:",
            checklist,
            ep["repair_heading"]
            + " Follow any moderator repair instructions strictly if provided.",
        ]
    )


def format_round3_prompt(
    *,
    expert_name: str,
    specialty: str,
    case_id: str,
    demographics: str,
    clinical_notes: str,
    qpath: str | Path,
    debate_status: str = "",
    debate_plan: str = "",
) -> str:
    ep = get_expert_prompts()["r3"]
    ia = get_iraki_assessment()["round3"]
    ci = get_conf_instructions()["ci_instructions"].strip()

    require_keys(
        ep,
        ["preamble", "schema_block", "checklist", "repair_heading"],
        "expert_prompts.r3",
    )
    require_keys(ia, ["base_prompt", "instructions"], "iraki_assessment.round3")
    if not ci:
        raise KeyError("missing ci_instructions in confidence_instructions.json")

    qtexts = load_qtexts(qpath)
    qb = mk_qids_placeholders(qtexts.keys())
    schema_block = safe_schema_format(ep["schema_block"], qb)

    base = ia["base_prompt"].format(
        expert_name=expert_name,
        specialty=specialty,
        case_id=case_id,
        demographics=demographics,
        clinical_notes=clinical_notes,
        questions=questions_block(qtexts),
    )

    checklist = "\n".join(f"- {item}" for item in ep["checklist"])

    debate_bits = []
    if debate_status:
        debate_bits += [ep.get("debate_status_heading", ""), debate_status]
    if debate_plan:
        debate_bits += [ep.get("debate_plan_heading", ""), debate_plan]

    return "\n\n".join(
        [
            ep["preamble"],
            base,
            *(debate_bits if debate_bits else []),
            "INSTRUCTIONS:",
            ia["instructions"],
            "CONFIDENCE & CI GUIDANCE:",
            ci,
            "SCHEMA:",
            schema_block,
            "CHECK BEFORE RETURNING JSON:",
            checklist,
            ep["repair_heading"]
            + " Follow any moderator repair instructions strictly if provided.",
        ]
    )
