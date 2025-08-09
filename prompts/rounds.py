# prompts/rounds.py
# round-specific prompt construction

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .formatter import (
    load_qtexts,
    mk_qids_placeholders,
    questions_block,
    require_keys,
    safe_schema_format,
)
from .loader import (
    get_conf_instructions,
    get_expert_prompts,
    get_iraki_assessment,
    load_triplet,
)


def _load_cfgs_with_optional_overrides(
    prompts_path: Optional[str | Path],
    assessment_path: Optional[str | Path],
    confidence_path: Optional[str | Path],
):
    # if any override is provided, load the triplet from explicit paths
    if prompts_path or assessment_path or confidence_path:
        ep, ia, ci = load_triplet(
            expert_prompts_path=prompts_path,
            iraki_assessment_path=assessment_path,
            conf_instructions_path=confidence_path,
        )
        return ep, ia, ci
    # else use cached defaults
    return (
        get_expert_prompts(),
        get_iraki_assessment(),
        {"ci_instructions": get_conf_instructions()["ci_instructions"]},
    )


def format_round1_prompt(
    *,
    expert_name: str,
    specialty: str,
    case_id: str,
    demographics: str,
    clinical_notes: str,
    qpath: str | Path,
    # backward-compatible optional overrides
    prompts_path: Optional[str | Path] = None,
    assessment_path: Optional[str | Path] = None,
    confidence_path: Optional[str | Path] = None,
    **_: Any,  # swallow unexpected kwargs to fail less noisily at call sites
) -> str:
    ep_all, ia_all, ci_all = _load_cfgs_with_optional_overrides(
        prompts_path, assessment_path, confidence_path
    )
    ep = ep_all["r1"]
    ia = ia_all["round1"]
    ci = ci_all["ci_instructions"].strip()

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
    # backward-compatible optional overrides
    prompts_path: Optional[str | Path] = None,
    assessment_path: Optional[str | Path] = None,
    confidence_path: Optional[str | Path] = None,
    **_: Any,  # swallow unexpected kwargs
) -> str:
    ep_all, ia_all, ci_all = _load_cfgs_with_optional_overrides(
        prompts_path, assessment_path, confidence_path
    )
    ep = ep_all["r3"]
    ia = ia_all["round3"]
    ci = ci_all["ci_instructions"].strip()

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
        if ep.get("debate_status_heading"):
            debate_bits += [ep["debate_status_heading"], debate_status]
        else:
            debate_bits += ["Debate status:", debate_status]
    if debate_plan:
        if ep.get("debate_plan_heading"):
            debate_bits += [ep["debate_plan_heading"], debate_plan]
        else:
            debate_bits += ["Debate plan summary:", debate_plan]

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
