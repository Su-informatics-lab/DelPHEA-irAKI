# prompts/loader.py
# handles loading and caching of static JSON config files for prompt generation

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]

# defaults
_DEFAULT_EXPERT_PROMPTS = _REPO_ROOT / "expert_prompts.json"
_DEFAULT_IRAKI_ASSESSMENT = _REPO_ROOT / "iraki_assessment.json"
_DEFAULT_CONF_INSTRUCTIONS = _REPO_ROOT / "confidence_instructions.json"

# lazy cache (kept for callers that rely on globals)
_expert_prompts = None
_iraki_assessment = None
_conf_instructions = None


def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"required config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_loaded() -> None:
    """lazy‐load JSON configs into module globals."""
    global _expert_prompts, _iraki_assessment, _conf_instructions
    if _expert_prompts is None:
        _expert_prompts = _load_json(_DEFAULT_EXPERT_PROMPTS)
    if _iraki_assessment is None:
        _iraki_assessment = _load_json(_DEFAULT_IRAKI_ASSESSMENT)
    if _conf_instructions is None:
        _conf_instructions = _load_json(_DEFAULT_CONF_INSTRUCTIONS)


def get_expert_prompts() -> Dict:
    ensure_loaded()
    return _expert_prompts


def get_iraki_assessment() -> Dict:
    ensure_loaded()
    return _iraki_assessment


def get_conf_instructions() -> Dict:
    ensure_loaded()
    return _conf_instructions


def load_triplet(
    expert_prompts_path: Optional[Path | str] = None,
    iraki_assessment_path: Optional[Path | str] = None,
    conf_instructions_path: Optional[Path | str] = None,
) -> Tuple[Dict, Dict, Dict]:
    """load three config jsons, using overrides if provided, otherwise defaults.

    returns:
        (expert_prompts, iraki_assessment, confidence_instructions)
    """
    ep_path = (
        Path(expert_prompts_path) if expert_prompts_path else _DEFAULT_EXPERT_PROMPTS
    )
    ia_path = (
        Path(iraki_assessment_path)
        if iraki_assessment_path
        else _DEFAULT_IRAKI_ASSESSMENT
    )
    ci_path = (
        Path(conf_instructions_path)
        if conf_instructions_path
        else _DEFAULT_CONF_INSTRUCTIONS
    )

    ep = _load_json(ep_path)
    ia = _load_json(ia_path)
    ci = _load_json(ci_path)
    return ep, ia, ci
