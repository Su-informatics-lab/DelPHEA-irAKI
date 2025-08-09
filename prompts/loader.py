# prompts/loader.py
# handles loading and caching of static JSON config files for prompt generation

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]

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
        _expert_prompts = _load_json(_REPO_ROOT / "expert_prompts.json")
    if _iraki_assessment is None:
        _iraki_assessment = _load_json(_REPO_ROOT / "iraki_assessment.json")
    if _conf_instructions is None:
        _conf_instructions = _load_json(_REPO_ROOT / "confidence_instructions.json")


def get_expert_prompts() -> Dict:
    ensure_loaded()
    return _expert_prompts


def get_iraki_assessment() -> Dict:
    ensure_loaded()
    return _iraki_assessment


def get_conf_instructions() -> Dict:
    ensure_loaded()
    return _conf_instructions
