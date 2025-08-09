# prompts/loader.py
# load prompt json configs with robust path resolution and simple caching.
# defaults to the prompts/ directory; supports env override DELPHEA_PROMPTS_DIR.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# dirs
_PKG_DIR = Path(__file__).resolve().parent  # .../prompts
_ROOT_DIR = _PKG_DIR.parent  # repo root
_BASE_DIR = Path(os.getenv("DELPHEA_PROMPTS_DIR", str(_PKG_DIR)))

# caches
_expert_prompts: Optional[Dict[str, Any]] = None
_iraki_assessment: Optional[Dict[str, Any]] = None
_confidence_instructions: Optional[Dict[str, Any]] = None


def _read_json(path: Path) -> Dict[str, Any]:
    """read a json file with explicit errors."""
    if not path.exists():
        raise FileNotFoundError(f"missing required json file: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid json in {path}: {e}") from e


def _resolve_default(name: str) -> Path:
    """resolve a default file path under the prompts dir."""
    p = _BASE_DIR / name
    if not p.exists():
        # legacy flat layout fallback: repo root
        alt = _ROOT_DIR / name
        if alt.exists():
            return alt
    return p


def ensure_loaded() -> None:
    """load all three json configs if not cached."""
    global _expert_prompts, _iraki_assessment, _confidence_instructions
    if _expert_prompts is None:
        _expert_prompts = _read_json(_resolve_default("expert_prompts.json"))
    if _iraki_assessment is None:
        _iraki_assessment = _read_json(_resolve_default("iraki_assessment.json"))
    if _confidence_instructions is None:
        _confidence_instructions = _read_json(
            _resolve_default("confidence_instructions.json")
        )


def get_expert_prompts() -> Dict[str, Any]:
    """return expert_prompts.json content (cached)."""
    ensure_loaded()
    assert _expert_prompts is not None
    return _expert_prompts


def get_iraki_assessment() -> Dict[str, Any]:
    """return iraki_assessment.json content (cached)."""
    ensure_loaded()
    assert _iraki_assessment is not None
    return _iraki_assessment


def get_confidence_instructions() -> Dict[str, Any]:
    """return confidence_instructions.json content (cached)."""
    ensure_loaded()
    assert _confidence_instructions is not None
    return _confidence_instructions


def get_conf_instructions() -> Dict[str, Any]:
    """back-compat shim; prefer get_confidence_instructions()."""
    return get_confidence_instructions()


def load_triplet(
    *,
    expert_prompts_path: Optional[str | Path] = None,
    iraki_assessment_path: Optional[str | Path] = None,
    conf_instructions_path: Optional[str | Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """load (expert_prompts, iraki_assessment, confidence_instructions).

    if any explicit path is provided, each may be a file or a directory.
    when a directory is given, the standard filename inside it is used.
    any None path falls back to the default location under the prompts dir.
    """

    def _resolve(p: Optional[str | Path], default_name: str) -> Path:
        if p is None:
            return _resolve_default(default_name)
        p = Path(p)
        return (p / default_name) if p.is_dir() else p

    ep = _read_json(_resolve(expert_prompts_path, "expert_prompts.json"))
    ia = _read_json(_resolve(iraki_assessment_path, "iraki_assessment.json"))
    ci = _read_json(_resolve(conf_instructions_path, "confidence_instructions.json"))
    return ep, ia, ci


def set_prompts_dir(path: str | Path) -> None:
    """optional: override base prompts directory at runtime and clear caches."""
    global _BASE_DIR, _expert_prompts, _iraki_assessment, _confidence_instructions
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"prompts dir not found: {p}")
    _BASE_DIR = p
    _expert_prompts = None
    _iraki_assessment = None
    _confidence_instructions = None


__all__ = [
    "ensure_loaded",
    "get_expert_prompts",
    "get_iraki_assessment",
    "get_confidence_instructions",
    "get_conf_instructions",
    "load_triplet",
    "set_prompts_dir",
]
