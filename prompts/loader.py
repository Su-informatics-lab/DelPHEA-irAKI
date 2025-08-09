# prompts/loader.py
# load prompt json configs with robust path resolution and simple caching.
# defaults to the prompts/ directory; supports env override DELPHEA_PROMPTS_DIR.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# dirs
_PKG_DIR = Path(__file__).resolve().parent  # .../prompts
_ROOT_DIR = _PKG_DIR.parent  # repo root
_BASE_DIR = Path(os.getenv("DELPHEA_PROMPTS_DIR", str(_PKG_DIR)))

# default files under prompts/
_DEFAULT_EXPERT_PROMPTS = _BASE_DIR / "expert_prompts.json"
_DEFAULT_IRAKI_ASSESSMENT = _BASE_DIR / "iraki_assessment.json"
_DEFAULT_CONFIDENCE = _BASE_DIR / "confidence_instructions.json"

# in-memory caches
_expert_prompts: Optional[Dict[str, Any]] = None
_iraki_assessment: Optional[Dict[str, Any]] = None
_confidence_instructions: Optional[Dict[str, Any]] = None


def _candidate_paths(target: Path) -> list[Path]:
    """build plausible locations to search for a json file."""
    name = target.name
    paths = []

    # 1) honor absolute/relative input directly
    if target.is_absolute():
        paths.append(target)
    else:
        paths.append(target)

    # 2) prompts/ (module dir) and env-resolved base
    paths.append(_BASE_DIR / name)
    paths.append(_PKG_DIR / name)

    # 3) repo root variations
    paths.append(_ROOT_DIR / "prompts" / name)
    paths.append(_ROOT_DIR / name)

    # 4) cwd variations (useful when running from repo root)
    cwd = Path.cwd()
    paths.append(cwd / "prompts" / name)
    paths.append(cwd / name)

    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _load_json(path: Path) -> Dict[str, Any]:
    """load json with robust search across prompts/ and repo root; fail fast with detail."""
    candidates = _candidate_paths(path)
    for p in candidates:
        try:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            # explicit surface of parse errors for faster debugging
            raise ValueError(f"failed to parse json at {p}: {e}") from e
    tried = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"required config not found. tried:\n  - {tried}")


def ensure_loaded() -> None:
    """idempotently load all required prompt configs into memory."""
    global _expert_prompts, _iraki_assessment, _confidence_instructions
    if _expert_prompts is None:
        _expert_prompts = _load_json(_DEFAULT_EXPERT_PROMPTS)
    if _iraki_assessment is None:
        _iraki_assessment = _load_json(_DEFAULT_IRAKI_ASSESSMENT)
    if _confidence_instructions is None:
        _confidence_instructions = _load_json(_DEFAULT_CONFIDENCE)


def get_expert_prompts() -> Dict[str, Any]:
    """return expert prompt blocks (json dict)."""
    ensure_loaded()
    return _expert_prompts  # type: ignore[return-value]


def get_iraki_assessment() -> Dict[str, Any]:
    """return irAKI assessment instructions (json dict)."""
    ensure_loaded()
    return _iraki_assessment  # type: ignore[return-value]


def get_confidence_instructions() -> Dict[str, Any]:
    """return confidence calibration instructions (json dict)."""
    ensure_loaded()
    return _confidence_instructions  # type: ignore[return-value]


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
    "set_prompts_dir",
]
