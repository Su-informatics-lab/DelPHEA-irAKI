# schema.py
# helpers to load questionnaire and extract qids + consensus rules

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ConsensusRules:
    minimum_agreement: float
    debate_threshold_points: int


def load_qids(questionnaire_path: str) -> List[str]:
    # read questionnaire_full.json and return ordered list of qids, e.g., ["Q1", ..., "Q16"]
    with open(questionnaire_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    q = data["questionnaire"]["questions"]
    return [item["id"] for item in q]


def load_question_texts(path: str) -> Dict[str, str]:
    """return mapping {qid: text}; robust to various questionnaire shapes."""
    data = json.loads(Path(path).read_text())
    out: Dict[str, str] = {}
    if isinstance(data, dict) and "questions" in data:
        for item in data["questions"]:
            qid = item.get("id") or item.get("qid")
            txt = item.get("text") or item.get("question") or ""
            if qid:
                out[qid] = txt
    # fallback: echo qid if text not available
    if not out:
        for q in load_qids(path):
            out[q] = ""
    return out


def load_consensus_rules(questionnaire_path: str) -> ConsensusRules:
    with open(questionnaire_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cc = data["questionnaire"]["scoring_guidelines"]["consensus_calculation"]
    # example string values: "≥70% ..." and "≥3 point disagreement ..."
    # parse minimal integers/floats robustly
    import re

    min_agree_text = str(cc.get("minimum_agreement", "70"))
    debate_text = str(cc.get("debate_threshold", "3"))
    pct = re.findall(r"(\d+(?:\.\d+)?)\s*%?", min_agree_text)
    thr = re.findall(r"(\d+)", debate_text)
    minimum_agreement = (
        float(pct[0]) / (100.0 if "%" in min_agree_text else 1.0) if pct else 0.7
    )
    debate_threshold_points = int(thr[0]) if thr else 3
    return ConsensusRules(
        minimum_agreement=minimum_agreement,
        debate_threshold_points=debate_threshold_points,
    )
