# delphea_iraki.py
# cli entrypoint wiring config → agents → run

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from consensus import WeightedMeanAggregator
from expert import Expert  # requires the lightweight Expert with assess_round1/3/debate
from llm_backend import LLMBackend
from moderator import Moderator
from router import FullRouter, SparseRouter
from schema import load_qids  # single source of truth


class DummyBackend(LLMBackend):
    # simple backend stub for smoke tests; replace with real transport
    def assess_round1(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
        qids = load_qids(expert_ctx["questionnaire_path"])
        return {
            "scores": {q: 7 for q in qids},
            "evidence": {q: "placeholder evidence" for q in qids},
            "clinical_reasoning": "placeholder reasoning",
            "p_iraki": 0.7,
            "ci_iraki": [0.6, 0.8],
            "confidence": 0.8,
            "differential_diagnosis": ["ATIN", "ATN"],
            "primary_diagnosis": "irAKI",
        }

    def assess_round3(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
        qids = load_qids(expert_ctx["questionnaire_path"])
        return {
            "scores": {q: 8 for q in qids},
            "evidence": {q: "updated evidence after debate" for q in qids},
            "p_iraki": 0.75,
            "ci_iraki": [0.65, 0.85],
            "confidence": 0.85,
            "changes_from_round1": {"overall": "nudged up after debate"},
            "debate_influence": "convinced by nephrology/rheum arguments",
            "verdict": True,
            "final_diagnosis": "irAKI",
            "confidence_in_verdict": 0.85,
            "recommendations": ["hold ICI", "consider steroids", "nephrology consult"],
        }

    def debate(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": "constructive rebuttal text",
            "citations": [],
            "satisfied": True,
        }


def _load_panel(panel_path: str):
    with open(panel_path, "r", encoding="utf-8") as f:
        return json.load(f)["expert_panel"]["experts"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questionnaire", default="questionnaire_full.json")
    parser.add_argument("--panel", default="panel.json")
    parser.add_argument("--router", choices=["sparse", "full"], default="sparse")
    parser.add_argument("--backend", default="dummy", choices=["dummy"])
    parser.add_argument(
        "--case", default="{}", help="json string or path to .json with case context"
    )
    args = parser.parse_args()

    qpath = args.questionnaire

    # load case from string or file
    case_arg = args.case
    if Path(case_arg).exists():
        with open(case_arg, "r", encoding="utf-8") as f:
            case: Dict[str, Any] = json.load(f)
    else:
        case = json.loads(case_arg)

    backend = DummyBackend()
    experts = [
        Expert(
            expert_id=ex["id"],
            specialty=ex["specialty"],
            persona=ex,
            backend=backend,
        )
        for ex in _load_panel(args.panel)
    ]

    router = SparseRouter() if args.router == "sparse" else FullRouter()
    mod = Moderator(
        experts=experts,
        questionnaire_path=qpath,
        router=router,
        aggregator=WeightedMeanAggregator(),
    )
    result = mod.run_case(case)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
