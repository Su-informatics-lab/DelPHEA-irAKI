import logging
import types
from typing import Any, Dict, List, Tuple

import moderator as mod
from aggregator import Aggregator
from expert import Expert
from moderator import Moderator
from router import DebatePlan, Router

# --- lightweight stubs -------------------------------------------------------


class R1Lite:
    def __init__(self, scores: Dict[str, int], evidence: Dict[str, str]):
        self.scores = scores
        self.evidence = evidence


class ScriptedBackend:
    """Backend that returns a dict based on (expert_id, role)."""

    def __init__(self, script: Dict[tuple, Dict[str, Any]]):
        self.script = script

    def debate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        eid = payload.get("expert_id")
        role = (payload.get("clinical_context") or {}).get("role") or "participant"
        qid = payload.get("qid")
        out = {
            "expert_id": eid,
            "qid": qid,
            "round_no": 2,
            "text": f"{eid}:{role}",
            "satisfied": True,
        }
        out.update(self.script.get((eid, role), {}))
        return out


class DummyRouter(Router):
    """Always solicit the provided minority for Q1; majority is everyone else."""

    def __init__(self, minority_ids: List[str]):
        self._minority = list(minority_ids)

    def plan(self, r1: List[Tuple[str, Any]], rules) -> DebatePlan:  # type: ignore[override]
        return DebatePlan(by_qid={"Q1": list(self._minority)})


# --- tests -------------------------------------------------------------------


def test_handoff_loop_prioritizes_requested_targets(monkeypatch):
    # Patch questionnaire helpers so Moderator.__init__ doesn't read files.
    monkeypatch.setattr(mod, "load_qids", lambda _p: ["Q1"])
    monkeypatch.setattr(
        mod,
        "load_consensus_rules",
        lambda _p: types.SimpleNamespace(
            minimum_agreement=0.7, debate_threshold_points=2
        ),
    )

    script = {
        ("E1", "minority_open"): {"satisfied": False, "handoff_to": "E2"},
        ("E2", "majority_rebuttal"): {"satisfied": False, "handoff_to": "E3"},
        ("E3", "participant"): {"satisfied": True},  # handoff target
        ("E1", "minority_followup"): {"satisfied": True},  # closes
    }

    e1 = Expert("E1", "nephro", {}, backend=ScriptedBackend(script))
    e2 = Expert("E2", "nephro", {}, backend=ScriptedBackend(script))
    e3 = Expert("E3", "nephro", {}, backend=ScriptedBackend(script))

    M = Moderator(
        experts=[e1, e2, e3],
        questionnaire_path="dummy.json",
        router=DummyRouter(minority_ids=["E1"]),
        aggregator=Aggregator(),
        logger=logging.getLogger("moderator-test"),
        debate_rounds=3,
        max_history_turns=5,
        max_turns_per_expert=2,
        max_total_turns_per_qid=10,
        quiet_turn_limit=2,
    )

    r1 = [
        ("E1", R1Lite(scores={"Q1": 3}, evidence={"Q1": "ev1"})),
        ("E2", R1Lite(scores={"Q1": 7}, evidence={"Q1": "ev2"})),
        ("E3", R1Lite(scores={"Q1": 7}, evidence={"Q1": "ev3"})),
    ]
    case = {"case_id": "iraki_case_1"}

    out = M.detect_and_run_debates(r1, case)
    turns = out["transcripts"]["Q1"]

    speakers = [t["expert_id"] for t in turns]
    assert speakers[:4] == ["E1", "E2", "E3", "E1"]

    roles = [t["speaker_role"] for t in turns[:4]]
    assert roles == ["minority", "majority", "participant", "minority"]

    assert turns[3]["expert_id"] == "E1"
    assert turns[3].get("satisfied", False) is True


def test_per_expert_turn_cap_is_enforced(monkeypatch):
    monkeypatch.setattr(mod, "load_qids", lambda _p: ["Q1"])
    monkeypatch.setattr(
        mod,
        "load_consensus_rules",
        lambda _p: types.SimpleNamespace(
            minimum_agreement=0.7, debate_threshold_points=2
        ),
    )

    script = {
        ("E1", "minority_open"): {"satisfied": False},
        ("E2", "majority_rebuttal"): {"satisfied": False},
        ("E3", "majority_rebuttal"): {"satisfied": False},
    }
    e1 = Expert("E1", "nephro", {}, backend=ScriptedBackend(script))
    e2 = Expert("E2", "nephro", {}, backend=ScriptedBackend(script))
    e3 = Expert("E3", "nephro", {}, backend=ScriptedBackend(script))

    M = Moderator(
        experts=[e1, e2, e3],
        questionnaire_path="dummy.json",
        router=DummyRouter(minority_ids=["E1"]),
        aggregator=Aggregator(),
        debate_rounds=3,
        max_turns_per_expert=1,  # <-- enforce per-expert limit
        max_total_turns_per_qid=10,
        quiet_turn_limit=5,
    )

    r1 = [
        ("E1", R1Lite(scores={"Q1": 3}, evidence={"Q1": "ev1"})),
        ("E2", R1Lite(scores={"Q1": 7}, evidence={"Q1": "ev2"})),
        ("E3", R1Lite(scores={"Q1": 8}, evidence={"Q1": "ev3"})),
    ]
    case = {"case_id": "iraki_case_2"}
    out = M.detect_and_run_debates(r1, case)
    turns = out["transcripts"]["Q1"]
    speakers = [t["expert_id"] for t in turns]

    assert speakers.count("E1") <= 1
    assert speakers.count("E2") <= 1
    assert speakers.count("E3") <= 1
