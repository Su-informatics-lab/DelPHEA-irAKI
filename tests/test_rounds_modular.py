import logging
import os
import socket
import types

import pytest
import requests

from aggregator import Aggregator
from expert import Expert
from llm_backend import LLMBackend
from models import AssessmentR1, AssessmentR3
from moderator import Moderator
from router import Router

# ---------- helpers ----------


def _maybe_live_url():
    """
    Return a base URL for a live OpenAI-compatible endpoint if available,
    else None. We look at ENDPOINT_URL, OPENAI_BASE_URL, or LIVE_BACKEND_URL.
    """
    for env in ("ENDPOINT_URL", "OPENAI_BASE_URL", "LIVE_BACKEND_URL"):
        url = os.getenv(env)
        if url:
            return url.rstrip("/")
    return None


def _reachable(url: str) -> bool:
    try:
        # Try a super-fast probe to /v1/models (works for vLLM)
        r = requests.get(url.rstrip("/") + "/v1/models", timeout=1.5)
        return r.status_code < 500
    except Exception:
        # Fallback: Try TCP connect to the host:port
        try:
            from urllib.parse import urlparse

            u = urlparse(url)
            host, port = u.hostname, u.port or 80
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except Exception:
            return False


class ScriptedDebateBackend:
    """
    Tiny backend for Expert.debate() path: returns deterministic debate turns.
    It ONLY implements .debate(payload) (Expert.debate uses this method directly).
    """

    def __init__(self, script):
        self.script = script

    def debate(self, payload):
        eid = payload.get("expert_id")
        role = (payload.get("clinical_context") or {}).get("role", "participant")
        key = (eid, role)
        spec = self.script.get(key, {})
        text = spec.get("text", f"{eid} speaks as {role}.")
        sat = spec.get("satisfied", True)
        hand = spec.get("handoff_to", None)
        return {
            "expert_id": eid,
            "qid": payload.get("qid"),
            "round_no": payload.get("round_no", 2),
            "text": text,
            "satisfied": bool(sat),
            "handoff_to": hand,
        }


# ---------- R1 (scripted) ----------


def test_round1_scripted(monkeypatch):
    """
    Scripted/fast check that Expert.assess_round1:
      - uses load_qids set for this test,
      - injects case_id/expert_id,
      - returns a validated AssessmentR1.
    """
    # Avoid reading questionnaire from disk
    import expert as expert_mod

    monkeypatch.setattr(expert_mod, "load_qids", lambda _p: ["Q1", "Q2"])

    # Keep prompts minimal/fast
    monkeypatch.setattr(expert_mod, "format_round1_prompt", lambda **_: "R1 PROMPT")

    def _fake_call_llm_with_schema(response_model, **kwargs):
        if response_model is AssessmentR1:
            qids = ["Q1", "Q2"]
            return AssessmentR1(
                case_id="iraki_case_1",
                expert_id="E1",
                scores={"Q1": 3, "Q2": 4},
                evidence={"Q1": "ev minority", "Q2": "ev minority 2"},
                clinical_reasoning="x" * 220,
                primary_diagnosis="ATIN?",
                differential_diagnosis=["ATN", "Prerenal"],
                rationale={q: "minority rationale" for q in qids},
                q_confidence={q: 0.65 for q in qids},
                importance={"Q1": 60, "Q2": 40},  # <-- sum to 100
                p_iraki=0.5,
                ci_iraki=(0.4, 0.6),
                confidence=0.7,
            )
        elif response_model is AssessmentR3:
            return AssessmentR3(
                case_id="iraki_case_1",
                expert_id="E1",
                scores={"Q1": 6, "Q2": 5},
                evidence={"Q1": "ev updated", "Q2": "ev updated 2"},
                changes_from_round1={"summary": "updated", "debate_influence": "minor"},
                final_diagnosis="ATIN",
                recommendations=["Rx A"],
                p_iraki=0.65,
                ci_iraki=(0.5, 0.8),
                confidence=0.7,
                rationale={"Q1": "updated rationale", "Q2": "updated rationale 2"},
                q_confidence={"Q1": 0.7, "Q2": 0.68},
                importance={"Q1": 55, "Q2": 45},  # <-- sum to 100
                verdict=True,
                confidence_in_verdict=0.72,
            )
        raise AssertionError("Unexpected response_model")

    monkeypatch.setattr(expert_mod, "call_llm_with_schema", _fake_call_llm_with_schema)

    e = Expert("E1", "nephro", {}, backend=ScriptedDebateBackend({}))
    case = {
        "case_id": "iraki_case_1",
        "demographics": {"age": 66, "sex": "F"},
        "clinical_notes": [{"text": "AKI with sterile pyuria."}],
    }
    a1 = e.assess_round1(case, questionnaire_path="dummy.json")
    assert isinstance(a1, AssessmentR1)
    assert a1.case_id == "iraki_case_1"
    assert a1.expert_id == "E1"
    assert set(a1.scores) == {"Q1", "Q2"}
    assert set(a1.evidence) == {"Q1", "Q2"}


# ---------- R2 (live, skippable) ----------


@pytest.mark.integration
def test_round2_live_backend_debate_smoke(monkeypatch):
    """
    Live smoke test for the debate path with your running server
    (skips if ENDPOINT_URL/OPENAI_BASE_URL/LIVE_BACKEND_URL is not reachable).
    """
    url = _maybe_live_url()
    if not url or not _reachable(url):
        pytest.skip("Live backend not reachable; set ENDPOINT_URL or OPENAI_BASE_URL.")

    backend = LLMBackend(
        endpoint_url=url, model_name=os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
    )

    # Use Expert.debate (which calls backend.debate); keep everything minimal.
    e = Expert("E_live", "nephro", {}, backend=backend)
    case = {"case_id": "iraki_case_live", "clinical_notes": [{"text": "short note"}]}

    turn = e.debate(
        qid="Q_live",
        round_no=2,
        clinical_context={"case": case, "peer_turns": [], "role": "participant"},
        minority_view="(none)",
    )
    d = turn.model_dump()
    assert d["expert_id"] == "E_live"
    assert d["qid"] == "Q_live"
    assert isinstance(d["text"], str) and len(d["text"].strip()) > 0
    assert isinstance(d["satisfied"], bool)


# ---------- R3 (scripted) ----------


def test_round3_scripted(monkeypatch):
    """
    Scripted/fast check that Expert.assess_round3 returns a valid AssessmentR3.
    """
    import expert as expert_mod

    monkeypatch.setattr(expert_mod, "load_qids", lambda _p: ["Q1", "Q2"])
    monkeypatch.setattr(expert_mod, "format_round3_prompt", lambda **_: "R3 PROMPT")

    def _fake_call_llm_with_schema(response_model, **kwargs):
        assert response_model is AssessmentR3
        qids = ["Q1", "Q2"]
        return AssessmentR3(
            case_id="iraki_case_1",
            expert_id="E1",
            scores={"Q1": 6, "Q2": 4},
            evidence={"Q1": "ev1'", "Q2": "ev2'"},
            changes_from_round1={"summary": "updated", "debate_influence": "minor"},
            final_diagnosis="ATIN",
            recommendations=["Rx A", "Rx B"],
            p_iraki=0.7,
            ci_iraki=(0.6, 0.8),
            confidence=0.8,
            rationale={q: "updated rationale" for q in qids},
            q_confidence={q: 0.75 for q in qids},
            importance={"Q1": 50, "Q2": 50},  # <-- sum to 100
            verdict=True,
            confidence_in_verdict=0.7,
        )

    monkeypatch.setattr(expert_mod, "call_llm_with_schema", _fake_call_llm_with_schema)

    e = Expert("E1", "nephro", {}, backend=ScriptedDebateBackend({}))
    case = {
        "case_id": "iraki_case_1",
        "demographics": {"age": 66, "sex": "F"},
        "clinical_notes": [{"text": "AKI with sterile pyuria."}],
    }
    a3 = e.assess_round3(
        case, questionnaire_path="dummy.json", debate_context={"debate_skipped": True}
    )
    assert isinstance(a3, AssessmentR3)
    assert a3.p_iraki == pytest.approx(0.7)
    lo, hi = a3.ci_iraki
    assert lo == pytest.approx(0.6)
    assert hi == pytest.approx(0.8)
    assert 0.0 <= a3.confidence <= 1.0


# ---------- Full scripted pipeline (R1 → R2 → R3) ----------


class DummyRouter(Router):
    """Plan: debate only Q1, minority=['E1']."""

    def __init__(self, minority_ids):
        self._min = list(minority_ids)

    def plan(self, r1, rules):
        # single QID present in r1 payloads
        qids = list(next(iter(r1))[1].scores.keys()) if r1 else []
        return types.SimpleNamespace(by_qid={qid: self._min[:] for qid in qids})


def test_full_pipeline_scripted(monkeypatch):
    """
    End-to-end scripted run: R1 (scripted) → R2 (scripted debate) → R3 (scripted) → aggregate.
    Verifies the moving parts work together without a live model.
    """
    import expert as expert_mod
    import moderator as mod

    # Stable qids and rules
    monkeypatch.setattr(expert_mod, "load_qids", lambda _p: ["Q1"])
    monkeypatch.setattr(mod, "load_qids", lambda _p: ["Q1"])
    monkeypatch.setattr(
        mod,
        "load_consensus_rules",
        lambda _p: types.SimpleNamespace(
            minimum_agreement=0.7, debate_threshold_points=2
        ),
    )

    # Minimal prompts
    monkeypatch.setattr(expert_mod, "format_round1_prompt", lambda **_: "R1 PROMPT")
    monkeypatch.setattr(expert_mod, "format_round3_prompt", lambda **_: "R3 PROMPT")

    # Script R1 and R3 schema calls
    def _fake_call_llm_with_schema(response_model, **kwargs):
        if response_model is AssessmentR1:
            return AssessmentR1(
                case_id="iraki_case_1",
                expert_id="E1",
                scores={"Q1": 3},
                evidence={"Q1": "ev minority"},
                clinical_reasoning="x" * 220,
                primary_diagnosis="ATIN?",
                differential_diagnosis=["ATN", "Prerenal"],
                rationale={"Q1": "minority rationale"},
                q_confidence={"Q1": 0.65},
                importance={"Q1": 100},  # <-- single QID -> 100
                p_iraki=0.5,
                ci_iraki=(0.4, 0.6),
                confidence=0.7,
            )
        elif response_model is AssessmentR3:
            return AssessmentR3(
                case_id="iraki_case_1",
                expert_id="E1",
                scores={"Q1": 6},
                evidence={"Q1": "ev updated"},
                changes_from_round1={"summary": "updated", "debate_influence": "minor"},
                final_diagnosis="ATIN",
                recommendations=["Rx A"],
                p_iraki=0.65,
                ci_iraki=(0.5, 0.8),
                confidence=0.7,
                rationale={"Q1": "updated reasoning"},
                q_confidence={"Q1": 0.72},
                importance={"Q1": 100},  # <-- single QID -> 100
                verdict=True,
                confidence_in_verdict=0.75,
            )
        raise AssertionError("Unexpected response_model")

    monkeypatch.setattr(expert_mod, "call_llm_with_schema", _fake_call_llm_with_schema)

    # Debate script: E1 opens (not satisfied, handoff→E2); E2 rebuts (satisfied); E1 follow-up (satisfied).
    script = {
        ("E1", "minority_open"): {
            "satisfied": False,
            "handoff_to": "E2",
            "text": "E1 open",
        },
        ("E2", "majority_rebuttal"): {"satisfied": True, "text": "E2 rebut"},
        ("E1", "minority_followup"): {"satisfied": True, "text": "E1 close"},
    }

    e1 = Expert("E1", "nephro", {}, backend=ScriptedDebateBackend(script))
    e2 = Expert("E2", "nephro", {}, backend=ScriptedDebateBackend(script))

    M = Moderator(
        experts=[e1, e2],
        questionnaire_path="dummy.json",
        router=DummyRouter(minority_ids=["E1"]),
        aggregator=Aggregator(),
        logger=logging.getLogger("moderator-test"),
        debate_rounds=3,
        max_history_turns=4,
        max_turns_per_expert=2,
        max_total_turns_per_qid=10,
        quiet_turn_limit=2,
    )

    # R1 results injected directly (mirrors scripted R1 above)
    r1 = [
        (
            "E1",
            AssessmentR1(
                case_id="iraki_case_1",
                expert_id="E1",
                scores={"Q1": 3},
                evidence={"Q1": "ev minority"},
                clinical_reasoning="x" * 220,
                primary_diagnosis="ATIN?",
                differential_diagnosis=["ATN", "Prerenal"],
                rationale={"Q1": "minority rationale"},
                q_confidence={"Q1": 0.65},
                importance={"Q1": 100},  # <-- single QID -> 100
                p_iraki=0.5,
                ci_iraki=(0.4, 0.6),
                confidence=0.7,
            ),
        ),
        (
            "E2",
            AssessmentR1(
                case_id="iraki_case_1",
                expert_id="E2",
                scores={"Q1": 7},
                evidence={"Q1": "ev majority"},
                clinical_reasoning="y" * 220,
                primary_diagnosis="ATN",
                differential_diagnosis=["ATIN", "Prerenal"],
                rationale={"Q1": "majority rationale"},
                q_confidence={"Q1": 0.75},
                importance={"Q1": 100},  # <-- single QID -> 100
                p_iraki=0.55,
                ci_iraki=(0.45, 0.65),
                confidence=0.72,
            ),
        ),
    ]

    case = {"case_id": "iraki_case_1", "clinical_notes": [{"text": "short note"}]}
    debate = M.detect_and_run_debates(r1, case)

    # We expect at least the 3 scripted turns in order
    q_turns = debate["transcripts"]["Q1"]
    speakers = [t["expert_id"] for t in q_turns[:3]]
    assert speakers == ["E1", "E2", "E1"]
    assert q_turns[0]["speaker_role"] == "minority"
    assert q_turns[1]["speaker_role"] == "majority"

    # Now run round 3 using the same experts (scripted schema call), then aggregate
    r3 = M.assess_round(3, case, debate_ctx=debate)
    assert len(r3) == 2
    cons = M.aggregator.aggregate([a for _, a in r3])
    cd = cons.model_dump()
    assert 0.0 <= cd["iraki_probability"] <= 1.0
    assert cd["ci_iraki"][0] <= cd["iraki_probability"] <= cd["ci_iraki"][1]


def test_r1_importance_must_sum_100():
    with pytest.raises(Exception):  # or ValidationError if you import it
        AssessmentR1(
            case_id="c1",
            expert_id="E1",
            scores={"Q1": 3, "Q2": 4},
            evidence={"Q1": "e1", "Q2": "e2"},
            clinical_reasoning="x" * 220,
            primary_diagnosis="Dx",
            differential_diagnosis=["Alt1", "Alt2"],
            rationale={"Q1": "r1", "Q2": "r2"},
            q_confidence={"Q1": 0.6, "Q2": 0.4},
            importance={"Q1": 70, "Q2": 20},  # sums to 90 -> should fail
            p_iraki=0.5,
            ci_iraki=(0.4, 0.6),
            confidence=0.7,
        )
