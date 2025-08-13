import logging
import types

import pytest

import moderator as mod
from aggregator import Aggregator
from expert import Expert
from models import AssessmentR1, AssessmentR3


class ScriptedDebateBackend:
    def __init__(self, script=None):
        self.script = script or {}

    def debate(self, payload):
        return {"text": "n/a", "satisfied": True}


def _mk_case():
    return {
        "case_id": "iraki_case_retry",
        "clinical_notes": [{"text": "brief"}],
        "demographics": {"age": 60, "sex": "F"},
    }


class FakeVE(Exception):
    """Duck-typed ValidationError with .errors()."""

    def __init__(self, errors_list):
        self._errors = errors_list

    def errors(self):
        return self._errors


@pytest.mark.parametrize("bad_sum", [90, 2020])
def test_r1_importance_sum_triggers_retry_with_hint(monkeypatch, bad_sum):
    import expert as expert_mod

    # Use our fake ValidationError inside moderator
    monkeypatch.setattr(mod, "ValidationError", FakeVE, raising=True)

    # stable qids and prompt
    monkeypatch.setattr(expert_mod, "load_qids", lambda _p: ["Q1", "Q2"])
    monkeypatch.setattr(expert_mod, "format_round1_prompt", lambda **_: "R1 PROMPT")

    # schema call produces a valid model; we inject the error via validator
    def _fake_schema(response_model, **kwargs):
        assert response_model is AssessmentR1
        return AssessmentR1(
            case_id="iraki_case_retry",
            expert_id="E1",
            scores={"Q1": 3, "Q2": 4},
            evidence={"Q1": "ev1", "Q2": "ev2"},
            clinical_reasoning="x" * 220,
            primary_diagnosis="ATIN?",
            differential_diagnosis=["ATN", "Prerenal"],
            rationale={"Q1": "r", "Q2": "r2"},
            q_confidence={"Q1": 0.6, "Q2": 0.7},
            importance={"Q1": 50, "Q2": 50},
            p_iraki=0.5,
            ci_iraki=(0.4, 0.6),
            confidence=0.7,
        )

    monkeypatch.setattr(expert_mod, "call_llm_with_schema", _fake_schema)

    first = {"flag": True}

    def _fail_once(vd, required_evidence=12):
        if first["flag"]:
            first["flag"] = False
            raise FakeVE(
                [
                    {
                        "loc": ("importance",),
                        "msg": f"importance must sum to 100 (got {bad_sum})",
                    }
                ]
            )
        # else pass

    monkeypatch.setattr(mod, "validate_round1_payload", _fail_once)

    # capture repair_hint regardless of Expert signature
    recorded_hints = []
    orig = expert_mod.Expert.assess_round1

    def _wrapped(self, case, questionnaire_path, repair_hint=None, **kw):
        recorded_hints.append(repair_hint)
        try:
            return orig(self, case, questionnaire_path, repair_hint=repair_hint, **kw)
        except TypeError:
            return orig(self, case, questionnaire_path, **kw)

    monkeypatch.setattr(expert_mod.Expert, "assess_round1", _wrapped)

    # build moderator and run R1
    monkeypatch.setattr(mod, "load_qids", lambda _p: ["Q1", "Q2"])
    monkeypatch.setattr(
        mod,
        "load_consensus_rules",
        lambda _p: types.SimpleNamespace(
            minimum_agreement=0.7, debate_threshold_points=2
        ),
    )

    e1 = Expert("E1", "nephro", {}, backend=ScriptedDebateBackend())
    M = mod.Moderator(
        experts=[e1],
        questionnaire_path="dummy.json",
        router=types.SimpleNamespace(plan=lambda *_: types.SimpleNamespace(by_qid={})),
        aggregator=Aggregator(),
        logger=logging.getLogger("moderator-test"),
    )

    out = M.assess_round(1, _mk_case())
    assert isinstance(out[0][1], AssessmentR1)
    assert recorded_hints[0] is None
    assert isinstance(recorded_hints[1], str)
    assert f"importance must sum to 100 (got {bad_sum})" in recorded_hints[1]


@pytest.mark.parametrize("bad_sum", [90, 2020])
def test_r3_importance_sum_triggers_retry_with_hint(monkeypatch, bad_sum):
    import expert as expert_mod

    # Use our fake ValidationError inside moderator
    monkeypatch.setattr(mod, "ValidationError", FakeVE, raising=True)

    monkeypatch.setattr(expert_mod, "load_qids", lambda _p: ["Q1", "Q2"])
    monkeypatch.setattr(expert_mod, "format_round3_prompt", lambda **_: "R3 PROMPT")

    def _fake_schema(response_model, **kwargs):
        assert response_model is AssessmentR3
        return AssessmentR3(
            case_id="iraki_case_retry",
            expert_id="E1",
            scores={"Q1": 6, "Q2": 5},
            evidence={"Q1": "ev1'", "Q2": "ev2'"},
            changes_from_round1={"summary": "upd", "debate_influence": "minor"},
            final_diagnosis="ATIN",
            recommendations=["Rx A"],
            rationale={"Q1": "r", "Q2": "r2"},
            q_confidence={"Q1": 0.7, "Q2": 0.65},
            importance={"Q1": 50, "Q2": 50},
            p_iraki=0.65,
            ci_iraki=(0.5, 0.8),
            confidence=0.7,
            verdict=True,
            confidence_in_verdict=0.72,
        )

    monkeypatch.setattr(expert_mod, "call_llm_with_schema", _fake_schema)

    first = {"flag": True}

    def _fail_once(vd):
        if first["flag"]:
            first["flag"] = False
            raise FakeVE(
                [
                    {
                        "loc": ("importance",),
                        "msg": f"importance must sum to 100 (got {bad_sum})",
                    }
                ]
            )
        # else pass

    monkeypatch.setattr(mod, "validate_round3_payload", _fail_once)

    recorded_hints = []
    orig = expert_mod.Expert.assess_round3

    def _wrapped(
        self, case, questionnaire_path, debate_context, repair_hint=None, **kw
    ):
        recorded_hints.append(repair_hint)
        try:
            return orig(
                self,
                case,
                questionnaire_path,
                debate_context,
                repair_hint=repair_hint,
                **kw,
            )
        except TypeError:
            return orig(self, case, questionnaire_path, debate_context, **kw)

    monkeypatch.setattr(expert_mod.Expert, "assess_round3", _wrapped)

    monkeypatch.setattr(mod, "load_qids", lambda _p: ["Q1", "Q2"])
    monkeypatch.setattr(
        mod,
        "load_consensus_rules",
        lambda _p: types.SimpleNamespace(
            minimum_agreement=0.7, debate_threshold_points=2
        ),
    )

    e1 = Expert("E1", "nephro", {}, backend=ScriptedDebateBackend())
    M = mod.Moderator(
        experts=[e1],
        questionnaire_path="dummy.json",
        router=types.SimpleNamespace(plan=lambda *_: types.SimpleNamespace(by_qid={})),
        aggregator=Aggregator(),
        logger=logging.getLogger("moderator-test"),
    )

    out = M.assess_round(3, _mk_case(), debate_ctx={"debate_skipped": True})
    assert isinstance(out[0][1], AssessmentR3)
    assert recorded_hints[0] is None
    assert isinstance(recorded_hints[1], str)
    assert f"importance must sum to 100 (got {bad_sum})" in recorded_hints[1]
