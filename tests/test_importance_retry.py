import logging
import types

import pytest

import moderator as mod
from aggregator import Aggregator
from expert import Expert
from models import AssessmentR1, AssessmentR3


class ScriptedDebateBackend:
    """Just to satisfy Expert() ctor; debate path unused here."""

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


@pytest.mark.parametrize("bad_sum", [90, 2020])
def test_r1_importance_sum_triggers_retry_with_hint(monkeypatch, bad_sum):
    """
    If R1 validation complains 'importance must sum to 100 (got X)',
    Moderator should retry Expert.assess_round1 with a repair_hint that
    includes that exact message.
    """
    import expert as expert_mod

    # Stable qids and trivial prompts so we don't touch disk
    monkeypatch.setattr(expert_mod, "load_qids", lambda _p: ["Q1", "Q2"])
    monkeypatch.setattr(expert_mod, "format_round1_prompt", lambda **_: "R1 PROMPT")

    # 1) Make schema call always produce a valid AssessmentR1 (importance OK).
    #    We'll inject the *error* via moderator.validate_round1_payload to force a retry.
    def _fake_call_llm_with_schema(response_model, **kwargs):
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
            importance={
                "Q1": 50,
                "Q2": 50,
            },  # model is valid; we force error downstream
            p_iraki=0.5,
            ci_iraki=(0.4, 0.6),
            confidence=0.7,
        )

    monkeypatch.setattr(expert_mod, "call_llm_with_schema", _fake_call_llm_with_schema)

    # 2) Patch moderator.ValidationError to a local class we can raise,
    #    and patch validate_round1_payload to fail once with our custom message.
    class FakeVE(mod.ValidationError):  # keep isinstance checks happy
        def __init__(self, errors_list):
            super().__init__("fake validation error")
            self._errors = errors_list

        def errors(self):
            return self._errors

    first_call = {"flag": True}

    def _fail_once_validate_round1_payload(vd, required_evidence=12):
        if first_call["flag"]:
            first_call["flag"] = False
            raise FakeVE(
                [
                    {
                        "loc": ("importance",),
                        "msg": f"importance must sum to 100 (got {bad_sum})",
                    }
                ]
            )
        # pass on retry

    monkeypatch.setattr(
        mod, "validate_round1_payload", _fail_once_validate_round1_payload
    )

    # 3) Wrap Expert.assess_round1 to record the repair_hint passed on retry.
    recorded_hints = []

    orig_assess_round1 = expert_mod.Expert.assess_round1

    def _wrapped_assess_round1(self, case, questionnaire_path, *args, **kwargs):
        recorded_hints.append(kwargs.get("repair_hint"))
        return orig_assess_round1(self, case, questionnaire_path, *args, **kwargs)

    monkeypatch.setattr(expert_mod.Expert, "assess_round1", _wrapped_assess_round1)

    # 4) Build a Moderator (no debate invoked here).
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

    # 5) Run R1: should call assess_round1 twice (None, then a hint with our message).
    out = M.assess_round(1, _mk_case())
    assert isinstance(out[0][1], AssessmentR1)

    assert recorded_hints[0] is None, "first call should have no repair hint"
    assert (
        isinstance(recorded_hints[1], str) and recorded_hints[1]
    ), "retry must pass a repair hint"
    assert f"importance must sum to 100 (got {bad_sum})" in recorded_hints[1]


@pytest.mark.parametrize("bad_sum", [90, 2020])
def test_r3_importance_sum_triggers_retry_with_hint(monkeypatch, bad_sum):
    """
    Same as above, but for R3 path via Moderator._call_round3_with_repair.
    """
    import expert as expert_mod

    # Stable qids and trivial prompts
    monkeypatch.setattr(expert_mod, "load_qids", lambda _p: ["Q1", "Q2"])
    monkeypatch.setattr(expert_mod, "format_round3_prompt", lambda **_: "R3 PROMPT")

    # Valid R3 returned by the schema call; we'll fail validation once in moderator.
    def _fake_call_llm_with_schema(response_model, **kwargs):
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
            importance={"Q1": 50, "Q2": 50},  # valid; error injected downstream
            p_iraki=0.65,
            ci_iraki=(0.5, 0.8),
            confidence=0.7,
            verdict=True,
            confidence_in_verdict=0.72,
        )

    monkeypatch.setattr(expert_mod, "call_llm_with_schema", _fake_call_llm_with_schema)

    # Patch moderator's validator to fail once with the 'importance' message.
    class FakeVE(mod.ValidationError):
        def __init__(self, errors_list):
            super().__init__("fake validation error")
            self._errors = errors_list

        def errors(self):
            return self._errors

    first_call = {"flag": True}

    def _fail_once_validate_round3_payload(vd):
        if first_call["flag"]:
            first_call["flag"] = False
            raise FakeVE(
                [
                    {
                        "loc": ("importance",),
                        "msg": f"importance must sum to 100 (got {bad_sum})",
                    }
                ]
            )
        # pass on retry

    monkeypatch.setattr(
        mod, "validate_round3_payload", _fail_once_validate_round3_payload
    )

    # Capture repair_hint on Expert.assess_round3
    recorded_hints = []

    orig_assess_round3 = expert_mod.Expert.assess_round3

    def _wrapped_assess_round3(
        self, case, questionnaire_path, debate_context, *args, **kwargs
    ):
        recorded_hints.append(kwargs.get("repair_hint"))
        return orig_assess_round3(
            self, case, questionnaire_path, debate_context, *args, **kwargs
        )

    monkeypatch.setattr(expert_mod.Expert, "assess_round3", _wrapped_assess_round3)

    # Build Moderator and run round 3
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
    assert isinstance(recorded_hints[1], str) and recorded_hints[1]
    assert f"importance must sum to 100 (got {bad_sum})" in recorded_hints[1]
