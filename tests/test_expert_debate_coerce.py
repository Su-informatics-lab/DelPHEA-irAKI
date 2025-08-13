from expert import Expert


class NullBackend:
    def debate(self, payload):
        return {}


def test_coerce_turn_passes_optional_fields_and_normalizes():
    e = Expert(expert_id="E1", specialty="nephro", persona={}, backend=NullBackend())
    raw = {
        "text": "Clear, specific rebuttal.",
        "satisfied": False,
        "revised_score": 8,
        "handoff_to": "E2",
        "citations": ["PMID:123", 456],
    }
    turn = e._coerce_debate_turn(raw, qid="Q1", round_no=2)
    assert turn.qid == "Q1"
    assert turn.round_no == 2
    assert turn.text.startswith("Clear")
    assert turn.satisfied is False
    assert turn.revised_score == 8
    assert turn.handoff_to == "E2"
    assert turn.citations == ["PMID:123", "456"]


def test_coerce_turn_infers_satisfaction_from_length():
    e = Expert(expert_id="E1", specialty="nephro", persona={}, backend=NullBackend())

    short = e._coerce_debate_turn({"text": "too short"}, qid="Q1", round_no=2)
    assert short.satisfied is False

    long_text = "this is deliberately made a bit longer than twenty characters."
    long = e._coerce_debate_turn({"text": long_text}, qid="Q1", round_no=2)
    assert long.satisfied is True
