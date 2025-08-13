from llm_backend import LLMBackend


class FakeBackend(LLMBackend):
    """Override generate() so we never hit the network."""

    def __init__(self, script):
        super().__init__(endpoint_url="http://unused", model_name="unused")
        self._script = script  # function(prompt)->str or fixed string

    def generate(
        self, prompt, *, max_tokens=512, temperature=0.3, system="", prefer_json=None
    ) -> str:
        return self._script(prompt) if callable(self._script) else str(self._script)


def _mk_payload():
    return {
        "expert_id": "E1",
        "specialty": "nephrology",
        "qid": "Q1",
        "round_no": 2,
        "clinical_context": {"role": "minority_open", "peer_turns": []},
        "minority_view": "E1: score=3 evidence=foo",
        "temperature": 0.2,
    }


def test_debate_parses_control_lines_and_strips_from_text():
    def script(_prompt):
        return (
            "Argument body about biopsy timing and PPI exposure.\n"
            "SATISFIED: no\n"
            "REVISED_SCORE: 7\n"
            "HANDOFF: E2\n"
        )

    b = FakeBackend(script)
    out = b.debate(_mk_payload())
    assert out["expert_id"] == "E1"
    assert out["qid"] == "Q1"
    assert out.get("satisfied") is False
    assert out.get("revised_score") == 7
    assert out.get("handoff_to") == "E2"
    assert "SATISFIED:" not in out["text"]
    assert "REVISED_SCORE:" not in out["text"]
    assert "HANDOFF:" not in out["text"]


def test_debate_handles_same_and_none_controls():
    b = FakeBackend(
        "Short point.\nSATISFIED: yes\nREVISED_SCORE: same\nHANDOFF: none\n"
    )
    out = b.debate(_mk_payload())
    assert out.get("satisfied") is True
    assert out.get("revised_score") is None
    assert out.get("handoff_to") is None


def test_debate_salvages_when_model_returns_json():
    json_blob = '{"text":"JSON-y argument","satisfied":true,"revised_score":9,"handoff_to":"E3"}'
    b = FakeBackend(json_blob)
    out = b.debate(_mk_payload())
    assert out["text"] == "JSON-y argument"
    assert out.get("satisfied") is True
    assert out.get("revised_score") == 9
    assert out.get("handoff_to") == "E3"
