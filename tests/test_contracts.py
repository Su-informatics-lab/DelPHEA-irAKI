import json

import pytest
from pydantic import ValidationError

from models import AssessmentR1, AssessmentR3
from validators import call_llm_with_schema

QIDS = ["Q1", "Q2"]

BASE_R1 = {
    "case_id": "C-001",
    "expert_id": "nephro_1",
    "scores": {"Q1": 7, "Q2": 3},
    "rationale": {
        "Q1": "ATIN pattern with sterile pyuria.",
        "Q2": "No obstruction on US.",
    },
    "evidence": {
        "Q1": "UA: WBC casts; recent PPI.",
        "Q2": "Renal US 8/1: no hydronephrosis.",
    },
    "q_confidence": {"Q1": 0.8, "Q2": 0.6},
    "importance": {"Q1": 60, "Q2": 40},
    "p_iraki": 0.62,
    "ci_iraki": (0.45, 0.77),
    "confidence": 0.7,
    "clinical_reasoning": "Narrative " * 50,
    "differential_diagnosis": ["irAKI (ATIN)", "ATN"],
    "primary_diagnosis": "irAKI (ATIN)",
}

BASE_R3 = {
    **{
        k: BASE_R1[k]
        for k in [
            "case_id",
            "expert_id",
            "scores",
            "rationale",
            "evidence",
            "q_confidence",
            "importance",
            "p_iraki",
            "ci_iraki",
            "confidence",
        ]
    },
    "changes_from_round1": {"overall": "Upweighted UA and timing after debate."},
    "verdict": True,
    "final_diagnosis": "irAKI (ATIN)",
    "confidence_in_verdict": 0.75,
    "recommendations": ["Hold ICI; start steroids; nephrology follow-up."],
}

# --------------------------- schema-only tests ---------------------------


def test_r1_valid_with_expected_qids_context():
    AssessmentR1.model_validate(BASE_R1, context={"expected_qids": QIDS})


def test_r3_valid_with_expected_qids_context():
    AssessmentR3.model_validate(BASE_R3, context={"expected_qids": QIDS})


def test_importance_must_sum_100():
    bad = {**BASE_R1, "importance": {"Q1": 50, "Q2": 30}}
    with pytest.raises(ValidationError):
        AssessmentR1.model_validate(bad, context={"expected_qids": QIDS})


def test_importance_zeros_allowed_if_sum_100():
    ok = {**BASE_R1, "importance": {"Q1": 100, "Q2": 0}}
    AssessmentR1.model_validate(ok, context={"expected_qids": QIDS})


def test_q_confidence_range():
    bad = {**BASE_R1, "q_confidence": {"Q1": 1.2, "Q2": 0.4}}
    with pytest.raises(ValidationError):
        AssessmentR1.model_validate(bad, context={"expected_qids": QIDS})


def test_parallel_qids_identical_and_ordered():
    # order mismatch relative to expected_qids
    bad = {**BASE_R1, "rationale": {"Q2": "x", "Q1": "y"}}
    with pytest.raises(ValidationError):
        AssessmentR1.model_validate(bad, context={"expected_qids": QIDS})


def test_ci_contains_point_and_is_ordered():
    bad = {**BASE_R1, "ci_iraki": (0.7, 0.6)}
    with pytest.raises(ValidationError):
        AssessmentR1.model_validate(bad, context={"expected_qids": QIDS})


# --------------------------- call_llm_with_schema ---------------------------


class DummyBackend:
    model_name = "dummy"

    def capabilities(self):
        return {"context_window": 4096}

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        # by default, return a fenced JSON blob with correct shape
        payload = {
            **BASE_R1,
            "case_id": "C-001",
            "expert_id": "nephro_1",
        }
        return "```json\n" + json.dumps(payload) + "\n```"


def test_call_llm_with_schema_ok_r1_with_fences():
    backend = DummyBackend()
    prompt = '... IMPORTANT: In your JSON output, you MUST include precisely these keys:\n- "case_id": "C-001"\n- "expert_id": "nephro_1"\n'
    out = call_llm_with_schema(
        response_model=AssessmentR1,
        prompt_text=prompt,
        backend=backend,
        expected_qids=QIDS,
    )
    assert out.case_id == "C-001"
    assert out.expert_id == "nephro_1"
    assert out.scores == {"Q1": 7, "Q2": 3}


def test_call_llm_with_schema_enforces_expected_qids_order():
    class ShuffledBackend(DummyBackend):
        def generate(
            self, prompt: str, max_tokens: int, temperature: float = 0.0
        ) -> str:
            payload = {**BASE_R1}
            # shuffle per-question keys
            payload["scores"] = {"Q2": 3, "Q1": 7}
            payload["rationale"] = {"Q2": "b", "Q1": "a"}
            payload["evidence"] = {"Q2": "e2", "Q1": "e1"}
            payload["q_confidence"] = {"Q2": 0.6, "Q1": 0.8}
            payload["importance"] = {"Q2": 40, "Q1": 60}
            payload["case_id"] = "C-001"
            payload["expert_id"] = "nephro_1"
            return json.dumps(payload)

    backend = ShuffledBackend()
    prompt = '... "case_id": "C-001"\n- "expert_id": "nephro_1"\n'
    with pytest.raises(ValidationError):
        call_llm_with_schema(
            response_model=AssessmentR1,
            prompt_text=prompt,
            backend=backend,
            expected_qids=QIDS,
        )


def test_call_llm_with_schema_verdict_string_coercion_r3():
    class StringVerdictBackend(DummyBackend):
        def generate(
            self, prompt: str, max_tokens: int, temperature: float = 0.0
        ) -> str:
            payload = {
                **BASE_R3,
                "case_id": "C-001",
                "expert_id": "nephro_1",
                "verdict": "likely",  # should coerce to True
            }
            return json.dumps(payload)

    backend = StringVerdictBackend()
    prompt = '... "case_id": "C-001"\n- "expert_id": "nephro_1"\n'
    out = call_llm_with_schema(
        response_model=AssessmentR3,
        prompt_text=prompt,
        backend=backend,
        expected_qids=QIDS,
    )
    assert out.verdict is True
