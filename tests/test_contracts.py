# tests/test_contracts.py
import pytest
from pydantic import ValidationError

from models import AssessmentR1, AssessmentR3

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
    "clinical_reasoning": "Long narrative …" * 10,
    "differential_diagnosis": ["ATIN (irAKI)", "ATN"],
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


def test_r1_valid_with_expected_qids_context():
    AssessmentR1.model_validate(BASE_R1, context={"expected_qids": QIDS})


def test_r3_valid_with_expected_qids_context():
    AssessmentR3.model_validate(BASE_R3, context={"expected_qids": QIDS})


def test_importance_must_sum_100():
    bad = {**BASE_R1, "importance": {"Q1": 50, "Q2": 30}}
    with pytest.raises(ValidationError):
        AssessmentR1.model_validate(bad, context={"expected_qids": QIDS})


def test_q_confidence_range():
    bad = {**BASE_R1, "q_confidence": {"Q1": 1.2, "Q2": 0.4}}
    with pytest.raises(ValidationError):
        AssessmentR1.model_validate(bad, context={"expected_qids": QIDS})


def test_parallel_qids_identical_and_ordered():
    bad = {**BASE_R1, "rationale": {"Q2": "x", "Q1": "y"}}  # order mismatch
    with pytest.raises(ValidationError):
        AssessmentR1.model_validate(bad, context={"expected_qids": QIDS})


def test_ci_contains_point_and_is_ordered():
    bad = {**BASE_R1, "ci_iraki": (0.7, 0.6)}
    with pytest.raises(ValidationError):
        AssessmentR1.model_validate(bad, context={"expected_qids": QIDS})
