#!/usr/bin/env python3
"""
Comprehensive Beta Pooling Tests for DelPHEA-irAKI
====================================================

Critical validation of beta opinion pooling mathematics for Nature Medicine.
Tests cover edge cases, mathematical properties, and clinical scenarios.

Mathematical Properties Tested:
- Consensus always within expert range
- Confidence increases with agreement
- Weights properly influence outcome
- Edge cases don't break computation
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# add parent directory to import consensus module
sys.path.append(str(Path(__file__).parent.parent))

from orchestration.consensus import beta_pool_confidence


class TestBetaPoolingMathematics:
    """Test suite for beta pooling mathematical correctness."""

    def test_unanimous_agreement_high_probability(self):
        """When all experts strongly agree on irAKI, consensus should reflect that."""
        # all experts very confident of irAKI
        p_vec = np.array([0.9, 0.92, 0.88, 0.91, 0.89, 0.93, 0.90, 0.91])
        ci_mat = np.array(
            [
                [0.85, 0.95],
                [0.88, 0.96],
                [0.83, 0.93],
                [0.87, 0.95],
                [0.84, 0.94],
                [0.89, 0.97],
                [0.86, 0.94],
                [0.87, 0.95],
            ]
        )
        w_vec = np.array([0.9, 0.85, 0.88, 0.92, 0.87, 0.90, 0.86, 0.89])

        result = beta_pool_confidence(p_vec, ci_mat, w_vec)

        # consensus should be high with narrow CI
        assert (
            0.88 <= result["pooled_mean"] <= 0.93
        ), f"Consensus {result['pooled_mean']} outside expected range"
        assert (
            result["pooled_ci"][1] - result["pooled_ci"][0] < 0.15
        ), "CI should be narrow with agreement"
        assert (
            result["consensus_conf"] > 0.8
        ), "Confidence should be high with agreement"

    def test_unanimous_agreement_low_probability(self):
        """When all experts agree it's NOT irAKI, consensus should reflect that."""
        # all experts very confident it's not irAKI
        p_vec = np.array([0.1, 0.08, 0.12, 0.09, 0.11, 0.07, 0.10, 0.09])
        ci_mat = np.array(
            [
                [0.05, 0.15],
                [0.04, 0.12],
                [0.07, 0.17],
                [0.05, 0.13],
                [0.06, 0.16],
                [0.03, 0.11],
                [0.06, 0.14],
                [0.05, 0.13],
            ]
        )
        w_vec = np.array([0.9, 0.85, 0.88, 0.92, 0.87, 0.90, 0.86, 0.89])

        result = beta_pool_confidence(p_vec, ci_mat, w_vec)

        # consensus should be low with narrow CI
        assert (
            0.07 <= result["pooled_mean"] <= 0.13
        ), f"Consensus {result['pooled_mean']} outside expected range"
        assert (
            result["pooled_ci"][1] - result["pooled_ci"][0] < 0.15
        ), "CI should be narrow with agreement"
        assert (
            result["consensus_conf"] > 0.8
        ), "Confidence should be high with agreement"

    def test_complete_disagreement(self):
        """When experts completely disagree, confidence should be low."""
        # half think definitely irAKI, half think definitely not
        p_vec = np.array([0.9, 0.92, 0.88, 0.91, 0.1, 0.08, 0.12, 0.09])
        ci_mat = np.array(
            [
                [0.85, 0.95],
                [0.88, 0.96],
                [0.83, 0.93],
                [0.87, 0.95],
                [0.05, 0.15],
                [0.04, 0.12],
                [0.07, 0.17],
                [0.05, 0.13],
            ]
        )
        w_vec = np.ones(8)  # equal weights

        result = beta_pool_confidence(p_vec, ci_mat, w_vec)

        # consensus should be near 0.5 with wide CI
        assert (
            0.4 <= result["pooled_mean"] <= 0.6
        ), f"Consensus {result['pooled_mean']} should be uncertain"
        assert (
            result["pooled_ci"][1] - result["pooled_ci"][0] > 0.3
        ), "CI should be wide with disagreement"
        assert (
            result["consensus_conf"] < 0.5
        ), "Confidence should be low with disagreement"

    def test_single_outlier_expert(self):
        """Single outlier shouldn't dominate consensus."""
        # 7 experts agree, 1 strongly disagrees
        p_vec = np.array([0.8, 0.82, 0.78, 0.81, 0.79, 0.83, 0.80, 0.15])
        ci_mat = np.array(
            [
                [0.75, 0.85],
                [0.78, 0.86],
                [0.73, 0.83],
                [0.77, 0.85],
                [0.74, 0.84],
                [0.79, 0.87],
                [0.76, 0.84],
                [0.10, 0.20],
            ]
        )
        w_vec = np.ones(8)  # equal weights

        result = beta_pool_confidence(p_vec, ci_mat, w_vec)

        # consensus should lean toward majority
        assert (
            0.65 <= result["pooled_mean"] <= 0.80
        ), f"Consensus {result['pooled_mean']} should favor majority"
        # but confidence should be reduced
        assert (
            result["consensus_conf"] < 0.75
        ), "Confidence should be reduced by outlier"

    def test_weight_influence(self):
        """High-confidence experts should have more influence."""
        # two groups with different confidence levels
        p_vec = np.array([0.7, 0.72, 0.3, 0.28])
        ci_mat = np.array(
            [
                [0.65, 0.75],
                [0.67, 0.77],  # narrow CI (confident)
                [0.20, 0.40],
                [0.18, 0.38],  # wide CI (uncertain)
            ]
        )
        w_vec = np.array([0.9, 0.92, 0.3, 0.28])  # weight by confidence

        result_weighted = beta_pool_confidence(p_vec, ci_mat, w_vec)

        # now equal weights for comparison
        w_vec_equal = np.ones(4)
        result_equal = beta_pool_confidence(p_vec, ci_mat, w_vec_equal)

        # weighted should lean more toward confident experts
        assert (
            result_weighted["pooled_mean"] > result_equal["pooled_mean"]
        ), "Weighting should favor high-confidence experts"

    def test_edge_case_all_zeros(self):
        """All experts giving P(irAKI)=0 shouldn't break computation."""
        p_vec = np.zeros(5)
        ci_mat = np.array([[0, 0.05] for _ in range(5)])

        result = beta_pool_confidence(p_vec, ci_mat)

        assert result["pooled_mean"] < 0.1, "Consensus should be very low"
        assert result["pooled_ci"][0] >= 0, "CI lower bound should be non-negative"
        assert result["pooled_ci"][1] <= 1, "CI upper bound should be <= 1"

    def test_edge_case_all_ones(self):
        """All experts giving P(irAKI)=1 shouldn't break computation."""
        p_vec = np.ones(5)
        ci_mat = np.array([[0.95, 1.0] for _ in range(5)])

        result = beta_pool_confidence(p_vec, ci_mat)

        assert result["pooled_mean"] > 0.9, "Consensus should be very high"
        assert result["pooled_ci"][0] >= 0, "CI lower bound should be non-negative"
        assert result["pooled_ci"][1] <= 1, "CI upper bound should be <= 1"

    def test_single_expert(self):
        """Single expert input should return valid results."""
        p_vec = np.array([0.7])
        ci_mat = np.array([[0.6, 0.8]])

        result = beta_pool_confidence(p_vec, ci_mat)

        assert (
            abs(result["pooled_mean"] - 0.7) < 0.1
        ), "Should be close to single expert"
        assert (
            result["consensus_conf"] < 0.5
        ), "Single expert shouldn't have high consensus confidence"

    def test_mathematical_properties(self):
        """Test fundamental mathematical properties of beta pooling."""
        # generate random valid inputs
        np.random.seed(42)  # reproducibility
        n_experts = 8

        for _ in range(10):  # test multiple random scenarios
            # random probabilities
            p_vec = np.random.uniform(0.1, 0.9, n_experts)

            # CIs around the probabilities
            ci_widths = np.random.uniform(0.1, 0.3, n_experts)
            ci_mat = np.column_stack(
                [
                    np.maximum(0, p_vec - ci_widths / 2),
                    np.minimum(1, p_vec + ci_widths / 2),
                ]
            )

            # random weights
            w_vec = np.random.uniform(0.5, 1.0, n_experts)

            result = beta_pool_confidence(p_vec, ci_mat, w_vec)

            # property 1: consensus within [0, 1]
            assert 0 <= result["pooled_mean"] <= 1, "Probability must be in [0,1]"

            # property 2: CI within [0, 1]
            assert 0 <= result["pooled_ci"][0] <= 1, "CI lower bound must be in [0,1]"
            assert 0 <= result["pooled_ci"][1] <= 1, "CI upper bound must be in [0,1]"

            # property 3: CI contains consensus
            assert (
                result["pooled_ci"][0]
                <= result["pooled_mean"]
                <= result["pooled_ci"][1]
            ), "Consensus must be within CI"

            # property 4: confidence in [0, 1]
            assert 0 <= result["consensus_conf"] <= 1, "Confidence must be in [0,1]"


class TestClinicalScenarios:
    """Test realistic clinical consensus scenarios."""

    def test_typical_iraki_case(self):
        """Typical irAKI case with moderate expert agreement."""
        # based on real clinical patterns: most experts lean toward irAKI
        p_vec = np.array(
            [
                0.75,  # Nephrologist - sees typical AIN pattern
                0.80,  # Oncologist - timing fits, no other irAEs
                0.70,  # Pharmacist - no major drug interactions
                0.65,  # Hospitalist - some uncertainty
                0.78,  # Rheumatologist - immune markers elevated
                0.72,  # Intensivist - hemodynamically stable
                0.68,  # Emergency - acute presentation
                0.77,  # Pathologist - if biopsy available
            ]
        )

        # realistic confidence intervals
        ci_mat = np.array(
            [
                [0.65, 0.85],
                [0.70, 0.90],
                [0.55, 0.85],
                [0.50, 0.80],
                [0.68, 0.88],
                [0.60, 0.84],
                [0.53, 0.83],
                [0.65, 0.89],
            ]
        )

        # specialty-based weights (some experts more relevant)
        w_vec = np.array(
            [
                0.95,  # Nephrologist - highest weight
                0.90,  # Oncologist - very relevant
                0.75,  # Pharmacist - relevant
                0.70,  # Hospitalist - moderate
                0.80,  # Rheumatologist - relevant for immune
                0.65,  # Intensivist - less central
                0.60,  # Emergency - initial assessment
                0.85,  # Pathologist - if data available
            ]
        )

        result = beta_pool_confidence(p_vec, ci_mat, w_vec)

        # should indicate probable irAKI with reasonable confidence
        assert 0.70 <= result["pooled_mean"] <= 0.80, "Should indicate probable irAKI"
        assert (
            0.60 <= result["consensus_conf"] <= 0.80
        ), "Should have moderate-high confidence"

    def test_complex_differential_case(self):
        """Complex case with multiple possible etiologies."""
        # experts divided: could be irAKI, ATN, or prerenal
        p_vec = np.array(
            [
                0.45,  # Nephrologist - sees mixed picture
                0.60,  # Oncologist - timing suggestive
                0.30,  # Pharmacist - recent nephrotoxins
                0.50,  # Hospitalist - clinical uncertainty
                0.40,  # Intensivist - volume issues present
                0.55,  # Radiologist - non-specific findings
                0.35,  # ID specialist - possible infection
                0.48,  # Geriatrician - multifactorial in elderly
            ]
        )

        # wide CIs reflecting uncertainty
        ci_mat = np.array(
            [
                [0.25, 0.65],
                [0.40, 0.80],
                [0.15, 0.45],
                [0.30, 0.70],
                [0.20, 0.60],
                [0.35, 0.75],
                [0.20, 0.50],
                [0.28, 0.68],
            ]
        )

        result = beta_pool_confidence(p_vec, ci_mat)

        # should show uncertainty
        assert 0.35 <= result["pooled_mean"] <= 0.55, "Should be uncertain"
        assert result["consensus_conf"] < 0.6, "Confidence should be low"
        assert (
            result["pooled_ci"][1] - result["pooled_ci"][0] > 0.25
        ), "Wide CI expected"

    def test_clear_alternative_diagnosis(self):
        """Clear non-irAKI case (e.g., contrast nephropathy)."""
        # experts agree it's not irAKI
        p_vec = np.array(
            [
                0.15,  # Nephrologist - pattern inconsistent
                0.20,  # Oncologist - timing wrong
                0.10,  # Radiologist - recent contrast
                0.18,  # Hospitalist - clear alternative
                0.12,  # Pharmacist - no ICI relationship
                0.22,  # Intensivist - prerenal features
                0.08,  # Emergency - obvious alternative
                0.16,  # Nurse practitioner - clinical picture clear
            ]
        )

        ci_mat = np.array(
            [
                [0.08, 0.22],
                [0.12, 0.28],
                [0.05, 0.15],
                [0.10, 0.26],
                [0.06, 0.18],
                [0.14, 0.30],
                [0.03, 0.13],
                [0.09, 0.23],
            ]
        )

        result = beta_pool_confidence(p_vec, ci_mat)

        # should clearly indicate not irAKI
        assert result["pooled_mean"] < 0.25, "Should indicate unlikely irAKI"
        assert (
            result["consensus_conf"] > 0.7
        ), "Should have high confidence in alternative diagnosis"


class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_probability_values(self):
        """Probabilities outside [0,1] should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid probability"):
            p_vec = np.array([0.5, 1.2, 0.7])  # 1.2 is invalid
            ci_mat = np.array([[0.4, 0.6], [1.0, 1.4], [0.6, 0.8]])
            beta_pool_confidence(p_vec, ci_mat)

    def test_negative_probability(self):
        """Negative probabilities should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid probability"):
            p_vec = np.array([0.5, -0.1, 0.7])
            ci_mat = np.array([[0.4, 0.6], [-0.2, 0.0], [0.6, 0.8]])
            beta_pool_confidence(p_vec, ci_mat)

    def test_mismatched_dimensions(self):
        """Mismatched array dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="dimension"):
            p_vec = np.array([0.5, 0.6, 0.7])  # 3 experts
            ci_mat = np.array([[0.4, 0.6], [0.5, 0.7]])  # only 2 CIs
            beta_pool_confidence(p_vec, ci_mat)

    def test_invalid_ci_ordering(self):
        """CI lower > upper should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid confidence interval"):
            p_vec = np.array([0.5, 0.6])
            ci_mat = np.array([[0.4, 0.6], [0.7, 0.5]])  # second CI reversed
            beta_pool_confidence(p_vec, ci_mat)

    def test_empty_input(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            p_vec = np.array([])
            ci_mat = np.array([]).reshape(0, 2)
            beta_pool_confidence(p_vec, ci_mat)

    def test_negative_weights(self):
        """Negative weights should raise ValueError."""
        with pytest.raises(ValueError, match="weight"):
            p_vec = np.array([0.5, 0.6])
            ci_mat = np.array([[0.4, 0.6], [0.5, 0.7]])
            w_vec = np.array([1.0, -0.5])  # negative weight
            beta_pool_confidence(p_vec, ci_mat, w_vec)


def run_all_tests():
    """Run all tests with detailed output."""
    print("\n" + "=" * 70)
    print("BETA POOLING VALIDATION FOR NATURE MEDICINE")
    print("=" * 70)

    # run pytest with verbose output
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
