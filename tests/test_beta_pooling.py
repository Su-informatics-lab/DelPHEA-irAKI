#!/usr/bin/env python3
"""
Comprehensive Test Suite for Beta Opinion Pooling
==================================================

Tests the mathematical correctness and edge case handling of the beta pooling
consensus mechanism for DelPHEA-irAKI. Ensures Nature Medicine standards for
reproducibility and fail-loud validation.

Run with: pytest tests/test_beta_pool.py -v --tb=short
"""

import os

# assuming the module structure
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestration.consensus import beta_pool_confidence


class TestBetaPoolingMathematics:
    """Test mathematical properties of beta pooling."""

    def test_unanimous_agreement_high(self):
        """Test when all experts strongly agree on irAKI."""
        # 5 experts all think irAKI very likely
        p_vec = np.array([0.95, 0.92, 0.94, 0.96, 0.93])
        ci_mat = np.array(
            [[0.90, 0.99], [0.88, 0.96], [0.89, 0.98], [0.92, 0.99], [0.88, 0.97]]
        )
        weight_vec = np.array([0.9, 0.85, 0.88, 0.92, 0.87])

        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # should have high consensus probability
        assert (
            result["pooled_mean"] > 0.9
        ), "Unanimous high agreement should yield high P(irAKI)"
        # should have narrow CI due to agreement
        ci_width = result["pooled_ci"][1] - result["pooled_ci"][0]
        assert ci_width < 0.15, "Unanimous agreement should yield narrow CI"
        # should have high confidence
        assert (
            result["consensus_conf"] > 0.8
        ), "Unanimous agreement should yield high confidence"

    def test_unanimous_agreement_low(self):
        """Test when all experts strongly agree NOT irAKI."""
        # 5 experts all think irAKI very unlikely
        p_vec = np.array([0.05, 0.08, 0.06, 0.04, 0.07])
        ci_mat = np.array(
            [[0.01, 0.10], [0.04, 0.12], [0.02, 0.11], [0.01, 0.08], [0.03, 0.12]]
        )
        weight_vec = np.array([0.9, 0.85, 0.88, 0.92, 0.87])

        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # should have low consensus probability
        assert (
            result["pooled_mean"] < 0.1
        ), "Unanimous low agreement should yield low P(irAKI)"
        # should have narrow CI due to agreement
        ci_width = result["pooled_ci"][1] - result["pooled_ci"][0]
        assert ci_width < 0.15, "Unanimous agreement should yield narrow CI"
        # should have high confidence
        assert (
            result["consensus_conf"] > 0.8
        ), "Unanimous agreement should yield high confidence"

    def test_complete_disagreement(self):
        """Test when experts are completely split."""
        # experts completely disagree
        p_vec = np.array([0.1, 0.9, 0.2, 0.8, 0.5])
        ci_mat = np.array(
            [[0.05, 0.15], [0.85, 0.95], [0.15, 0.25], [0.75, 0.85], [0.40, 0.60]]
        )
        weight_vec = np.array([0.9, 0.9, 0.9, 0.9, 0.9])  # all equally confident

        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # should have middling probability
        assert (
            0.3 < result["pooled_mean"] < 0.7
        ), "Complete disagreement should yield uncertain P(irAKI)"
        # should have wide CI due to disagreement
        ci_width = result["pooled_ci"][1] - result["pooled_ci"][0]
        assert ci_width > 0.3, "Complete disagreement should yield wide CI"
        # should have low confidence
        assert (
            result["consensus_conf"] < 0.5
        ), "Complete disagreement should yield low confidence"

    def test_single_outlier_expert(self):
        """Test robustness to single outlier expert."""
        # 4 experts agree, 1 outlier
        p_vec = np.array([0.85, 0.82, 0.88, 0.86, 0.15])  # last is outlier
        ci_mat = np.array(
            [
                [0.80, 0.90],
                [0.77, 0.87],
                [0.83, 0.93],
                [0.81, 0.91],
                [0.10, 0.20],  # outlier
            ]
        )
        # outlier has lower confidence
        weight_vec = np.array([0.9, 0.88, 0.92, 0.89, 0.4])

        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # consensus should still lean toward majority
        assert (
            result["pooled_mean"] > 0.7
        ), "Majority should dominate with low-weight outlier"
        # confidence should be moderate due to disagreement
        assert 0.4 < result["consensus_conf"] < 0.8, "Outlier should reduce confidence"


class TestBetaPoolingEdgeCases:
    """Test edge cases and error handling."""

    def test_single_expert(self):
        """Test handling of single expert (no pooling possible)."""
        p_vec = np.array([0.7])
        ci_mat = np.array([[0.6, 0.8]])
        weight_vec = np.array([1.0])

        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # should return expert's estimate
        assert (
            abs(result["pooled_mean"] - 0.7) < 0.05
        ), "Single expert should return their estimate"
        # confidence should reflect single opinion
        assert (
            result["consensus_conf"] < 0.6
        ), "Single expert should have lower confidence"

    def test_extreme_probabilities(self):
        """Test handling of boundary probabilities (0 and 1)."""
        # mix of extreme and moderate probabilities
        p_vec = np.array([0.0, 1.0, 0.5, 0.99, 0.01])
        ci_mat = np.array(
            [[0.0, 0.05], [0.95, 1.0], [0.4, 0.6], [0.97, 1.0], [0.0, 0.03]]
        )
        weight_vec = None  # test equal weighting

        # should not crash
        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # check result is valid
        assert 0 <= result["pooled_mean"] <= 1, "Probability must be in [0,1]"
        assert (
            0 <= result["pooled_ci"][0] <= result["pooled_ci"][1] <= 1
        ), "CI must be valid"

    def test_missing_confidence_intervals(self):
        """Test graceful handling when CIs are missing."""
        p_vec = np.array([0.7, 0.65, 0.75])
        # simulate missing CIs with NaN
        ci_mat = np.array([[np.nan, np.nan], [0.55, 0.75], [0.65, 0.85]])
        weight_vec = np.array([0.8, 0.9, 0.85])

        # should handle gracefully
        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # should still produce valid result
        assert not np.isnan(result["pooled_mean"]), "Should handle missing CIs"
        assert (
            result["consensus_conf"] > 0
        ), "Should compute confidence despite missing CIs"

    def test_invalid_probabilities(self):
        """Test that invalid probabilities raise errors."""
        # probability outside [0,1]
        p_vec = np.array([0.5, 1.5, 0.7])  # 1.5 is invalid
        ci_mat = np.array([[0.4, 0.6], [0.8, 0.9], [0.6, 0.8]])

        with pytest.raises(ValueError, match="probabilities must be in"):
            beta_pool_confidence(p_vec, ci_mat)

    def test_inverted_confidence_intervals(self):
        """Test auto-correction of inverted CI bounds."""
        p_vec = np.array([0.7, 0.75])
        # second CI is inverted
        ci_mat = np.array([[0.6, 0.8], [0.85, 0.65]])  # inverted
        weight_vec = np.array([0.9, 0.85])

        # should auto-correct and not crash
        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        assert result["pooled_ci"][0] < result["pooled_ci"][1], "CI should be corrected"

    def test_zero_weights(self):
        """Test handling of zero weights (expert abstention)."""
        p_vec = np.array([0.7, 0.75, 0.2])
        ci_mat = np.array([[0.6, 0.8], [0.65, 0.85], [0.1, 0.3]])
        # middle expert has zero weight
        weight_vec = np.array([0.9, 0.0, 0.85])

        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # result should be between non-zero weight experts
        assert 0.3 < result["pooled_mean"] < 0.8, "Zero weight expert should be ignored"


class TestBetaPoolingClinicalScenarios:
    """Test clinically relevant scenarios."""

    def test_high_confidence_nephrologist_scenario(self):
        """Test scenario where nephrologist has high confidence in irAKI."""
        # nephrologist very confident, others less sure
        p_vec = np.array(
            [
                0.95,  # nephrologist - high confidence irAKI
                0.70,  # oncologist - moderate confidence
                0.60,  # pharmacist - uncertain
                0.75,  # intensivist - moderate confidence
            ]
        )
        ci_mat = np.array(
            [
                [0.90, 0.99],  # narrow CI - high certainty
                [0.55, 0.85],  # moderate CI
                [0.40, 0.80],  # wide CI - uncertain
                [0.60, 0.90],  # moderate CI
            ]
        )
        # nephrologist has highest weight due to expertise
        weight_vec = np.array([0.95, 0.75, 0.60, 0.70])

        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # should lean toward nephrologist's assessment
        assert (
            result["pooled_mean"] > 0.75
        ), "Should weight expert opinion appropriately"
        assert (
            result["consensus_conf"] > 0.6
        ), "High-weight expert should increase confidence"

    def test_conflicting_specialties_scenario(self):
        """Test when different specialties disagree based on their perspective."""
        # simulate realistic specialty disagreement
        p_vec = np.array(
            [
                0.80,  # nephrologist - sees kidney injury pattern
                0.30,  # infectious disease - suspects infection
                0.75,  # rheumatologist - sees immune activation
                0.25,  # intensivist - thinks prerenal
                0.85,  # oncologist - timing matches ICI
            ]
        )
        ci_mat = np.array(
            [[0.70, 0.90], [0.20, 0.40], [0.65, 0.85], [0.15, 0.35], [0.75, 0.95]]
        )
        weight_vec = np.array([0.85, 0.80, 0.75, 0.70, 0.82])

        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # should reflect the disagreement
        assert (
            0.4 < result["pooled_mean"] < 0.7
        ), "Disagreement should yield uncertain consensus"
        assert (
            result["consensus_conf"] < 0.6
        ), "Specialty conflict should reduce confidence"
        # CI should be wide
        ci_width = result["pooled_ci"][1] - result["pooled_ci"][0]
        assert ci_width > 0.25, "Disagreement should yield wide uncertainty"

    def test_post_debate_convergence(self):
        """Test Round 3 scenario where debate led to convergence."""
        # after debate, experts converge
        p_vec = np.array([0.72, 0.68, 0.75, 0.70, 0.73])
        ci_mat = np.array(
            [[0.65, 0.79], [0.61, 0.75], [0.68, 0.82], [0.63, 0.77], [0.66, 0.80]]
        )
        # confidence increases after debate
        weight_vec = np.array([0.92, 0.88, 0.90, 0.89, 0.91])

        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # should have moderate-high probability
        assert (
            0.65 < result["pooled_mean"] < 0.75
        ), "Convergence should yield clear direction"
        # should have good confidence due to agreement
        assert (
            result["consensus_conf"] > 0.75
        ), "Post-debate convergence should increase confidence"
        # CI should be relatively narrow
        ci_width = result["pooled_ci"][1] - result["pooled_ci"][0]
        assert ci_width < 0.2, "Convergence should narrow uncertainty"


class TestBetaPoolingMathematicalProperties:
    """Test that mathematical properties of beta pooling hold."""

    def test_weight_normalization(self):
        """Test that weight normalization preserves relative importance."""
        p_vec = np.array([0.7, 0.6, 0.8])
        ci_mat = np.array([[0.6, 0.8], [0.5, 0.7], [0.7, 0.9]])

        # test with different weight scales
        weight_vec1 = np.array([1.0, 0.5, 0.75])
        weight_vec2 = np.array([10.0, 5.0, 7.5])  # scaled by 10

        result1 = beta_pool_confidence(p_vec, ci_mat, weight_vec1)
        result2 = beta_pool_confidence(p_vec, ci_mat, weight_vec2)

        # results should be very similar despite different scales
        assert abs(result1["pooled_mean"] - result2["pooled_mean"]) < 0.01
        assert abs(result1["consensus_conf"] - result2["consensus_conf"]) < 0.01

    def test_beta_distribution_properties(self):
        """Test that beta distribution is properly formed."""
        # moderate agreement case
        p_vec = np.array([0.65, 0.70, 0.60, 0.68])
        ci_mat = np.array([[0.55, 0.75], [0.60, 0.80], [0.50, 0.70], [0.58, 0.78]])
        weight_vec = np.array([0.8, 0.85, 0.75, 0.82])

        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # verify CI contains the mean
        assert (
            result["pooled_ci"][0] <= result["pooled_mean"] <= result["pooled_ci"][1]
        ), "Mean should be within CI"

        # verify CI is reasonable width (not degenerate)
        ci_width = result["pooled_ci"][1] - result["pooled_ci"][0]
        assert 0.05 < ci_width < 0.5, "CI width should be reasonable"

    def test_confidence_harmonic_mean(self):
        """Test that confidence uses harmonic mean of components."""
        # case with high between-expert agreement but wide CIs
        p_vec = np.array([0.71, 0.69, 0.72, 0.70])  # high agreement
        ci_mat = np.array(
            [
                [0.50, 0.92],  # wide CI
                [0.48, 0.90],  # wide CI
                [0.51, 0.93],  # wide CI
                [0.49, 0.91],  # wide CI
            ]
        )
        weight_vec = np.array([0.9, 0.9, 0.9, 0.9])

        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # confidence should be moderate (high agreement but low certainty)
        assert (
            0.3 < result["consensus_conf"] < 0.7
        ), "Wide CIs should limit confidence despite agreement"


class TestBetaPoolingIntegration:
    """Test integration with DelPHEA-irAKI system."""

    def test_round1_to_round3_improvement(self):
        """Test that Round 3 typically improves consensus."""
        # Round 1: high disagreement
        round1_p = np.array([0.3, 0.8, 0.5, 0.9, 0.2])
        round1_ci = np.array(
            [[0.2, 0.4], [0.7, 0.9], [0.3, 0.7], [0.8, 0.95], [0.1, 0.3]]
        )
        round1_w = np.array([0.6, 0.7, 0.5, 0.8, 0.6])

        round1_result = beta_pool_confidence(round1_p, round1_ci, round1_w)

        # Round 3: after debate, convergence
        round3_p = np.array([0.65, 0.70, 0.68, 0.72, 0.63])
        round3_ci = np.array(
            [[0.58, 0.72], [0.63, 0.77], [0.61, 0.75], [0.65, 0.79], [0.56, 0.70]]
        )
        round3_w = np.array([0.85, 0.88, 0.82, 0.90, 0.83])  # increased confidence

        round3_result = beta_pool_confidence(round3_p, round3_ci, round3_w)

        # Round 3 should have better consensus
        assert (
            round3_result["consensus_conf"] > round1_result["consensus_conf"]
        ), "Debate should improve consensus confidence"

        # Round 3 should have narrower CI
        round1_width = round1_result["pooled_ci"][1] - round1_result["pooled_ci"][0]
        round3_width = round3_result["pooled_ci"][1] - round3_result["pooled_ci"][0]
        assert round3_width < round1_width, "Debate should narrow uncertainty"

    def test_eleven_expert_panel(self):
        """Test with full 11-expert panel as in DelPHEA-irAKI."""
        # simulate full expert panel
        np.random.seed(42)  # for reproducibility

        # create realistic distribution of opinions
        base_prob = 0.65  # moderate irAKI likelihood
        p_vec = np.random.normal(base_prob, 0.15, 11)
        p_vec = np.clip(p_vec, 0.1, 0.9)  # keep in valid range

        # create CIs around probabilities
        ci_mat = np.zeros((11, 2))
        for i in range(11):
            width = np.random.uniform(0.1, 0.25)
            ci_mat[i] = [max(0, p_vec[i] - width / 2), min(1, p_vec[i] + width / 2)]

        # varied confidence weights
        weight_vec = np.random.uniform(0.6, 0.95, 11)

        result = beta_pool_confidence(p_vec, ci_mat, weight_vec)

        # check all outputs are present and valid
        assert "pooled_mean" in result
        assert "pooled_ci" in result
        assert "consensus_conf" in result
        assert 0 <= result["pooled_mean"] <= 1
        assert len(result["pooled_ci"]) == 2
        assert 0 <= result["consensus_conf"] <= 1


if __name__ == "__main__":
    # run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
