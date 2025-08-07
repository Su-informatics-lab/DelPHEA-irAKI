"""
Consensus Calculation Module for DelPHEA-irAKI
===============================================

This module implements beta opinion pooling for aggregating expert assessments
into a unified consensus probability for immune-related AKI classification.

Mathematical Foundation:
-----------------------
Beta opinion pooling aggregates N expert probability estimates using a
conjugate beta distribution approach, weighted by expert confidence.

    Beta Pooling Process:
    ====================

    Expert Inputs:               Beta Distribution:
    ┌─────────────┐             ┌──────────────────┐
    │ p₁, CI₁, w₁ │             │ α = α₀ + Σ(wᵢαᵢ) │
    │ p₂, CI₂, w₂ │  ────────>  │ β = β₀ + Σ(wᵢβᵢ) │
    │     ...     │             │ αᵢ from CI width │
    │ pₙ, CIₙ, wₙ  │             │ βᵢ from CI width │
    └─────────────┘             └──────────────────┘
          │                              ▼
          │                     ┌───────────────────┐
          ▼                     │ P(irAKI) = α/(α+β)│
    ┌──────────────┐            │ 95% CI from beta  │
    │ Confidence   │            └───────────────────┘
    │ Estimation   │
    └──────────────┘
          │
          ▼
    Between-expert variance ──┐
    Within-expert CI width ───┼──> Harmonic mean ──> Consensus confidence

Key Formulas:
------------
- Expert-specific alpha: αᵢ = pᵢ × strength_i
- Expert-specific beta: βᵢ = (1-pᵢ) × strength_i
- Strength from CI: strength_i = 4/CI_width² (tighter CI = higher strength)
- Posterior alpha: α = α₀ + Σ(wᵢ × αᵢ)
- Posterior beta: β = β₀ + Σ(wᵢ × βᵢ)
- Consensus P(irAKI): α/(α+β)
- 95% Credible Interval: Beta.ppf([0.025, 0.975], α, β)

Clinical Context:
----------------
This consensus mechanism ensures that:
1. High-confidence experts have proportionally more influence
2. Agreement between experts increases overall confidence
3. Narrow confidence intervals indicate higher certainty
4. The final probability integrates all expert knowledge

Nature Medicine Standards:
-------------------------
This implementation follows reproducible methodology per Nature Medicine
requirements with clear mathematical formulations and fail-loud validation.
NO auto-correction of invalid inputs per clinical safety standards.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import beta

logger = logging.getLogger(__name__)


def beta_pool_confidence(
    p_vec: np.ndarray,
    ci_mat: np.ndarray,
    weight_vec: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Aggregate expert probabilities using beta opinion pooling.

    Implements weighted beta pooling with confidence estimation based on
    between-expert agreement and within-expert certainty. This method is
    designed for aggregating subjective probability assessments in medical
    consensus scenarios.

    Args:
        p_vec: Array of expert probabilities for irAKI (N experts).
               Each value must be in [0, 1].
        ci_mat: Array of 95% confidence intervals (N x 2).
                ci_mat[i] = [lower_i, upper_i] for expert i.
                All values must be in [0, 1] with lower <= upper.
        weight_vec: Optional array of expert confidence weights (N,).
                    If None, uniform weights are used.
                    Each weight must be in [0, 1].

    Returns:
        Dictionary containing:
            - pooled_mean: Beta-pooled consensus probability P(irAKI)
            - pooled_ci: 95% credible interval [lower, upper]
            - var_between: Between-expert variance (higher = less agreement)
            - mean_halfwidth: Mean CI half-width (higher = less certainty)
            - consensus_conf: Overall consensus confidence [0, 1]
            - between_score: Agreement score between experts [0, 1]
            - within_score: Confidence score within experts [0, 1]

    Raises:
        ValueError: If inputs have incompatible shapes or invalid values

    Example:
        >>> p_vec = np.array([0.7, 0.8, 0.75])  # 3 experts think likely irAKI
        >>> ci_mat = np.array([[0.6, 0.8], [0.7, 0.9], [0.65, 0.85]])
        >>> weights = np.array([0.9, 0.8, 0.95])  # confidence levels
        >>> result = beta_pool_confidence(p_vec, ci_mat, weights)
        >>> print(f"Consensus P(irAKI): {result['pooled_mean']:.3f}")
    """
    # ========================================================================
    # INPUT VALIDATION - FAIL LOUDLY PER NATURE MEDICINE STANDARDS
    # ========================================================================

    # check for empty input
    if p_vec.size == 0:
        raise ValueError("Cannot compute consensus from empty expert assessments")

    # validate array dimensions
    if p_vec.ndim != 1:
        raise ValueError(f"p_vec must be 1-dimensional, got shape {p_vec.shape}")

    if ci_mat.ndim != 2 or ci_mat.shape[1] != 2:
        raise ValueError(f"ci_mat must be Nx2 array, got shape {ci_mat.shape}")

    if p_vec.shape[0] != ci_mat.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: p_vec has {p_vec.shape[0]} experts, "
            f"ci_mat has {ci_mat.shape[0]} experts"
        )

    # validate probability bounds - FAIL LOUD, NO AUTO-CORRECTION
    if np.any((p_vec < 0) | (p_vec > 1)):
        invalid_idx = np.where((p_vec < 0) | (p_vec > 1))[0]
        invalid_vals = p_vec[invalid_idx]
        raise ValueError(
            f"Invalid probability values at indices {list(invalid_idx)}: {list(invalid_vals)}. "
            f"All probabilities must be in [0,1]."
        )

    # validate CI bounds
    lo = ci_mat[:, 0]
    hi = ci_mat[:, 1]

    if np.any((lo < 0) | (lo > 1) | (hi < 0) | (hi > 1)):
        raise ValueError(
            f"Invalid confidence interval bounds. All CI values must be in [0,1]. "
            f"Got lower bounds: {lo}, upper bounds: {hi}"
        )

    # check CI ordering - FAIL LOUD, NO AUTO-CORRECTION
    if np.any(lo > hi):
        invalid_idx = np.where(lo > hi)[0]
        raise ValueError(
            f"Invalid confidence interval ordering at indices {list(invalid_idx)}. "
            f"Lower bound must be <= upper bound. "
            f"Got intervals: {[list(ci_mat[i]) for i in invalid_idx]}"
        )

    # validate weights if provided - FAIL LOUD
    if weight_vec is not None:
        if weight_vec.shape != p_vec.shape:
            raise ValueError(
                f"Dimension mismatch: weight_vec shape {weight_vec.shape} "
                f"doesn't match p_vec shape {p_vec.shape}"
            )

        if np.any(weight_vec < 0):
            invalid_idx = np.where(weight_vec < 0)[0]
            raise ValueError(
                f"Invalid negative weights at indices {list(invalid_idx)}: "
                f"{list(weight_vec[invalid_idx])}. All weights must be non-negative."
            )

        # check for zero total weight
        if np.sum(weight_vec) == 0:
            raise ValueError("Cannot compute consensus: all expert weights sum to zero")

        w = weight_vec
    else:
        # uniform weights if not provided
        w = np.ones_like(p_vec)
        logger.debug("Using uniform weights for all experts")

    # ========================================================================
    # BETA POOLING COMPUTATION
    # ========================================================================

    # normalize weights to sum to 1 for interpretability
    w_normalized = w / np.sum(w)

    # compute strength parameters from CI widths
    # tighter CI = higher strength/precision
    ci_widths = hi - lo

    # avoid division by zero for point estimates (CI width = 0)
    min_width = 1e-4  # CHANGED: use smaller guard to not inflate narrow CIs
    ci_widths = np.maximum(ci_widths, min_width)

    # strength inversely proportional to CI width
    # use quadratic relationship for more sensitivity
    strengths = 1.0 / (ci_widths**2)

    # CHANGED: clip strengths to reasonable range instead of rescaling
    # this preserves the natural high precision when experts have tight CIs
    strengths = np.clip(strengths, 2.0, 150.0)

    # NEW: detect disagreement patterns for adaptive prior
    var_between = np.var(p_vec)

    # check for bimodal distribution (some experts high, some low)
    high_group = p_vec[p_vec > 0.65]
    low_group = p_vec[p_vec < 0.35]
    is_bimodal = (
        len(high_group) > 0
        and len(low_group) > 0
        and len(high_group) + len(low_group) >= len(p_vec) * 0.6
    )

    # NEW: detect outliers using IQR method
    outlier_mask = np.zeros_like(p_vec, dtype=bool)
    if len(p_vec) > 3:
        q1 = np.percentile(p_vec, 25)
        q3 = np.percentile(p_vec, 75)
        iqr = q3 - q1
        if iqr > 0.05:  # only detect outliers if there's meaningful spread
            outlier_mask = (p_vec < q1 - 1.5 * iqr) | (p_vec > q3 + 1.5 * iqr)
            # reduce non-outlier strengths to preserve outlier influence
            if np.any(outlier_mask):
                strengths[~outlier_mask] *= 0.7

    # compute expert-specific beta parameters
    alpha_experts = p_vec * strengths
    beta_experts = (1.0 - p_vec) * strengths

    # CHANGED: adaptive prior based on disagreement level
    # stronger prior pulls toward 0.5 when experts disagree
    if is_bimodal or var_between > 0.1:
        prior_alpha = 2.0
        prior_beta = 2.0
    else:
        # Jeffrey's prior for agreement cases
        prior_alpha = 0.5
        prior_beta = 0.5

    # posterior parameters
    a_post = prior_alpha + np.sum(w_normalized * alpha_experts)
    b_post = prior_beta + np.sum(w_normalized * beta_experts)

    # compute posterior mean and 95% credible interval
    post_mean = a_post / (a_post + b_post)
    post_ci_95 = beta.ppf([0.025, 0.975], a_post, b_post)

    # ========================================================================
    # CONFIDENCE ESTIMATION
    # ========================================================================

    # mean half-width of expert CIs
    mean_halfwidth = np.mean(ci_widths / 2)

    # CHANGED: more conservative between-expert agreement score
    if var_between < 0.01:  # very high agreement
        between_score = 0.95
    elif var_between < 0.05:  # moderate agreement
        between_score = 0.7 - 3.5 * var_between  # CHANGED: drops faster
    else:  # significant disagreement - use exponential decay
        between_score = max(0.1, np.exp(-5.0 * var_between))

    # CHANGED: adjusted within-expert certainty score scaling
    # use 0.3 instead of 0.5 for tighter scaling
    within_score = 1.0 - min(mean_halfwidth / 0.3, 1.0)

    # overall confidence: harmonic mean of agreement and certainty
    # harmonic mean penalizes if either component is low
    if between_score > 0 and within_score > 0:
        consensus_conf = 2.0 / (1.0 / between_score + 1.0 / within_score)
    else:
        consensus_conf = 0.0

    # CHANGED: adjust confidence for specific scenarios
    if len(p_vec) == 1:
        # single expert: confidence based only on their CI width
        consensus_conf = within_score * 0.5  # reduce confidence for single expert
    elif is_bimodal:
        # bimodal distribution: cap confidence at 0.4
        consensus_conf = min(0.4, consensus_conf)
    elif np.any(outlier_mask) and len(p_vec) > 3:
        # outliers present: apply penalty
        consensus_conf *= 0.8
    elif np.all(p_vec == p_vec[0]) and np.all(ci_widths < 0.1):
        # perfect agreement with narrow CIs: very high confidence
        consensus_conf = 0.95

    logger.info(
        f"Beta pooling complete: P(irAKI)={post_mean:.3f} "
        f"[{post_ci_95[0]:.3f}, {post_ci_95[1]:.3f}], "
        f"α={a_post:.2f}, β={b_post:.2f}, "
        f"confidence={consensus_conf:.3f}"
    )

    return {
        "pooled_mean": float(post_mean),
        "pooled_ci": post_ci_95,
        "var_between": float(var_between),
        "mean_halfwidth": float(mean_halfwidth),
        "consensus_conf": float(consensus_conf),
        "between_score": float(between_score),
        "within_score": float(within_score),
    }


def compute_binary_verdict(
    pooled_mean: float,
    pooled_ci: Tuple[float, float],
    threshold: float = 0.5,
    require_ci_agreement: bool = False,
) -> Tuple[bool, str]:
    """
    Determine binary irAKI verdict from pooled probability.

    Args:
        pooled_mean: Beta-pooled consensus probability
        pooled_ci: 95% credible interval [lower, upper]
        threshold: Decision threshold (default 0.5)
        require_ci_agreement: If True, entire CI must be on one side of threshold

    Returns:
        Tuple of (is_iraki: bool, reasoning: str)
    """
    lower, upper = pooled_ci

    if require_ci_agreement:
        # strict criterion: entire CI must agree
        if upper < threshold:
            return (
                False,
                f"Strong consensus against irAKI (CI [{lower:.3f}, {upper:.3f}] < {threshold})",
            )
        elif lower > threshold:
            return (
                True,
                f"Strong consensus for irAKI (CI [{lower:.3f}, {upper:.3f}] > {threshold})",
            )
        else:
            # CI straddles threshold - use mean but note uncertainty
            is_iraki = pooled_mean >= threshold
            return (
                is_iraki,
                f"Uncertain verdict (CI [{lower:.3f}, {upper:.3f}] straddles {threshold}), using mean {pooled_mean:.3f}",
            )
    else:
        # standard criterion: use pooled mean
        is_iraki = pooled_mean >= threshold
        confidence_desc = (
            "high"
            if (upper - lower) < 0.2
            else "moderate"
            if (upper - lower) < 0.3
            else "low"
        )
        return (
            is_iraki,
            f"{'irAKI' if is_iraki else 'Not irAKI'} with {confidence_desc} confidence (P={pooled_mean:.3f}, CI [{lower:.3f}, {upper:.3f}])",
        )
