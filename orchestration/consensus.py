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
    │ p₁, CI₁, w₁ │             │ α = 1 + Σ(wᵢpᵢ)  │
    │ p₂, CI₂, w₂ │  ────────>  │ β = 1 + Σ(wᵢqᵢ)  │
    │     ...     │             │   where qᵢ=1-pᵢ  │
    │ pₙ, CIₙ, wₙ  │              └──────────────────┘
    └─────────────┘                      │
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
- Normalized weights: w'ᵢ = wᵢ × N / Σwᵢ
- Posterior alpha: α = 1 + Σ(w'ᵢ × pᵢ)
- Posterior beta: β = 1 + Σ(w'ᵢ × (1-pᵢ))
- Consensus P(irAKI): α/(α+β)
- 95% Credible Interval: Beta.ppf([0.025, 0.975], α, β)
- Between-expert agreement: 1 - Var(p)/0.25
- Within-expert certainty: 1 - mean(CI_width)/0.5
- Overall confidence: 2/(1/between + 1/within) [harmonic mean]

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
                    Each weight should be in [0, 1].

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
        ValueError: If inputs have incompatible shapes
        ValueError: If all expert weights sum to zero
        ValueError: If any probability or CI bound is outside [0, 1]

    Example:
        >>> p_vec = np.array([0.7, 0.8, 0.75])  # 3 experts think likely irAKI
        >>> ci_mat = np.array([[0.6, 0.8], [0.7, 0.9], [0.65, 0.85]])
        >>> weights = np.array([0.9, 0.8, 0.95])  # confidence levels
        >>> result = beta_pool_confidence(p_vec, ci_mat, weights)
        >>> print(f"Consensus P(irAKI): {result['pooled_mean']:.3f}")
        >>> print(f"95% CI: [{result['pooled_ci'][0]:.3f}, {result['pooled_ci'][1]:.3f}]")

    References:
        - Genest & Zidek (1986). Combining Probability Distributions.
        - Cooke (1991). Experts in Uncertainty: Opinion and Subjective Probability.
    """
    # input validation - fail loudly per Nature Medicine standards
    if p_vec.ndim != 1:
        raise ValueError(f"p_vec must be 1-dimensional, got shape {p_vec.shape}")

    if ci_mat.ndim != 2 or ci_mat.shape[1] != 2:
        raise ValueError(f"ci_mat must be Nx2 array, got shape {ci_mat.shape}")

    if p_vec.shape[0] != ci_mat.shape[0]:
        raise ValueError(
            f"Incompatible shapes: p_vec has {p_vec.shape[0]} experts, "
            f"ci_mat has {ci_mat.shape[0]} experts"
        )

    # validate probability bounds
    if np.any((p_vec < 0) | (p_vec > 1)):
        invalid_idx = np.where((p_vec < 0) | (p_vec > 1))[0]
        raise ValueError(
            f"Probabilities must be in [0,1]. Invalid values at indices {invalid_idx}: "
            f"{p_vec[invalid_idx]}"
        )

    # validate CI bounds
    lo = ci_mat[:, 0]
    hi = ci_mat[:, 1]

    if np.any((lo < 0) | (lo > 1) | (hi < 0) | (hi > 1)):
        raise ValueError(f"CI bounds must be in [0,1]. Got lower={lo}, upper={hi}")

    if np.any(lo > hi):
        invalid_idx = np.where(lo > hi)[0]
        logger.warning(f"CI bounds inverted at indices {invalid_idx}. Auto-correcting.")
        # auto-correct inverted bounds
        ci_mat = ci_mat.copy()
        ci_mat[invalid_idx] = ci_mat[invalid_idx, ::-1]
        lo = ci_mat[:, 0]
        hi = ci_mat[:, 1]

    # compute between-expert variance (measure of disagreement)
    # normalized by maximum possible variance (0.25 when p=0.5)
    var_between = np.var(p_vec, ddof=1) if p_vec.size > 1 else 0.0
    between_score = 1.0 - (var_between / 0.25)
    between_score = np.clip(between_score, 0.0, 1.0)

    # compute within-expert uncertainty (CI width)
    # normalized by maximum reasonable width (0.5)
    half_widths = (hi - lo) / 2.0
    mean_half = np.mean(half_widths)
    within_score = 1.0 - (mean_half / 0.5)
    within_score = np.clip(within_score, 0.0, 1.0)

    # harmonic mean for overall consensus confidence
    # uses harmonic mean to penalize if either component is low
    if between_score == 0 or within_score == 0:
        consensus_conf = 0.0
    else:
        consensus_conf = 2.0 / (1.0 / between_score + 1.0 / within_score)

    # prepare weights (default to uniform if not provided)
    if weight_vec is None:
        w = np.ones_like(p_vec)
        logger.debug("Using uniform weights for all experts")
    else:
        if weight_vec.shape != p_vec.shape:
            raise ValueError(
                f"weight_vec shape {weight_vec.shape} doesn't match "
                f"p_vec shape {p_vec.shape}"
            )
        w = weight_vec.copy()

        # validate weight bounds
        if np.any((w < 0) | (w > 1)):
            logger.warning(
                f"Weights should be in [0,1]. Got min={w.min()}, max={w.max()}. "
                f"Clipping to valid range."
            )
            w = np.clip(w, 0.0, 1.0)

    # check for zero total weight - critical failure
    total_weight = np.sum(w)
    if total_weight == 0:
        raise ValueError(
            "All expert confidences sum to zero - cannot compute consensus. "
            "At least one expert must have non-zero confidence."
        )

    # normalize weights to sum to N for interpretable scaling
    # this ensures the posterior is comparable regardless of expert count
    w_normalized = w * len(p_vec) / total_weight

    logger.debug(
        f"Weight normalization: original sum={total_weight:.3f}, "
        f"normalized sum={np.sum(w_normalized):.3f}"
    )

    # beta opinion pooling
    # posterior parameters with Jeffrey's prior (α=β=1)
    a_post = 1.0 + np.sum(w_normalized * p_vec)
    b_post = 1.0 + np.sum(w_normalized * (1.0 - p_vec))

    # posterior mean and credible interval
    post_mean = a_post / (a_post + b_post)
    post_ci_95 = beta.ppf([0.025, 0.975], a_post, b_post).tolist()

    logger.info(
        f"Beta pooling complete: P(irAKI)={post_mean:.3f} "
        f"[{post_ci_95[0]:.3f}, {post_ci_95[1]:.3f}], "
        f"confidence={consensus_conf:.3f}"
    )

    return {
        "pooled_mean": float(post_mean),
        "pooled_ci": post_ci_95,
        "var_between": float(var_between),
        "mean_halfwidth": float(mean_half),
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
        require_ci_agreement: If True, entire CI must be above/below threshold
                              for high confidence verdict

    Returns:
        Tuple of (verdict, confidence_level) where:
            - verdict: True if irAKI likely, False otherwise
            - confidence_level: "high", "moderate", or "low"

    Example:
        >>> verdict, conf = compute_binary_verdict(0.75, [0.65, 0.85])
        >>> print(f"irAKI: {verdict}, Confidence: {conf}")
    """
    lower, upper = pooled_ci

    # determine verdict
    verdict = pooled_mean > threshold

    # assess confidence level
    if require_ci_agreement:
        if lower > threshold:
            # entire CI above threshold - high confidence irAKI
            confidence_level = "high"
            verdict = True
        elif upper < threshold:
            # entire CI below threshold - high confidence not irAKI
            confidence_level = "high"
            verdict = False
        else:
            # CI crosses threshold - lower confidence
            confidence_level = (
                "moderate" if abs(pooled_mean - threshold) > 0.2 else "low"
            )
    else:
        # simpler confidence based on distance from threshold
        distance = abs(pooled_mean - threshold)
        if distance > 0.3:
            confidence_level = "high"
        elif distance > 0.15:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

    logger.debug(
        f"Binary verdict: {'irAKI' if verdict else 'not irAKI'} "
        f"({confidence_level} confidence)"
    )

    return verdict, confidence_level
