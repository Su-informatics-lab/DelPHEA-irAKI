"""
Consensus calculation for DelPHEA-irAKI.

Beta opinion pooling and confidence estimation for expert consensus.
"""

import numpy as np
from scipy.stats import beta


def beta_pool_confidence(
    p_vec: np.ndarray, ci_mat: np.ndarray, weight_vec: np.ndarray = None
) -> dict:
    """Beta opinion pooling with confidence estimation.

    Aggregates expert probabilities using beta pooling and computes
    consensus confidence metrics.

    Args:
        p_vec: Array of expert probabilities for irAKI
        ci_mat: Array of confidence intervals (N x 2)
        weight_vec: Optional array of expert confidence weights

    Returns:
        Dict containing:
            - pooled_mean: Beta-pooled consensus probability
            - pooled_ci: 95% credible interval [lower, upper]
            - var_between: Between-expert variance
            - mean_halfwidth: Mean CI half-width
            - consensus_conf: Overall consensus confidence
            - between_score: Agreement score between experts
            - within_score: Confidence score within experts

    Raises:
        ValueError: If all expert confidences are zero
    """
    lo = ci_mat[:, 0]
    hi = ci_mat[:, 1]

    # between-expert variance
    var_between = np.var(p_vec, ddof=1) if p_vec.size > 1 else 0.0
    between_score = 1.0 - (var_between / 0.25)
    between_score = np.clip(between_score, 0.0, 1.0)

    # within-expert CI width
    half_widths = (hi - lo) / 2.0
    mean_half = half_widths.mean()
    within_score = 1.0 - (mean_half / 0.5)
    within_score = np.clip(within_score, 0.0, 1.0)

    # harmonic-mean panel confidence
    if between_score == 0 or within_score == 0:
        consensus_conf = 0.0
    else:
        consensus_conf = 2.0 / (1.0 / between_score + 1.0 / within_score)

    # normalize weights to sum to N for predictable scaling
    w = weight_vec if weight_vec is not None else np.ones_like(p_vec)

    # check for zero weights to avoid division by zero
    total_weight = w.sum()
    if total_weight == 0:
        raise ValueError("All expert confidences are zero – cannot pool.")

    w_normalized = w * len(p_vec) / total_weight

    # beta opinion pool
    a_post = 1.0 + np.sum(w_normalized * p_vec)
    b_post = 1.0 + np.sum(w_normalized * (1.0 - p_vec))

    post_mean = a_post / (a_post + b_post)
    post_ci_95 = beta.ppf([0.025, 0.975], a_post, b_post).tolist()

    return {
        "pooled_mean": post_mean,
        "pooled_ci": post_ci_95,
        "var_between": var_between,
        "mean_halfwidth": mean_half,
        "consensus_conf": consensus_conf,
        "between_score": between_score,
        "within_score": within_score,
    }
