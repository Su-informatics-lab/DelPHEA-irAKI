# aggregator.py
# consensus aggregation for DelPHEA-irAKI
# yagnified: validate inputs, drop invalid experts, compute confidence-weighted mean,
# and form a credible interval that widens with disagreement (between-expert variance)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class AggregationResult:
    """lightweight container for consensus outputs.

    attributes:
        iraki_probability: pooled probability of irAKI in [0, 1]
        ci_iraki: 95% interval as (lower, upper), clipped to [0, 1]
        verdict: boolean decision using a threshold on pooled probability
        consensus_confidence: mean of expert confidences for valid experts
        expert_count: number of valid experts used
        p_iraki: alias of iraki_probability for backward compatibility
    """

    iraki_probability: float
    ci_iraki: Tuple[float, float]
    verdict: bool
    consensus_confidence: float
    expert_count: int
    p_iraki: float


class WeightedMeanAggregator:
    """confidence-weighted pooling with variance inflation for disagreement.

    design goals (yagni):
    - drop assessments that are structurally invalid
    - weight each expert by confidence (fallback to equal weights if all zero)
    - produce intervals that widen with between-expert disagreement
      by combining between-expert variance with within-expert uncertainty
    - fail fast when nothing valid is available

    statistical notes:
    - pooled mean: m = sum_i w_i * p_i / sum_i w_i
    - between-expert variance (B): sum_i w_i * (p_i - m)^2 / sum_i w_i
    - within-expert variance (W): from each expert's ci via normal approx:
        sigma_i ≈ (upper_i - lower_i) / (2 * z), z = 1.96
      then W = sum_i w_i * sigma_i^2 / sum_i w_i
    - total variance V = B + W; 95% interval m ± z * sqrt(V), clipped to [0, 1]
    """

    def __init__(self, decision_threshold: float = 0.5, z_value: float = 1.96) -> None:
        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be in [0, 1]")
        if z_value <= 0:
            raise ValueError("z_value must be positive")
        self.decision_threshold = decision_threshold
        self.z = z_value

    def aggregate(self, expert_results: Iterable[Dict[str, Any]]) -> AggregationResult:
        """aggregate expert assessments into a consensus.

        args:
            expert_results: iterable of expert payloads (dicts) with keys:
                - "_status": "ok" when validated upstream
                - "p_iraki": float in [0, 1]
                - "ci_iraki": [lower, upper], both in [0, 1]
                - "confidence": float in [0, 1]

        returns:
            AggregationResult with pooled probability and 95% interval.

        raises:
            ValueError: if no valid assessments are provided.
        """
        valid = [r for r in expert_results if self._is_valid(r)]
        if not valid:
            raise ValueError("no valid expert assessments; aborting aggregation")

        p = [float(r["p_iraki"]) for r in valid]
        conf = [float(r.get("confidence", 0.0)) for r in valid]
        ci = [tuple(r["ci_iraki"]) for r in valid]

        weights = self._normalize_weights(conf)
        m = sum(w * x for w, x in zip(weights, p))

        # between-expert variance (disagreement)
        between = sum(w * (x - m) ** 2 for w, x in zip(weights, p))

        # within-expert variance from each expert's ci; use conservative floor
        sigmasq = [self._ci_to_sigma_sq(c) for c in ci]
        within = sum(w * s for w, s in zip(weights, sigmasq))

        total_var = max(1e-8, between + within)
        half_width = self.z * math.sqrt(total_var)

        lower = max(0.0, m - half_width)
        upper = min(1.0, m + half_width)

        # consensus confidence: simple mean of confidences among valid experts
        consensus_conf = sum(conf) / len(conf) if valid else 0.0

        verdict = bool(m >= self.decision_threshold)

        return AggregationResult(
            iraki_probability=m,
            ci_iraki=(lower, upper),
            verdict=verdict,
            consensus_confidence=consensus_conf,
            expert_count=len(valid),
            p_iraki=m,
        )

    # ------------------------- helpers (yagni, no extra deps) -------------------------

    def _is_valid(self, r: Dict[str, Any]) -> bool:
        """structural validation for an expert record."""
        if r is None or not isinstance(r, dict):
            return False
        if r.get("_status") != "ok":
            return False
        try:
            p = float(r["p_iraki"])
            ci = r["ci_iraki"]
            c = float(r.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError):
            return False
        if not (0.0 <= p <= 1.0):
            return False
        if not (isinstance(ci, (list, tuple)) and len(ci) == 2):
            return False
        try:
            lo, hi = float(ci[0]), float(ci[1])
        except (TypeError, ValueError):
            return False
        if not (0.0 <= lo <= p <= hi <= 1.0):
            return False
        if not (0.0 <= c <= 1.0):
            return False
        return True

    def _normalize_weights(self, conf: List[float]) -> List[float]:
        """turn confidences into normalized weights; fallback to equal if all zero."""
        eps = 1e-9
        s = sum(conf)
        if s <= eps:
            n = len(conf)
            return [1.0 / n] * n
        return [x / s for x in conf]

    def _ci_to_sigma_sq(self, ci: Tuple[float, float]) -> float:
        """convert a 95% ci on probability scale to variance via normal approx.

        uses a conservative floor to avoid zero-width intervals from overconfident models.
        """
        lo, hi = float(ci[0]), float(ci[1])
        width = max(0.0, hi - lo)
        if width <= 0.0:
            return 1e-6
        sigma = width / (2.0 * self.z)
        return max(sigma * sigma, 1e-6)


# legacy-compatible wrapper expected by moderator.py
class Aggregator:
    """legacy facade so existing imports `from aggregator import Aggregator` keep working.

    aggregate() returns a dict shaped like the historical 'consensus' block.
    use aggregate_result() if you prefer the typed AggregationResult.
    """

    def __init__(self, decision_threshold: float = 0.5) -> None:
        self._impl = WeightedMeanAggregator(decision_threshold=decision_threshold)

    def aggregate(self, expert_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        res = self._impl.aggregate(expert_results)
        return _result_as_dict(res)

    # optional helper if callers want the object form
    def aggregate_result(
        self, expert_results: Iterable[Dict[str, Any]]
    ) -> AggregationResult:
        return self._impl.aggregate(expert_results)


# convenience function for legacy callers
def aggregate_consensus(
    expert_results: Iterable[Dict[str, Any]], decision_threshold: float = 0.5
) -> Dict[str, Any]:
    agg = WeightedMeanAggregator(decision_threshold=decision_threshold)
    res = agg.aggregate(expert_results)
    return _result_as_dict(res)


# ------------------------------ utils -----------------------------------------


def _result_as_dict(res: AggregationResult) -> Dict[str, Any]:
    return {
        "iraki_probability": res.iraki_probability,
        "verdict": res.verdict,
        "consensus_confidence": res.consensus_confidence,
        "ci_iraki": [res.ci_iraki[0], res.ci_iraki[1]],
        "expert_count": res.expert_count,
        "p_iraki": res.p_iraki,
    }


if __name__ == "__main__":
    # quick self-check with a disagreement scenario
    demo = [
        {"_status": "ok", "p_iraki": 0.8, "ci_iraki": [0.7, 0.86], "confidence": 0.8},
        {"_status": "ok", "p_iraki": 0.5, "ci_iraki": [0.3, 0.7], "confidence": 0.5},
        {"_status": "ok", "p_iraki": 0.0, "ci_iraki": [0.0, 0.05], "confidence": 0.2},
        {"_status": "invalid_assessment"},  # dropped
    ]
    print(Aggregator().aggregate(demo))
