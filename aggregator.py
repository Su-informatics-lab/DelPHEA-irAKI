# aggregator.py
# consensus aggregation for DelPHEA-irAKI
# yagnified: tolerate missing _status, drop only explicit invalids, compute
# confidence-weighted mean, and widen ci with disagreement

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class AggregationResult:
    iraki_probability: float
    ci_iraki: Tuple[float, float]
    verdict: bool
    consensus_confidence: float
    expert_count: int
    p_iraki: float


class WeightedMeanAggregator:
    def __init__(self, decision_threshold: float = 0.5, z_value: float = 1.96) -> None:
        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be in [0, 1]")
        if z_value <= 0:
            raise ValueError("z_value must be positive")
        self.decision_threshold = decision_threshold
        self.z = z_value

    def aggregate(self, expert_results: Iterable[Dict[str, Any]]) -> AggregationResult:
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

        # within-expert variance from expert cis
        sigmasq = [self._ci_to_sigma_sq(c) for c in ci]
        within = sum(w * s for w, s in zip(weights, sigmasq))

        total_var = max(1e-8, between + within)
        half_width = self.z * math.sqrt(total_var)

        lower = max(0.0, m - half_width)
        upper = min(1.0, m + half_width)

        consensus_conf = sum(conf) / len(conf)
        verdict = bool(m >= self.decision_threshold)

        return AggregationResult(
            iraki_probability=m,
            ci_iraki=(lower, upper),
            verdict=verdict,
            consensus_confidence=consensus_conf,
            expert_count=len(valid),
            p_iraki=m,
        )

    # ------------------------- helpers -------------------------

    def _is_valid(self, r: Dict[str, Any]) -> bool:
        # drop only explicit invalids; tolerate missing _status
        if r is None or not isinstance(r, dict):
            return False
        if r.get("_status") == "invalid_assessment":
            return False
        try:
            p = float(r["p_iraki"])
            ci = r["ci_iraki"]
            c = float(r.get("confidence", 0.0))  # default if missing
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
        eps = 1e-9
        s = sum(conf)
        if s <= eps:
            n = len(conf)
            return [1.0 / n] * n
        return [x / s for x in conf]

    def _ci_to_sigma_sq(self, ci: Tuple[float, float]) -> float:
        lo, hi = float(ci[0]), float(ci[1])
        width = max(0.0, hi - lo)
        if width <= 0.0:
            return 1e-6
        sigma = width / (2.0 * self.z)
        return max(sigma * sigma, 1e-6)


# legacy-compatible wrapper expected by moderator.py
class Aggregator:
    def __init__(self, decision_threshold: float = 0.5) -> None:
        self._impl = WeightedMeanAggregator(decision_threshold=decision_threshold)

    def aggregate(self, expert_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        res = self._impl.aggregate(expert_results)
        return _result_as_dict(res)

    def aggregate_result(
        self, expert_results: Iterable[Dict[str, Any]]
    ) -> AggregationResult:
        return self._impl.aggregate(expert_results)


def aggregate_consensus(
    expert_results: Iterable[Dict[str, Any]], decision_threshold: float = 0.5
) -> Dict[str, Any]:
    agg = WeightedMeanAggregator(decision_threshold=decision_threshold)
    res = agg.aggregate(expert_results)
    return _result_as_dict(res)


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
    demo = [
        {"p_iraki": 0.8, "ci_iraki": [0.7, 0.86], "confidence": 0.8},  # no _status → ok
        {"p_iraki": 0.5, "ci_iraki": [0.3, 0.7], "confidence": 0.5},
        {"p_iraki": 0.5, "ci_iraki": [0.0, 1.0], "confidence": 0.5},
        {"_status": "invalid_assessment"},  # dropped
    ]
    print(Aggregator().aggregate(demo))
