# aggregator.py
# consensus aggregation for DelPHEA-irAKI
# yagni: tolerate dicts or pydantic objects, drop explicit invalids,
# confidence-weighted mean, ci widens with disagreement, return models.Consensus

from __future__ import annotations

import math
from typing import Any, Iterable, List, Tuple

# pydantic models used by moderator and report
from models import Consensus  # must exist with fields used below


class WeightedMeanAggregator:
    """confidence-weighted pooling with variance inflation for disagreement.

    pooled mean: m = sum_i w_i * p_i / sum_i w_i
    between-expert variance B = sum_i w_i * (p_i - m)^2 / sum_i w_i
    within-expert variance W from each expert's ci via normal approx:
        sigma_i ≈ (upper_i - lower_i) / (2 * z), z = 1.96
        W = sum_i w_i * sigma_i^2 / sum_i w_i
    total variance V = B + W; 95% ci = m ± z * sqrt(V), clipped to [0, 1]
    """

    def __init__(self, decision_threshold: float = 0.5, z_value: float = 1.96) -> None:
        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be in [0, 1]")
        if z_value <= 0:
            raise ValueError("z_value must be positive")
        self.decision_threshold = decision_threshold
        self.z = z_value

    # public api expected by Moderator
    def aggregate(self, expert_results: Iterable[Any]) -> Consensus:
        valid = [r for r in expert_results if self._is_valid(r)]
        if not valid:
            raise ValueError("no valid expert assessments; aborting aggregation")

        p = [self._get_float(r, "p_iraki") for r in valid]
        conf = [self._get_float(r, "confidence", default=0.0) for r in valid]
        ci = [self._get_ci(r) for r in valid]

        weights = self._normalize_weights(conf)
        m = sum(w * x for w, x in zip(weights, p))

        # between-expert variance (disagreement)
        between = sum(w * (x - m) ** 2 for w, x in zip(weights, p))

        # within-expert variance from experts' cis
        sigmasq = [self._ci_to_sigma_sq(c) for c in ci]
        within = sum(w * s for w, s in zip(weights, sigmasq))

        total_var = max(1e-8, between + within)
        half = self.z * math.sqrt(total_var)

        lower = max(0.0, m - half)
        upper = min(1.0, m + half)

        consensus_conf = sum(conf) / len(conf)
        verdict = bool(m >= self.decision_threshold)

        # return pydantic Consensus so moderator can .model_dump()
        return Consensus(
            iraki_probability=m,
            ci_iraki=[lower, upper],
            verdict=verdict,
            consensus_confidence=consensus_conf,
            expert_count=len(valid),
            p_iraki=m,
        )

    # ------------------------- helpers -------------------------

    def _is_valid(self, r: Any) -> bool:
        # drop only explicit invalids; tolerate missing _status
        if r is None:
            return False
        status = self._get(r, "_status", default=None)
        if status == "invalid_assessment":
            return False
        try:
            p = self._get_float(r, "p_iraki")
            lo, hi = self._get_ci(r)
            c = self._get_float(r, "confidence", default=0.0)
        except Exception:
            return False
        if not (0.0 <= p <= 1.0):
            return False
        if not (0.0 <= lo <= p <= hi <= 1.0):
            return False
        if not (0.0 <= c <= 1.0):
            return False
        return True

    def _get(self, obj: Any, key: str, default: Any = ...):
        # dict or pydantic/object attribute access
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            if default is ...:
                raise KeyError(key)
            return default
        # pydantic model or simple object
        val = getattr(obj, key, ...)
        if val is ...:
            if default is ...:
                raise KeyError(key)
            return default
        return val

    def _get_float(self, obj: Any, key: str, default: float | None = None) -> float:
        v = self._get(obj, key, default if default is not None else ...)
        f = float(v)
        return f

    def _get_ci(self, obj: Any) -> Tuple[float, float]:
        v = self._get(obj, "ci_iraki")
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError("ci_iraki must be [lower, upper]")
        return float(v[0]), float(v[1])

    def _normalize_weights(self, conf: List[float]) -> List[float]:
        s = sum(conf)
        if s <= 1e-9:
            n = len(conf)
            return [1.0 / n] * n
        return [x / s for x in conf]

    def _ci_to_sigma_sq(self, ci: Tuple[float, float]) -> float:
        lo, hi = ci
        width = max(0.0, hi - lo)
        if width <= 0.0:
            return 1e-6
        sigma = width / (2.0 * self.z)
        return max(sigma * sigma, 1e-6)


# legacy-compatible facade so `from aggregator import Aggregator` works
class Aggregator:
    """facade returning pydantic Consensus, matching Moderator expectations."""

    def __init__(self, decision_threshold: float = 0.5) -> None:
        self._impl = WeightedMeanAggregator(decision_threshold=decision_threshold)

    def aggregate(self, expert_results: Iterable[Any]) -> Consensus:
        return self._impl.aggregate(expert_results)

    # optional: expose the underlying impl if needed
    def aggregate_result(self, expert_results: Iterable[Any]) -> Consensus:
        return self._impl.aggregate(expert_results)


if __name__ == "__main__":
    # quick self-check with a disagreement scenario mixing dicts and dummy objects
    class R3Obj:
        def __init__(self, p, ci, conf):
            self.p_iraki = p
            self.ci_iraki = ci
            self.confidence = conf

    demo = [
        {"p_iraki": 0.8, "ci_iraki": [0.7, 0.86], "confidence": 0.8},
        R3Obj(0.5, [0.3, 0.7], 0.5),  # pydantic-like object
        {"p_iraki": 0.1, "ci_iraki": [0.0, 0.2], "confidence": 0.2},
        {"_status": "invalid_assessment"},  # dropped
    ]
    cons = Aggregator().aggregate(demo)
    print(cons.model_dump())
