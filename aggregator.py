# aggregator.py
# consensus aggregation strategies

from __future__ import annotations

from typing import List

from models import AssessmentR3, Consensus


class Aggregator:
    """abstract aggregator strategy."""

    def aggregate(self, r3: List[AssessmentR3]) -> Consensus:
        raise NotImplementedError


class WeightedMeanAggregator(Aggregator):
    """confidence-weighted mean of p_iraki with simple ci pooling.

    math:
        let w_i = max(confidence_i, eps)
        p_hat = sum_i w_i * p_i / sum_i w_i
        ci_lo = sum_i w_i * lo_i / sum_i w_i
        ci_hi = sum_i w_i * hi_i / sum_i w_i
        verdict = p_hat >= 0.5 (yagni default; replace via config if needed)
        consensus_confidence = mean(confidence_i)
    """

    def __init__(self, verdict_threshold: float = 0.5):
        self.verdict_threshold = verdict_threshold

    def aggregate(self, r3: List[AssessmentR3]) -> Consensus:
        if not r3:
            raise ValueError("cannot aggregate empty r3 assessments")
        eps = 1e-6
        ws = [max(a.confidence, eps) for a in r3]
        denom = sum(ws)
        p_hat = sum(w * a.p_iraki for w, a in zip(ws, r3)) / denom
        lo = sum(w * a.ci_iraki[0] for w, a in zip(ws, r3)) / denom
        hi = sum(w * a.ci_iraki[1] for w, a in zip(ws, r3)) / denom
        conf = sum(a.confidence for a in r3) / len(r3)
        verdict = p_hat >= self.verdict_threshold
        return Consensus(
            iraki_probability=p_hat,
            verdict=verdict,
            consensus_confidence=conf,
            ci_iraki=(min(lo, p_hat), max(hi, p_hat)),
            expert_count=len(r3),
        )


class BetaPoolAggregator(Aggregator):
    """placeholder for beta-pooling; not implemented to keep yagni.

    raise to make missing implementation explicit.
    """

    def aggregate(self, r3: List[AssessmentR3]) -> Consensus:
        raise NotImplementedError("beta pooling not yet implemented")
