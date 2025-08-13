# router.py
# debate routing strategies (sparse and full) with pluggable disagreement detectors

from __future__ import annotations

import random
import statistics as stats
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from models import AssessmentR1
from schema import ConsensusRules


@dataclass
class DebatePlan:
    # map qid -> list of expert_ids to solicit a debate turn from (minority set)
    by_qid: Dict[str, List[str]]


DisagreementMethod = Literal["range", "stdev", "mad"]


def _minority_by_range(
    scores: Dict[str, List[Tuple[str, int]]], qid: str, threshold_points: int
) -> List[str]:
    vec = [s for _, s in scores[qid]]
    if not vec or (max(vec) - min(vec) < threshold_points):
        return []
    med = stats.median(vec)
    max_dev = max(abs(s - med) for s in vec)
    return [eid for eid, s in scores[qid] if abs(s - med) == max_dev]


def _minority_by_stdev(
    scores: Dict[str, List[Tuple[str, int]]], qid: str, min_sigma: float
) -> List[str]:
    xs = [s for _, s in scores[qid]]
    if len(xs) < 2:
        return []
    mu = stats.mean(xs)
    try:
        sigma = stats.pstdev(xs) if len(xs) > 1 else 0.0
    except Exception:
        sigma = 0.0
    if sigma < min_sigma:
        return []
    # pick farthest from mean (ties allowed)
    max_dev = max(abs(s - mu) for s in xs)
    return [eid for eid, s in scores[qid] if abs(s - mu) == max_dev]


def _minority_by_mad(
    scores: Dict[str, List[Tuple[str, int]]], qid: str, min_mad: float
) -> List[str]:
    xs = [s for _, s in scores[qid]]
    if not xs:
        return []
    med = stats.median(xs)
    abs_dev = [abs(s - med) for s in xs]
    mad = stats.median(abs_dev)
    if mad < min_mad:
        return []
    max_dev = max(abs_dev)
    return [eid for (eid, s), d in zip(scores[qid], abs_dev) if d == max_dev]


class Router:
    """abstract router strategy."""

    def plan(
        self, r1: Sequence[Tuple[str, AssessmentR1]], rules: ConsensusRules
    ) -> DebatePlan:
        raise NotImplementedError


class SparseRouter(Router):
    """route only qids with disagreement per chosen method."""

    def __init__(
        self,
        *,
        method: DisagreementMethod = "range",
        tie_break: Literal["stable", "random"] = "stable",
        seed: Optional[int] = None,
        range_threshold_points: int | None = None,
        stdev_threshold: float = 1.5,
        mad_threshold: float = 1.5,
    ) -> None:
        self.method = method
        self.tie_break = tie_break
        self.seed = seed
        self.range_threshold_points = (
            range_threshold_points  # fallback to rules if None
        )
        self.stdev_threshold = float(stdev_threshold)
        self.mad_threshold = float(mad_threshold)
        self._rng = random.Random(seed)

    def plan(
        self, r1: Sequence[Tuple[str, AssessmentR1]], rules: ConsensusRules
    ) -> DebatePlan:
        if not r1:
            return DebatePlan(by_qid={})

        # assumed validated upstream: all assessments share qids
        qids = list(next(iter(r1))[1].scores.keys())
        # map qid -> [(expert_id, score), ...]
        scores: Dict[str, List[Tuple[str, int]]] = {qid: [] for qid in qids}
        for eid, a in r1:
            for qid, s in a.scores.items():
                scores[qid].append((eid, s))

        plan: Dict[str, List[str]] = {}

        for qid in qids:
            if self.method == "range":
                thr = (
                    self.range_threshold_points
                    if self.range_threshold_points is not None
                    else getattr(rules, "debate_threshold_points", 3)
                )
                minority = _minority_by_range(scores, qid, int(thr))
            elif self.method == "stdev":
                minority = _minority_by_stdev(scores, qid, self.stdev_threshold)
            else:  # "mad"
                minority = _minority_by_mad(scores, qid, self.mad_threshold)

            if not minority:
                continue

            # deterministic order among ties:
            #   primary: distance from median (descending)
            #   secondary: stable expert_id or seeded random if tie_break="random"
            vec = scores[qid]
            med = stats.median([s for _, s in vec])

            def _dist(eid: str) -> float:
                s = next(s for x, s in vec if x == eid)
                return abs(s - med)

            minority_sorted = sorted(minority, key=lambda e: (-_dist(e), e))
            if self.tie_break == "random":
                # shuffle among equal-distance blocks for variety but reproducibility
                # group by distance
                groups: Dict[float, List[str]] = {}
                for e in minority_sorted:
                    groups.setdefault(_dist(e), []).append(e)
                minority_sorted = []
                for d in sorted(groups.keys(), reverse=True):
                    g = groups[d][:]
                    self._rng.shuffle(g)
                    minority_sorted.extend(g)

            plan[qid] = minority_sorted

        return DebatePlan(by_qid=plan)


class FullRouter(Router):
    """route all qids to all experts (approximation of full pairwise)."""

    def plan(
        self, r1: Sequence[Tuple[str, AssessmentR1]], rules: ConsensusRules
    ) -> DebatePlan:
        qids = list(next(iter(r1))[1].scores.keys()) if r1 else []
        all_experts = [expert_id for expert_id, _ in r1]
        return DebatePlan(by_qid={qid: list(all_experts) for qid in qids})
