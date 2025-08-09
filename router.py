# router.py
# debate routing strategies (sparse and full)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from models import AssessmentR1
from schema import ConsensusRules


@dataclass
class DebatePlan:
    # map qid -> list of expert_ids to solicit a debate turn from
    by_qid: Dict[str, List[str]]


class Router:
    """abstract router strategy."""

    def plan(
        self, r1: Sequence[Tuple[str, AssessmentR1]], rules: ConsensusRules
    ) -> DebatePlan:
        """create a debate plan from round-1 payloads.

        Args:
            r1: sequence of (expert_id, AssessmentR1)
            rules: parsed consensus rules (minimum_agreement, debate_threshold_points)

        Returns:
            DebatePlan mapping qid->experts_to_message
        """
        raise NotImplementedError


class SparseRouter(Router):
    """route only qids with disagreement ≥ debate_threshold_points."""

    def plan(
        self, r1: Sequence[Tuple[str, AssessmentR1]], rules: ConsensusRules
    ) -> DebatePlan:
        # compute per-qid score vectors across experts
        if not r1:
            return DebatePlan(by_qid={})

        # assume all assessments share the same qid set (validated upstream)
        qids = list(next(iter(r1))[1].scores.keys())
        # disagreement by qid = max - min
        by_qid: Dict[str, List[int]] = {qid: [] for qid in qids}
        for _, a in r1:
            for qid, s in a.scores.items():
                by_qid[qid].append(s)

        plan: Dict[str, List[str]] = {}
        for qid, vec in by_qid.items():
            if not vec:
                continue
            if max(vec) - min(vec) >= rules.debate_threshold_points:
                # minority = experts with scores at or near the extreme away from the median
                import statistics as stats

                med = stats.median(vec)
                # pick experts where |score - median| is maximal
                max_dev = max(abs(s - med) for s in vec)
                expert_ids = [
                    expert_id
                    for expert_id, a in r1
                    if abs(a.scores[qid] - med) == max_dev
                ]
                plan[qid] = expert_ids
        return DebatePlan(by_qid=plan)


class FullRouter(Router):
    """route all qids to all experts (approximation of full pairwise)."""

    def plan(
        self, r1: Sequence[Tuple[str, AssessmentR1]], rules: ConsensusRules
    ) -> DebatePlan:
        qids = list(next(iter(r1))[1].scores.keys()) if r1 else []
        all_experts = [expert_id for expert_id, _ in r1]
        return DebatePlan(by_qid={qid: list(all_experts) for qid in qids})
