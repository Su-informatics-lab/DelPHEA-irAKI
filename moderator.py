"""
moderator: orchestrates r1 → debate → r3 and aggregates to consensus.

purpose
-------
single orchestrator for n experts × m questions with pluggable routing (sparse/full)
and pluggable aggregation. validates io contracts, fails fast, and returns a
serializable run report.

ascii: control & data flow (single case)
----------------------------------------
+------------------+         +-----------------+         +-------------------+
|   delphea_cli    |         |    moderator    |         |    expert[i]      |
| (entrypoint)     |         | (this module)   |         |  (llm-backed)     |
+---------+--------+         +---------+-------+         +---------+---------+
          |                            |                           ^
          | load panel & questionnaire |                           |
          |--------------------------->|                           |
          |                            |  R1: assess_all(Q set)    |
          |                            +------------------------->>+  (for all i)
          |                            |   returns AssessmentR1    |
          |                            |<<-------------------------+
          |                            |  detect disagreement      |
          |                            |  (router strategy)        |
          |                            |-- if any --> Debate (R2) -+
          |                            |       minority prompts    |
          |                            |  R3: reassess_all(Q set)  |
          |                            +------------------------->>+
          |                            |  returns AssessmentR3     |
          |                            |<<-------------------------+
          |                            |  aggregate consensus      |
          |<---------------------------+  build report/artifacts
          | write outputs (json, logs) |
          v                            v
        files                     results dict

attention analogy
-----------------
q = questionnaire items; k = experts; v = dialog memory (r1 results + debate turns + r3).
current mode is 'sparse attention' via the router; swap to 'full' without changing call sites.

contracts
---------
- experts must return pydantic models: AssessmentR1/AssessmentR3/DebateTurn (via .model_dump()).
- router.plan consumes r1 payloads and returns a DebatePlan.
- aggregator.aggregate consumes r3 payloads and returns a Consensus object.

"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from aggregator import Aggregator
from models import AssessmentR1, AssessmentR3, Consensus
from router import DebatePlan, Router
from schema import load_consensus_rules, load_qids


@dataclass
class _CaseBuffers:
    r1: List[Tuple[str, AssessmentR1]]
    debate_ctx: Dict[str, Any]
    r3: List[Tuple[str, AssessmentR3]]
    consensus: Consensus | None


class Moderator:
    """single orchestrator for n experts × m questions, with sparse/full routing."""

    def __init__(
        self,
        experts,
        questionnaire_path: str,
        router: Router,
        aggregator: Aggregator,
        logger: logging.Logger | None = None,
    ):
        if not experts:
            raise ValueError("experts cannot be empty")
        self.experts = experts
        self.qpath = questionnaire_path
        self.qids = load_qids(questionnaire_path)
        self.rules = load_consensus_rules(questionnaire_path)
        self.router = router
        self.aggregator = aggregator
        self.logger = logger or logging.getLogger("moderator")

        # basic expert id sanity
        ids = [getattr(e, "expert_id", None) for e in self.experts]
        if any(i is None for i in ids):
            raise ValueError("all experts must expose .expert_id")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate expert_id detected")

    # --------- public api ---------

    def assess_round(
        self,
        round_no: int,
        case: Dict[str, Any],
        debate_ctx: Dict[str, Any] | None = None,
    ) -> List[Tuple[str, AssessmentR1 | AssessmentR3]]:
        """fan-out round requests to all experts; return structured pydantic models."""
        if round_no not in (1, 3):
            raise ValueError(f"unsupported round: {round_no}")
        self.logger.info(f"assessing round {round_no} for {len(self.experts)} experts")

        outputs: List[Tuple[str, AssessmentR1 | AssessmentR3]] = []
        if round_no == 1:
            for e in self.experts:
                a1 = e.assess_round1(case, self.qpath)
                self._validate_qids_exact(a1)
                outputs.append((e.expert_id, a1))
            return outputs

        # round 3
        ctx = debate_ctx or {}
        for e in self.experts:
            a3 = e.assess_round3(case, self.qpath, ctx)
            self._validate_qids_exact(a3)
            outputs.append((e.expert_id, a3))
        return outputs

    def detect_and_run_debates(
        self, r1: Sequence[Tuple[str, AssessmentR1]], case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """compute debate plan via router and collect debate turns."""
        plan: DebatePlan = self.router.plan(r1, self.rules)
        transcripts: Dict[str, List[Dict[str, Any]]] = {}
        self.logger.info(
            f"debate planning complete: {sum(len(v) for v in plan.by_qid.values())} solicitations "
            f"across {len(plan.by_qid)} qids"
        )

        # derive a simple minority_view summary per qid from r1 spread
        for qid, expert_ids in plan.by_qid.items():
            if not expert_ids:
                continue
            # textual minority view = highest-deviation explanation snippets
            minority_text = []
            for eid, a in r1:
                if eid in expert_ids:
                    ev = a.evidence.get(qid, "")
                    minority_text.append(f"{eid}: score={a.scores[qid]} evidence={ev}")
            mv = "\n".join(minority_text) or "minority perspective not available"

            transcripts[qid] = []
            for e in self.experts:
                if e.expert_id in expert_ids:
                    turn = e.debate(
                        qid=qid,
                        round_no=2,
                        clinical_context={"case": case},
                        minority_view=mv,
                    )
                    transcripts[qid].append(turn.model_dump())

        return {"debate_plan": plan.by_qid, "transcripts": transcripts}

    def run_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """run the full r1 → debate → r3 → aggregate pipeline and return a report dict."""
        buffers = _CaseBuffers(r1=[], debate_ctx={}, r3=[], consensus=None)

        # round 1
        buffers.r1 = self.assess_round(1, case)  # [(eid, AssessmentR1)]
        self._validate_r1_coherence(buffers.r1)

        # debate
        buffers.debate_ctx = self.detect_and_run_debates(buffers.r1, case)

        # round 3
        buffers.r3 = self.assess_round(
            3, case, buffers.debate_ctx
        )  # [(eid, AssessmentR3)]
        self._validate_r3_coherence(buffers.r1, buffers.r3)

        # aggregate
        buffers.consensus = self.aggregator.aggregate([a for _, a in buffers.r3])

        report = {
            "round1": [(eid, a.model_dump()) for eid, a in buffers.r1],
            "debate": buffers.debate_ctx,
            "round3": [(eid, a.model_dump()) for eid, a in buffers.r3],
            "consensus": buffers.consensus.model_dump(),
        }
        # optional pretty logging
        self.logger.debug(json.dumps(report["consensus"], indent=2))
        return report

    # --------- validators (fail fast) ---------

    def _validate_qids_exact(self, assessed: AssessmentR1 | AssessmentR3) -> None:
        """ensure strict schema echo: qids in scores/evidence match questionnaire ids."""
        s_keys = set(assessed.scores.keys())
        e_keys = set(assessed.evidence.keys())
        expected = set(self.qids)
        if s_keys != expected:
            missing = sorted(expected - s_keys)
            extra = sorted(s_keys - expected)
            raise ValueError(
                f"scores qids must match questionnaire. missing={missing} extra={extra}"
            )
        if e_keys != expected:
            missing = sorted(expected - e_keys)
            extra = sorted(e_keys - expected)
            raise ValueError(
                f"evidence qids must match questionnaire. missing={missing} extra={extra}"
            )

    def _validate_r1_coherence(self, r1: Sequence[Tuple[str, AssessmentR1]]) -> None:
        """basic coherence checks across experts for r1 payloads."""
        if not r1:
            raise ValueError("r1 assessments cannot be empty")
        # ensure all experts covered the same qids
        for eid, a in r1:
            self._validate_qids_exact(a)

    def _validate_r3_coherence(
        self,
        r1: Sequence[Tuple[str, AssessmentR1]],
        r3: Sequence[Tuple[str, AssessmentR3]],
    ) -> None:
        """ensure r3 payloads exist for each r1 expert and qids match."""
        if len(r3) != len(r1):
            raise ValueError(
                f"r3 count ({len(r3)}) must equal r1 count ({len(r1)}) for the same expert set"
            )
        r1_ids = [eid for eid, _ in r1]
        r3_ids = [eid for eid, _ in r3]
        if set(r1_ids) != set(r3_ids):
            raise ValueError("r3 expert set must match r1 expert set")
        for eid, a3 in r3:
            self._validate_qids_exact(a3)
