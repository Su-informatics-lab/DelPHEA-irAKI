"""
Moderator Agent for DelPHEA-irAKI Orchestration
================================================
Master agent coordinating the Delphi consensus process across all rounds.
Manages expert interactions, conflict resolution, and consensus computation.

Architecture:
------------
    Moderator (this module)
         │
    ┌────┼────┐
    ▼    ▼    ▼
  Round1 Round2 Round3
    │     │     │
    │  Debates  │
    │     │     │
    └─────┴─────┘
         │
    Beta Pooling
    Consensus

Delphi Process Flow:
-------------------
1. Round 1: Independent assessments from all experts
2. Round 2: Debate on conflicting questions (if threshold exceeded)
3. Round 3: Final assessments incorporating debate insights
4. Consensus: Beta pooling to compute P(irAKI) with confidence

Clinical Context:
----------------
The moderator ensures comprehensive evaluation by orchestrating
diverse expert opinions, identifying key conflicts, and driving
evidence-based consensus for irAKI classification.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Set

import numpy as np
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler, rpc

from config.core import DelphiConfig
from config.loader import ConfigurationLoader
from dataloader import DataLoader  # FIX: Import DataLoader from dataloader.py
from orchestration.consensus import beta_pool_confidence
from orchestration.messages import (
    AckMsg,
    DebatePrompt,
    ExpertRound1Reply,
    ExpertRound3Reply,
    HumanReviewExport,
    QuestionnaireMsg,
    StartCase,
    TerminateDebate,
)


# helper local to _run_round1 and _run_round3
def _build_documents(pd: Dict[str, Any]) -> Dict[str, List[str]]:
    # Try common keys; fall back to empty lists. Convert scalars to single-item lists.
    def as_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return [str(t) for t in x]
        return [str(x)]

    return {
        "notes": as_list(pd.get("notes") or pd.get("clinical_notes")),
        "labs_text": as_list(pd.get("labs_text") or pd.get("lab_values_text")),
        "imaging_text": as_list(pd.get("imaging_text") or pd.get("imaging_reports")),
        "meds_text": as_list(pd.get("meds_text") or pd.get("medication_history")),
        # add other buckets as needed with zero cost:
        # "path_text": as_list(pd.get("path_text")),
    }


class irAKIModeratorAgent(RoutedAgent):
    """Master agent coordinating irAKI Delphi consensus process.

    Manages three rounds of expert assessment, debate facilitation,
    and final consensus computation using beta pooling.
    """

    def __init__(
        self,
        case_id: str,
        config_loader: ConfigurationLoader,
        data_loader: DataLoader,  # FIX: Changed from DataLoaderWrapper to DataLoader
        delphi_config: DelphiConfig,
    ) -> None:
        """Initialize moderator for case coordination.

        Args:
            case_id: Case identifier
            config_loader: Configuration loader instance
            data_loader: Data loader instance (changed from DataLoaderWrapper)
            delphi_config: Delphi methodology configuration
        """
        super().__init__(f"irAKI Moderator for case {case_id}")
        self._case_id = case_id
        self._config_loader = config_loader
        self._data_loader = data_loader
        self._delphi_config = delphi_config

        # use expert_count from config to select subset
        all_expert_ids = [
            e["id"] for e in self._config_loader.expert_panel["expert_panel"]["experts"]
        ]
        self._expert_ids = all_expert_ids[: delphi_config.expert_count]

        # round tracking - store actual replies
        self._round1_replies: List[ExpertRound1Reply] = []
        self._round3_replies: List[ExpertRound3Reply] = []

        # debate tracking
        self._debate_summaries: Dict[str, str] = {}  # q_id -> summary

        # synchronization primitives
        self._pending_round1: Set[str] = set()
        self._pending_round3: Set[str] = set()
        self._round1_done = asyncio.Event()
        self._round3_done = asyncio.Event()

        self.logger = logging.getLogger(f"moderator.{case_id}")
        self.logger.info(
            f"Initialized moderator for case {case_id} with {len(self._expert_ids)} experts: "
            f"{self._expert_ids}"
        )

    @message_handler
    async def handle_start_case(self, message: StartCase, ctx: MessageContext) -> None:
        self.logger.info(f"=== STARTING DELPHI PROCESS FOR CASE {message.case_id} ===")
        patient_data = self._data_loader.load_patient_case(message.case_id)
        self._patient_data = patient_data
        await self._run_round1()

    async def _run_round1(self) -> None:
        self.logger.info("=== ROUND 1: Independent Expert Assessments ===")
        questions = self._config_loader.get_questions()
        self._pending_round1 = set(self._expert_ids)
        self._round1_done.clear()
        self._round1_replies.clear()
        questionnaire = QuestionnaireMsg(
            case_id=self._case_id,
            round_phase="round1",
            patient_info=self._patient_data.get("patient_info", {}),
            documents=_build_documents(self._patient_data),
            questions=questions,
        )
        for ex_id in self._expert_ids:
            target = AgentId(type=f"expert_{ex_id}", key=self._case_id)
            await self.send_message(questionnaire, target)
        await self._wait_for_round_completion("round1")

    @rpc
    async def record_round1(
        self, message: ExpertRound1Reply, ctx: MessageContext
    ) -> AckMsg:
        """Collect Round 1 expert replies via RPC.

        Args:
            message: Expert's Round 1 reply
            ctx: Message context

        Returns:
            AckMsg: Acknowledgment
        """
        # validate expert
        if message.expert_id not in self._expert_ids:
            self.logger.warning(
                f"Received reply from unknown expert: {message.expert_id}"
            )
            return AckMsg(ok=False, message=f"Unknown expert ID: {message.expert_id}")

        # store reply
        self._round1_replies.append(message)
        self._pending_round1.discard(message.expert_id)
        self.logger.info(
            "✓ Round 1 reply from %s; pending: %d -> %s",
            message.expert_id,
            len(self._pending_round1),
            sorted(self._pending_round1),
        )
        self.logger.info(
            "✓ Round 1 reply from %s; pending: %d -> %s",
            message.expert_id,
            len(self._pending_round1),
            sorted(self._pending_round1),
        )

        # flip event when done
        if not self._pending_round1:
            self._round1_done.set()

        return AckMsg(ok=True, message="Round 1 reply recorded")

    async def _wait_for_round_completion(self, round_phase: str) -> None:
        event = self._round1_done if round_phase == "round1" else self._round3_done
        timeout = getattr(self._delphi_config, f"{round_phase}_timeout", 300)
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            self.logger.info(f"All {round_phase} replies received successfully")
        except asyncio.TimeoutError:
            pending = (
                self._pending_round1
                if round_phase == "round1"
                else self._pending_round3
            )
            self.logger.warning(
                f"Timeout in {round_phase} after {timeout}s. Missing experts: {pending}"
            )
        if round_phase == "round1":
            await self._run_round2()
        else:
            await self._compute_final_consensus()

    async def _run_round2(self) -> None:
        self.logger.info("=== ROUND 2: Conflict Resolution via Debate ===")
        conflicts = self._identify_conflicts()
        if not conflicts:
            self.logger.info("No significant conflicts identified, skipping debate")
            await self._run_round3(ctx)
            return
        self.logger.info(f"Identified {len(conflicts)} questions with conflicts")
        for q_id, conflict_info in conflicts.items():
            await self._run_single_debate(q_id, conflict_info)
        await self._run_round3()

    async def _run_single_debate(
        self, ctx: MessageContext, q_id: str, conflict_info: Dict
    ) -> None:
        self.logger.info(f"Starting debate for question {q_id}")
        debate_prompt = DebatePrompt(
            case_id=self._case_id,
            q_id=q_id,
            question=conflict_info["question"],
            score_distribution=conflict_info["score_distribution"],
            score_range=conflict_info["score_range"],
            conflict_severity=conflict_info["conflict_severity"],
            conflicting_evidence=conflict_info["conflicting_evidence"],
            clinical_importance=conflict_info["clinical_importance"],
        )
        participants = (
            list(conflict_info["score_distribution"].keys()) or self._expert_ids
        )
        for ex_id in participants:
            target = AgentId(type=f"expert_{ex_id}", key=self._case_id)
            await self.send_message(debate_prompt, target)
        await asyncio.sleep(min(30, self._delphi_config.debate_timeout))
        terminate_msg = TerminateDebate(
            case_id=self._case_id, q_id=q_id, reason="timeout"
        )
        for ex_id in participants:
            target = AgentId(type=f"expert_{ex_id}", key=self._case_id)
            await self.send_message(terminate_msg, target)

    def _identify_conflicts(self) -> Dict[str, Dict]:
        """Identify questions with significant disagreement using category method.

        Returns:
            Dict mapping question IDs to conflict information
        """
        conflicts = {}
        questions = self._config_loader.get_questions()

        for question in questions:
            q_id = question["id"]

            # collect scores for this question
            scores = []
            score_by_expert = {}
            for reply in self._round1_replies:
                if q_id in reply.scores:
                    score = reply.scores[q_id]
                    scores.append(score)
                    score_by_expert[reply.expert_id] = score

            if len(scores) < 2:
                continue

            # use category-based conflict detection (more scientific for 9-point Likert)
            # LOW: 1-3 (unlikely irAKI), NEUTRAL: 4-6 (uncertain), HIGH: 7-9 (likely irAKI)
            low_scores = [s for s in scores if s <= 3]
            neutral_scores = [s for s in scores if 4 <= s <= 6]
            high_scores = [s for s in scores if s >= 7]

            # trigger debate if experts are in opposite camps (low vs high)
            has_conflict = False
            conflict_severity = "moderate"

            if low_scores and high_scores:
                # clear disagreement: some think unlikely, others think likely
                has_conflict = True
                if len(low_scores) >= 2 and len(high_scores) >= 2:
                    # multiple experts in each camp = severe conflict
                    conflict_severity = "severe"
            elif (
                len(
                    set(
                        [s <= 3 for s in scores]
                        + [4 <= s <= 6 for s in scores]
                        + [s >= 7 for s in scores]
                    )
                )
                >= 3
            ):
                # experts spread across all three categories
                has_conflict = True
                conflict_severity = "moderate"

            # alternative: use standard deviation method if configured
            if hasattr(self._delphi_config, "conflict_method"):
                if self._delphi_config.conflict_method == "std":
                    std_dev = np.std(scores)
                    has_conflict = (
                        std_dev > 2.0
                    )  # >2 SD indicates significant disagreement
                    conflict_severity = "severe" if std_dev > 2.5 else "moderate"
                elif self._delphi_config.conflict_method == "range":
                    # fallback to simple range method
                    score_range = max(scores) - min(scores)
                    has_conflict = score_range >= self._delphi_config.conflict_threshold
                    conflict_severity = "severe" if score_range >= 5 else "moderate"

            if has_conflict:
                # collect evidence from conflicting experts
                conflicting_evidence = {}
                for reply in self._round1_replies:
                    if q_id in reply.evidence:
                        conflicting_evidence[reply.expert_id] = reply.evidence[q_id]

                conflicts[q_id] = {
                    "question": question,
                    "score_distribution": score_by_expert,
                    "score_range": f"{min(scores)}-{max(scores)}",
                    "conflict_severity": conflict_severity,
                    "conflicting_evidence": conflicting_evidence,
                    "clinical_importance": question.get("clinical_context", {}).get(
                        "importance",
                        "Assessment of this factor is important for irAKI diagnosis",
                    ),
                    "category_distribution": {
                        "low": len(low_scores),
                        "neutral": len(neutral_scores),
                        "high": len(high_scores),
                    },
                }

        return conflicts

    async def _run_round3(self) -> None:
        self.logger.info("=== ROUND 3: Final Consensus Assessments ===")
        self._pending_round3 = set(self._expert_ids)
        self._round3_done.clear()
        self._round3_replies.clear()
        debate_summary = (
            "\n".join(f"- {s}" for s in self._debate_summaries.values())
            or "No debates were conducted."
        )
        questions = self._config_loader.get_questions()
        questionnaire = QuestionnaireMsg(
            case_id=self._case_id,
            round_phase="round3",
            patient_info=self._patient_data.get("patient_info", {}),
            documents=_build_documents(self._patient_data),
            questions=questions,
            debate_summary=debate_summary,
        )
        for ex_id in self._expert_ids:
            target = AgentId(type=f"expert_{ex_id}", key=self._case_id)
            await self.send_message(questionnaire, target)
        await self._wait_for_round_completion("round3")

    @rpc
    async def record_round3(
        self, message: ExpertRound3Reply, ctx: MessageContext
    ) -> AckMsg:
        """Collect Round 3 expert replies via RPC.

        Args:
            message: Expert's Round 3 reply
            ctx: Message context

        Returns:
            AckMsg: Acknowledgment
        """
        # validate expert
        if message.expert_id not in self._expert_ids:
            return AckMsg(ok=False, message=f"Unknown expert ID: {message.expert_id}")

        # store reply
        self._round3_replies.append(message)
        self._pending_round3.discard(message.expert_id)

        self.logger.info(
            "✓ Round 3 reply from %s; pending: %d -> %s",
            message.expert_id,
            len(self._pending_round3),
            sorted(self._pending_round3),
        )

        if not self._pending_round3:
            self._round3_done.set()

        return AckMsg(ok=True, message="Round 3 reply recorded")

    async def _compute_final_consensus(self) -> None:
        """Compute beta pooling consensus for irAKI classification."""
        self.logger.info("=== COMPUTING FINAL CONSENSUS ===")

        if not self._round3_replies:
            self.logger.error("No Round 3 replies received, cannot compute consensus")
            return

        # extract data for beta pooling
        p_vec = np.array([r.p_iraki for r in self._round3_replies])
        ci_mat = np.array([list(r.ci_iraki) for r in self._round3_replies])
        w_vec = np.array([r.confidence for r in self._round3_replies])

        # compute beta pooling consensus
        try:
            consensus_stats = beta_pool_confidence(p_vec, ci_mat, w_vec)
        except Exception as e:
            self.logger.error(f"Failed to compute beta pooling: {e}")
            # fallback to simple average
            consensus_stats = {
                "pooled_mean": float(np.mean(p_vec)),
                "pooled_ci": [
                    float(np.percentile(p_vec, 2.5)),
                    float(np.percentile(p_vec, 97.5)),
                ],
                "consensus_conf": float(np.mean(w_vec)),
            }

        # compute majority vote for comparison
        votes_iraki = sum(1 for r in self._round3_replies if r.verdict)
        consensus_verdict = votes_iraki > len(self._round3_replies) / 2

        # log results
        self.logger.info("=" * 80)
        self.logger.info(f"CASE {self._case_id} irAKI CONSENSUS RESULTS:")
        self.logger.info("-" * 80)
        self.logger.info(
            f"Beta Pooled P(irAKI):    {consensus_stats['pooled_mean']:.3f}"
        )
        self.logger.info(
            f"95% Credible Interval:   [{consensus_stats['pooled_ci'][0]:.3f}, "
            f"{consensus_stats['pooled_ci'][1]:.3f}]"
        )
        self.logger.info(
            f"Consensus Confidence:    {consensus_stats['consensus_conf']:.3f}"
        )
        self.logger.info(
            f"Majority Vote:           {'irAKI' if consensus_verdict else 'Other AKI'} "
            f"({votes_iraki}/{len(self._round3_replies)})"
        )
        self.logger.info(
            f"Expert Participation:    {len(self._round3_replies)}/{len(self._expert_ids)}"
        )
        self.logger.info("=" * 80)

        # export for human review if configured
        if self._delphi_config.export_full_transcripts:
            await self._export_for_human_review(consensus_stats, consensus_verdict)

    async def _export_for_human_review(
        self, consensus_stats: Dict, consensus_verdict: bool
    ) -> None:
        """Export complete case for human expert review.

        Args:
            consensus_stats: Beta pooling statistics
            consensus_verdict: Binary verdict from majority vote
        """
        self.logger.info("Exporting case for human review...")

        # prepare export data
        export_data = HumanReviewExport(
            case_id=self._case_id,
            export_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            delphea_version="1.0.0",
            # consensus results
            final_consensus={
                "p_iraki": consensus_stats["pooled_mean"],
                "ci_95": consensus_stats["pooled_ci"],
                "confidence": consensus_stats["consensus_conf"],
                "beta_pooling_stats": consensus_stats,
            },
            majority_verdict=consensus_verdict,
            expert_agreement_level=self._calculate_agreement_level(),
            # detailed assessments
            expert_assessments=[
                {
                    "round": "round1",
                    "expert_id": r.expert_id,
                    "p_iraki": r.p_iraki,
                    "confidence": r.confidence,
                    "primary_diagnosis": r.primary_diagnosis,
                    "differential": r.differential_diagnosis,
                }
                for r in self._round1_replies
            ]
            + [
                {
                    "round": "round3",
                    "expert_id": r.expert_id,
                    "p_iraki": r.p_iraki,
                    "verdict": r.verdict,
                    "final_diagnosis": r.final_diagnosis,
                    "recommendations": r.recommendations,
                }
                for r in self._round3_replies
            ],
            debate_transcripts=[
                {"q_id": q_id, "summary": summary}
                for q_id, summary in self._debate_summaries.items()
            ],
            # clinical summary
            reasoning_summary=self._synthesize_reasoning(),
            clinical_timeline=self._patient_data.get("timeline", {}),
            differential_summary=self._consolidate_differentials(),
            # recommendations
            consensus_recommendations=self._consolidate_recommendations(),
            # quality metrics
            confidence_metrics={
                "mean_confidence": float(
                    np.mean([r.confidence for r in self._round3_replies])
                ),
                "min_confidence": float(
                    np.min([r.confidence for r in self._round3_replies])
                ),
                "max_confidence": float(
                    np.max([r.confidence for r in self._round3_replies])
                ),
            },
        )

        # save to file
        output_file = (
            f"iraki_consensus_{self._case_id}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(export_data.model_dump(), f, indent=2)

        self.logger.info(f"Exported consensus results to {output_file}")

    def _calculate_agreement_level(self) -> float:
        """Calculate inter-expert agreement level."""
        if len(self._round3_replies) < 2:
            return 1.0

        # simple agreement based on verdict consistency
        verdicts = [r.verdict for r in self._round3_replies]
        majority = max(verdicts.count(True), verdicts.count(False))
        return majority / len(verdicts)

    def _synthesize_reasoning(self) -> str:
        """Synthesize clinical reasoning from all experts."""
        # simplified - in production would use NLP to summarize
        high_confidence_experts = [
            r for r in self._round3_replies if r.confidence > 0.7
        ]
        if high_confidence_experts:
            return (
                f"High-confidence assessment from {len(high_confidence_experts)} experts. "
                f"Primary consensus: {high_confidence_experts[0].final_diagnosis}"
            )
        return "Mixed confidence levels across expert panel."

    def _consolidate_differentials(self) -> List[str]:
        """Consolidate differential diagnoses from all experts."""
        all_differentials = set()
        for reply in self._round1_replies:
            all_differentials.update(reply.differential_diagnosis)
        return sorted(list(all_differentials))

    def _consolidate_recommendations(self) -> List[str]:
        """Consolidate clinical recommendations from Round 3."""
        # collect all unique recommendations
        all_recommendations = set()
        for reply in self._round3_replies:
            all_recommendations.update(reply.recommendations)

        # prioritize by frequency (simplified)
        return sorted(list(all_recommendations))[:5]  # top 5 recommendations
