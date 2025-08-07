"""
Moderator agent for DelPHEA-irAKI orchestration.

Master agent coordinating the Delphi consensus process across all rounds.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    TopicId,
    message_handler,
    rpc,
)

from config.core import DelphiConfig
from config.loader import ConfigurationLoader
from data.loader import DataLoaderWrapper
from orchestration.consensus import beta_pool_confidence
from orchestration.messages import (
    AckMsg,
    DebatePrompt,
    ExpertRound1Reply,
    ExpertRound3Reply,
    QuestionnaireMsg,
    StartCase,
)


class irAKIModeratorAgent(RoutedAgent):
    """Master agent coordinating irAKI Delphi process."""

    def __init__(
        self,
        case_id: str,
        config_loader: ConfigurationLoader,
        data_loader: DataLoaderWrapper,
        delphi_config: DelphiConfig,
    ) -> None:
        """Initialize moderator for case coordination.

        Args:
            case_id: Case identifier
            config_loader: Configuration loader instance
            data_loader: Data loader wrapper instance
            delphi_config: Delphi methodology configuration
        """
        super().__init__(f"irAKI Moderator for case {case_id}")
        self._case_id = case_id
        self._config_loader = config_loader
        self._data_loader = data_loader
        self._delphi_config = delphi_config

        # use expert_count from config
        self._expert_ids = self._config_loader.get_available_expert_ids()[
            : delphi_config.expert_count
        ]

        # round tracking
        self._round1_replies: List[ExpertRound1Reply] = []
        self._round3_replies: List[ExpertRound3Reply] = []
        self._chat_logs: Dict[str, List[Dict]] = defaultdict(list)

        # debate management
        self._active_debates: dict[str, dict] = {}

        # synchronization
        self._pending_round1: set = set()
        self._pending_round3: set = set()
        self._round1_done = asyncio.Event()
        self._round3_done = asyncio.Event()

        self.logger = logging.getLogger(f"iraki_moderator.{case_id}")
        self.logger.info(f"Initialized moderator with experts: {self._expert_ids}")

    @message_handler
    async def handle_start_case(self, message: StartCase, ctx: MessageContext) -> None:
        """Bootstrap irAKI Delphi process.

        Args:
            message: Start case signal
            ctx: Message context
        """
        self.logger.info(f"Starting irAKI Delphi process for case {message.case_id}")
        patient_data = self._data_loader.load_patient_case(message.case_id)
        await self._run_round1(patient_data)

    async def _run_round1(self, patient_data: Dict) -> None:
        """Execute Round 1: individual irAKI assessments.

        Args:
            patient_data: Loaded patient case data
        """
        self.logger.info("=== ROUND 1: Individual irAKI Assessments ===")

        # load questions with full context from configuration
        questions = self._config_loader.get_questions()

        # initialize pending tracking with actual expert IDs
        self._pending_round1 = set(self._expert_ids)
        self._round1_done.clear()

        # create questionnaire message
        questionnaire = QuestionnaireMsg(
            case_id=self._case_id,
            patient_info=patient_data.get("patient_info", {}),
            icu_summary=patient_data.get("patient_summary", ""),
            medication_history=patient_data.get("medication_history", {}),
            lab_values=patient_data.get("lab_values", {}),
            imaging_reports=str(patient_data.get("imaging_reports", [])),
            questions=questions,
            round_phase="round1",
        )

        await self.publish_message(questionnaire, TopicId("case", self._case_id))
        self.logger.info(
            f"Broadcast Round 1 questionnaire to {len(self._expert_ids)} experts"
        )

        await self._wait_for_round_completion("round1")

    @rpc
    async def record_round1(
        self, message: ExpertRound1Reply, ctx: MessageContext
    ) -> AckMsg:
        """Collect Round 1 expert replies.

        Args:
            message: Expert's Round 1 reply
            ctx: Message context

        Returns:
            AckMsg: Acknowledgment
        """
        # validate expert ID
        if message.expert_id not in self._expert_ids:
            return AckMsg(ok=False, message=f"Unknown expert ID: {message.expert_id}")

        self._chat_logs[message.expert_id].append(
            {
                "role": "expert_reply",
                "payload": message.model_dump(),
                "timestamp": time.time(),
            }
        )

        self._round1_replies.append(message)
        self._pending_round1.discard(message.expert_id)

        self.logger.debug(
            f"Round 1 reply from {message.expert_id} ({len(self._pending_round1)} pending)"
        )

        if not self._pending_round1:
            self._round1_done.set()

        return AckMsg(ok=True, message="Round 1 reply recorded")

    @rpc
    async def record_round3(
        self, message: ExpertRound3Reply, ctx: MessageContext
    ) -> AckMsg:
        """Collect Round 3 expert replies.

        Args:
            message: Expert's Round 3 reply
            ctx: Message context

        Returns:
            AckMsg: Acknowledgment
        """
        # validate expert ID
        if message.expert_id not in self._expert_ids:
            return AckMsg(ok=False, message=f"Unknown expert ID: {message.expert_id}")

        self._chat_logs[message.expert_id].append(
            {
                "role": "expert_reply",
                "payload": message.model_dump(),
                "timestamp": time.time(),
            }
        )

        self._round3_replies.append(message)
        self._pending_round3.discard(message.expert_id)

        self.logger.debug(
            f"Round 3 reply from {message.expert_id} ({len(self._pending_round3)} pending)"
        )

        if not self._pending_round3:
            self._round3_done.set()

        return AckMsg(ok=True, message="Round 3 reply recorded")

    async def _wait_for_round_completion(self, round_phase: str) -> None:
        """Wait for round completion with timeout.

        Args:
            round_phase: "round1" or "round3"
        """
        event = self._round1_done if round_phase == "round1" else self._round3_done
        timeout = getattr(self._delphi_config, f"{round_phase}_timeout", 300)

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            self.logger.info(f"All {round_phase} replies received")

            if round_phase == "round1":
                await self._run_round2()
            else:
                await self._compute_final_consensus()

        except asyncio.TimeoutError:
            pending = (
                self._pending_round1
                if round_phase == "round1"
                else self._pending_round3
            )
            self.logger.warning(
                f"Timeout in {round_phase}: {len(pending)} experts pending"
            )

            if round_phase == "round1":
                await self._run_round2()
            else:
                await self._compute_final_consensus()

    async def _run_round2(self) -> None:
        """Execute Round 2: debate conflicts with responsive timeout."""
        self.logger.info("=== ROUND 2: Conflict Resolution ===")

        conflicts = self._detect_conflicts()

        if not conflicts:
            self.logger.info("No conflicts detected, proceeding to Round 3")
            await self._run_round3()
            return

        self.logger.info(f"Detected conflicts in {len(conflicts)} questions")

        # start debates for all conflicts
        debate_tasks = {}
        for q_id, meta in conflicts.items():
            prompt = DebatePrompt(
                case_id=self._case_id,
                q_id=q_id,
                minority_view=f"Score range: {meta['score_range']}",
                round_no=2,
                participating_experts=self._expert_ids,
                clinical_context=meta.get("clinical_context"),
            )

            # send debate prompt to each expert
            for expert_id in self._expert_ids:
                try:
                    await self.send_message(
                        prompt, AgentId(f"Expert_{expert_id}", self._case_id)
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to send debate prompt to {expert_id}: {e}"
                    )

            # create task to wait for debate completion on this question
            debate_tasks[q_id] = asyncio.create_task(
                self._await_debate_completion(q_id, self._delphi_config.debate_timeout)
            )

        # wait for all debates to complete or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*debate_tasks.values(), return_exceptions=True),
                timeout=self._delphi_config.debate_timeout,
            )
            self.logger.info("All debates completed")
        except asyncio.TimeoutError:
            self.logger.info("Debate window expired, proceeding to Round 3")
            # cancel remaining tasks
            for task in debate_tasks.values():
                if not task.done():
                    task.cancel()

        await self._run_round3()

    def _detect_conflicts(self) -> Dict[str, Dict]:
        """Detect questions with significant disagreement.

        Returns:
            Dict mapping question IDs to conflict metadata
        """
        conflicts = {}
        questions = self._config_loader.get_questions()

        for i, question in enumerate(questions):
            q_id = question["id"]
            scores = [
                reply.scores.get(q_id, 5)
                for reply in self._round1_replies
                if q_id in reply.scores
            ]

            if (
                len(scores) > 1
                and max(scores) - min(scores) >= self._delphi_config.conflict_threshold
            ):
                conflicts[q_id] = {
                    "question": question,
                    "score_range": f"{min(scores)}-{max(scores)}",
                    "clinical_context": question.get("clinical_context", {}),
                }

        return conflicts

    async def _await_debate_completion(self, q_id: str, timeout: float) -> None:
        """Wait for debate completion on a specific question.

        Args:
            q_id: Question identifier
            timeout: Maximum wait time in seconds
        """
        # placeholder for debate completion logic
        # in a full implementation, this would track debate comments
        await asyncio.sleep(min(10, timeout))  # simulate debate time

    async def _run_round3(self) -> None:
        """Execute Round 3: final consensus."""
        self.logger.info("=== ROUND 3: Final Consensus ===")

        patient_data = self._data_loader.load_patient_case(self._case_id)
        questions = self._config_loader.get_questions()

        self._pending_round3 = set(self._expert_ids)
        self._round3_done.clear()

        questionnaire = QuestionnaireMsg(
            case_id=self._case_id,
            patient_info=patient_data.get("patient_info", {}),
            icu_summary=patient_data.get("patient_summary", ""),
            medication_history=patient_data.get("medication_history", {}),
            lab_values=patient_data.get("lab_values", {}),
            imaging_reports=str(patient_data.get("imaging_reports", [])),
            questions=questions,
            round_phase="round3",
        )

        await self.publish_message(questionnaire, TopicId("case", self._case_id))
        self.logger.info("Broadcast Round 3 questionnaire")

        await self._wait_for_round_completion("round3")

    async def _compute_final_consensus(self) -> None:
        """Compute beta pooling consensus for irAKI classification."""
        self.logger.info("=== COMPUTING FINAL CONSENSUS ===")

        if not self._round3_replies:
            self.logger.error("No Round 3 replies received")
            return

        # extract data for beta pooling
        p_vec = np.array([r.p_iraki for r in self._round3_replies])
        ci_mat = np.array([r.ci_iraki for r in self._round3_replies])
        w_vec = np.array([r.confidence for r in self._round3_replies])

        # compute beta pooling with confidence estimation
        stats = beta_pool_confidence(p_vec, ci_mat, w_vec)

        # traditional majority vote for comparison
        consensus_verdict = (
            sum(r.verdict for r in self._round3_replies) > len(self._round3_replies) / 2
        )

        # log results
        self.logger.info("=" * 80)
        self.logger.info(f"CASE {self._case_id} irAKI CONSENSUS RESULTS:")
        self.logger.info("-" * 80)
        self.logger.info(f"Beta Pooled P(irAKI):    {stats['pooled_mean']:.3f}")
        self.logger.info(
            f"95% Credible Interval:   [{stats['pooled_ci'][0]:.3f}, {stats['pooled_ci'][1]:.3f}]"
        )
        self.logger.info(f"Consensus Confidence:    {stats['consensus_conf']:.3f}")
        self.logger.info(
            f"Majority Vote Verdict:   {'irAKI' if consensus_verdict else 'Other AKI'}"
        )
        self.logger.info(f"Expert Count:            {len(self._round3_replies)}")
        self.logger.info("=" * 80)

        # export for human review if configured
        if self._delphi_config.export_full_transcripts:
            await self._export_for_human_review(stats, consensus_verdict)

    async def _export_for_human_review(
        self, stats: Dict, consensus_verdict: bool
    ) -> None:
        """Export complete case for human expert review.

        Args:
            stats: Consensus statistics
            consensus_verdict: Binary verdict from majority vote
        """
        self.logger.info("Exporting case for human review...")
        # implementation for full export would go here
        # this would create HumanReviewExport message and save to file
