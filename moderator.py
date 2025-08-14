"""
moderator: orchestrates r1 → debate → r3 and aggregates to consensus.

purpose
-------
single orchestrator for n experts × m questions with pluggable routing (sparse/full)
and pluggable aggregation. validates io contracts, fails fast but attempts a
one-shot repair per expert, and returns a serializable run report.

multi-turn debate (this version)
--------------------------------
For each QID with disagreement:
  1) Minority opening (all experts selected by router for that QID)
  2) Majority rebuttals (all other experts speak once)
  3) Handoff-aware loop:
       - speakers may append control lines: SATISFIED / REVISED_SCORE / HANDOFF
       - moderator prioritizes requested HANDOFF targets if eligible
       - experts exit when SATISFIED; loop ends when all satisfied, quiet, or capped
       - early-stop guard: if agreement ≥ minimum_agreement after any turn, stop QID
Each turn receives the prior turns for that QID via clinical_context["peer_turns"].

contracts
---------
- experts must return pydantic models: AssessmentR1/AssessmentR3/DebateTurn (via .model_dump()).
- router.plan consumes r1 payloads and returns a DebatePlan.
- aggregator.aggregate consumes r3 payloads and returns a Consensus object.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Sequence, Tuple

from aggregator import Aggregator
from models import AssessmentR1, AssessmentR3, Consensus
from router import DebatePlan, Router
from schema import load_consensus_rules, load_qids
from validators import (
    ValidationError,  # NOTE: tests may monkeypatch this to a duck-typed error.
)
from validators import (
    log_debate_status,
    validate_round1_payload,
    validate_round3_payload,
)


@dataclass
class _CaseBuffers:
    """in-memory buffers for a single case run."""

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
        max_retries: int = 1,
        *,
        debate_rounds: int = 3,
        max_history_turns: int = 6,
        max_turns_per_expert: int = 2,
        max_total_turns_per_qid: int = 12,
        quiet_turn_limit: int = 3,
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
        self.max_retries = max_retries

        # debate knobs
        self.debate_rounds = max(1, int(debate_rounds))
        self.max_history_turns = max(0, int(max_history_turns))
        self.max_turns_per_expert = max(1, int(max_turns_per_expert))
        self.max_total_turns_per_qid = max(1, int(max_total_turns_per_qid))
        self.quiet_turn_limit = max(1, int(quiet_turn_limit))

        # basic expert id sanity
        ids = [getattr(e, "expert_id", None) for e in self.experts]
        if any(i is None for i in ids):
            raise ValueError("all experts must expose .expert_id")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate expert_id detected")

        # transient context used for autopatching identifiers
        self.current_case_id: str | None = None
        self.current_expert_id: str | None = None

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
        self.logger.info(
            "assessing round %d for %d experts", round_no, len(self.experts)
        )

        outputs: List[Tuple[str, AssessmentR1 | AssessmentR3]] = []
        if round_no == 1:
            for e in self.experts:
                a1 = self._call_round1_with_repair(e, case)
                self._validate_qids_exact(a1)
                outputs.append((e.expert_id, a1))
            return outputs

        # round 3
        ctx = debate_ctx or {}
        for e in self.experts:
            a3 = self._call_round3_with_repair(e, case, ctx)
            self._validate_qids_exact(a3)
            outputs.append((e.expert_id, a3))
        return outputs

    # --------- helpers: ids & logging ---------

    def _extract_case_id(self, case: Dict[str, Any]) -> str:
        if isinstance(case, dict):
            for k in ("case_id", "id", "patient_id", "person_id"):
                v = case.get(k)
                if v:
                    return str(v)
        return "unknown_case"

    def _expert_by_id(self, expert_id: str):
        for e in self.experts:
            if e.expert_id == expert_id:
                return e
        raise KeyError(f"expert not found: {expert_id}")

    # --------- tiny helpers: agreement rule ---------

    def _min_agreement(self) -> float:
        """pull minimum_agreement from rules with a sensible default."""
        # accept attribute, dict, or nested dict
        val = getattr(self.rules, "minimum_agreement", None)
        if val is None and isinstance(self.rules, dict):
            val = self.rules.get("minimum_agreement", None)
        try:
            v = float(val)
            if 0.0 < v <= 1.0:
                return v
        except Exception:
            pass
        return 0.70  # default if not specified

    @staticmethod
    def _agreement_fraction(scores: Dict[str, int]) -> Tuple[float, int]:
        """compute fraction of experts within ±1 of the median (Delphi-style)."""
        vals = [int(v) for v in scores.values() if isinstance(v, (int, float))]
        if not vals:
            return 0.0, 0
        med = int(round(median(vals)))
        agree = sum(1 for v in vals if abs(int(v) - med) <= 1) / len(vals)
        return agree, med

    # --------- debate orchestration ---------

    def detect_and_run_debates(
        self, r1: Sequence[Tuple[str, AssessmentR1]], case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """plan debates via router and run a small, handoff-aware turn loop per QID."""
        plan: DebatePlan = self.router.plan(r1, self.rules)
        solicitations = sum(len(v) for v in plan.by_qid.values())
        disagreement_present = solicitations > 0

        cid = self._extract_case_id(case)
        log_debate_status(
            logger=self.logger,
            case_id=cid,
            question_id="__summary__",
            expert_id="moderator",
            stage="debate",
            status="start",
            meta={
                "solicitations": solicitations,
                "qids": len(plan.by_qid),
                "rounds": self.debate_rounds,
            },
        )

        transcripts: Dict[str, List[Dict[str, Any]]] = {}
        minority_views: Dict[str, str] = {}

        if not disagreement_present:
            log_debate_status(
                logger=self.logger,
                case_id=cid,
                question_id="__summary__",
                expert_id="moderator",
                stage="debate",
                status="skip",
                reason="no_disagreement_detected",
                meta={"solicitations": 0},
            )
            return {
                "debate_plan": plan.by_qid,
                "transcripts": {},
                "minority_views": {},
                "debate_skipped": True,
                "rounds": self.debate_rounds,
            }

        self.logger.info(
            "debate planning complete: %d solicitations across %d qids",
            solicitations,
            len(plan.by_qid),
        )

        # helpers
        all_ids = [eid for eid, _ in r1]
        r1_by_id = {eid: a for eid, a in r1}

        def _append_turn(qid: str, turn_dict: Dict[str, Any]) -> None:
            transcripts.setdefault(qid, []).append(turn_dict)

        def _peer_history(qid: str) -> List[Dict[str, Any]]:
            hist = transcripts.get(qid, [])
            if self.max_history_turns <= 0:
                return []
            return hist[-self.max_history_turns :]

        for qid, minority_ids in plan.by_qid.items():
            if not minority_ids:
                continue
            majority_ids = [eid for eid in all_ids if eid not in minority_ids]

            # status before collecting turns
            log_debate_status(
                logger=self.logger,
                case_id=cid,
                question_id=qid,
                expert_id="moderator",
                stage="debate",
                status="turn",
                meta={
                    "asked_experts": minority_ids,
                    "majority_candidates": majority_ids,
                    "pattern": "minority→majority(+handoff)→minority_followup",
                },
            )

            # build a compact minority view text from r1
            mv_lines = []
            for eid in minority_ids:
                a = r1_by_id.get(eid)
                if not a:
                    continue
                ev = a.evidence.get(qid, "")
                score = a.scores.get(qid, None)
                mv_lines.append(f"{eid}: score={score} evidence={ev}")
            mv = "\n".join(mv_lines) or "minority perspective not available"
            minority_views[qid] = mv

            # queue: minority_open (all), majority_rebuttal (all), then minority_followup (first minority)
            opener = minority_ids[0]
            queue: List[Tuple[str, str]] = []
            for eid in minority_ids:
                queue.append(("minority_open", eid))
            for eid in majority_ids:
                queue.append(("majority_rebuttal", eid))
            queue.append(("minority_followup", opener))  # will be bubbled forward later

            # per-qid tracking
            turns_by_expert: Dict[str, int] = {eid: 0 for eid in all_ids}
            satisfied: set[str] = set()

            # maintain current scores for this qid (seeded from r1; updated on revised_score)
            current_scores: Dict[str, int] = {}
            for eid in all_ids:
                try:
                    s = int(r1_by_id[eid].scores[qid])
                except Exception:
                    s = None
                if isinstance(s, int) and 1 <= s <= 9:
                    current_scores[eid] = s

            def _bubble_minority_followup(front_offset: int = 0) -> None:
                """move the opener's follow-up near the front (after any new handoff)."""
                try:
                    idx = next(
                        i
                        for i, (role, eid) in enumerate(queue)
                        if role == "minority_followup" and eid == opener
                    )
                except StopIteration:
                    return
                item = queue.pop(idx)
                queue.insert(0 + max(0, front_offset), item)

            # main loop
            while (
                queue and len(transcripts.get(qid, [])) < self.max_total_turns_per_qid
            ):
                role, eid = queue.pop(0)

                # skip if capped or already satisfied
                if (
                    eid in satisfied
                    or turns_by_expert.get(eid, 0) >= self.max_turns_per_expert
                ):
                    continue

                # build ctx and call expert
                e = self._expert_by_id(eid)
                ctx = {"case": case, "peer_turns": _peer_history(qid), "role": role}
                turn = e.debate(
                    qid=qid, round_no=2, clinical_context=ctx, minority_view=mv
                )
                td = turn.model_dump()

                speaker_role = (
                    "minority"
                    if role.startswith("minority")
                    else "majority"
                    if role.startswith("majority")
                    else "participant"
                )
                td.update(
                    {
                        "expert_id": eid,
                        "qid": qid,
                        "turn_index": len(transcripts.get(qid, [])),
                        "speaker_role": speaker_role,
                    }
                )
                _append_turn(qid, td)

                # update counters / satisfaction
                turns_by_expert[eid] = turns_by_expert.get(eid, 0) + 1
                if bool(td.get("satisfied", False)):
                    satisfied.add(eid)

                # apply revised score (if provided) to current scores for agreement checks
                rs = td.get("revised_score", None)
                if isinstance(rs, int) and 1 <= rs <= 9:
                    current_scores[eid] = rs

                # honor handoff: prioritize requested target if eligible
                inserted_handoff = False
                target = td.get("handoff_to")
                if isinstance(target, str):
                    target = target.strip()
                if (
                    isinstance(target, str)
                    and target
                    and target in all_ids
                    and target not in satisfied
                    and turns_by_expert.get(target, 0) < self.max_turns_per_expert
                ):
                    idx_in_queue = next(
                        (i for i, (_role, _eid) in enumerate(queue) if _eid == target),
                        None,
                    )
                    if idx_in_queue is not None:
                        planned_role, _ = queue.pop(idx_in_queue)
                        if role == "minority_open" and idx_in_queue == 0:
                            new_role = planned_role
                        else:
                            new_role = "participant"
                        queue.insert(0, (new_role, target))
                        inserted_handoff = not (
                            role == "minority_open" and idx_in_queue == 0
                        )
                    else:
                        queue.insert(0, ("participant", target))
                        inserted_handoff = True

                # after any majority or participant, ensure opener's follow-up comes next
                if role in ("majority_rebuttal", "participant"):
                    _bubble_minority_followup(front_offset=1 if inserted_handoff else 0)

                # early-stop guard: stop once agreement reaches the questionnaire threshold
                agree_frac, med = self._agreement_fraction(current_scores)
                min_agree = self._min_agreement()
                if agree_frac >= min_agree:
                    log_debate_status(
                        logger=self.logger,
                        case_id=cid,
                        question_id=qid,
                        expert_id="moderator",
                        stage="debate",
                        status="early_stop",
                        meta={
                            "agreement": round(agree_frac, 3),
                            "minimum_agreement": min_agree,
                            "median_score": med,
                            "turns_so_far": len(transcripts.get(qid, [])),
                        },
                    )
                    break

                # stop if everyone is done or capped
                if all(
                    (eid_ in satisfied)
                    or (turns_by_expert.get(eid_, 0) >= self.max_turns_per_expert)
                    for eid_ in set(minority_ids + majority_ids)
                ):
                    break

        # structured status: summary "end"
        log_debate_status(
            logger=self.logger,
            case_id=cid,
            question_id="__summary__",
            expert_id="moderator",
            stage="debate",
            status="end",
            meta={"qids_with_transcripts": len(transcripts)},
        )

        return {
            "debate_plan": plan.by_qid,
            "transcripts": transcripts,
            "minority_views": minority_views,
            "debate_skipped": False,
            "rounds": self.debate_rounds,
        }

    def run_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """run the full r1 → debate → r3 → aggregate pipeline and return a report dict."""
        buffers = _CaseBuffers(r1=[], debate_ctx={}, r3=[], consensus=None)

        # round 1
        buffers.r1 = self.assess_round(1, case)  # [(eid, AssessmentR1)]
        self._validate_r1_coherence(buffers.r1)

        # debate (may be skipped)
        buffers.debate_ctx = self.detect_and_run_debates(buffers.r1, case)

        # round 3
        buffers.r3 = self.assess_round(
            3, case, buffers.debate_ctx
        )  # [(eid, AssessmentR3)]
        self._validate_r3_coherence(buffers.r1, buffers.r3)

        # aggregate
        buffers.consensus = self.aggregator.aggregate([a for _, a in buffers.r3])

        # include case_id for easier downstream bundling
        cid = self._extract_case_id(case)

        report = {
            "case_id": cid,
            "round1": [(eid, a.model_dump()) for eid, a in buffers.r1],
            "debate": buffers.debate_ctx,
            "round3": [(eid, a.model_dump()) for eid, a in buffers.r3],
            "consensus": buffers.consensus.model_dump(),
        }
        self.logger.debug(json.dumps(report["consensus"], indent=2))
        return report

    # --------- private: robust r1/r3 callers with repair ---------

    def _validation_payload_with_ids(
        self, assessed: AssessmentR1 | AssessmentR3, expert, case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """return a dict suitable for validators, guaranteeing case_id/expert_id fields."""
        d = assessed.model_dump() if hasattr(assessed, "model_dump") else dict(assessed)
        # derive case_id deterministically from the case payload or transient context
        cid = None
        if isinstance(case, dict):
            cid = (
                case.get("case_id")
                or case.get("id")
                or case.get("patient_id")
                or case.get("person_id")
            )
        d["case_id"] = (
            d.get("case_id")
            or cid
            or getattr(self, "current_case_id", None)
            or "unknown_case"
        )
        d["expert_id"] = (
            d.get("expert_id")
            or getattr(self, "current_expert_id", None)
            or "unknown_expert"
        )
        return d

    def _call_round1_with_repair(self, expert, case: Dict[str, Any]) -> AssessmentR1:
        """call expert.assess_round1 with one-shot retry and auto-repair fallback."""
        attempt = 0
        last_err: ValidationError | None = None
        self.current_expert_id = getattr(expert, "expert_id", None)
        self.current_case_id = case.get("case_id") if isinstance(case, dict) else None

        a1 = None
        while attempt <= self.max_retries:
            a1 = expert.assess_round1(case, self.qpath)
            vd = self._validation_payload_with_ids(a1, expert, case)
            try:
                validate_round1_payload(vd, required_evidence=12)
                if attempt > 0:
                    self.logger.info(
                        "round1 validation succeeded after retry for %s",
                        expert.expert_id,
                    )
                return a1
            except ValidationError as ve:  # duck-typed in tests
                last_err = ve
                self.logger.error(
                    "round1 validation failed for %s: %s", expert.expert_id, ve
                )
                attempt += 1
                if attempt > self.max_retries:
                    break
                hint = self._build_repair_hint(ve, round_no=1)
                a1 = self._retry_assess_round1(expert, case, hint)

        self.logger.warning(
            "auto-repairing round1 payload for %s (last error: %s)",
            expert.expert_id,
            last_err,
        )
        vd = self._validation_payload_with_ids(
            a1 or expert.assess_round1(case, self.qpath), expert, case
        )
        patched = self._autopatch_round1(vd)
        return AssessmentR1(**patched)

    def _call_round3_with_repair(
        self, expert, case: Dict[str, Any], ctx: Dict[str, Any]
    ) -> AssessmentR3:
        """call expert.assess_round3 with one-shot retry and auto-repair fallback."""
        attempt = 0
        last_err: ValidationError | None = None
        self.current_expert_id = getattr(expert, "expert_id", None)
        self.current_case_id = case.get("case_id") if isinstance(case, dict) else None

        a3 = None
        while attempt <= self.max_retries:
            a3 = expert.assess_round3(case, self.qpath, ctx)
            vd = self._validation_payload_with_ids(a3, expert, case)
            try:
                validate_round3_payload(vd)
                if attempt > 0:
                    self.logger.info(
                        "round3 validation succeeded after retry for %s",
                        expert.expert_id,
                    )
                return a3
            except ValidationError as ve:  # duck-typed in tests
                last_err = ve
                self.logger.error(
                    "round3 validation failed for %s: %s", expert.expert_id, ve
                )
                attempt += 1
                if attempt > self.max_retries:
                    break
                hint = self._build_repair_hint(ve, round_no=3)
                a3 = self._retry_assess_round3(expert, case, ctx, hint)

        self.logger.warning(
            "auto-repairing round3 payload for %s (last error: %s)",
            expert.expert_id,
            last_err,
        )
        vd = self._validation_payload_with_ids(
            a3 or expert.assess_round3(case, self.qpath, ctx), expert, case
        )
        patched = self._autopatch_round3(vd)
        return AssessmentR3(**patched)

    # --------- helpers: retry & autopatch ---------

    def _retry_assess_round1(
        self, expert, case: Dict[str, Any], hint: str
    ) -> AssessmentR1:
        """retry r1 with a repair hint if the expert api supports it; else plain retry."""
        kwargs = {}
        try:
            sig = inspect.signature(expert.assess_round1)
            if "repair_hint" in sig.parameters:
                kwargs["repair_hint"] = hint
        except Exception:
            pass
        return expert.assess_round1(case, self.qpath, **kwargs)

    def _retry_assess_round3(
        self, expert, case: Dict[str, Any], ctx: Dict[str, Any], hint: str
    ) -> AssessmentR3:
        """retry r3 with a repair hint if supported; else plain retry."""
        kwargs = {}
        try:
            sig = inspect.signature(expert.assess_round3)
            if "repair_hint" in sig.parameters:
                kwargs["repair_hint"] = hint
        except Exception:
            pass
        return expert.assess_round3(case, self.qpath, ctx, **kwargs)

    def _build_repair_hint(self, ve: ValidationError, round_no: int) -> str:
        """compact, model-friendly hint listing what failed and how to fix.
        works with pydantic.ValidationError or any duck-typed object exposing .errors().
        """
        # collect msg/loc from any error-like payload.
        msgs: List[str] = []
        try:
            items = list(getattr(ve, "errors")() or [])
        except Exception:
            items = []
        for err in items:
            loc = " → ".join(str(x) for x in err.get("loc", ()))
            msg = err.get("msg", "invalid value")
            msgs.append(f"{loc}: {msg}")

        # friendly, round-specific prefix.
        if round_no == 1:
            prefix = (
                "please fix: provide ≥200-char clinical_reasoning; non-empty primary_diagnosis; "
                "≥2 differential_diagnosis items; and fill ≥12 evidence entries with concise text.\n"
            )
        else:
            prefix = (
                "please fix: non-empty changes_from_round1.summary and .debate_influence; "
                "non-empty final_diagnosis; at least 1 recommendation.\n"
            )

        # if we detect the common 'importance must sum to 100' case, add a very explicit instruction.
        extra = ""
        if any("importance must sum to 100" in m for m in msgs):
            extra = (
                "ensure per-QID 'importance' are integers that sum to exactly 100 "
                "(no rounding; adjust values, then regenerate).\n"
            )

        core = (
            "violations:\n- " + "\n- ".join(msgs)
            if msgs
            else "violations: none provided"
        )
        return prefix + extra + core

    def _autopatch_round1(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """make a minimal, explicit placeholder fix so downstream code can proceed."""
        patched = dict(payload)

        # identifiers (deterministic, not llm-dependent)
        if not isinstance(patched.get("case_id"), str) or not patched["case_id"]:
            patched["case_id"] = (
                getattr(self, "current_case_id", None)
                or payload.get("case_id")
                or "unknown_case"
            )
        if not isinstance(patched.get("expert_id"), str) or not patched["expert_id"]:
            patched["expert_id"] = (
                getattr(self, "current_expert_id", None) or "unknown_expert"
            )

        # ensure evidence values are non-empty
        ev = dict(patched.get("evidence") or {})
        for q in self.qids:
            txt = ev.get(q) or ""
            if not isinstance(txt, str) or not txt.strip():
                ev[
                    q
                ] = "[auto-repair] evidence text not provided by expert; please review source notes."
        patched["evidence"] = ev

        # clinical_reasoning
        cr = patched.get("clinical_reasoning") or ""
        if not isinstance(cr, str):
            cr = ""
        cr = cr.strip()
        if len(cr) < 200:
            parts = [
                "[auto-repair] clinical reasoning synthesized due to empty expert response."
            ]
            for q in self.qids[:8]:
                s = payload.get("scores", {}).get(q, None)
                parts.append(f"{q}: score={s}, ev={ev.get(q)}")
            cr = " ".join(parts)
            if len(cr) < 200:
                cr += " this placeholder meets minimum length for validation and flags the need for human review."
        patched["clinical_reasoning"] = cr

        # primary_diagnosis
        pdx = patched.get("primary_diagnosis")
        if not isinstance(pdx, str) or not pdx.strip():
            patched[
                "primary_diagnosis"
            ] = "AKI—etiology uncertain [auto-repair placeholder]"

        # differential_diagnosis
        ddx = patched.get("differential_diagnosis")
        if not (
            isinstance(ddx, list)
            and len([x for x in ddx if isinstance(x, str) and x.strip()]) >= 2
        ):
            patched["differential_diagnosis"] = [
                "prerenal azotemia [placeholder]",
                "acute tubular injury vs. AIN [placeholder]",
            ]

        return patched

    def _autopatch_round3(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """minimal placeholder fixes for round 3 structure."""
        patched = dict(payload)

        # identifiers, same rationale as r1
        if not isinstance(patched.get("case_id"), str) or not patched["case_id"]:
            patched["case_id"] = (
                getattr(self, "current_case_id", None)
                or payload.get("case_id")
                or "unknown_case"
            )
        if not isinstance(patched.get("expert_id"), str) or not patched["expert_id"]:
            patched["expert_id"] = (
                getattr(self, "current_expert_id", None) or "unknown_expert"
            )

        ch = dict(patched.get("changes_from_round1") or {})
        if (
            not isinstance(ch.get("summary", ""), str)
            or not ch.get("summary", "").strip()
        ):
            ch[
                "summary"
            ] = "[auto-repair] no changes documented; summary synthesized for continuity."
        if (
            not isinstance(ch.get("debate_influence", ""), str)
            or not ch.get("debate_influence", "").strip()
        ):
            ch[
                "debate_influence"
            ] = "[auto-repair] debate skipped or not recorded; no influence on final judgment."
        patched["changes_from_round1"] = ch

        fd = patched.get("final_diagnosis")
        if not isinstance(fd, str) or not fd.strip():
            patched[
                "final_diagnosis"
            ] = "final diagnosis not specified [auto-repair placeholder]"

        recs = patched.get("recommendations")
        if not (
            isinstance(recs, list)
            and any(isinstance(x, str) and x.strip() for x in recs)
        ):
            patched["recommendations"] = [
                "review case with nephrology; validate placeholder content."
            ]

        return patched

    # --------- validators (fail fast on id mismatches) ---------

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
        for _, a in r1:
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
        for _, a3 in r3:
            self._validate_qids_exact(a3)


if __name__ == "__main__":
    import sys

    print("moderator module loaded ok", file=sys.stderr)
