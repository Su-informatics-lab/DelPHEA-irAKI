# validators.py — add this near the other helpers
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict


def log_debate_status(*args, **kwargs) -> None:
    """Structured logger for debate lifecycle events (compat shim).

    supports both:
        log_debate_status(logger, case_id, question_id, expert_id, status, reason=None, meta=None)
    and:
        log_debate_status(
            logger=..., case_id=..., question_id=..., expert_id=..., status=..., stage="debate",
            reason=None, meta=None
        )

    Args:
        logger: logging.Logger instance (first positional arg or 'logger=' kwarg).
        case_id: normalized case id string like 'iraki_case_...'.
        question_id: questionnaire key for this turn.
        expert_id: panel expert identifier (e.g., 'nephrologist_1').
        status: one of {'start','turn','repair','skip','timeout','error','end'}.
        stage: optional lifecycle stage {'r1','debate','r3'} (default 'debate').
        reason: optional short reason for non-normal states (e.g., 'timeout').
        meta: optional JSON-serializable dict with extra fields.

    Raises:
        TypeError: if logger is missing or invalid.
        ValueError: if required fields are missing/invalid or meta is not JSON-serializable.
    """
    # extract logger
    logger = kwargs.pop("logger", args[0] if args else None)
    if not isinstance(logger, logging.Logger):
        raise TypeError(
            "log_debate_status: first arg or 'logger=' must be a logging.Logger"
        )

    # attempt positional signature first: (logger, case_id, qid, expert_id, status, [reason], [meta])
    if len(args) >= 5:
        _, case_id, question_id, expert_id, status, *rest = args
        reason = rest[0] if len(rest) >= 1 else kwargs.pop("reason", None)
        meta = rest[1] if len(rest) >= 2 else kwargs.pop("meta", None)
        stage = kwargs.pop("stage", "debate")
    else:
        # keyword signature
        try:
            case_id = kwargs.pop("case_id")
            question_id = kwargs.pop("question_id")
            expert_id = kwargs.pop("expert_id")
            status = kwargs.pop("status")
        except KeyError as e:
            raise ValueError(
                f"log_debate_status: missing required field {e.args[0]!r}"
            ) from e
        stage = kwargs.pop("stage", "debate")
        reason = kwargs.pop("reason", None)
        meta = kwargs.pop("meta", None)

    allowed_status = {"start", "turn", "repair", "skip", "timeout", "error", "end"}
    allowed_stage = {"r1", "debate", "r3"}

    if stage not in allowed_stage:
        raise ValueError(
            f"log_debate_status: invalid stage={stage!r}; expected one of {sorted(allowed_stage)}"
        )
    if status not in allowed_status:
        raise ValueError(
            f"log_debate_status: invalid status={status!r}; expected one of {sorted(allowed_status)}"
        )

    # ensure meta is json-serializable
    if meta is not None:
        try:
            json.dumps(meta)
        except Exception as e:
            raise ValueError(
                f"log_debate_status: 'meta' must be JSON-serializable: {e}"
            ) from e

    payload: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "case_id": case_id,
        "question_id": question_id,
        "expert_id": expert_id,
        "status": status,
    }
    if reason:
        payload["reason"] = reason
    if meta is not None:
        payload["meta"] = meta

    # single line for easy grep/parse
    logger.info("debate_event %s", json.dumps(payload, sort_keys=True))
