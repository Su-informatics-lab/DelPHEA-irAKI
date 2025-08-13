import logging
import os
import types

import pytest
import requests

from aggregator import Aggregator
from expert import Expert
from llm_backend import LLMBackend
from moderator import Moderator
from router import Router


def _maybe_live_url():
    for env in ("ENDPOINT_URL", "OPENAI_BASE_URL", "LIVE_BACKEND_URL"):
        url = os.getenv(env)
        if url:
            return url.rstrip("/")
    return None


def _reachable(url: str) -> bool:
    try:
        r = requests.get(url.rstrip("/") + "/v1/models", timeout=1.5)
        return r.status_code < 500
    except Exception:
        return False


class DummyRouter(Router):
    """Always solicit minority=['E1'] for any QID present."""

    def plan(self, r1, rules):
        qids = list(next(iter(r1))[1].scores.keys()) if r1 else []
        return types.SimpleNamespace(by_qid={qid: ["E1"] for qid in qids})


class R1Lite:
    """Pydantic-like minimal for Moderator routing."""

    def __init__(self, scores, evidence):
        self.scores = scores
        self.evidence = evidence


@pytest.mark.integration
def test_moderator_detect_and_run_debates_live_backend(monkeypatch):
    """
    Skips if live backend not reachable.
    Uses live backend only for R2 debate turns (R1 is injected).
    """
    url = _maybe_live_url()
    if not url or not _reachable(url):
        pytest.skip("Live backend not reachable; set ENDPOINT_URL or OPENAI_BASE_URL.")

    # Make Moderator init independent from files
    import moderator as mod

    monkeypatch.setattr(mod, "load_qids", lambda _p: ["Q1"])
    monkeypatch.setattr(
        mod,
        "load_consensus_rules",
        lambda _p: types.SimpleNamespace(
            minimum_agreement=0.7, debate_threshold_points=2
        ),
    )

    backend = LLMBackend(
        endpoint_url=url, model_name=os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
    )
    e1 = Expert("E1", "nephro", {}, backend=backend)
    e2 = Expert("E2", "nephro", {}, backend=backend)

    M = Moderator(
        experts=[e1, e2],
        questionnaire_path="dummy.json",
        router=DummyRouter(),
        aggregator=Aggregator(),
        logger=logging.getLogger("moderator-live"),
        debate_rounds=3,
        max_history_turns=4,
        max_turns_per_expert=2,
        max_total_turns_per_qid=8,
        quiet_turn_limit=2,
    )

    # Inject R1 with disagreement so router triggers a debate
    r1 = [
        ("E1", R1Lite(scores={"Q1": 3}, evidence={"Q1": "ev1"})),
        ("E2", R1Lite(scores={"Q1": 7}, evidence={"Q1": "ev2"})),
    ]
    case = {"case_id": "iraki_case_live", "clinical_notes": [{"text": "short note"}]}

    out = M.detect_and_run_debates(r1, case)
    transcripts = out["transcripts"]["Q1"]
    # Expect at least minority_open and one majority_rebuttal (model may mark satisfied early)
    assert len(transcripts) >= 2
    roles = {t["speaker_role"] for t in transcripts[:2]}
    assert "minority" in roles and "majority" in roles
