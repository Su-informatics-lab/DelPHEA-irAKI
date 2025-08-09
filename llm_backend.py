# llm_backend.py
# minimal interface that hides transport details (openai, vllm, etc.)

from __future__ import annotations

from typing import Any, Dict


class LLMBackend:
    """abstract transport for expert prompting."""

    def assess_round1(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def assess_round3(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def debate(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
