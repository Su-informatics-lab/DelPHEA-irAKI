# prompts/__init__.py
"""Public API for the prompts package.

Exposes only the top-level functions used by other modules:
- format_round1_prompt
- format_round3_prompt
"""

from .rounds import format_round1_prompt, format_round3_prompt

__all__ = ["format_round1_prompt", "format_round3_prompt"]
