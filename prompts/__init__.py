# prompts/__init__.py
"""public api for the prompts package.

exports only the top-level formatters used elsewhere:
- format_round1_prompt
- format_round3_prompt
"""

from .rounds import format_round1_prompt, format_round3_prompt

__all__ = ["format_round1_prompt", "format_round3_prompt"]
