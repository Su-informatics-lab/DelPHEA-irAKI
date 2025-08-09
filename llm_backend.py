# llm_backend.py
# single minimal backend for text generation against a vLLM/OpenAI-compatible server.
# provides a tiny, reliable .generate(prompt) api and hides transport details.

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import requests


class LLMBackend:
    """thin client for vLLM/OpenAI-compatible inference servers.

    the project expects one capability at this layer: generate a completion
    for a raw prompt string. higher-level orchestration (round prompts, debate,
    aggregation) lives outside this transport.

    Args:
        endpoint_url: base http url, e.g. "http://localhost:8000".
        model_name: model identifier; optional if 'model' is provided.
        model: alias for model_name for compatibility with upstream code.
        timeout: request timeout in seconds.
        api_key: optional bearer token.
        **_: ignore unexpected kwargs gracefully.

    Raises:
        ValueError: if neither model_name nor model is provided.
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: Optional[str] = None,
        *,
        model: Optional[str] = None,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
        **_: Any,  # ignore legacy kwargs gracefully
    ) -> None:
        model_id = model_name or model
        if not model_id:
            raise ValueError("LLMBackend requires 'model' (or 'model_name')")
        self._base = endpoint_url.rstrip("/")
        self._model = model_id
        self._timeout = timeout
        self._api_key = api_key

        # endpoints
        self._completions = f"{self._base}/v1/completions"
        self._chat = f"{self._base}/v1/chat/completions"

    # -------------------- public api --------------------

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """return raw text completion for a single prompt string.

        attempts /v1/completions first; if unavailable or blank, falls back to
        /v1/chat/completions with json mode enabled.

        Raises:
            RuntimeError: on network, http, or schema errors, or if both routes
            return no usable text.
        """
        # try legacy text completions first
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
        if isinstance(extra, dict):
            payload.update(extra)

        text = self._post_and_parse_text(self._completions, payload, mode="completions")
        text = self._clean_and_maybe_extract_json(text)
        if text is not None:
            return text

        # fallback to chat api; request json if supported
        chat_payload: Dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if stop:
            chat_payload["stop"] = stop
        if seed is not None:
            chat_payload["seed"] = seed
        if isinstance(extra, dict):
            chat_payload.update(extra)

        text = self._post_and_parse_text(self._chat, chat_payload, mode="chat")
        text = self._clean_and_maybe_extract_json(text)
        if text is not None:
            return text

        raise RuntimeError("LLMBackend.generate: no usable text from either route")

    # -------------------- internals --------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _post_and_parse_text(
        self, url: str, payload: Dict[str, Any], *, mode: str
    ) -> Optional[str]:
        try:
            resp = requests.post(
                url, headers=self._headers(), json=payload, timeout=self._timeout
            )
        except requests.RequestException as e:
            # for completions mode allow chat fallback by returning None
            if mode == "completions":
                return None
            raise RuntimeError(f"LLMBackend: request error for {url}: {e}") from e

        # allow fallback on missing route
        if resp.status_code == 404 and mode == "completions":
            return None

        if resp.status_code // 100 != 2:
            body_head = resp.text[:500]
            raise RuntimeError(
                f"LLMBackend: HTTP {resp.status_code} for {url}; body head: {body_head}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLMBackend: non-JSON response from {url}: {e}") from e

        try:
            if mode == "completions":
                choices = data.get("choices") or []
                if not choices:
                    return None
                text = choices[0].get("text")
                return text if isinstance(text, str) and text.strip() else None

            # chat mode
            choices = data.get("choices") or []
            if not choices:
                return None
            msg = choices[0].get("message") or {}

            # primary: regular assistant content
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content

            # secondary: function/tool calls
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, (dict, list)):
                    return json.dumps(args)
                if isinstance(args, str) and args.strip():
                    return args

            return None

        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"LLMBackend: failed to parse response schema from {url}: {e}"
            ) from e

    # -------------------- helpers --------------------

    def _clean_and_maybe_extract_json(self, text: Optional[str]) -> Optional[str]:
        """normalize empty to None; if text contains json, extract and return it."""
        if not isinstance(text, str):
            return None
        s = text.strip()
        if not s:
            return None

        # fast path: already valid json
        if self._is_valid_json(s):
            return s

        # fenced code block: ```json ...```
        m = re.search(
            r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", s, flags=re.IGNORECASE
        )
        if m and self._is_valid_json(m.group(1)):
            return m.group(1)

        # first balanced-looking json object/array substring (simple heuristic)
        # try object
        obj = self._extract_first_braced(s, opener="{", closer="}")
        if obj and self._is_valid_json(obj):
            return obj
        # try array
        arr = self._extract_first_braced(s, opener="[", closer="]")
        if arr and self._is_valid_json(arr):
            return arr

        # no valid json found; return None so caller can fallback/raise
        return None

    @staticmethod
    def _is_valid_json(s: str) -> bool:
        try:
            json.loads(s)
            return True
        except Exception:
            return False

    @staticmethod
    def _extract_first_braced(s: str, *, opener: str, closer: str) -> Optional[str]:
        # simple bracket matching to find the first top-level balanced segment
        start = s.find(opener)
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(s)):
            if s[i] == opener:
                depth += 1
            elif s[i] == closer:
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return None
