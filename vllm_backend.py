# vllm_backend.py
# minimal openai-compatible client with a simple .generate(prompt) api
# works with vLLM's server (e.g., http://localhost:8000)

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests


class VLLMBackend:
    """thin client for vLLM's OpenAI-compatible server.

    accepts either 'model' or 'model_name' for compatibility.
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: Optional[str] = None,
        *,
        model: Optional[str] = None,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
        **_: Any,  # swallow unexpected kwargs from upstream
    ) -> None:
        # normalize model arg
        model_id = model_name or model
        if not model_id:
            raise ValueError("VLLMBackend requires 'model' (or 'model_name')")
        self.base = endpoint_url.rstrip("/")
        self.model = model_id
        self.timeout = timeout
        self.api_key = api_key

        # precompute endpoints
        self._completions = f"{self.base}/v1/completions"
        self._chat = f"{self.base}/v1/chat/completions"

    # -------------------- public api --------------------

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stop: Optional[list[str]] = None,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """return raw text completion for a single prompt string.

        tries /v1/completions first; if unavailable, falls back to /v1/chat/completions.
        raises RuntimeError with helpful context on HTTP or schema errors.
        """
        payload = {
            "model": self.model,
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
        if text is not None:
            return text

        chat_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            chat_payload["stop"] = stop
        if seed is not None:
            chat_payload["seed"] = seed
        if isinstance(extra, dict):
            chat_payload.update(extra)

        text = self._post_and_parse_text(self._chat, chat_payload, mode="chat")
        if text is not None:
            return text

        raise RuntimeError("vLLMBackend.generate: server responded without usable text")

    # -------------------- internals --------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post_and_parse_text(
        self, url: str, payload: Dict[str, Any], *, mode: str
    ) -> Optional[str]:
        try:
            resp = requests.post(
                url, headers=self._headers(), json=payload, timeout=self.timeout
            )
        except requests.RequestException as e:
            # network / connection / timeout
            if mode == "completions":
                return None  # allow chat fallback
            raise RuntimeError(f"vLLMBackend: request error for {url}: {e}") from e

        # some servers return 404 for unsupported routes -> allow fallback
        if resp.status_code == 404 and mode == "completions":
            return None

        if resp.status_code // 100 != 2:
            body_head = resp.text[:500]
            raise RuntimeError(
                f"vLLMBackend: HTTP {resp.status_code} for {url}; body head: {body_head}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"vLLMBackend: non-JSON response from {url}: {e}") from e

        try:
            if mode == "completions":
                choices = data.get("choices") or []
                if not choices:
                    return None
                text = choices[0].get("text")
                return str(text) if text is not None else None
            choices = data.get("choices") or []
            if not choices:
                return None
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            return str(content) if content is not None else None
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"vLLMBackend: failed to parse response schema from {url}: {e}"
            ) from e
