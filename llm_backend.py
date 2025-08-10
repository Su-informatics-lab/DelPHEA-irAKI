# llm_backend.py (exposes endpoint_url/model_name properties for validators.py)
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class LLMBackend:
    def __init__(
        self,
        *,
        endpoint_url: str,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 120.0,
        api_key: Optional[str] = None,
        prefer_json_mode: Optional[bool] = None,
        **_: Any,
    ) -> None:
        model_id = model_name or model
        if not model_id:
            raise ValueError("LLMBackend requires 'model' (or 'model_name')")
        if not endpoint_url:
            raise ValueError("LLMBackend requires 'endpoint_url'")
        self._base = endpoint_url.rstrip("/")
        self._model = model_id
        self._timeout = timeout
        self._api_key = api_key
        self._prefer_json_mode = (
            bool(prefer_json_mode) if prefer_json_mode is not None else False
        )

        self._completions = f"{self._base}/v1/completions"
        self._chat = f"{self._base}/v1/chat/completions"

    # convenient properties used by validators.py
    @property
    def endpoint_url(self) -> str:
        return self._base

    @property
    def model_name(self) -> str:
        return self._model

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
        # keep a simple text path as a fallback (unchanged from earlier simplified version)
        chat_payload: Dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            chat_payload["stop"] = stop
        if seed is not None:
            chat_payload["seed"] = seed
        if isinstance(extra, dict):
            for k, v in extra.items():
                chat_payload[k] = v

        text = self._post_and_parse_text(self._chat, chat_payload, mode="chat")
        if isinstance(text, str) and text.strip():
            return text.strip()

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
        if isinstance(text, str) and text.strip():
            return text.strip()

        raise RuntimeError("LLMBackend.generate: no usable text from either route")

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
        except requests.RequestException:
            return None

        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            # surface body head for diagnostics
            raise RuntimeError(f"HTTP {resp.status_code} for {url}: {resp.text[:200]}")

        try:
            data = resp.json()
        except Exception:
            return None

        if mode == "chat":
            choices = data.get("choices") or []
            if not choices:
                return None
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
            delta = choices[0].get("delta") or {}
            dcontent = delta.get("content")
            return dcontent if isinstance(dcontent, str) and dcontent.strip() else None

        # completions
        choices = data.get("choices") or []
        if not choices:
            return None
        text = choices[0].get("text")
        return text if isinstance(text, str) and text.strip() else None
