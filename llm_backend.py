# llm_backend.py
# minimal openai-compatible client with optional json mode

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests


class LLMBackend:
    """thin client for an openai-compatible /v1/chat/completions endpoint.

    attributes:
        endpoint_url: base url, e.g., http://localhost:8000/v1
        model_name: model id
        api_key: optional bearer token
        session: requests session
        supports_json_mode: feature flag toggled at first request
    """

    def __init__(
        self, endpoint_url: str, model_name: str, api_key: Optional[str] = None
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", None)
        self.session = requests.Session()
        self.supports_json_mode: Optional[bool] = None  # unknown until first try

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.0,
        system: str = "you are a service that returns only json objects.",
    ) -> str:
        """return raw text content; prefers json mode if supported."""
        url = f"{self.endpoint_url}/v1/chat/completions"

        def _post(payload: Dict[str, Any]) -> requests.Response:
            return self.session.post(
                url, headers=self._headers(), data=json.dumps(payload), timeout=120
            )

        # try json mode on first attempt if unknown/true
        try_json_mode = True if self.supports_json_mode in (None, True) else False

        payload_base = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        # attempt json mode
        if try_json_mode:
            payload = dict(payload_base)
            payload["response_format"] = {"type": "json_object"}
            r = _post(payload)
            if r.status_code == 200:
                self.supports_json_mode = True
                data = r.json()
                return data["choices"][0]["message"]["content"]
            # feature not supported; mark and fall through
            self.supports_json_mode = False

        # plain text fallback
        r = _post(payload_base)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
