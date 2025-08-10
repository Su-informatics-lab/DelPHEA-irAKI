# llm_backend.py
# minimal openai-compatible client with optional json mode; tolerant init

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests


class LLMBackend:
    """thin client for an openai-compatible /v1/chat/completions endpoint.

    accepts multiple aliases so callers can pass model/model_name and endpoint/endpoint_url/api_base.
    unknown kwargs are ignored (yagni-friendly).
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # accept common aliases without failing
        endpoint = kwargs.pop("endpoint", None)
        api_base = kwargs.pop("api_base", None)
        base_url = kwargs.pop("base_url", None)
        model = kwargs.pop("model", None)

        # prefer explicit args, then aliases, then env
        self.endpoint_url = (
            endpoint_url
            or endpoint
            or api_base
            or base_url
            or os.getenv("OPENAI_BASE_URL")
            or "http://localhost:8000"
        ).rstrip(
            "/"
        )  # noqa: E501
        self.model_name = (
            model_name or model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", None)

        # stash remaining kwargs in case future features want them (do not break)
        self._extra: Dict[str, Any] = kwargs

        self.session = requests.Session()
        self.supports_json_mode: Optional[bool] = None  # unknown until first try

    # ------------------------ internal helpers ------------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, payload: Dict[str, Any]) -> requests.Response:
        url = f"{self.endpoint_url}/v1/chat/completions"
        return self.session.post(
            url, headers=self._headers(), data=json.dumps(payload), timeout=120
        )

    # ----------------------------- api -------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.0,
        system: str = "you are a service that returns only json objects.",
    ) -> str:
        """return raw text content; prefers json mode if supported; falls back to plain."""
        payload_base = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        # try json mode first unless we know it's unsupported
        try_json_mode = True if self.supports_json_mode in (None, True) else False
        if try_json_mode:
            payload = dict(payload_base)
            payload["response_format"] = {"type": "json_object"}
            r = self._post(payload)
            if r.status_code == 200:
                self.supports_json_mode = True
                data = r.json()
                return data["choices"][0]["message"]["content"]
            # if the server doesn’t support json mode, fall through
            self.supports_json_mode = False

        # plain text fallback
        r = self._post(payload_base)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
