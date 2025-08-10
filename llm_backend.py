# llm_backend.py
# thin, robust client for vLLM/OpenAI-compatible servers with JSON repair.
#
# key changes in this version
# ---------------------------
# - try chat completions first; fallback to legacy completions
# - optionally attempt openai "json mode" first; if the server returns empty content,
#   we gracefully fall back
# - aggressive json extraction + repair (fixes unescaped newlines in strings, smart quotes,
#   trailing commas, and strips code fences)
# - loud, actionable errors on http/schema failures
#
# this is intentionally minimal (yagni) and fails fast with clear messages.

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import requests


class LLMBackend:
    """minimal backend for text generation with optional json-first behavior.

    the `generate()` method returns a string. when a json object/array is present,
    it is extracted and repaired into valid json so that downstream `json.loads()`
    succeeds. otherwise, the raw text is returned.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        model: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 120.0,
        api_key: Optional[str] = None,
        prefer_json_mode: bool = True,  # try response_format json_object on chat route
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
        self._prefer_json_mode = prefer_json_mode

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
        """return text completion for a prompt; extract/repair json when found.

        tries chat api (optionally with json mode) → falls back to completions.
        """
        # first attempt: chat with optional json mode
        chat_payload: Dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._prefer_json_mode:
            chat_payload["response_format"] = {"type": "json_object"}
        if stop:
            chat_payload["stop"] = stop
        if seed is not None:
            chat_payload["seed"] = seed
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k == "response_format" and not self._prefer_json_mode:
                    # allow caller to force json mode off or custom formats
                    chat_payload[k] = v
                elif k != "response_format":
                    chat_payload[k] = v

        text = self._post_and_parse_text(self._chat, chat_payload, mode="chat")
        text = self._normalize_text(text)
        if text is not None:
            return text

        # second attempt: legacy completions (no json mode here)
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
            payload.update({k: v for k, v in extra.items() if k != "response_format"})

        text = self._post_and_parse_text(self._completions, payload, mode="completions")
        text = self._normalize_text(text)
        if text is not None:
            return text

        raise RuntimeError("LLMBackend.generate: no usable text from either route")

    # -------------------- http helpers --------------------

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
            body_head = resp.text[:500]
            raise RuntimeError(
                f"LLMBackend: HTTP {resp.status_code} for {url}; body head: {body_head}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLMBackend: non-JSON response from {url}: {e}") from e

        try:
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
                return (
                    dcontent if isinstance(dcontent, str) and dcontent.strip() else None
                )

            # completions
            choices = data.get("choices") or []
            if not choices:
                return None
            text = choices[0].get("text")
            return text if isinstance(text, str) and text.strip() else None

        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"LLMBackend: unexpected response schema from {url}: {data!r}"
            ) from e

    # -------------------- normalization & json repair --------------------

    def _normalize_text(self, text: Optional[str]) -> Optional[str]:
        if not isinstance(text, str):
            return None
        s = text.strip()
        if not s:
            return None

        # cut code fences if present
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
        if fence:
            s = fence.group(1).strip()

        # if it already parses as json, return as-is
        if self._json_ok(s):
            return s

        # try to isolate first balanced json object/array (string-aware)
        for candidate in self._extract_candidates(s):
            fixed = self._repair_json(candidate)
            if self._json_ok(fixed):
                return fixed

        # last chance: attempt repair on full string
        fixed = self._repair_json(s)
        if self._json_ok(fixed):
            return fixed

        # give the caller raw text; upstream may retry/repair further
        return s

    @staticmethod
    def _json_ok(s: str) -> bool:
        try:
            json.loads(s)
            return True
        except Exception:
            return False

    def _extract_candidates(self, s: str) -> List[str]:
        """yield plausible json substrings respecting quoted strings."""
        out: List[str] = []
        for opener, closer in (("{", "}"), ("[", "]")):
            start = s.find(opener)
            if start == -1:
                continue
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            out.append(s[start : i + 1])
                            break
        return out

    def _repair_json(self, s: str) -> str:
        """fix common json issues: smart quotes, unescaped newlines in strings, trailing commas."""
        # normalize quotes
        s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

        # strip leading/trailing junk backticks/boms
        s = s.strip().lstrip("`").rstrip("`").lstrip("\ufeff").rstrip("\ufeff")

        # remove trailing commas before } or ]
        s = re.sub(r",(\s*[}\]])", r"\1", s)

        # escape bare newlines that occur inside json strings
        s = self._escape_newlines_inside_strings(s)

        return s

    @staticmethod
    def _escape_newlines_inside_strings(s: str) -> str:
        out: List[str] = []
        in_str = False
        esc = False
        for ch in s:
            if in_str:
                if esc:
                    out.append(ch)
                    esc = False
                else:
                    if ch == "\\":
                        out.append(ch)
                        esc = True
                    elif ch == '"':
                        out.append(ch)
                        in_str = False
                    elif ch == "\n" or ch == "\r":
                        out.append("\\n")
                    else:
                        out.append(ch)
            else:
                if ch == '"':
                    out.append(ch)
                    in_str = True
                else:
                    out.append(ch)
        return "".join(out)
