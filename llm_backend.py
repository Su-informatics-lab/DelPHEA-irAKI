# llm_backend.py
# minimal openai-compatible client with optional json mode; advertises capabilities
# so validators can size prompts for large context windows.

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests


class LLMBackend:
    """thin client for an OpenAI-compatible /v1/chat/completions endpoint.

    accepts multiple aliases so callers can pass model/model_name and endpoint/endpoint_url/api_base.
    unknown kwargs are ignored (yagni-friendly).
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        context_window: Optional[int] = None,
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
        ).rstrip("/")
        self.model_name = (
            model_name or model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # stash remaining kwargs in case future features want them (do not break)
        self._extra: Dict[str, Any] = kwargs

        self.session = requests.Session()

        # capability advertisement for validators
        env_ctx = os.getenv("CTX_WINDOW")
        self._context_window: int = (
            int(env_ctx) if (env_ctx and env_ctx.isdigit()) else 0
        )
        if context_window:
            self._context_window = int(context_window)
        self.supports_json_mode: Optional[bool] = None  # unknown until first try

        # optional per-round temps; safe for setattr from caller
        self.temperature_r1: Optional[float] = None
        self.temperature_r2: Optional[float] = None
        self.temperature_r3: Optional[float] = None

    # ------------------------ capabilities / tokenization ------------------------

    def set_context_window(self, tokens: int) -> None:
        """set the advertised context window (tokens) for validators/prompt sizing."""
        try:
            self._context_window = int(tokens)
        except Exception:
            # keep prior value if parse fails
            pass

    def capabilities(self) -> Dict[str, Any]:
        """Advertise backend limits so validators can size prompts correctly."""
        ctx = int(self._context_window) if self._context_window else 32768
        return {"context_window": ctx, "json_mode": bool(self.supports_json_mode)}

    def count_tokens(self, text: str) -> int:
        """approx token count used by validators; server-independent heuristic."""
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                return 1
        # ~4 chars per token heuristic
        return max(1, len(text) // 4)

    # ------------------------ internal helpers ------------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, payload: Dict[str, Any]) -> requests.Response:
        url = f"{self.endpoint_url}/v1/chat/completions"
        # generous timeout; tune per deployment as needed
        return self.session.post(
            url, headers=self._headers(), data=json.dumps(payload), timeout=180
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
            # if the server doesn’t support json mode (e.g., 400/404), fall through
            self.supports_json_mode = False

        # plain text fallback
        r = self._post(payload_base)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    # ----------------------------- debate api ------------------------------

    def debate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a single debate turn (used by Expert.debate).

        Input payload keys (expected):
          - expert_id: str
          - specialty: str
          - qid: str
          - round_no: int
          - clinical_context: dict
          - minority_view: str
          - temperature: float
        Returns:
          dict suitable for models.DebateTurn(**dict)
        """
        expert_id = payload.get("expert_id", "unknown_expert")
        specialty = payload.get("specialty", "expert")
        qid = payload.get("qid", "unknown_qid")
        round_no = int(payload.get("round_no", 2))
        minority_view = payload.get("minority_view", "")
        clinical_context = payload.get("clinical_context", {})
        temperature = float(payload.get("temperature", 0.3))

        system = (
            "You are a clinical expert participating in a structured, single-turn debate. "
            "Return ONLY one JSON object with your argument."
        )
        user = (
            f"Role: {specialty} ({expert_id})\n"
            f"Question ID: {qid}\n"
            f"Round: {round_no}\n\n"
            "Clinical context (JSON):\n"
            f"{json.dumps(clinical_context, ensure_ascii=False)}\n\n"
            "Minority view summary:\n"
            f"{minority_view}\n\n"
            "TASK: Write a concise, evidence-grounded argument (<=150 words) addressing the question.\n"
            "OUTPUT: Return ONLY a JSON object with fields:\n"
            "{\n"
            f'  "expert_id": "{expert_id}",\n'
            f'  "qid": "{qid}",\n'
            f'  "round_no": {round_no},\n'
            '  "content": "short argument text",\n'
            '  "stance": "support" | "oppose" | "nuanced",\n'
            '  "citations": ["optional strings"]\n'
            "}\n"
        )

        # try JSON-mode completion first
        try:
            text = self.generate(
                user, max_tokens=600, temperature=temperature, system=system
            )
            obj = self._safe_first_json(text)
            if isinstance(obj, dict):
                # ensure required identifiers present
                obj.setdefault("expert_id", expert_id)
                obj.setdefault("qid", qid)
                obj.setdefault("round_no", round_no)
                # compatibility: some schemas name the field "argument" instead of "content"
                if "argument" not in obj and "content" in obj:
                    obj["argument"] = obj["content"]
                if "content" not in obj and "argument" in obj:
                    obj["content"] = obj["argument"]
                if "citations" not in obj:
                    obj["citations"] = []
                return obj
        except Exception:
            # fall through to plain completion path below
            pass

        # fallback: ask plain completion and wrap it
        try:
            plain = self.generate(
                f"{user}\nReturn just the argument text (no JSON) if you cannot return JSON.",
                max_tokens=400,
                temperature=temperature,
                system="You are concise and factual.",
            )
        except Exception:
            plain = "Unable to generate debate content."

        return {
            "expert_id": expert_id,
            "qid": qid,
            "round_no": round_no,
            "content": plain.strip(),
            "argument": plain.strip(),  # compatibility with alternate schemas
            "citations": [],
            "stance": "nuanced",
        }

    # ----------------------------- utils ----------------------------------

    @staticmethod
    def _safe_first_json(text: str) -> Any:
        """Extract and parse the first JSON object in text; return dict on success or None."""
        if not isinstance(text, str):
            return None
        # fenced block first
        fence = text.find("```json")
        if fence == -1:
            fence = text.find("```JSON")
        if fence != -1:
            end = text.find("```", fence + 7)
            if end != -1:
                text = text[fence + 7 : end]

        # find first '{' and balance braces
        start = text.find("{")
        if start == -1:
            return None
        brace = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
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
                elif ch == "{":
                    brace += 1
                elif ch == "}":
                    brace -= 1
                    if brace == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except Exception:
                            return None
        return None
