# llm_backend.py
# minimal openai-compatible client with optional json mode; advertises capabilities
# so validators can size prompts for large context windows.

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        prefer_json: Optional[bool] = None,
    ) -> str:
        """return raw text content; prefers json mode if supported; falls back to plain.

        prefer_json:
          - None (default): try JSON mode if the server supports it (back-compat).
          - True: force a JSON-mode attempt first.
          - False: skip JSON mode and use plain text.
        """
        payload_base = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        try_json_mode = (
            True
            if prefer_json is True
            else (
                False
                if prefer_json is False
                else (self.supports_json_mode in (None, True))
            )
        )

        # try json mode first unless explicitly disabled
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

    # ----------------------------- safe dumping (optional) ------------------------------

    @staticmethod
    def _safe_name(x: Any, default: str) -> str:
        s = str(x) if x is not None else default
        # allow letters, digits, _, -, .
        cleaned = "".join(ch for ch in s if ch.isalnum() or ch in "._-")
        return cleaned or default

    @staticmethod
    def _dump_write(base: Optional[Path], rel: str, content: str) -> None:
        if not base:
            return
        try:
            p = base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        except Exception:
            pass

    @staticmethod
    def _dump_json(base: Optional[Path], rel: str, obj: Any) -> None:
        if not base:
            return
        try:
            p = base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        except Exception:
            pass

    # ----------------------------- debate helpers ------------------------------

    @staticmethod
    def _format_peer_turns(
        peer_turns: Any, max_turns: int = 8, max_chars: int = 320
    ) -> str:
        """Render a compact, readable transcript snippet for inclusion in the prompt."""
        if not isinstance(peer_turns, list) or not peer_turns:
            return ""
        # keep last N in chronological order
        tail: List[Dict[str, Any]] = peer_turns[-max_turns:]
        lines: List[str] = []
        for t in tail:
            try:
                speaker = str(t.get("expert_id") or t.get("speaker", "expert"))
                role = str(t.get("speaker_role") or t.get("role") or "")
                txt = str(t.get("text") or "").strip().replace("\n", " ")
                if len(txt) > max_chars:
                    txt = txt[: max_chars - 1].rstrip() + "…"
                who = f"{speaker} ({role})" if role else speaker
                lines.append(f"- {who}: {txt}")
            except Exception:
                continue
        return "\n".join(lines)

    # ----------------------------- debate api ------------------------------

    def debate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a single debate turn (used by Expert.debate).

        Input payload keys (expected):
          - expert_id: str
          - specialty: str
          - qid: str
          - round_no: int
          - clinical_context: dict  (may include 'role' and 'peer_turns')
          - minority_view: str
          - temperature: float

        Returns:
          dict suitable for models.DebateTurn(**dict), i.e.
          {
            "expert_id": "...",
            "qid": "...",
            "round_no": 2,
            "text": "<plain argument>",
            "satisfied": true
          }
        """
        expert_id = str(payload.get("expert_id") or "unknown_expert")
        specialty = str(payload.get("specialty") or "expert")
        qid = str(payload.get("qid") or "unknown_qid")
        round_no = int(payload.get("round_no") or 2)
        minority_view = str(payload.get("minority_view") or "")
        clinical_context = payload.get("clinical_context") or {}
        temperature = float(payload.get("temperature") or self.temperature_r2 or 0.3)

        role_hint = ""
        try:
            r = (
                clinical_context.get("role")
                if isinstance(clinical_context, dict)
                else None
            )
            if r:
                role_hint = str(r)
        except Exception:
            role_hint = ""

        # peer history
        peer_snippet = ""
        try:
            peer_turns = (
                clinical_context.get("peer_turns")
                if isinstance(clinical_context, dict)
                else None
            )
            peer_snippet = self._format_peer_turns(
                peer_turns, max_turns=8, max_chars=320
            )
        except Exception:
            peer_snippet = ""

        # Best-effort case id (optional, for prompt context only)
        case_id = None
        try:
            case = (
                clinical_context.get("case")
                if isinstance(clinical_context, dict)
                else None
            )
            if isinstance(case, dict):
                for k in ("case_id", "id", "patient_id", "person_id"):
                    v = case.get(k)
                    if v:
                        case_id = str(v)
                        break
        except Exception:
            case_id = None

        header = f"Debate turn for question {qid} (round {round_no})."
        role = f"Role: {expert_id} ({specialty})."
        stage = f"Turn type: {role_hint or 'participant'}."
        cid = f"Case: {case_id or '(unspecified)'}."
        mv = (
            f"Minority view summary:\n{minority_view}"
            if minority_view
            else "Minority view: (not provided)."
        )
        hist = (
            f"\nPrior turns (most recent last):\n{peer_snippet}" if peer_snippet else ""
        )

        prompt = (
            f"{header}\n{role}\n{stage}\n{cid}\n\n"
            f"{mv}{hist}\n\n"
            "Write a concise, evidence-grounded argument (≈120–180 words) that advances the discussion.\n"
            "- If you are majority: directly rebut the minority’s strongest claims with specific evidence.\n"
            "- If you are minority (closing): address key rebuttals succinctly and clarify residual uncertainty.\n"
            "Avoid repetition. No lists, no markdown, no JSON. Return PLAIN TEXT only."
        )

        # optional dump root
        dump_root_env = os.getenv("DELPHEA_OUT_DIR")
        dump_base: Optional[Path] = Path(dump_root_env) if dump_root_env else None
        safe_case = self._safe_name(case_id or "unknown_case", "unknown_case")
        safe_eid = self._safe_name(expert_id, "unknown_expert")
        safe_qid = self._safe_name(qid, "Q")
        subdir = f"{safe_case}/experts/{safe_eid}/debate/{safe_qid}"

        # dump prompt + meta (best-effort, never raise)
        self._dump_write(dump_base, f"{subdir}/prompt.txt", prompt)
        self._dump_json(
            dump_base,
            f"{subdir}/meta.json",
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "model_name": self.model_name,
                "temperature": temperature,
                "round_no": round_no,
                "expert_id": expert_id,
                "qid": qid,
                "case_id": case_id,
                "minority_view_len": len(minority_view),
                "role": role_hint or "participant",
                "peer_turn_count": len(peer_snippet.splitlines())
                if peer_snippet
                else 0,
            },
        )

        try:
            text = self.generate(
                prompt,
                max_tokens=512,
                temperature=temperature,
                system="You are a clinical expert generating short, plain-text debate arguments. No markdown, no JSON.",
                prefer_json=False,  # critical: force plain completion
            )
        except Exception:
            text = "Unable to generate debate content."

        # normalize & guard against None, and salvage if JSON slipped through
        try:
            s = (text or "").strip()
        except Exception:
            s = "Unable to generate debate content."
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                for key in ("text", "content", "argument", "message"):
                    if isinstance(obj.get(key), str) and obj.get(key).strip():
                        s = obj[key].strip()
                        break
                else:
                    s = json.dumps(obj, ensure_ascii=False)
            except Exception:
                pass

        if not s:
            s = "No argument produced."

        result = {
            "expert_id": expert_id,
            "qid": qid,
            "round_no": round_no,
            "text": s,
            "satisfied": True,
        }

        # dump turn (best-effort)
        self._dump_json(dump_base, f"{subdir}/turn.json", result)

        return result
