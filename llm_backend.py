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

        Accepts plain-text with control lines or a JSON blob, and returns a normalized dict:
          {
            "expert_id": str, "qid": str, "round_no": int,
            "text": str, "satisfied": bool,
            "revised_score": Optional[int], "handoff_to": Optional[str]
          }
        """
        expert_id = str(payload.get("expert_id") or "unknown_expert")
        specialty = str(payload.get("specialty") or "expert")
        qid = str(payload.get("qid") or "unknown_qid")
        round_no = int(payload.get("round_no") or 2)
        minority_view = str(payload.get("minority_view") or "")
        clinical_context = payload.get("clinical_context") or {}
        temperature = float(payload.get("temperature") or self.temperature_r2 or 0.3)

        # role hint & prior turns
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
            pass

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
            pass

        # optional case id for prompt
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
            pass

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
            "- If majority: directly rebut the minority’s strongest claims with specific evidence.\n"
            "- If minority (closing/followup): address rebuttals succinctly and clarify residual uncertainty.\n"
            "Avoid repetition. No lists, no markdown, no JSON. Return PLAIN TEXT only.\n\n"
            "At the VERY END, append EXACTLY these 3 control lines (no extra words):\n"
            "SATISFIED: yes|no\n"
            "REVISED_SCORE: 1-9|same\n"
            "HANDOFF: <expert_id>|none\n"
        )

        # optional dumps
        dump_root_env = os.getenv("DELPHEA_OUT_DIR")
        dump_base: Optional[Path] = Path(dump_root_env) if dump_root_env else None
        safe_case = self._safe_name(case_id or "unknown_case", "unknown_case")
        safe_eid = self._safe_name(expert_id, "unknown_expert")
        safe_qid = self._safe_name(qid, "Q")
        subdir = f"{safe_case}/experts/{safe_eid}/debate/{safe_qid}"
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

        # --- call model (force plain completion) ---
        try:
            text = self.generate(
                prompt,
                max_tokens=512,
                temperature=temperature,
                system="You are a clinical expert generating short, plain-text debate arguments. No markdown, no JSON.",
                prefer_json=False,
            )
        except Exception:
            text = "Unable to generate debate content."

        # --- normalize result, supporting accidental JSON replies ---
        satisfied: Optional[bool] = None
        revised_score: Optional[int] = None
        handoff_to: Optional[str] = None

        s = (text or "").strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                if isinstance(obj.get("text"), str) and obj["text"].strip():
                    s = obj["text"].strip()
                # pass through optional controls if present
                if isinstance(obj.get("satisfied"), bool):
                    satisfied = obj["satisfied"]
                rs = obj.get("revised_score", None)
                if isinstance(rs, int) and 1 <= rs <= 9:
                    revised_score = rs
                ht = obj.get("handoff_to", None)
                if isinstance(ht, str) and ht.strip().lower() not in {"", "none"}:
                    handoff_to = ht.strip()
            except Exception:
                # leave as original s
                pass

        # --- parse control lines from plain text (and strip them from body) ---
        lines = [ln.rstrip() for ln in s.splitlines()]
        body: List[str] = []
        for ln in lines:
            low = ln.strip().lower()
            if low.startswith("satisfied:"):
                val = low.split(":", 1)[1].strip()
                if val in {"yes", "no"}:
                    satisfied = val == "yes"
                continue
            if low.startswith("revised_score:"):
                val = low.split(":", 1)[1].strip()
                if val != "same":
                    try:
                        num = int(val)
                        if 1 <= num <= 9:
                            revised_score = num
                    except Exception:
                        pass
                continue
            if low.startswith("handoff:"):
                val = low.split(":", 1)[1].strip()
                if val and val.lower() not in {"none", "null", "-"}:
                    handoff_to = val
                else:
                    handoff_to = None
                continue
            body.append(ln)

        cleaned_text = "\n".join(body).strip()
        if not cleaned_text:
            cleaned_text = "No argument produced."

        if satisfied is None:
            satisfied = bool(len(cleaned_text) >= 20)

        result = {
            "expert_id": expert_id,
            "qid": qid,
            "round_no": round_no,
            "text": cleaned_text,
            "satisfied": bool(satisfied),
        }
        if revised_score is not None:
            result["revised_score"] = revised_score
        if handoff_to:
            result["handoff_to"] = handoff_to

        self._dump_json(dump_base, f"{subdir}/turn.json", result)
        return result
