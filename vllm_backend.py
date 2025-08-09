# vllm_backend.py
# minimal openai-compatible client for a running vllm server.
# robust to /v1/chat/completions vs /v1/responses vs /v1/completions,
# retries once with a self-repair prompt if the model returns non-json,
# and sanitizes fields to satisfy downstream validation.

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import requests

from llm_backend import LLMBackend
from schema import load_qids

# ---------- prompt builders (schema-aligned, concise) ----------


def _build_round_prompt(
    phase: str,
    persona: Dict[str, Any],
    case: Dict[str, Any],
    qids: List[str],
    debate_context: Dict[str, Any] | None = None,
) -> List[Dict[str, str]]:
    """compose system/user messages for round1/round3."""
    system = (
        "you are a clinical expert. respond in strict json only. "
        "do not include any text outside a single valid json object."
    )
    user_lines = [
        f"specialty: {persona.get('specialty','unknown')}",
        f"phase: {phase}",
        "case json:",
        json.dumps(case)[:4000],
        "question_ids (answer each):",
        ", ".join(qids),
    ]
    if phase == "round1":
        schema = {
            "scores": {qid: "int 1..9" for qid in qids},
            "evidence": {qid: "string" for qid in qids},
            "clinical_reasoning": "string",
            "p_iraki": "float 0..1",
            "ci_iraki": ["float 0..1", "float 0..1"],
            "confidence": "float 0..1",
            "differential_diagnosis": ["string"],
            "primary_diagnosis": "string",
        }
    else:
        user_lines += ["debate_context:", json.dumps(debate_context or {})[:2000]]
        schema = {
            "scores": {qid: "int 1..9" for qid in qids},
            "evidence": {qid: "string" for qid in qids},
            "p_iraki": "float 0..1",
            "ci_iraki": ["float 0..1", "float 0..1"],
            "confidence": "float 0..1",
            "changes_from_round1": {"summary": "string"},
            "debate_influence": "string",
            "verdict": "boolean",
            "final_diagnosis": "string",
            "confidence_in_verdict": "float 0..1",
            "recommendations": ["string"],
        }
    user_lines += [
        "output schema (return exactly one json object using these keys; use null/empty if unsure):",
        json.dumps(schema),
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def _build_debate_prompt(
    persona: Dict[str, Any],
    qid: str,
    clinical_context: Dict[str, Any],
    minority_view: str,
) -> List[Dict[str, str]]:
    """compose messages for debate turn."""
    system = (
        "you are a clinical expert debating a disputed question. "
        "respond in strict json only (single json object)."
    )
    user_lines = [
        f"specialty: {persona.get('specialty','unknown')}",
        f"qid: {qid}",
        "clinical_context:",
        json.dumps(clinical_context)[:3000],
        "minority_view_summary:",
        minority_view[:2000],
        "output schema:",
        json.dumps({"text": "string", "citations": ["string"], "satisfied": "boolean"}),
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


class VLLMBackend(LLMBackend):
    """openai-compatible transport with route detection, retry, and sanitization."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        # normalize base url and pick defaults from env
        self.base_url = (
            base_url
            or os.getenv("VLLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "http://127.0.0.1:8000"
        )
        self.model = (
            model
            or os.getenv("VLLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "openai/gpt-oss-120b"
        )
        self.api_key = (
            api_key or os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or None
        )
        self.timeout = timeout
        self._route: tuple[str, str] | None = None  # (tag, url)
        self._normalize_base()

    # ---------------- public api ----------------

    def assess_round1(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
        qids = load_qids(expert_ctx["questionnaire_path"])
        messages = _build_round_prompt(
            phase="round1",
            persona=expert_ctx["persona"],
            case=expert_ctx["case"],
            qids=qids,
        )
        raw = self._chat_json_with_retry(messages, max_attempts=3)
        return self._sanitize_output(raw, phase="round1", qids=qids)

    def assess_round3(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
        qids = load_qids(expert_ctx["questionnaire_path"])
        messages = _build_round_prompt(
            phase="round3",
            persona=expert_ctx["persona"],
            case=expert_ctx["case"],
            qids=qids,
            debate_context=expert_ctx.get("debate_context", {}),
        )
        raw = self._chat_json_with_retry(messages, max_attempts=3)
        return self._sanitize_output(raw, phase="round3", qids=qids)

    def debate(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
        messages = _build_debate_prompt(
            persona=expert_ctx["persona"],
            qid=expert_ctx["qid"],
            clinical_context=expert_ctx.get("clinical_context", {}),
            minority_view=expert_ctx.get("minority_view", ""),
        )
        raw = self._chat_json_with_retry(messages, max_attempts=3)
        # debate schema is simple; still sanitize string/boolean shapes if needed
        if not isinstance(raw, dict):
            raise ValueError("debate: expected json object")
        if "text" in raw and raw["text"] is None:
            raw["text"] = ""
        if "citations" in raw and not isinstance(raw["citations"], list):
            raw["citations"] = []
        if "satisfied" in raw and not isinstance(raw["satisfied"], bool):
            raw["satisfied"] = False
        return raw

    # ---------------- helpers: io plumbing ----------------

    def _normalize_base(self):
        # strip trailing slash and any accidental /v1
        self.base_url = re.sub(r"/v1/?$", "", self.base_url.rstrip("/"))

    def _post_json(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return requests.post(url, headers=headers, json=payload, timeout=self.timeout)

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        # join chat messages into a plain prompt for legacy routes
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}".strip())
        parts.append("assistant:")
        return "\n".join(parts)

    def _extract_json_obj(self, text: str) -> Dict[str, Any]:
        # try raw json
        try:
            obj = json.loads((text or "").strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # try fenced block
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # try first {...}
        m = re.search(r"(\{.*\})", text or "", flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        raise ValueError(
            "model did not return valid json; snippet: " + (text or "")[:300]
        )

    def _detect_route(self) -> tuple[str, str]:
        # detect once: prefer chat → responses → completions
        if self._route is not None:
            return self._route
        base = self.base_url
        probes: List[tuple[str, str, Dict[str, Any]]] = [
            (
                "chat",
                f"{base}/v1/chat/completions",
                {"model": self.model, "messages": []},
            ),
            ("responses", f"{base}/v1/responses", {"model": self.model, "input": ""}),
            ("comp", f"{base}/v1/completions", {"model": self.model, "prompt": ""}),
        ]
        tried: List[str] = []
        for tag, url, body in probes:
            try:
                r = self._post_json(url, body)
            except Exception as e:
                tried.append(f"{url} -> {e}")
                continue
            if r.status_code == 404:
                tried.append(f"{url} -> 404")
                continue
            # any non-404 means route exists (400 is expected for empty body)
            self._route = (tag, url)
            return self._route
        raise RuntimeError(
            "no compatible openai route found. tried:\n" + "\n".join(tried)
        )

    def _parse_openai_response(self, tag: str, obj: Dict[str, Any]) -> str:
        # unify text extraction for chat/responses/completions
        if tag == "chat":
            return obj.get("choices", [{}])[0].get("message", {}).get("content", "")
        if tag == "comp":
            return obj.get("choices", [{}])[0].get("text", "")
        # responses api
        if "output_text" in obj:
            return obj.get("output_text", "")
        out = obj.get("output", [])
        try:
            for seg in out:
                for c in seg.get("content", []):
                    if "text" in c:
                        return c["text"]
        except Exception:
            pass
        # some servers also echo chat-style choices
        return obj.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _chat_once(
        self, tag: str, url: str, messages: List[Dict[str, str]]
    ) -> requests.Response:
        # build payload per route; only chat supports response_format
        if tag == "chat":
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "stream": False,
                "response_format": {"type": "json_object"},
            }
        elif tag == "responses":
            payload = {
                "model": self.model,
                "input": self._messages_to_text(messages),
                "temperature": 0,
            }
        else:  # "comp"
            payload = {
                "model": self.model,
                "prompt": self._messages_to_text(messages),
                "temperature": 0,
            }
        return self._post_json(url, payload)

    def _repair_messages(
        self, original: List[Dict[str, str]], error_msg: str, bad_text: str
    ) -> List[Dict[str, str]]:
        # append a terse correction turn, preserving original context
        repair = {
            "role": "user",
            "content": (
                "your previous reply was not a single valid json object.\n"
                f"error: {error_msg}\n"
                "return the json object now, using the exact keys previously requested. "
                "do not include any explanations, markdown, or extra text."
            ),
        }
        # we do not include the bad_text to avoid the model copying it back
        return [*original, repair]

    def _chat_json_with_retry(
        self, messages: List[Dict[str, str]], max_attempts: int = 3
    ) -> Dict[str, Any]:
        tag, url = self._detect_route()
        last_err = None
        msgs = list(messages)
        for attempt in range(1, max_attempts + 1):
            r = self._chat_once(tag, url, msgs)

            if r.status_code == 404:
                # route may have changed due to server reload
                self._route = None
                tag, url = self._detect_route()
                r = self._chat_once(tag, url, msgs)

            if r.status_code >= 400:
                raise RuntimeError(f"vllm backend http {r.status_code}: {r.text[:500]}")

            obj = r.json()
            text = self._parse_openai_response(tag, obj)
            try:
                return self._extract_json_obj(text)
            except Exception as e:
                last_err = e
                if attempt == max_attempts:
                    raise
                # add a short, explicit repair turn and try again
                msgs = self._repair_messages(messages, str(e), text)
        # should not reach here
        if last_err:
            raise last_err
        raise RuntimeError("unexpected empty retry loop")

    # ---------------- sanitization (tolerate minor drift) ----------------

    def _sanitize_output(
        self, obj: Dict[str, Any], *, phase: str, qids: List[str]
    ) -> Dict[str, Any]:
        # float coercion
        def _f(x, default=None):
            try:
                return float(x)
            except Exception:
                return default

        def _clip01(x, default=0.5):
            v = _f(x, default)
            if v is None:
                v = default
            return 0.0 if v < 0 else 1.0 if v > 1 else v

        # p_iraki and ci_iraki
        p = _clip01(obj.get("p_iraki"), 0.5)
        ci = obj.get("ci_iraki") or []
        if not isinstance(ci, (list, tuple)) or len(ci) != 2:
            lo, hi = max(0.0, p - 0.2), min(1.0, p + 0.2)
        else:
            lo, hi = _f(ci[0], 0.0), _f(ci[1], 1.0)
            if lo is None:
                lo = 0.0
            if hi is None:
                hi = 1.0
            if lo > hi:
                lo, hi = hi, lo
            lo, hi = max(0.0, lo), min(1.0, hi)
        # ensure p lies within [lo, hi]
        if p < lo:
            lo = p
        if p > hi:
            hi = p
        if (hi - lo) < 1e-6:
            eps = 0.05
            lo, hi = max(0.0, p - eps), min(1.0, p + eps)

        obj["p_iraki"] = p
        obj["ci_iraki"] = [lo, hi]

        # normalize confidences
        if "confidence" in obj:
            obj["confidence"] = _clip01(obj["confidence"], 0.5)
        if "confidence_in_verdict" in obj:
            obj["confidence_in_verdict"] = _clip01(obj["confidence_in_verdict"], 0.5)

        # scores 1..9 for all qids
        scores = obj.get("scores") if isinstance(obj.get("scores"), dict) else {}
        fixed_scores = {}
        for q in qids:
            v = scores.get(q, 5)
            try:
                v = int(round(float(v)))
            except Exception:
                v = 5
            v = 1 if v < 1 else 9 if v > 9 else v
            fixed_scores[q] = v
        obj["scores"] = fixed_scores

        # evidence strings
        if "evidence" in obj and isinstance(obj["evidence"], dict):
            obj["evidence"] = {
                k: ("" if v is None else str(v)) for k, v in obj["evidence"].items()
            }

        return obj
