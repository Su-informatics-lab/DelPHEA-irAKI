# vllm_backend.py
# minimal openai-compatible client for a running vllm server.
# expects an openai-style /v1/chat/completions endpoint.
# env fallbacks: VLLM_BASE_URL, OPENAI_BASE_URL, VLLM_API_KEY/OPENAI_API_KEY, VLLM_MODEL/OPENAI_MODEL

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

import requests

from llm_backend import LLMBackend
from schema import load_qids


def _build_round_prompt(
    phase: str,
    persona: Dict[str, Any],
    case: Dict[str, Any],
    qids: list[str],
    debate_context: Dict[str, Any] | None = None,
) -> List[Dict[str, str]]:
    system = (
        "you are a clinical expert. respond in strict json only. "
        "do not include any text outside valid json."
    )
    user_lines = [
        f"specialty: {persona.get('specialty','unknown')}",
        f"phase: {phase}",
        "case: (summarized json below)",
        json.dumps(case)[:4000],
        "",
        "question_ids (answer for each id):",
        ", ".join(qids),
        "",
    ]
    if phase == "round1":
        user_lines += [
            "output schema:",
            json.dumps(
                {
                    "scores": {qid: "int 1..9" for qid in qids},
                    "evidence": {qid: "string" for qid in qids},
                    "clinical_reasoning": "string",
                    "p_iraki": "float 0..1",
                    "ci_iraki": [0.1, 0.9],
                    "confidence": 0.7,
                    "differential_diagnosis": ["string", "string"],
                    "primary_diagnosis": "string",
                }
            ),
        ]
    else:
        user_lines += [
            "debate context:",
            json.dumps(debate_context or {})[:2000],
            "output schema:",
            json.dumps(
                {
                    "scores": {qid: "int 1..9" for qid in qids},
                    "evidence": {qid: "string" for qid in qids},
                    "p_iraki": 0.5,
                    "ci_iraki": [0.2, 0.8],
                    "confidence": 0.7,
                    "changes_from_round1": {"summary": "string"},
                    "debate_influence": "string",
                    "verdict": True,
                    "final_diagnosis": "string",
                    "confidence_in_verdict": 0.7,
                    "recommendations": ["string"],
                }
            ),
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
    system = (
        "you are a clinical expert debating a disputed question. "
        "respond in strict json only."
    )
    user_lines = [
        f"specialty: {persona.get('specialty','unknown')}",
        f"qid: {qid}",
        "clinical context:",
        json.dumps(clinical_context)[:3000],
        "minority view summary:",
        minority_view[:2000],
        "output schema:",
        json.dumps({"text": "string", "citations": ["string"], "satisfied": True}),
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def _ensure_json_dict(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("top-level json is not an object")
        return obj
    except Exception as e:
        raise ValueError(
            f"backend returned non-json or malformed json: {e}\ncontent: {s[:300]}"
        ) from e


class VLLMBackend(LLMBackend):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
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
        self._route = None  # cached (tag, url)
        self._normalize_base()

    # ---------------- public api ----------------

    def assess_round1(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
        qids = load_qids(expert_ctx["questionnaire_path"])
        content = _build_round_prompt(
            phase="round1",
            persona=expert_ctx["persona"],
            case=expert_ctx["case"],
            qids=qids,
        )
        return self._chat_json(content)

    def assess_round3(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
        qids = load_qids(expert_ctx["questionnaire_path"])
        content = _build_round_prompt(
            phase="round3",
            persona=expert_ctx["persona"],
            case=expert_ctx["case"],
            qids=qids,
            debate_context=expert_ctx.get("debate_context", {}),
        )
        return self._chat_json(content)

    def debate(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
        content = _build_debate_prompt(
            persona=expert_ctx["persona"],
            qid=expert_ctx["qid"],
            clinical_context=expert_ctx.get("clinical_context", {}),
            minority_view=expert_ctx.get("minority_view", ""),
        )
        return self._chat_json(content)

    # ---------------- helpers ----------------

    def _normalize_base(self):
        # strip trailing slash and any accidental /v1
        self.base_url = re.sub(r"/v1/?$", "", self.base_url.rstrip("/"))

    def _post_json(self, url, payload):
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return requests.post(url, headers=headers, json=payload, timeout=self.timeout)

    def _messages_to_text(self, messages):
        # join chat messages into a plain prompt for legacy routes
        if isinstance(messages, list):
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role}: {content}".strip())
            parts.append("assistant:")
            return "\n".join(parts)
        return str(messages)

    def _extract_json_obj(self, text):
        # be forgiving: try raw json, then fenced blocks, then first {...}
        text = (text or "").strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        m = re.search(r"(\{.*\})", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        raise ValueError("model did not return valid json; got snippet: " + text[:300])

    def _detect_route(self):
        # detect once: prefer chat → responses → completions
        if self._route is not None:
            return self._route
        base = self.base_url
        probes = [
            (
                "chat",
                f"{base}/v1/chat/completions",
                {"model": self.model, "messages": []},
            ),
            ("responses", f"{base}/v1/responses", {"model": self.model, "input": ""}),
            ("comp", f"{base}/v1/completions", {"model": self.model, "prompt": ""}),
        ]
        tried = []
        for tag, url, body in probes:
            try:
                r = self._post_json(url, body)
            except Exception as e:
                tried.append(f"{url} -> {e}")
                continue
            if r.status_code == 404:
                tried.append(f"{url} -> 404")
                continue
            # any non-404 means the route exists (400 is fine for empty body)
            self._route = (tag, url)
            return self._route
        raise RuntimeE
