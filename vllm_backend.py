# vllm_backend.py
# minimal openai-compatible client for a running vllm server.
# expects an openai-style /v1/chat/completions endpoint.
# env fallbacks: VLLM_BASE_URL, OPENAI_BASE_URL, VLLM_API_KEY/OPENAI_API_KEY, VLLM_MODEL/OPENAI_MODEL

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests

from llm_backend import LLMBackend
from schema import load_qids


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
        # base url like http://localhost:8000/v1
        self.base_url = (
            base_url
            or os.getenv("VLLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "http://localhost:8000/v1"
        ).rstrip("/")
        self.model = (
            model
            or os.getenv("VLLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        self.api_key = (
            api_key
            or os.getenv("VLLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "EMPTY"
        )
        self.timeout = timeout

    # ------------- public api -------------

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

    # ------------- transport -------------

    def _chat_json(self, system_and_user: List[Dict[str, str]]) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            # many vllm deployments ignore auth; include if configured
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": system_and_user,
            # response_format helps some model families stay json-only
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code >= 300:
            raise RuntimeError(
                f"vllm backend http {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"unexpected vllm payload: {data}") from e
        return _ensure_json_dict(content)


# ------------- prompt helpers (yagni) -------------


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
    # minimal, self-describing instructions that match our pydantic contracts
    user_lines = [
        f"specialty: {persona.get('specialty','unknown')}",
        f"phase: {phase}",
        "case: (summarized json below)",
        json.dumps(case)[:4000],  # keep concise
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


def _normalize_base(self):
    # strip trailing slash and any accidental /v1
    import re

    self.base_url = re.sub(r"/v1/?$", "", self.base_url.rstrip("/"))


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
    import json
    import re

    text = (text or "").strip()
    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # try ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # try first json object
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    raise ValueError("model did not return valid json; got snippet: " + text[:300])


def _post_json(self, url, payload):
    import requests

    headers = {"content-type": "application/json"}
    if getattr(self, "api_key", None):
        headers["authorization"] = f"Bearer {self.api_key}"
    resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
    return resp


def _detect_route(self):
    # detect once: prefer chat → responses → completions
    if getattr(self, "_route", None):
        return self._route
    self._normalize_base()
    base = self.base_url
    probes = [
        ("chat", f"{base}/v1/chat/completions", {"model": self.model, "messages": []}),
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
    raise RuntimeError("no compatible openai route found. tried:\n" + "\n".join(tried))


def _parse_openai_response(self, tag, obj):
    # unify text extraction for chat/responses/completions
    # returns plain text content
    if tag == "chat":
        # openai chat: choices[0].message.content
        return obj.get("choices", [{}])[0].get("message", {}).get("content", "")
    if tag == "comp":
        # legacy completions: choices[0].text
        return obj.get("choices", [{}])[0].get("text", "")
    # responses api
    # try openai-style 'output_text' first, then compact 'output' arrays
    if "output_text" in obj:
        return obj.get("output_text", "")
    out = obj.get("output", [])
    # vllm often returns [{"content":[{"type":"output_text","text": "..."}]}]
    try:
        for seg in out:
            content = seg.get("content", [])
            for c in content:
                if "text" in c:
                    return c["text"]
    except Exception:
        pass
    # last resort: some servers also echo choices like chat
    return obj.get("choices", [{}])[0].get("message", {}).get("content", "")


def _chat_json(self, messages):
    """
    send messages to the server using whichever openai-style route it supports,
    then parse the model's reply text and return it as a python dict (json).
    """
    tag, url = self._detect_route()

    if tag == "chat":
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "stream": False,
        }
    elif tag == "responses":
        payload = {
            "model": self.model,
            # responses api: safest is to pass a compact text input
            "input": self._messages_to_text(messages),
            "temperature": 0,
        }
    else:  # "comp"
        payload = {
            "model": self.model,
            "prompt": self._messages_to_text(messages),
            "temperature": 0,
        }

    r = self._post_json(url, payload)

    # retry once if a 404 slips through due to server reloads
    if r.status_code == 404:
        self._route = None
        tag, url = self._detect_route()
        r = self._post_json(url, payload)

    if r.status_code >= 400:
        raise RuntimeError(f"vllm backend http {r.status_code}: {r.text}")

    obj = r.json()
    text = self._parse_openai_response(tag, obj)
    return self._extract_json_obj(text)
