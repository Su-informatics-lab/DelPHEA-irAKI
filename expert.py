"""
expert agent: specialty-conditioned assessor that emits strictly validated json.

- Round 1: independent assessment (strict schema)
- Debate (Round 2): single-turn producer with optional control lines parsed upstream
- Round 3: independent re-assessment WITH a compact recap of ONLY THIS EXPERT'S prior
  debate turns (not the whole transcript), so the agent "remembers" what they said
  without leaking other panelists' content.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_backend import LLMBackend
from models import AssessmentR1, AssessmentR3, DebateTurn
from prompts.rounds import format_round1_prompt, format_round3_prompt
from schema import load_qids
from validators import call_llm_with_schema


class Expert:
    """specialty-conditioned assessor with strict schema validation."""

    # recap display knobs for R3 prompts (kept small to avoid context bloat)
    _RECAP_MAX_QIDS = 6
    _RECAP_MAX_TURNS_PER_QID = 3
    _RECAP_MAX_CHARS = 240

    def __init__(
        self,
        expert_id: str,
        specialty: str,
        persona: Dict[str, Any],
        backend: LLMBackend,
        prompts_path: str = "prompts/expert_prompts.json",
        *,
        temperature_r1: float = 0.0,
        temperature_r2: float = 0.3,
        temperature_r3: float = 0.0,
        dump_root: Optional[str | Path] = None,  # optional: where to dump io traces
    ) -> None:
        # validate
        if not expert_id:
            raise ValueError("expert_id cannot be empty")
        if backend is None:
            raise ValueError("backend cannot be None")

        self.expert_id = expert_id
        self.specialty = specialty
        self.persona = persona or {}
        self.backend = backend
        self.prompts_path = prompts_path

        # per-round temperatures (cli-controlled via delphea_iraki.py)
        self.temperature_r1 = float(temperature_r1)
        self.temperature_r2 = float(temperature_r2)  # debate
        self.temperature_r3 = float(temperature_r3)

        # optional io dump root (env fallback)
        env_root = os.getenv("DELPHEA_OUT_DIR")
        self._dump_root: Optional[Path] = (
            Path(dump_root) if dump_root else (Path(env_root) if env_root else None)
        )

        # logger
        self.logger = logging.getLogger(f"expert.{self.expert_id}")

    # --------- round 1 / round 3 ---------

    def assess_round1(
        self,
        case: Dict[str, Any],
        questionnaire_path: str,
        repair_hint: Optional[str] = None,  # reserved for future use
    ) -> AssessmentR1:
        """produce a round-1 assessment using strict pydantic validation at the boundary."""
        # validate questionnaire and capture expected qids (fail loud if malformed/missing)
        expected_qids: List[str] = load_qids(questionnaire_path)

        info = self._extract_case_strings(case)
        prompt_text = format_round1_prompt(
            expert_name=self._expert_name(),
            expert_id=self.expert_id,  # accepted by formatter via **_ sink
            specialty=self.specialty,
            case_id=info["case_id"],
            demographics=info["demographics"],
            clinical_notes=info["clinical_notes"],
            qpath=questionnaire_path,
        )
        # require identifiers; per-question dicts contract is enforced by validators
        prompt_text += (
            "\n\nIMPORTANT: in your JSON output, include these exact keys for identification:\n"
            '- "case_id": "{case_id}"\n'
            '- "expert_id": "{expert_id}"\n'
            "return ONLY valid JSON."
        ).format(case_id=info["case_id"], expert_id=self.expert_id)

        # dump prompt (optional)
        self._dump_text(
            case_id=info["case_id"],
            relpath=f"experts/{self.expert_id}/r1.prompt.txt",
            text=prompt_text,
        )

        reply = call_llm_with_schema(
            response_model=AssessmentR1,
            prompt_text=prompt_text,
            backend=self.backend,
            temperature=self.temperature_r1,
            expected_qids=expected_qids,
        )

        # harden: inject identifiers if the model omitted them
        if hasattr(reply, "model_dump"):
            d = reply.model_dump()
            d.setdefault("case_id", info["case_id"])
            d.setdefault("expert_id", self.expert_id)
            reply = AssessmentR1(**d)

        # dump validated output (optional)
        self._dump_json(
            case_id=info["case_id"],
            relpath=f"experts/{self.expert_id}/r1.output.json",
            obj=reply.model_dump(),
        )

        self._log_preview(reply, "r1-validated")
        return reply  # already a validated AssessmentR1

    def assess_round3(
        self,
        case: Dict[str, Any],
        questionnaire_path: str,
        debate_context: Dict[str, Any],
        repair_hint: Optional[str] = None,  # reserved for future use
    ) -> AssessmentR3:
        """produce a round-3 reassessment using strict pydantic validation at the boundary.

        The prompt includes a compact recap of ONLY THIS EXPERT'S prior debate turns
        (last few per QID), not the full transcript. This respects the "experienced-only"
        recall requirement without leaking other panelists' content.
        """
        expected_qids: List[str] = load_qids(questionnaire_path)

        info = self._extract_case_strings(case)
        recap_text = self._my_debate_recap(
            self.expert_id,
            debate_context or {},
            max_qids=self._RECAP_MAX_QIDS,
            max_turns=self._RECAP_MAX_TURNS_PER_QID,
            max_chars=self._RECAP_MAX_CHARS,
        )

        prompt_text = format_round3_prompt(
            expert_name=self._expert_name(),
            expert_id=self.expert_id,  # accepted by formatter via **_ sink
            specialty=self.specialty,
            case_id=info["case_id"],
            demographics=info["demographics"],
            clinical_notes=info["clinical_notes"],
            qpath=questionnaire_path,
            peer_feedback_summary=recap_text,  # ← per-expert recap only
        )
        # require identifiers again for consistency
        prompt_text += (
            "\n\nIMPORTANT: in your JSON output, include these exact keys for identification:\n"
            '- "case_id": "{case_id}"\n'
            '- "expert_id": "{expert_id}"\n'
            "return ONLY valid JSON."
        ).format(case_id=info["case_id"], expert_id=self.expert_id)

        # dump prompt & r3 context (optional)
        self._dump_text(
            case_id=info["case_id"],
            relpath=f"experts/{self.expert_id}/r3.prompt.txt",
            text=prompt_text,
        )
        # we also dump the personal recap to make audits easy
        if recap_text:
            self._dump_text(
                case_id=info["case_id"],
                relpath=f"experts/{self.expert_id}/r3.recap.txt",
                text=recap_text,
            )

        reply = call_llm_with_schema(
            response_model=AssessmentR3,
            prompt_text=prompt_text,
            backend=self.backend,
            temperature=self.temperature_r3,
            expected_qids=expected_qids,  # strict enforcement of qid set and order
        )

        # harden: inject identifiers if the model omitted them
        if hasattr(reply, "model_dump"):
            d = reply.model_dump()
            d.setdefault("case_id", info["case_id"])
            d.setdefault("expert_id", self.expert_id)
            reply = AssessmentR3(**d)

        # dump validated output (optional)
        self._dump_json(
            case_id=info["case_id"],
            relpath=f"experts/{self.expert_id}/r3.output.json",
            obj=reply.model_dump(),
        )

        self._log_preview(reply, "r3-validated")
        return reply  # already a validated AssessmentR3

    # --------- debate (round 2 single turn) ---------

    def debate(
        self,
        qid: str,
        round_no: int,
        clinical_context: Dict[str, Any],
        minority_view: str,
    ) -> DebateTurn:
        """produce a single debate turn for a specific question.

        note: if your backend supports passing temperature for debate, it can
        honor 'temperature' in the payload below (r2 temperature).
        """
        if not qid:
            raise ValueError("debate requires a non-empty qid")
        if round_no not in (1, 2, 3):
            raise ValueError(f"unsupported round_no for debate: {round_no}")

        # best effort case_id extraction from clinical_context
        case_id = self._extract_case_id_from_context(clinical_context) or "unknown_case"

        payload = {
            "expert_id": self.expert_id,
            "specialty": self.specialty,
            "qid": qid,
            "round_no": round_no,
            "clinical_context": clinical_context,
            "minority_view": minority_view,
            "temperature": self.temperature_r2,  # r2 temperature
        }

        # dump the request we’re sending to the backend (optional)
        self._dump_json(
            case_id=case_id,
            relpath=f"experts/{self.expert_id}/debate/{qid}.request.json",
            obj=payload,
        )

        raw = self.backend.debate(payload)
        self._log_preview(raw, f"[debate {qid} raw]")

        # normalize backend reply into DebateTurn schema
        turn = self._coerce_debate_turn(raw, qid=qid, round_no=round_no)

        # dump the normalized/validated turn (optional)
        self._dump_json(
            case_id=case_id,
            relpath=f"experts/{self.expert_id}/debate/{qid}.turn.json",
            obj=turn.model_dump(),
        )

        return turn

    # --------- helpers ---------

    def _my_debate_recap(
        self,
        expert_id: str,
        debate_ctx: Dict[str, Any],
        *,
        max_qids: int,
        max_turns: int,
        max_chars: int,
    ) -> Optional[str]:
        """Return a compact recap of ONLY this expert's own debate turns.

        Format example:
          Q5:
          - minority turn 0: <snippet> (revised_score=7)
          - participant turn 3: <snippet> (revised_score=same)
        """
        tx = (debate_ctx or {}).get("transcripts", {}) or {}
        if not isinstance(tx, dict) or not tx:
            return None

        # stable traversal by QID name
        qids = sorted(tx.keys())
        lines: List[str] = []
        shown_qids = 0
        for qid in qids:
            if shown_qids >= max_qids:
                break
            turns = tx.get(qid) or []
            mine = [t for t in turns if str(t.get("expert_id")) == expert_id]
            if not mine:
                continue
            lines.append(f"{qid}:")
            for t in mine[-max_turns:]:
                role = (t.get("speaker_role") or "participant").strip()
                idx = t.get("turn_index", 0)
                rs = t.get("revised_score", "same")
                txt = (t.get("text") or t.get("raw") or "").strip().replace("\n", " ")
                if len(txt) > max_chars:
                    txt = txt[: max_chars - 1].rstrip() + "…"
                lines.append(f"- {role} turn {idx}: {txt} (revised_score={rs})")
            shown_qids += 1

        rec = "\n".join(lines).strip()
        return rec or None

    def _coerce_debate_turn(self, raw: Any, *, qid: str, round_no: int) -> DebateTurn:
        """Coerce backend reply into DebateTurn fields.

        Guarantees:
          - expert_id, qid, round_no, text, satisfied
          - citations: list[str]
          - revised_score: Optional[int] in [1,9]
          - handoff_to: Optional[str]
        """
        # accept dict, pydantic-like, or json string
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        elif isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {"text": raw}

        if not isinstance(raw, dict):
            raw = {}

        # choose best text field
        text = (
            (raw.get("text") if isinstance(raw.get("text"), str) else None)
            or (raw.get("content") if isinstance(raw.get("content"), str) else None)
            or (raw.get("argument") if isinstance(raw.get("argument"), str) else None)
            or ""
        ).strip()

        # satisfied: prefer explicit bool; else infer by length
        sat = raw.get("satisfied")
        if not isinstance(sat, bool):
            sat = bool(len(text) >= 20)

        # citations: normalize to list[str]
        citations = raw.get("citations")
        if not isinstance(citations, list):
            citations = []
        citations = [str(x) for x in citations]

        # revised_score (optional int 1..9)
        rs = raw.get("revised_score")
        if isinstance(rs, int) and 1 <= rs <= 9:
            revised_score = rs
        else:
            revised_score = None

        # handoff_to (optional str)
        ht = raw.get("handoff_to")
        handoff_to = None
        if (
            isinstance(ht, str)
            and ht.strip()
            and ht.strip().lower() not in {"none", "null", "-"}
        ):
            handoff_to = ht.strip()

        base = {
            "expert_id": self.expert_id,
            "qid": qid,
            "round_no": int(round_no),
            "text": text if text else "[auto-repair] debate content missing.",
            "satisfied": bool(sat),
            "citations": citations,
        }
        if revised_score is not None:
            base["revised_score"] = revised_score
        if handoff_to:
            base["handoff_to"] = handoff_to

        return DebateTurn(**base)

    def _expert_name(self) -> str:
        """derive display name for prompts."""
        name = self.persona.get("name") or self.persona.get("display_name") or ""
        return (
            name.strip() if isinstance(name, str) and name.strip() else self.expert_id
        )

    def _flatten_dict_lines(
        self, d: Dict[str, Any], allow_keys: Optional[List[str]] = None
    ) -> str:
        """turn a small dict into stable 'k: v' lines; optionally filter keys."""
        if not isinstance(d, dict):
            return str(d)
        items = d.items()
        if allow_keys:
            allow = set(allow_keys)
            items = ((k, v) for k, v in d.items() if k in allow)
        # ensure stable ordering
        return "\n".join(
            f"{k}: {v}"
            for k, v in sorted(items, key=lambda kv: str(kv[0]))
            if v is not None and str(v).strip()
        )

    def _coerce_demographics(self, case: Dict[str, Any]) -> str:
        """derive a demographics text block from multiple possible case layouts."""
        if case.get("demographics_text"):
            return str(case["demographics_text"]).strip()

        if isinstance(case.get("demographics"), dict):
            return self._flatten_dict_lines(case["demographics"])
        if "demographics" in case and isinstance(case["demographics"], str):
            return case["demographics"].strip()

        candidates = [
            ("person", None),
            ("patient", None),
            ("patient_info", None),
            ("person_demographics", None),
            ("summary", "demographics"),
            ("meta", "demographics"),
            ("profile", None),
        ]
        for top, sub in candidates:
            if top in case and case[top] is not None:
                node = case[top]
                if sub and isinstance(node, dict) and node.get(sub):
                    val = node[sub]
                    if isinstance(val, dict):
                        return self._flatten_dict_lines(val)
                    return str(val).strip()
                if isinstance(node, dict):
                    txt = self._flatten_dict_lines(
                        node, allow_keys=["age", "sex", "gender", "race", "ethnicity"]
                    )
                    if txt:
                        return txt

        return "demographics: not available in source case"

    def _coerce_notes(self, case: Dict[str, Any]) -> str:
        """normalize aggregated clinical notes into a single string (no truncation here).

        input-size control is handled upstream via budgeting and centrally
        in validators via token budgeting. this function only normalizes.
        """
        if case.get("notes_agg") is not None:
            notes_val = case["notes_agg"]
        elif case.get("clinical_notes") is not None:
            notes_val = case["clinical_notes"]
        elif case.get("notes_text") is not None:
            notes_val = case["notes_text"]
        else:
            raise ValueError(
                "notes_agg|clinical_notes|notes_text is required in case payload"
            )

        if isinstance(notes_val, list):
            parts: List[str] = []
            for item in notes_val:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    t = (
                        item.get("text")
                        or item.get("note")
                        or item.get("REPORT_TEXT")
                        or item.get("report_text")
                        or ""
                    )
                    if t:
                        parts.append(str(t))
            notes = "\n---\n".join(p for p in parts if p.strip())
        else:
            notes = str(notes_val)

        return notes

    def _extract_case_strings(self, case: Dict[str, Any]) -> Dict[str, str]:
        """normalize required text fields from heterogeneous case dicts."""
        if not isinstance(case, dict):
            raise ValueError("case must be a dict")

        cid = (
            case.get("case_id")
            or case.get("id")
            or case.get("patient_id")
            or case.get("person_id")
        )
        if not cid:
            raise ValueError(
                "case_id|id|patient_id|person_id is required in case payload"
            )

        demo = self._coerce_demographics(case)
        notes = self._coerce_notes(case)

        return {"case_id": str(cid), "demographics": demo, "clinical_notes": notes}

    # --------- dump helpers (optional) ---------

    def _mk_dump_dir(self, case_id: str, relpath: str) -> Optional[Path]:
        if not self._dump_root:
            return None
        base = Path(self._dump_root) / str(case_id)
        # allow absolute relpath? no — always under case dir
        full = base / relpath
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None
        return full

    def _dump_text(self, *, case_id: str, relpath: str, text: str) -> None:
        p = self._mk_dump_dir(case_id, relpath)
        if not p:
            return
        try:
            p.write_text(text if isinstance(text, str) else str(text))
        except Exception:
            pass

    def _dump_json(self, *, case_id: str, relpath: str, obj: Any) -> None:
        p = self._mk_dump_dir(case_id, relpath)
        if not p:
            return
        try:
            p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def _extract_case_id_from_context(self, ctx: Dict[str, Any]) -> Optional[str]:
        try:
            case = ctx.get("case") if isinstance(ctx, dict) else None
            if isinstance(case, dict):
                for k in ("case_id", "id", "patient_id", "person_id"):
                    v = case.get(k)
                    if v:
                        return str(v)
        except Exception:
            return None
        return None

    def _log_preview(self, obj: Any, tag: str) -> None:
        """log a short preview of backend output for debugging (best-effort)."""
        try:
            s = str(obj)
            self.logger.debug("%s %s", tag, s[:500])
        except Exception:
            # never let logging break the pipeline
            pass
