#!/usr/bin/env python3
"""
Render a per-expert narrative history from an out/<CASE_ID> bundle.

- Shows ONLY this expert's experiences:
  * Round 1: their scores & (short) evidence per QID
  * Debate: only their turns per QID, with role, text snippet, revised_score
            plus a per-QID score timeline: R1 -> (turn revisions) -> R3
  * Round 3: their final scores, delta vs R1
- Does NOT repeat shared boilerplate (patient info text, questionnaire text).
- YAGNI & fail-loud: requires Jinja2; simple CLI.

Usage:
  python render_report_per_expert.py --bundle out/iraki_case_123 --expert-id neph1 \
      [--outfile out/iraki_case_123/agent_neph1.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Tuple


def _die(msg: str) -> None:
    raise SystemExit(msg)


def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception as e:
        _die(f"failed to read {p}: {e}")


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"raw": line})
        return rows
    except FileNotFoundError:
        return []
    except Exception as e:
        _die(f"failed to read jsonl {p}: {e}")


def _round1_for_expert(round1_items: List[Any], expert_id: str) -> Dict[str, Any]:
    """Return this expert's R1 payload as dict (scores/evidence/clinical_reasoning)."""
    for item in round1_items:
        if isinstance(item, list) and len(item) == 2 and item[0] == expert_id:
            return item[1]
        if isinstance(item, dict) and item.get("expert_id") == expert_id:
            return item
    return {}


def _round3_for_expert(round3_items: List[Any], expert_id: str) -> Dict[str, Any]:
    for item in round3_items:
        if isinstance(item, list) and len(item) == 2 and item[0] == expert_id:
            return item[1]
        if isinstance(item, dict) and item.get("expert_id") == expert_id:
            return item
    return {}


def _r1_scores(round1: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for q, v in (round1.get("scores") or {}).items():
        try:
            out[q] = int(v)
        except Exception:
            pass
    return out


def _r3_scores(round3: Dict[str, Any]) -> Dict[str, int]:
    src = round3.get("final_scores", round3.get("scores", {})) or {}
    out: Dict[str, int] = {}
    for q, v in src.items():
        try:
            out[q] = int(v)
        except Exception:
            pass
    return out


def _collect_my_turns(
    bundle_dir: Path, expert_id: str
) -> Dict[str, List[Dict[str, Any]]]:
    """Load only this expert's turns from debate/*.jsonl, keyed by QID."""
    deb_dir = bundle_dir / "debate"
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not deb_dir.exists():
        return out
    for p in deb_dir.glob("*.jsonl"):
        qid = p.stem
        turns = _read_jsonl(p)
        mine = [t for t in turns if str(t.get("expert_id")) == expert_id]
        if mine:
            out[qid] = mine
    return out


def _score_timelines(
    qids: List[str],
    r1: Dict[str, int],
    r3: Dict[str, int],
    my_turns: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Tuple[str, int]]]:
    """Per QID, build a timeline: R1 -> debate revisions (if any) -> R3."""
    out: Dict[str, List[Tuple[str, int]]] = {}
    for q in qids:
        steps: List[Tuple[str, int]] = []
        if q in r1:
            steps.append(("R1", r1[q]))
        # use chronological turn_index order if present
        for t in sorted(my_turns.get(q, []), key=lambda x: x.get("turn_index", 0)):
            rs = t.get("revised_score", None)
            if isinstance(rs, int):
                steps.append((f"T{t.get('turn_index', len(steps))}", rs))
        steps.append(("R3", r3.get(q, r1.get(q))))
        out[q] = steps
    return out


def _agreement_fraction(panel_scores: Dict[str, int]) -> Tuple[float, int]:
    """Utility: fraction within ±1 of median; used only for optional context."""
    vals = [int(v) for v in panel_scores.values() if isinstance(v, (int, float))]
    if not vals:
        return 0.0, 0
    med = int(round(median(vals)))
    agree = sum(1 for v in vals if abs(int(v) - med) <= 1) / len(vals)
    return agree, med


def _minority_membership(debate_plan: Dict[str, Any], expert_id: str) -> Dict[str, str]:
    by_qid = (
        debate_plan.get("debate_plan") if isinstance(debate_plan, dict) else {}
    ) or debate_plan
    stance: Dict[str, str] = {}
    for qid, minority in (by_qid or {}).items():
        try:
            if expert_id in (minority or []):
                stance[qid] = "minority"
            else:
                stance[qid] = "majority"
        except Exception:
            pass
    return stance


def _render(bundle_dir: Path, expert_id: str, outfile: Path) -> None:
    try:
        from jinja2 import Template
    except Exception as e:
        _die(f"jinja2 is required: pip install jinja2  ({e})")

    report = _load_json(bundle_dir / "report.json")
    run_args = (
        _load_json(bundle_dir / "run_args.json")
        if (bundle_dir / "run_args.json").exists()
        else {}
    )
    debate_plan = (
        _load_json(bundle_dir / "debate" / "debate_plan.json")
        if (bundle_dir / "debate" / "debate_plan.json").exists()
        else {}
    )

    case_id = report.get("case_id") or bundle_dir.name
    round1_items = report.get("round1", [])
    round3_items = report.get("round3", [])
    consensus = report.get("consensus", {})

    r1_me = _round1_for_expert(round1_items, expert_id)
    r3_me = _round3_for_expert(round3_items, expert_id)
    if not r1_me:
        _die(
            f"expert_id={expert_id!r} not found in round1 payloads under {bundle_dir}/report.json"
        )

    r1_scores = _r1_scores(r1_me)
    r3_scores = _r3_scores(r3_me) if r3_me else {}
    qids = sorted(r1_scores.keys())
    my_turns = _collect_my_turns(bundle_dir, expert_id)
    timelines = _score_timelines(qids, r1_scores, r3_scores, my_turns)
    stance = _minority_membership(debate_plan, expert_id)

    # evidence snippets (short)
    ev_short: Dict[str, str] = {}
    for q in qids:
        txt = ((r1_me.get("evidence") or {}).get(q) or "").strip().replace("\n", " ")
        ev_short[q] = (txt[:280].rstrip() + " …") if len(txt) > 280 else txt

    # build per-qid sections filtered to this expert
    q_sections: List[Dict[str, Any]] = []
    for q in qids:
        q_sections.append(
            {
                "qid": q,
                "stance": stance.get(q, "(n/a)"),
                "r1": r1_scores.get(q),
                "r3": r3_scores.get(q, r1_scores.get(q)),
                "delta": (r3_scores.get(q, r1_scores.get(q)) - r1_scores.get(q))
                if q in r1_scores
                else 0,
                "evidence": ev_short.get(q, ""),
                "turns": my_turns.get(q, []),
                "timeline": timelines.get(q, []),
            }
        )

    tmpl = Template(
        """
# Per-Expert History — {{ expert_id }} (Case {{ case_id }})

**Model**: {{ run_args.get('model_name', '(unknown)') }}  
**Endpoint**: {{ run_args.get('endpoint_url', '(unknown)') }}

This report shows ONLY {{ expert_id }}'s own experience: their Round-1 ratings,
their own debate turns (roles, snippets, score changes), and their Round-3 ratings.

---

## Round 1 — Your Assessments
{% for q in q_sections %}
### QID: {{ q.qid }}  (stance: {{ q.stance }})
- **R1 score**: {{ q.r1 }}
- **Evidence (your note, short)**: {{ q.evidence or '(none)' }}

{% endfor %}

---

## Debate — Your Turns Only
{% set any_turns = false %}
{% for q in q_sections %}
{% if q.turns %}
### QID: {{ q.qid }}
{% set any_turns = true %}
{% for t in q.turns %}
- **{{ t.get('speaker_role','participant') }}** — turn {{ t.get('turn_index', loop.index0) }}{% if t.get('revised_score') is not none %}, revised_score={{ t.get('revised_score') }}{% endif %}{% if t.get('satisfied') %}, satisfied{% endif %}
  {{ (t.get('text','') or t.get('raw','')).strip() }}
{% endfor %}

**Your score timeline**: {%- for tag, val in q.timeline if val is not none -%} {{ tag }}={{ val }}{%- if not loop.last -%} →{%- endif -%}{%- endfor -%}

---
{% endif %}
{% endfor %}
{% if not any_turns %}
(no debate turns recorded for {{ expert_id }})
{% endif %}

## Round 3 — Your Final Ratings
| QID | R3 | Δ(R3−R1) |
|---|---:|---:|
{% for q in q_sections -%}
| {{ q.qid }} | {{ q.r3 }} | {{ q.delta }} |
{% endfor %}

---

## Study Consensus (context)
- Verdict: {{ consensus.get('verdict','(unknown)') }}
- P(irAKI): {{ consensus.get('p_iraki','(n/a)') }}
- Threshold: {{ consensus.get('decision_threshold','(n/a)') }}

*Note:* Boilerplate patient information and the questionnaire text are omitted here
to keep the report focused on your own actions and decisions.
"""
    )

    md = tmpl.render(
        case_id=case_id,
        expert_id=expert_id,
        run_args=run_args,
        q_sections=q_sections,
        consensus=consensus,
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(md)
    print(f"wrote {outfile}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a per-expert history from an out/<CASE_ID> bundle"
    )
    ap.add_argument(
        "--bundle",
        required=True,
        help="path to out/<case_id> directory containing report.json",
    )
    ap.add_argument(
        "--expert-id", required=True, help="expert id to render (e.g., neph1)"
    )
    ap.add_argument(
        "--outfile",
        default=None,
        help="output markdown path (default: <bundle>/agent_<expert>.md)",
    )
    args = ap.parse_args()

    bundle_dir = Path(args.bundle)
    if not bundle_dir.exists():
        _die(f"bundle dir not found: {bundle_dir}")
    if not (bundle_dir / "report.json").exists():
        _die(f"report.json not found under {bundle_dir}")

    expert_id = args.expert_id.strip()
    if not expert_id:
        _die("expert-id must be a non-empty string")

    outfile = (
        Path(args.outfile) if args.outfile else (bundle_dir / f"agent_{expert_id}.md")
    )
    _render(bundle_dir, expert_id, outfile)


if __name__ == "__main__":
    main()
