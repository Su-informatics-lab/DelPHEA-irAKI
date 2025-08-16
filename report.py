#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report.py — consolidated renderers for DelPHEA-irAKI run artifacts

what it does
------------
given an out/<CASE_ID> bundle produced by delphea_iraki.py, this script renders:
  1) a patient-level audit narrative (markdown) summarizing diagnosis & debate
  2) per-expert activity narratives (markdown), one file per expert

design
------
- yagni: no external deps beyond jinja2; fail loud with a clear message if missing
- robust to minor schema drift across repo versions (e.g., consensus key names)
- strict i/o: raise SystemExit on unreadable inputs

usage
-----
  # everything (audit + all agents)
  python report.py --bundle out/iraki_case_123

  # audit only
  python report.py --bundle out/iraki_case_123 --mode audit --outfile audit.md

  # one agent only
  python report.py --bundle out/iraki_case_123 --mode expert --expert-id nephrologist

  # both, writing into a reports/ subdir
  python report.py --bundle out/iraki_case_123 --mode both --outdir out/iraki_case_123/reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ------------- small utils -------------
def _die(msg: str) -> None:
    """fail loud with a short message."""
    raise SystemExit(msg)


def _load_json(p: Path, *, missing_ok: bool = False) -> Any:
    """read a json file, optionally returning {} on missing."""
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        if missing_ok:
            return {}
        _die(f"file not found: {p}")
    except Exception as e:
        _die(f"failed to read {p}: {e}")


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    """read a jsonl file; keep non-json lines as {'raw': line} for transparency."""
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


def _collect_debate(bundle_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """load debate/*.jsonl keyed by qid."""
    deb_dir = bundle_dir / "debate"
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not deb_dir.exists():
        return out
    for p in sorted(deb_dir.glob("*.jsonl")):
        qid = p.stem
        out[qid] = _read_jsonl(p)
    return out


def _discover_experts_from_round_items(round_items: Iterable[Any]) -> List[str]:
    """extract expert ids with stable ordering from a round payload list."""
    order: List[str] = []
    for item in round_items:
        if isinstance(item, list) and len(item) == 2:
            eid = str(item[0])
        elif isinstance(item, dict) and "expert_id" in item:
            eid = str(item["expert_id"])
        else:
            continue
        order.append(eid)
    # dedupe preserving order
    return list(dict.fromkeys(order).keys())


# ------------- score helpers -------------
def _round1_scores(
    round1_items: List[Any],
) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """returns (expert_ids, per_qid: {expert_id -> score})."""
    expert_ids = _discover_experts_from_round_items(round1_items)
    per_qid: Dict[str, Dict[str, int]] = {}
    for item in round1_items:
        if isinstance(item, list) and len(item) == 2:
            eid, payload = item
        elif isinstance(item, dict) and "expert_id" in item:
            eid, payload = item["expert_id"], item
        else:
            continue
        scores = (payload.get("scores") or {}) if isinstance(payload, dict) else {}
        for qid, v in scores.items():
            try:
                per_qid.setdefault(qid, {})[str(eid)] = int(v)
            except Exception:
                pass
    return expert_ids, per_qid


def _round3_scores(round3_items: List[Any]) -> Dict[str, Dict[str, int]]:
    """map qid -> {expert_id: score} from round 3 items."""
    per_qid: Dict[str, Dict[str, int]] = {}
    for item in round3_items:
        if isinstance(item, list) and len(item) == 2:
            eid, payload = item
        elif isinstance(item, dict) and "expert_id" in item:
            eid, payload = item["expert_id"], item
        else:
            continue
        src = {}
        if isinstance(payload, dict):
            src = payload.get("final_scores", payload.get("scores", {})) or {}
        for qid, v in src.items():
            try:
                per_qid.setdefault(qid, {})[str(eid)] = int(v)
            except Exception:
                pass
    return per_qid


def _score_deltas(
    r1: Dict[str, Dict[str, int]], r3: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, int]]:
    """per qid per expert: r3 − r1 (0 if missing)."""
    out: Dict[str, Dict[str, int]] = {}
    for qid, m in r1.items():
        out[qid] = {}
        r3m = r3.get(qid, {})
        for eid, s1 in m.items():
            s3 = r3m.get(eid, s1)
            try:
                out[qid][eid] = int(s3) - int(s1)
            except Exception:
                out[qid][eid] = 0
    return out


def _agreement_fraction(scores: Dict[str, int]) -> Tuple[float, int]:
    """fraction within ±1 of median and the median itself."""
    vals = [int(v) for v in scores.values() if isinstance(v, (int, float))]
    if not vals:
        return 0.0, 0
    med = int(round(median(vals)))
    agree = sum(1 for v in vals if abs(int(v) - med) <= 1) / len(vals)
    return agree, med


# ------------- debate plan & pros/cons -------------
def _minority_majority(
    plan: Dict[str, List[str]], expert_ids: List[str]
) -> Dict[str, Dict[str, List[str]]]:
    """build per-qid minority/majority sets from the router plan."""
    mm: Dict[str, Dict[str, List[str]]] = {}
    for qid, minority in plan.items():
        maj = [eid for eid in expert_ids if eid not in (minority or [])]
        mm[qid] = {"minority": list(minority or []), "majority": maj}
    return mm


def _summarize_pros_cons(
    qid: str,
    round1_items: List[Any],
    mm_for_q: Dict[str, List[str]],
    *,
    max_len: int = 400,
) -> Dict[str, str]:
    """heuristic: collect minority vs majority evidence sentences for a qid."""
    pros, cons = [], []
    ev_by_eid: Dict[str, str] = {}
    for item in round1_items:
        if isinstance(item, list) and len(item) == 2:
            eid, payload = item
        elif isinstance(item, dict) and "expert_id" in item:
            eid, payload = item["expert_id"], item
        else:
            continue
        txt = ""
        if isinstance(payload, dict):
            txt = ((payload.get("evidence") or {}).get(qid) or "").strip()
        ev_by_eid[str(eid)] = txt
    for eid in mm_for_q.get("minority", []):
        t = (ev_by_eid.get(eid) or "").strip()
        if t:
            pros.append(f"{eid}: {t}")
    for eid in mm_for_q.get("majority", []):
        t = (ev_by_eid.get(eid) or "").strip()
        if t:
            cons.append(f"{eid}: {t}")

    def _join(xs: List[str]) -> str:
        s = " ".join(xs)
        return (s[:max_len].rstrip() + " …") if len(s) > max_len else s

    return {"pros": _join(pros) or "(n/a)", "cons": _join(cons) or "(n/a)"}


# ------------- per-expert helpers -------------
def _round_payload_for_expert(round_items: List[Any], expert_id: str) -> Dict[str, Any]:
    """return this expert's payload dict from a round list; {} if not found."""
    for item in round_items:
        if isinstance(item, list) and len(item) == 2 and str(item[0]) == expert_id:
            return item[1] if isinstance(item[1], dict) else {}
        if isinstance(item, dict) and str(item.get("expert_id")) == expert_id:
            return item
    return {}


def _scores_from_payload(
    payload: Dict[str, Any], final: bool = False
) -> Dict[str, int]:
    """extract scores dict from a round payload."""
    if not isinstance(payload, dict):
        return {}
    src = (
        payload.get("final_scores" if final else "scores", payload.get("scores", {}))
        or {}
    )
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
    """filter debate turns to only those by this expert."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for qid, turns in _collect_debate(bundle_dir).items():
        mine = [t for t in turns if str(t.get("expert_id")) == expert_id]
        if mine:
            # keep chronological order if turn_index present
            mine = sorted(mine, key=lambda x: x.get("turn_index", 0))
            out[qid] = mine
    return out


def _score_timelines(
    qids: List[str],
    r1: Dict[str, int],
    r3: Dict[str, int],
    my_turns: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Tuple[str, int]]]:
    """per qid, R1 → debate revisions (if any) → R3."""
    out: Dict[str, List[Tuple[str, int]]] = {}
    for q in qids:
        steps: List[Tuple[str, int]] = []
        if q in r1:
            steps.append(("R1", r1[q]))
        for t in my_turns.get(q, []):
            rs = t.get("revised_score", None)
            if isinstance(rs, int):
                steps.append((f"T{t.get('turn_index', len(steps))}", rs))
        steps.append(("R3", r3.get(q, r1.get(q))))
        out[q] = steps
    return out


# ------------- consensus key helper -------------
def _cns(consensus: Dict[str, Any], *keys: str, default: Any = "(n/a)") -> Any:
    """read first present key among *keys from consensus dict."""
    for k in keys:
        if k in consensus:
            return consensus[k]
    return default


# ------------- panel info (optional) -------------
def _load_panel_map(repo_root: Path) -> Dict[str, Dict[str, str]]:
    """optional expert id → {name, specialty} mapping from panel_full.json."""
    try:
        p = repo_root / "panel_full.json"
        data = _load_json(p, missing_ok=True)
        exps = (
            (data.get("expert_panel", {}) or {}).get("experts", [])
            if isinstance(data, dict)
            else []
        )
        out: Dict[str, Dict[str, str]] = {}
        for e in exps:
            if not isinstance(e, dict):
                continue
            eid = str(e.get("id", "")).strip()
            if not eid:
                continue
            out[eid] = {
                "name": str(e.get("name", "")).strip(),
                "specialty": str(e.get("specialty", "")).strip(),
            }
        return out
    except Exception:
        return {}


# ------------- renderers -------------
def _render_audit(bundle_dir: Path, outfile: Path) -> None:
    """render patient-level audit narrative."""
    try:
        from jinja2 import Template
    except Exception as e:
        _die(f"jinja2 is required: pip install jinja2  ({e})")

    report = _load_json(bundle_dir / "report.json")
    run_args = _load_json(bundle_dir / "run_args.json", missing_ok=True)
    debate_plan = _load_json(
        bundle_dir / "debate" / "debate_plan.json", missing_ok=True
    )
    transcripts = _collect_debate(bundle_dir)

    case_id = report.get("case_id") or bundle_dir.name
    round1 = report.get("round1", [])
    round3 = report.get("round3", [])
    consensus = report.get("consensus", {})
    experts, r1_scores = _round1_scores(round1)
    r3_scores = _round3_scores(round3)
    deltas = _score_deltas(r1_scores, r3_scores)

    plan_by_qid = (
        debate_plan.get("debate_plan") if isinstance(debate_plan, dict) else {}
    ) or {}
    mm = _minority_majority(plan_by_qid, experts)

    # per-qid blocks
    qids = list(r1_scores.keys())
    q_sections: List[Dict[str, Any]] = []
    for qid in qids:
        r1_vec = r1_scores.get(qid, {})
        r3_vec = r3_scores.get(qid, {})
        agree_r1, med_r1 = _agreement_fraction(r1_vec)
        agree_r3, med_r3 = _agreement_fraction(r3_vec if r3_vec else r1_vec)
        turns = transcripts.get(qid, [])
        # expert timelines
        timelines: Dict[str, List[Tuple[str, Optional[int]]]] = {
            eid: [("R1", r1_vec.get(eid))] for eid in experts
        }
        for t in turns:
            eid = str(t.get("expert_id"))
            rs = t.get("revised_score", None)
            if eid in timelines and isinstance(rs, int):
                timelines[eid].append(
                    (f"T{t.get('turn_index', len(timelines[eid]))}", rs)
                )
        for eid in experts:
            timelines[eid].append(("R3", r3_vec.get(eid, r1_vec.get(eid))))

        pc = _summarize_pros_cons(qid, round1, mm.get(qid, {}))

        q_sections.append(
            {
                "qid": qid,
                "minority": mm.get(qid, {}).get("minority", []),
                "majority": mm.get(qid, {}).get("majority", []),
                "r1_scores": r1_vec,
                "r3_scores": r3_vec,
                "deltas": deltas.get(qid, {}),
                "agree_r1": round(agree_r1, 3),
                "agree_r3": round(agree_r3, 3),
                "median_r1": med_r1,
                "median_r3": med_r3,
                "transcript": turns,
                "timelines": timelines,
                "pros": pc["pros"],
                "cons": pc["cons"],
            }
        )

    # optional panel details for nicer header
    panel_map = _load_panel_map(Path.cwd())
    panel_decor = ", ".join(
        f"{eid}"
        + (
            f" ({panel_map[eid]['specialty']})"
            if eid in panel_map and panel_map[eid].get("specialty")
            else ""
        )
        for eid in experts
    )

    try:
        from jinja2 import Template  # re-import inside function scope for clarity
    except Exception as e:
        _die(f"jinja2 is required: pip install jinja2  ({e})")

    tmpl = Template(
        # markdown template; lean for audits
        """
# DelPHEA-irAKI Audit Report — {{ case_id }}

**Model**: {{ run_args.get('model_name', '(unknown)') }}  
**Endpoint**: {{ run_args.get('endpoint_url', '(unknown)') }}  
**Panel**: {{ experts|length }} experts — {{ panel_decor }}

---

## Round 1 — Independent Assessments
- Questions assessed: {{ q_sections|length }}
- Agreement (Delphi ±1 around median) varies by question; see per-qid tables below.

{% for q in q_sections %}
### QID: {{ q.qid }}
- Minority: {{ q.minority|join(', ') if q.minority else '(none)' }}
- Majority: {{ q.majority|join(', ') if q.majority else '(none)' }}
- R1 median={{ q.median_r1 }}, agreement={{ '%.0f%%' % (100*q.agree_r1) }}
- R3 median={{ q.median_r3 }}, agreement={{ '%.0f%%' % (100*q.agree_r3) }}

**R1 scores**
| Expert | Score |
|---|---:|
{% for eid, s in q.r1_scores.items() -%}
| {{ eid }} | {{ s }} |
{% endfor %}

{% if q.transcript %}
**Debate (key turns)**  
{% for t in q.transcript %}
- **{{ t.get('expert_id','?') }}** ({{ t.get('speaker_role','?') }}) — turn {{ t.get('turn_index', loop.index0) }}{% if t.get('revised_score') is not none %}, revised_score={{ t.get('revised_score') }}{% endif %}{% if t.get('satisfied') %}, satisfied{% endif %}  
  {{ (t.get('text','') or t.get('raw','')).strip() }}
{% endfor %}
{% else %}
**Debate**: (skipped)
{% endif %}

**Score timelines**
| Expert | Timeline |
|---|---|
{% for eid, steps in q.timelines.items() -%}
| {{ eid }} | {%- for tag, val in steps if val is not none -%} {{ tag }}={{ val }}{%- if not loop.last -%} →{%- endif -%}{%- endfor -%} |
{% endfor %}

**Pros (minority evidence, heuristic)**: {{ q.pros }}

**Cons (majority evidence, heuristic)**: {{ q.cons }}

**R3 scores & deltas**
| Expert | R3 | Δ(R3−R1) |
|---|---:|---:|
{% for eid, s3 in q.r3_scores.items() -%}
| {{ eid }} | {{ s3 }} | {{ q.deltas.get(eid, 0) }} |
{% endfor %}

---
{% endfor %}

## Consensus
- Verdict: {{ cns_verdict }}
- P(irAKI): {{ cns_p }}
- CI: {{ cns_ci }}
- Threshold: {{ cns_thresh }}

**notes**
- agreement uses Delphi band (±1 around median).  
- debate logs omit repeated patient/questionnaire boilerplate by design.  
"""
    )

    md = tmpl.render(
        case_id=case_id,
        run_args=run_args,
        experts=experts,
        panel_decor=panel_decor,
        q_sections=q_sections,
        cns_verdict=_cns(consensus, "verdict", "iraki_verdict", default="(unknown)"),
        cns_p=_cns(consensus, "p_iraki", "iraki_probability"),
        cns_ci=_cns(consensus, "ci_iraki", default="(n/a)"),
        cns_thresh=_cns(consensus, "decision_threshold", "threshold", default="(n/a)"),
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(md)
    print(f"wrote {outfile}")


def _render_expert(bundle_dir: Path, expert_id: str, outfile: Path) -> None:
    """render per-expert activity narrative."""
    try:
        from jinja2 import Template
    except Exception as e:
        _die(f"jinja2 is required: pip install jinja2  ({e})")

    report = _load_json(bundle_dir / "report.json")
    run_args = _load_json(bundle_dir / "run_args.json", missing_ok=True)
    debate_plan = _load_json(
        bundle_dir / "debate" / "debate_plan.json", missing_ok=True
    )

    case_id = report.get("case_id") or bundle_dir.name
    round1_items = report.get("round1", [])
    round3_items = report.get("round3", [])
    consensus = report.get("consensus", {})

    r1_me = _round_payload_for_expert(round1_items, expert_id)
    if not r1_me:
        _die(
            f"expert_id={expert_id!r} not found in round1 payloads under {bundle_dir}/report.json"
        )
    r3_me = _round_payload_for_expert(round3_items, expert_id)

    r1_scores = _scores_from_payload(r1_me, final=False)
    r3_scores = _scores_from_payload(r3_me, final=True) if r3_me else {}
    qids = sorted(r1_scores.keys())
    my_turns = _collect_my_turns(bundle_dir, expert_id)
    timelines = _score_timelines(qids, r1_scores, r3_scores, my_turns)

    # stance detection from router plan
    by_qid = (
        debate_plan.get("debate_plan") if isinstance(debate_plan, dict) else debate_plan
    ) or {}
    stance: Dict[str, str] = {}
    for qid, minority in (by_qid or {}).items():
        try:
            stance[qid] = "minority" if expert_id in (minority or []) else "majority"
        except Exception:
            pass

    # short evidence
    ev_short: Dict[str, str] = {}
    for q in qids:
        txt = ((r1_me.get("evidence") or {}).get(q) or "").strip().replace("\n", " ")
        ev_short[q] = (txt[:280].rstrip() + " …") if len(txt) > 280 else txt

    tmpl = Template(
        """
# Per-Expert History — {{ expert_id }} (Case {{ case_id }})

**Model**: {{ run_args.get('model_name', '(unknown)') }}  
**Endpoint**: {{ run_args.get('endpoint_url', '(unknown)') }}

this report shows ONLY {{ expert_id }}'s own experience: their round-1 ratings,
their own debate turns (roles, snippets, score changes), and their round-3 ratings.

---

## Round 1 — Your Assessments
{% for q in qids %}
### QID: {{ q }}  (stance: {{ stance.get(q, '(n/a)') }})
- **R1 score**: {{ r1_scores.get(q) }}
- **Evidence (your note, short)**: {{ ev_short.get(q, '(none)') }}

{% endfor %}

---

## Debate — Your Turns Only
{% set any_turns = false %}
{% for q in qids %}
{% if my_turns.get(q) %}
### QID: {{ q }}
{% set any_turns = true %}
{% for t in my_turns.get(q, []) %}
- **{{ t.get('speaker_role','participant') }}** — turn {{ t.get('turn_index', loop.index0) }}{% if t.get('revised_score') is not none %}, revised_score={{ t.get('revised_score') }}{% endif %}{% if t.get('satisfied') %}, satisfied{% endif %}
  {{ (t.get('text','') or t.get('raw','')).strip() }}
{% endfor %}

**your score timeline**: {%- for tag, val in timelines.get(q, []) if val is not none -%} {{ tag }}={{ val }}{%- if not loop.last -%} →{%- endif -%}{%- endfor -%}

---
{% endif %}
{% endfor %}
{% if not any_turns %}
(no debate turns recorded for {{ expert_id }})
{% endif %}

## Round 3 — Your Final Ratings
| QID | R3 | Δ(R3−R1) |
|---|---:|---:|
{% for q in qids -%}
| {{ q }} | {{ r3_scores.get(q, r1_scores.get(q)) }} | {{ (r3_scores.get(q, r1_scores.get(q)) - r1_scores.get(q)) if q in r1_scores else 0 }} |
{% endfor %}

---

## Study Consensus (context)
- Verdict: {{ cns_verdict }}
- P(irAKI): {{ cns_p }}
- CI: {{ cns_ci }}
- Threshold: {{ cns_thresh }}

*note:* boilerplate patient information and the questionnaire text are omitted here
to keep the report focused on your own actions and decisions.
"""
    )

    md = tmpl.render(
        case_id=case_id,
        expert_id=expert_id,
        run_args=run_args,
        qids=qids,
        r1_scores=r1_scores,
        r3_scores=r3_scores,
        my_turns=my_turns,
        timelines=timelines,
        stance=stance,
        ev_short=ev_short,
        cns_verdict=_cns(consensus, "verdict", "iraki_verdict", default="(unknown)"),
        cns_p=_cns(consensus, "p_iraki", "iraki_probability"),
        cns_ci=_cns(consensus, "ci_iraki", default="(n/a)"),
        cns_thresh=_cns(consensus, "decision_threshold", "threshold", default="(n/a)"),
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(md)
    print(f"wrote {outfile}")


# ------------- cli -------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="render audit and/or per-expert narratives from an out/<CASE_ID> bundle"
    )
    ap.add_argument(
        "--bundle",
        required=True,
        help="path to out/<case_id> directory containing report.json",
    )
    ap.add_argument(
        "--mode",
        choices=["audit", "expert", "both"],
        default="both",
        help="what to render (default: both)",
    )
    ap.add_argument(
        "--expert-id",
        help="if mode includes 'expert': render only this expert id; otherwise all experts discovered from Round-1",
    )
    ap.add_argument(
        "--outfile",
        default=None,
        help="audit markdown output path (only used when --mode=audit)",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="directory for outputs (default: <bundle>)",
    )
    return ap.parse_args()


def _main() -> None:
    args = _parse_args()
    bundle_dir = Path(args.bundle)
    if not bundle_dir.exists():
        _die(f"bundle dir not found: {bundle_dir}")
    if not (bundle_dir / "report.json").exists():
        _die(f"report.json not found under {bundle_dir}")

    outdir = Path(args.outdir) if args.outdir else bundle_dir
    outdir.mkdir(parents=True, exist_ok=True)

    report = _load_json(bundle_dir / "report.json")
    r1_items = report.get("round1", [])
    experts = _discover_experts_from_round_items(r1_items)

    wrote: List[Path] = []

    if args.mode in ("audit", "both"):
        audit_out = Path(args.outfile) if args.outfile else (outdir / "audit_report.md")
        _render_audit(bundle_dir, audit_out)
        wrote.append(audit_out)

    if args.mode in ("expert", "both"):
        if args.expert_id:
            target_ids = [args.expert_id.strip()]
        else:
            target_ids = experts
        if not target_ids:
            _die(
                "no experts found in round1 payloads; cannot render per-expert reports"
            )
        for eid in target_ids:
            per_out = outdir / f"agent_{eid}.md"
            _render_expert(bundle_dir, eid, per_out)
            wrote.append(per_out)

    # final friendly note
    if wrote:
        print("rendered:")
        for p in wrote:
            print(f"  - {p}")


if __name__ == "__main__":
    _main()
