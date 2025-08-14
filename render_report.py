if __name__ == "__main__":
    # renders a narrative delphea-iraki audit report (markdown) from an out/<case_id> bundle
    #
    # usage:
    #   python render_report.py --bundle out/iraki_case_123 [--outfile audit_report.md]
    #
    # fail-loud: requires jinja2. install with `pip install jinja2`.

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

    def _round1_scores(
        round1_items: List[Tuple[str, Dict[str, Any]]]
    ) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
        # returns (expert_ids, per_qid: {expert_id -> score})
        expert_ids: List[str] = []
        per_qid: Dict[str, Dict[str, int]] = {}
        for item in round1_items:
            if isinstance(item, list) and len(item) == 2:
                eid, payload = item
            elif isinstance(item, dict) and "expert_id" in item:
                eid, payload = item["expert_id"], item
            else:
                continue
            expert_ids.append(eid)
            scores = payload.get("scores", {}) or {}
            for qid, v in scores.items():
                try:
                    per_qid.setdefault(qid, {})[eid] = int(v)
                except Exception:
                    pass
        # keep unique order
        expert_ids = list(dict.fromkeys(expert_ids).keys())
        return expert_ids, per_qid

    def _round3_scores(
        round3_items: List[Tuple[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, int]]:
        per_qid: Dict[str, Dict[str, int]] = {}
        for item in round3_items:
            if isinstance(item, list) and len(item) == 2:
                eid, payload = item
            elif isinstance(item, dict) and "expert_id" in item:
                eid, payload = item["expert_id"], item
            else:
                continue
            scores = payload.get("final_scores", payload.get("scores", {})) or {}
            for qid, v in scores.items():
                try:
                    per_qid.setdefault(qid, {})[eid] = int(v)
                except Exception:
                    pass
        return per_qid

    def _score_deltas(
        r1: Dict[str, Dict[str, int]], r3: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, int]]:
        # per qid per expert: r3 - r1
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
        vals = [int(v) for v in scores.values() if isinstance(v, (int, float))]
        if not vals:
            return 0.0, 0
        med = int(round(median(vals)))
        agree = sum(1 for v in vals if abs(int(v) - med) <= 1) / len(vals)
        return agree, med

    def _collect_debate(bundle_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
        deb_dir = bundle_dir / "debate"
        out: Dict[str, List[Dict[str, Any]]] = {}
        if not deb_dir.exists():
            return out
        for p in deb_dir.glob("*.jsonl"):
            qid = p.stem
            out[qid] = _read_jsonl(p)
        return out

    def _minority_majority(
        plan: Dict[str, List[str]], expert_ids: List[str]
    ) -> Dict[str, Dict[str, List[str]]]:
        mm: Dict[str, Dict[str, List[str]]] = {}
        for qid, minority in plan.items():
            maj = [eid for eid in expert_ids if eid not in minority]
            mm[qid] = {"minority": list(minority), "majority": maj}
        return mm

    def _summarize_pros_cons(
        qid: str, round1: Dict[str, Any], mm: Dict[str, List[str]], max_len: int = 400
    ) -> Dict[str, str]:
        # heuristic: collect minority evidence snippets vs majority evidence snippets
        pros, cons = [], []
        ev_by_eid = {}
        for item in round1:
            if isinstance(item, list) and len(item) == 2:
                eid, payload = item
            elif isinstance(item, dict) and "expert_id" in item:
                eid, payload = item["expert_id"], item
            else:
                continue
            ev_by_eid[eid] = (payload.get("evidence", {}) or {}).get(qid, "")
        for eid in mm.get("minority", []):
            t = (ev_by_eid.get(eid) or "").strip()
            if t:
                pros.append(f"{eid}: {t}")
        for eid in mm.get("majority", []):
            t = (ev_by_eid.get(eid) or "").strip()
            if t:
                cons.append(f"{eid}: {t}")

        def _join(xs: List[str]) -> str:
            s = " ".join(xs)
            return (s[:max_len].rstrip() + " …") if len(s) > max_len else s

        return {"pros": _join(pros) or "(n/a)", "cons": _join(cons) or "(n/a)"}

    def _render(bundle_dir: Path, outfile: Path) -> None:
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
        transcripts = _collect_debate(bundle_dir)

        case_id = report.get("case_id") or Path(bundle_dir).name
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

        # build per-qid narrative payloads
        qids = list(r1_scores.keys())
        q_sections: List[Dict[str, Any]] = []
        min_agree = run_args.get("minimum_agreement", None)
        if min_agree is None:
            # try to recover from the Moderator summary saved by delphea_iraki (not guaranteed)
            min_agree = 0.70

        for qid in qids:
            r1_vec = r1_scores.get(qid, {})
            r3_vec = r3_scores.get(qid, {})
            agree_r1, med_r1 = _agreement_fraction(r1_vec)
            agree_r3, med_r3 = _agreement_fraction(r3_vec if r3_vec else r1_vec)
            turns = transcripts.get(qid, [])
            # compute a simple score timeline per expert
            timelines: Dict[str, List[Tuple[str, int]]] = {
                eid: [("R1", r1_vec.get(eid))] for eid in experts
            }
            for t in turns:
                eid = t.get("expert_id")
                rs = t.get("revised_score", None)
                if eid in timelines and isinstance(rs, int):
                    timelines[eid].append(
                        (f"T{t.get('turn_index', len(timelines[eid]))}", rs)
                    )
            for eid in experts:
                timelines[eid].append(("R3", r3_vec.get(eid, r1_vec.get(eid))))

            # pros/cons heuristics
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

        tmpl = Template(
            # markdown template; lean and readable for audits
            """
# DelPHEA-irAKI Audit Report — {{ case_id }}

**Model**: {{ run_args.get('model_name', '(unknown)') }}  
**Endpoint**: {{ run_args.get('endpoint_url', '(unknown)') }}  
**Panel**: {{ experts|length }} experts — {{ experts|join(', ') }}

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
- Verdict: {{ consensus.get('verdict','(unknown)') }}
- P(irAKI): {{ consensus.get('p_iraki','(n/a)') }}
- Threshold: {{ consensus.get('decision_threshold','(n/a)') }}

**Notes**
- Agreement uses Delphi band (±1 around median).  
- Debate logs omit repeated patient/questionnaire boilerplate by design.  
"""
        )

        md = tmpl.render(
            case_id=case_id,
            run_args=run_args,
            experts=experts,
            q_sections=q_sections,
            consensus=consensus,
        )
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_text(md)
        print(f"wrote {outfile}")

    # ------------- cli -------------
    ap = argparse.ArgumentParser(
        description="render a narrative audit report from out/<case_id> bundle"
    )
    ap.add_argument(
        "--bundle",
        required=True,
        help="path to out/<case_id> directory containing report.json",
    )
    ap.add_argument(
        "--outfile",
        default=None,
        help="output markdown path (default: <bundle>/audit_report.md)",
    )
    args = ap.parse_args()

    bundle_dir = Path(args.bundle)
    if not bundle_dir.exists():
        _die(f"bundle dir not found: {bundle_dir}")
    if not (bundle_dir / "report.json").exists():
        _die(f"report.json not found under {bundle_dir}")

    outfile = Path(args.outfile) if args.outfile else (bundle_dir / "audit_report.md")
    _render(bundle_dir, outfile)
