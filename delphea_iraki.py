# delphea_iraki.py
# cli entrypoint wiring config → dataloader → experts/moderator → run (single case)
# keeps to yagni: minimal flags, fail fast, clear logs. includes prompt budgeter and
# true ctx-window reporting to validators via backend.capabilities().

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from aggregator import Aggregator
from dataloader import DataLoader
from expert import Expert
from llm_backend import LLMBackend
from moderator import Moderator
from router import FullRouter, SparseRouter
from schema import load_qids

LOG = logging.getLogger("delphea_iraki")
_ID_RE = re.compile(r"(\d+)")


def _configure_logging(verbosity: int) -> None:
    level = (
        logging.WARNING
        if verbosity <= 0
        else logging.INFO
        if verbosity == 1
        else logging.DEBUG
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _normalize_case_id(x: str) -> str:
    if not x or not isinstance(x, str):
        raise ValueError("case id must be a non-empty string")
    if x.startswith("iraki_case_"):
        return x
    m = _ID_RE.search(x)
    if not m:
        raise ValueError(f"cannot parse numeric id from: {x!r}")
    return f"iraki_case_{m.group(1)}"


# ------------------------ panel loader (tolerant) ------------------------


def _load_panel(panel_path: str) -> List[Dict[str, Any]]:
    p = Path(panel_path)
    if not p.exists():
        raise FileNotFoundError(f"panel file not found: {panel_path}")
    try:
        cfg = json.loads(p.read_text())
    except Exception as e:
        raise ValueError(f"failed to parse panel json: {e}") from e

    # unwrap {"expert_panel": {"experts": [...] , ...}}
    if (
        isinstance(cfg, dict)
        and "expert_panel" in cfg
        and isinstance(cfg["expert_panel"], dict)
    ):
        cfg = cfg["expert_panel"]

    raw_list = None
    if isinstance(cfg, list):
        raw_list = cfg
    if raw_list is None and isinstance(cfg, dict) and "experts" in cfg:
        node = cfg["experts"]
        if isinstance(node, list):
            raw_list = node
        elif isinstance(node, dict):
            raw_list = [{"id": k, **(v or {})} for k, v in node.items()]
    if raw_list is None and isinstance(cfg, dict):
        for key in ("expert_panel", "panel", "members"):
            if key in cfg:
                node = cfg[key]
                if isinstance(node, list):
                    raw_list = node
                elif (
                    isinstance(node, dict)
                    and "experts" in node
                    and isinstance(node["experts"], list)
                ):
                    raw_list = node["experts"]
                elif isinstance(node, dict) and all(
                    isinstance(v, dict) for v in node.values()
                ):
                    raw_list = [{"id": k, **(v or {})} for k, v in node.items()]
                if raw_list is not None:
                    break
    if (
        raw_list is None
        and isinstance(cfg, dict)
        and cfg
        and all(isinstance(v, dict) for v in cfg.values())
    ):
        raw_list = [{"id": k, **(v or {})} for k, v in cfg.items()]

    if not isinstance(raw_list, list) or not raw_list:
        keys = list(cfg.keys()) if isinstance(cfg, dict) else type(cfg).__name__
        raise ValueError(
            "panel config must contain a non-empty experts collection. "
            f"top-level keys seen: {keys!r}"
        )

    normd: List[Dict[str, Any]] = []
    for i, e in enumerate(raw_list):
        if not isinstance(e, dict):
            raise ValueError(
                f"panel entry {i} must be an object, got {type(e).__name__}"
            )
        eid = e.get("id") or e.get("expert_id") or e.get("eid")
        if not eid or not isinstance(eid, str):
            raise ValueError(f"panel entry {i} missing expert id (id/expert_id/eid)")
        spec = e.get("specialty") or e.get("role") or e.get("discipline")
        if not spec or not isinstance(spec, str):
            raise ValueError(
                f"panel entry {i} missing specialty (specialty/role/discipline)"
            )
        persona = e.get("persona") or {}
        if not isinstance(persona, dict):
            raise ValueError(f"panel entry {i} persona must be an object if provided")
        normd.append({"id": eid, "specialty": spec, "persona": persona})

    ids = [x["id"] for x in normd]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate expert_id in panel: {ids}")

    return normd


# ------------------------ backend / experts / router ------------------------


def _build_backend(
    endpoint_url: str, model_name: str, t_r1: float, t_r2: float, t_r3: float
) -> LLMBackend:
    backend = LLMBackend(endpoint_url=endpoint_url, model_name=model_name)
    applied = []
    for attr, val in (
        ("temperature_r1", t_r1),
        ("temperature_r2", t_r2),
        ("temperature_r3", t_r3),
    ):
        try:
            setattr(backend, attr, val)
            applied.append(attr)
        except Exception:
            pass
    if len(applied) < 3:
        LOG.debug("backend may not support per-round temperatures; applied=%s", applied)
    return backend


def _build_experts(
    panel_cfg: List[Dict[str, Any]], backend: LLMBackend
) -> List[Expert]:
    experts: List[Expert] = []
    for e in panel_cfg:
        experts.append(
            Expert(
                expert_id=e["id"],
                specialty=e["specialty"],
                persona=e.get("persona") or {},
                backend=backend,
            )
        )
    ids = [x.expert_id for x in experts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate expert_id in panel: {ids}")
    return experts


def _select_router(name: str):
    n = (name or "sparse").lower()
    if n == "sparse":
        return SparseRouter()
    if n == "full":
        return FullRouter()
    raise ValueError(f"unknown router: {name!r} (use 'sparse' or 'full')")


# ------------------------ prompt budgeter (notes trimming) ------------------------

_AKI_TERMS = [
    "aki",
    "acute kidney",
    "creatinine",
    "bump in cr",
    "oliguria",
    "anuria",
    "urine output",
    "uop",
    "proteinuria",
    "hematuria",
    "casts",
    "ain",
    "atn",
    # immune checkpoint & common brand/generics
    "pembrolizumab",
    "nivolumab",
    "ipilimumab",
    "atezolizumab",
    "durvalumab",
    "avelumab",
    "cemiplimab",
    "checkpoint inhibitor",
    "ici",
    # concomitant nephrotoxins & differentials
    "ppi",
    "omeprazole",
    "pantoprazole",
    "nsaid",
    "ibuprofen",
    "naproxen",
    "vancomycin",
    "zosyn",
    "piperacillin",
    "contrast",
    "ct contrast",
    "bactrim",
    "tmp-smx",
]

_TEXT_KEYS = ("report_text", "text", "note", "content", "body")
_TIME_KEYS = ("physiologic_time", "time", "timestamp", "date", "datetime", "note_time")


def _find_notes_list(case: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """heuristic: prefer explicit 'clinical_notes' or 'notes'; else scan for a list of dicts with long text."""
    # explicit keys first
    for k in ("clinical_notes", "notes"):
        v = case.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    # scan any list that looks like notes
    for _, v in case.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            d0 = v[0]
            lower_keys = {str(k).lower() for k in d0.keys()}
            if any(tk in lower_keys for tk in _TEXT_KEYS):
                return v
    return None


def _get_lower(d: Dict[str, Any], key: str) -> Any:
    """case-insensitive dict get."""
    for k, v in d.items():
        if str(k).lower() == key:
            return v
    return None


def _score_note(n: Dict[str, Any], recency_rank: int) -> float:
    """simple relevance score: AKI/ICI term hits + recency bonus."""
    txt = ""
    for tk in _TEXT_KEYS:
        val = _get_lower(n, tk)
        if isinstance(val, str):
            txt = val
            break
    hits = 0
    if txt:
        low = txt.lower()
        for term in _AKI_TERMS:
            if term in low:
                hits += 1
    return hits + 0.1 * recency_rank  # mild recency bonus


def _trim_notes_for_prompt(
    case: Dict[str, Any],
    max_notes: int,
    note_char_cap: int,
    total_chars_cap: int,
) -> List[Dict[str, Any]]:
    """select a compact, high-signal subset of notes and truncate long texts."""
    notes = _find_notes_list(case) or []
    if not notes:
        return []

    # sort by time if possible; newest first
    def _to_time(n: Dict[str, Any]) -> Any:
        for tk in _TIME_KEYS:
            v = _get_lower(n, tk)
            if v is not None:
                return v
        return None

    try:
        notes_sorted = sorted(notes, key=_to_time)
    except Exception:
        notes_sorted = notes[:]
    notes_sorted = list(reversed(notes_sorted))  # newest first

    # score by relevance + recency
    scored = [(n, _score_note(n, i)) for i, n in enumerate(notes_sorted)]
    scored.sort(key=lambda t: t[1], reverse=True)

    # take top-k, then enforce character budgets
    selected: List[Dict[str, Any]] = []
    total = 0
    for n, _ in scored[: max_notes * 3]:  # slight over-select before char budgeting
        n2 = dict(n)
        for tk in _TEXT_KEYS:
            v = _get_lower(n2, tk)
            if isinstance(v, str):
                s = v
                if len(s) > note_char_cap:
                    s = s[:note_char_cap].rstrip() + " …[truncated]"
                for k in list(n2.keys()):
                    if str(k).lower() == tk:
                        n2[k] = s
                        break
        lens = [len(str(_get_lower(n2, tk)) or "") for tk in _TEXT_KEYS]
        add = max(lens) if lens else 0
        if total + add > total_chars_cap:
            break
        selected.append(n2)
        total += add
        if len(selected) >= max_notes:
            break

    return selected


def _apply_prompt_budget(
    case: Dict[str, Any], max_notes: int, note_char_cap: int, total_chars_cap: int
) -> Dict[str, Any]:
    """non-destructive: returns a shallow-copied case with a compact 'clinical_notes' field."""
    compact = dict(case)
    compact["clinical_notes"] = _trim_notes_for_prompt(
        compact, max_notes, note_char_cap, total_chars_cap
    )
    compact["_prompt_budget_applied"] = {
        "max_notes": max_notes,
        "note_char_cap": note_char_cap,
        "total_chars_cap": total_chars_cap,
        "selected_notes": len(compact["clinical_notes"]),
    }
    return compact


# ------------------------ case io & output helpers ------------------------


def _fetch_case(loader: Any, case_id: str) -> Dict[str, Any]:
    """fetch a single case dict using the dataloader's public api.

    prefers .load_patient_case(case_id); falls back to a few legacy names.
    """
    if hasattr(loader, "load_patient_case"):
        return loader.load_patient_case(case_id)
    for fn_name in ("get_case", "load_case", "fetch_case"):
        if hasattr(loader, fn_name):
            return getattr(loader, fn_name)(case_id)
    raise AttributeError(
        "dataloader must expose one of: .load_patient_case(case_id), .get_case(case_id), "
        ".load_case(case_id), .fetch_case(case_id)"
    )


def _write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dump_run_bundle(
    out_dir: str, case_id: str, report: Dict[str, Any], args: argparse.Namespace
) -> None:
    """Persist a browsable bundle under out/<case_id>/ without altering stdout behavior."""
    base = Path(out_dir) / case_id
    base.mkdir(parents=True, exist_ok=True)

    # full report for archival
    _write_json(base / "report.json", report)

    # run args (for reproducibility)
    try:
        _write_json(base / "run_args.json", vars(args))
    except Exception:
        pass

    # summary (same content you print when --out is not set)
    summary = {
        "case_id": case_id,
        "consensus": report.get("consensus", {}),
        "debate_skipped": report.get("debate", {}).get("debate_skipped", None),
    }
    _write_json(base / "summary.json", summary)

    # round1 per-expert
    r1_dir = base / "round1"
    r1_dir.mkdir(exist_ok=True)
    for item in report.get("round1", []):
        # item may be [eid, payload] (list/tuple) or {"expert_id":..., ...}
        if isinstance(item, (list, tuple)) and len(item) == 2:
            eid, payload = item[0], item[1]
        elif isinstance(item, dict) and "expert_id" in item:
            eid, payload = item["expert_id"], item
        else:
            # fallback unique filename
            eid, payload = "unknown_expert", item
        _write_json(r1_dir / f"{eid}.json", payload)

    # debate plan + transcripts
    debate = report.get("debate", {})
    deb_dir = base / "debate"
    deb_dir.mkdir(exist_ok=True)
    if debate:
        _write_json(
            deb_dir / "debate_plan.json",
            {
                "debate_plan": debate.get("debate_plan", {}),
                "debate_skipped": debate.get("debate_skipped", False),
            },
        )
        transcripts = debate.get("transcripts", {})
        if isinstance(transcripts, dict):
            for qid, turns in transcripts.items():
                rows: List[Dict[str, Any]] = []
                if isinstance(turns, list):
                    for t in turns:
                        if isinstance(t, dict):
                            rows.append(t)
                        else:
                            rows.append({"raw": t})
                _write_jsonl(deb_dir / f"{qid}.jsonl", rows)

    # round3 per-expert
    r3_dir = base / "round3"
    r3_dir.mkdir(exist_ok=True)
    for item in report.get("round3", []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            eid, payload = item[0], item[1]
        elif isinstance(item, dict) and "expert_id" in item:
            eid, payload = item["expert_id"], item
        else:
            eid, payload = "unknown_expert", item
        _write_json(r3_dir / f"{eid}.json", payload)


# ------------------------ cli ------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="delphea-iraki",
        description="run delphea irAKI multi-expert consensus for a single case",
    )
    ap.add_argument(
        "--case", required=True, help="case id (e.g., iraki_case_123 or raw numeric)"
    )
    ap.add_argument(
        "--q",
        dest="questionnaire",
        default="questionnaire_full.json",
        help="questionnaire json path",
    )
    ap.add_argument("--panel", default="panel.json", help="expert panel json path")
    ap.add_argument(
        "--router",
        choices=["sparse", "full"],
        default="sparse",
        help="debate routing strategy",
    )
    ap.add_argument("--out", default=None, help="optional output json path")
    ap.add_argument(
        "--out-dir",
        default="out",
        help="directory to write a browsable run bundle (default: out/CASE_ID)",
    )

    # backend config
    ap.add_argument(
        "--endpoint-url", default=os.getenv("ENDPOINT_URL", "http://localhost:8000")
    )
    ap.add_argument(
        "--model-name", default=os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
    )

    # per-round temperatures
    ap.add_argument("--temperature-r1", type=float, default=0.3)
    ap.add_argument("--temperature-r2", type=float, default=0.6)
    ap.add_argument("--temperature-r3", type=float, default=0.3)

    # moderator behavior
    ap.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="per-expert retry attempts on validation failure",
    )

    # context window and prompt budget knobs (defaults sized for big-context backends)
    ap.add_argument(
        "--ctx-window",
        type=int,
        default=102400,
        help="context window tokens supported by the backend/model",
    )
    ap.add_argument(
        "--max-notes",
        type=int,
        default=1024,
        help="max number of notes to pass into prompts",
    )
    ap.add_argument(
        "--note-char-cap",
        type=int,
        default=10240,
        help="truncate each note to this many chars",
    )
    ap.add_argument(
        "--total-chars-cap",
        type=int,
        default=600000,
        help="overall char budget across all notes",
    )

    # logging / consensus
    ap.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="increase verbosity (-v info, -vv debug)",
    )
    ap.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="consensus threshold for irAKI verdict (default 0.5)",
    )
    return ap.parse_args(argv)


# ------------------------ main ------------------------


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    _configure_logging(args.verbose)

    LOG.info("starting delphea-iraki")
    LOG.info("questionnaire=%s panel=%s", args.questionnaire, args.panel)
    LOG.info("endpoint_url=%s model_name=%s", args.endpoint_url, args.model_name)

    # questionnaire sanity
    qpath = Path(args.questionnaire)
    if not qpath.exists():
        raise FileNotFoundError(f"questionnaire file not found: {args.questionnaire}")
    try:
        qids = load_qids(args.questionnaire)
        LOG.info("questionnaire qids: %d", len(qids))
    except Exception as e:
        raise RuntimeError(
            f"failed to load questionnaire ids from {args.questionnaire}: {e}"
        ) from e

    # backend + experts
    backend = _build_backend(
        endpoint_url=args.endpoint_url,
        model_name=args.model_name,
        t_r1=args.temperature_r1,
        t_r2=args.temperature_r2,
        t_r3=args.temperature_r3,
    )
    # advertise true context window to validators
    if hasattr(backend, "set_context_window"):
        backend.set_context_window(args.ctx_window)
    elif not hasattr(backend, "capabilities"):

        def _caps(_ctx=args.ctx_window):
            return {
                "context_window": int(_ctx),
                "json_mode": getattr(backend, "supports_json_mode", False),
            }

        backend.capabilities = _caps  # type: ignore[attr-defined]

    panel_cfg = _load_panel(args.panel)
    experts = _build_experts(panel_cfg, backend)
    LOG.info("constructed expert panel: %s", [e.expert_id for e in experts])

    # router + aggregator + moderator
    router = _select_router(args.router)
    aggregator = Aggregator(decision_threshold=getattr(args, "decision_threshold", 0.5))
    moderator = Moderator(
        experts=experts,
        questionnaire_path=args.questionnaire,
        router=router,
        aggregator=aggregator,
        logger=logging.getLogger("moderator"),
        max_retries=args.max_retries,
    )

    # dataloader
    loader = DataLoader()

    # case fetch + prompt budget
    case_id = _normalize_case_id(args.case)
    raw_case = _fetch_case(loader, case_id)
    raw_case.setdefault("case_id", case_id)  # guarantees downstream availability
    if not isinstance(raw_case, dict) or not raw_case:
        raise ValueError(f"case not found or invalid structure for id: {case_id}")

    case = _apply_prompt_budget(
        raw_case,
        max_notes=args.max_notes,
        note_char_cap=args.note_char_cap,
        total_chars_cap=args.total_chars_cap,
    )
    b = case.get("_prompt_budget_applied", {})
    LOG.info(
        "prompt budget applied: selected_notes=%s max_notes=%s note_char_cap=%s total_chars_cap=%s",
        b.get("selected_notes"),
        b.get("max_notes"),
        b.get("note_char_cap"),
        b.get("total_chars_cap"),
    )

    LOG.info("[case %s] running r1 → debate → r3 → aggregate", case_id)
    report = moderator.run_case(case)

    # Always dump a browsable bundle under out_dir/case_id
    try:
        _dump_run_bundle(args.out_dir, case_id, report, args)
        LOG.info("wrote run bundle to %s", (Path(args.out_dir) / case_id))
    except Exception as e:
        LOG.warning("failed to write run bundle: %s", e)

    # Preserve existing behavior
    if args.out:
        _write_json(args.out, report)
        LOG.info("wrote results to %s", args.out)
    else:
        summary = {
            "case_id": case_id,
            "consensus": report.get("consensus", {}),
            "debate_skipped": report.get("debate", {}).get("debate_skipped", None),
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
