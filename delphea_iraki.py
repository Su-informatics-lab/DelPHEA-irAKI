# delphea_iraki.py
# cli entrypoint wiring config → dataloader → agents → run (single or batch)
# token policy is controlled via cli (ctx window, output budget, retries, etc).

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import validators as _validators  # to set token policy defaults
from aggregator import WeightedMeanAggregator
from dataloader import DataLoader
from expert import Expert
from llm_backend import LLMBackend
from moderator import Moderator
from router import FullRouter, SparseRouter
from schema import load_qids  # for logging qid count

# ------------------------------- logging setup -------------------------------

LOG = logging.getLogger("delphea_iraki")
logging.basicConfig(
    level=os.environ.get("DELPHEA_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# -------------------- helpers: id parsing & dataloader glue --------------------

_ID_RE = re.compile(r"(\d+)")


def _to_case_id_from_patient_id(x: str | int) -> str:
    """normalize arbitrary patient_id inputs (e.g., 'P123', '123') to 'iraki_case_123'."""
    if isinstance(x, int):
        return f"iraki_case_{x}"
    s = str(x)
    m = _ID_RE.search(s)
    if not m:
        raise ValueError(f"cannot parse patient/case id from {x!r}")
    return f"iraki_case_{m.group(1)}"


def _patch_backend_caps(backend: Any, ctx_window: int) -> None:
    """ensure backend.capabilities() reports our chosen context window (no recursion)."""
    orig_caps = getattr(backend, "capabilities", None)

    def _caps() -> Dict[str, Any]:
        base: Dict[str, Any] = {}
        try:
            if callable(orig_caps):
                got = orig_caps()
                if isinstance(got, dict):
                    base.update(got)
        except Exception:
            pass
        base["context_window"] = int(ctx_window)
        return base

    setattr(backend, "capabilities", _caps)


def _prune_case_text(case: Dict[str, Any], max_chars: int) -> Tuple[int, int]:
    """trim total free-text in case['notes'] to <= max_chars (best-effort, in-place).

    returns:
        (chars_before, chars_after)
    """
    if max_chars <= 0:
        return (0, 0)

    # discover note list
    notes = None
    for key in ("notes", "clinical_notes", "CLINICAL_NOTES"):
        if isinstance(case.get(key), list):
            notes = case[key]
            break
    if notes is None:
        return (0, 0)

    text_keys = ("REPORT_TEXT", "report_text", "NOTE_TEXT", "text")
    before = 0
    for n in notes:
        if isinstance(n, dict):
            for tk in text_keys:
                v = n.get(tk)
                if isinstance(v, str):
                    before += len(v)

    # short-circuit if already within budget
    if before <= max_chars:
        return (before, before)

    # greedy trim: walk notes in order and truncate spillover
    budget = max_chars
    for n in notes:
        if not isinstance(n, dict):
            continue
        for tk in text_keys:
            v = n.get(tk)
            if not isinstance(v, str) or not v:
                continue
            if budget <= 0:
                n[tk] = ""
                continue
            if len(v) <= budget:
                budget -= len(v)
            else:
                n[tk] = v[:budget]
                budget = 0

    # recount
    after = 0
    for n in notes:
        if isinstance(n, dict):
            for tk in text_keys:
                v = n.get(tk)
                if isinstance(v, str):
                    after += len(v)

    return (before, after)


def _load_panel(panel_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """load expert panel config; supports dict{specialty: persona} or list of entries."""
    with panel_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    panel: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(raw, dict):
        for spec, persona in raw.items():
            panel.append((str(spec), persona or {}))
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                spec = item.get("specialty") or item.get("name") or item.get("id")
                persona = item.get("persona", {})
                if not spec:
                    raise ValueError(f"panel entry missing specialty: {item}")
                panel.append((str(spec), persona or {}))
    else:
        raise ValueError("panel.json must be an object or array")
    return panel


def _build_router(kind: str) -> Any:
    if kind.lower() == "sparse":
        try:
            return SparseRouter()
        except Exception:
            LOG.warning("failed to init SparseRouter; falling back to FullRouter")
    return FullRouter()


# --------------------------------- argument io ---------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DelPHEA-irAKI — multi-expert Delphi consensus for irAKI"
    )
    parser.add_argument(
        "--case",
        required=True,
        help="case id or patient id (e.g., iraki_case_123 or P123)",
    )
    parser.add_argument("--q", required=True, help="path to questionnaire json")
    parser.add_argument(
        "--panel", default="panel.json", help="path to expert panel json"
    )
    parser.add_argument(
        "--router", choices=["full", "sparse"], default="full", help="routing strategy"
    )
    parser.add_argument(
        "--endpoint-url", default=os.getenv("DELPHEA_ENDPOINT", "http://localhost:8000")
    )
    parser.add_argument(
        "--model-name", default=os.getenv("DELPHEA_MODEL", "openai/gpt-oss-120b")
    )
    parser.add_argument(
        "--out", default="delphea_run.json", help="where to write the run report (json)"
    )

    # token policy knobs (centralized in validators via defaults we set below)
    parser.add_argument(
        "--ctx-window",
        type=int,
        default=200_000,
        help="model context window tokens (e.g., 200000)",
    )
    parser.add_argument(
        "--out-tokens-init",
        type=int,
        default=6_000,
        help="initial output token budget per attempt",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="number of repair retries on schema/JSON errors",
    )
    parser.add_argument(
        "--retry-factor",
        type=float,
        default=1.75,
        help="geometric escalation factor per retry",
    )
    parser.add_argument(
        "--reserve-tokens",
        type=int,
        default=1_024,
        help="safety buffer tokens not used by output",
    )

    # per-round temperatures (r1, r2/debate, r3)
    parser.add_argument(
        "--temperature-r1", type=float, default=0.0, help="temperature for round 1"
    )
    parser.add_argument(
        "--temperature-r2",
        type=float,
        default=0.3,
        help="temperature for round 2 (debate)",
    )
    parser.add_argument(
        "--temperature-r3", type=float, default=0.0, help="temperature for round 3"
    )

    # optional input length control (best-effort)
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=0,
        help="cap total free-text chars from notes (0=off)",
    )
    return parser.parse_args(argv)


# ------------------------------------- main ------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    LOG.info("starting delphea-iraki")
    LOG.info("questionnaire=%s panel=%s", args.q, args.panel)
    LOG.info("endpoint_url=%s model_name=%s", args.endpoint_url, args.model_name)

    # set validator token policy defaults based on cli (no env control)
    _validators.DEFAULT_CTX_WINDOW = int(args.ctx_window)
    _validators.DEFAULT_OUT_TOKENS_INIT = int(args.out_tokens_init)
    _validators.DEFAULT_RETRIES = int(args.retries)
    _validators.DEFAULT_RETRY_FACTOR = float(args.retry_factor)
    _validators.DEFAULT_RESERVE_TOKENS = int(args.reserve_tokens)

    # init backend and patch reported context window
    backend = LLMBackend(endpoint_url=args.endpoint_url, model_name=args.model_name)
    _patch_backend_caps(backend, ctx_window=args.ctx_window)

    # load data
    LOG.info("Loading patient data...")
    loader = DataLoader()
    case_id = (
        args.case
        if str(args.case).startswith("iraki_case_")
        else _to_case_id_from_patient_id(args.case)
    )

    # fetch the case (support common method names in DataLoader)
    case: Optional[Dict[str, Any]] = None
    method_names = (
        "get_case",
        "load_case",
        "get_patient_case",
        "select_case",
        "load_patient_case",
    )
    for fn in method_names:
        if hasattr(loader, fn):
            try:
                case = getattr(loader, fn)(case_id)
                break
            except TypeError:
                try:
                    case = getattr(loader, fn)(case_id=case_id)
                    break
                except Exception:
                    continue
            except Exception:
                continue
    if case is None:
        try:
            sample = loader.get_available_patients(limit=3)
            LOG.error("could not load case %r; samples: %s", case_id, sample)
        except Exception:
            pass
        raise RuntimeError(f"could not load case {case_id!r} via DataLoader")

    LOG.info(
        "Loaded %d patients",
        getattr(loader, "n_patients", 0) or getattr(loader, "num_patients", 0) or 0,
    )

    # optional input pruning to keep giant notes sane
    if args.max_input_chars and args.max_input_chars > 0:
        before, after = _prune_case_text(case, args.max_input_chars)
        LOG.info(
            "pruned case text: %d → %d chars (limit=%d)",
            before,
            after,
            args.max_input_chars,
        )

    # build expert panel
    panel = _load_panel(Path(args.panel))
    LOG.info("constructed expert panel: %s", [spec for spec, _ in panel])

    # show qid count (helpful for budgeting)
    try:
        qids = load_qids(args.q)
        LOG.info("questionnaire qids: %d", len(qids))
    except Exception:
        LOG.info("questionnaire qids: (unavailable)")

    experts: List[Expert] = [
        Expert(
            expert_id=str(spec),
            specialty=str(spec),
            persona=(persona or {}),
            backend=backend,
            temperature_r1=args.temperature_r1,
            temperature_r2=args.temperature_r2,  # ← r2, not "debate"
            temperature_r3=args.temperature_r3,
        )
        for spec, persona in panel
    ]

    # router & aggregator
    router = _build_router(args.router)
    aggregator = WeightedMeanAggregator()

    # moderator orchestrates rounds w/ questionnaire path
    moderator = Moderator(
        experts=experts, router=router, aggregator=aggregator, qpath=args.q
    )

    # run rounds (r1 → debate/r2 → r3)
    LOG.info("[case %s] round 1 start — fan-out %d experts", case_id, len(experts))
    r1 = moderator.assess_round(1, case)

    r_debate = None
    if hasattr(moderator, "run_debate"):
        LOG.info("[case %s] debate start", case_id)
        try:
            r_debate = moderator.run_debate(case, r1)
        except TypeError:
            r_debate = moderator.run_debate(case_id=case_id, r1=r1)

    r3 = None
    if hasattr(moderator, "assess_round"):
        LOG.info("[case %s] round 3 start", case_id)
        r3 = moderator.assess_round(3, case)

    # aggregate to consensus if Moderator doesn’t already do it
    consensus = None
    if hasattr(moderator, "aggregate"):
        try:
            consensus = moderator.aggregate(case_id, r1=r1, debate=r_debate, r3=r3)
        except TypeError:
            consensus = moderator.aggregate(case_id, r1, r_debate, r3)

    # build a serializable report
    report = {
        "case_id": case_id,
        "model_name": args.model_name,
        "endpoint_url": args.endpoint_url,
        "token_policy": {
            "ctx_window": args.ctx_window,
            "out_tokens_init": args.out_tokens_init,
            "retries": args.retries,
            "retry_factor": args.retry_factor,
            "reserve_tokens": args.reserve_tokens,
        },
        "temperatures": {
            "r1": args.temperature_r1,
            "r2": args.temperature_r2,
            "r3": args.temperature_r3,
        },
        "round1": r1,
        "debate": r_debate,
        "round3": r3,
        "consensus": consensus,
    }

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    LOG.info("run saved → %s", out_path.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        LOG.exception("fatal error: %s", e)
        sys.exit(1)
