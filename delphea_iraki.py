# delphea_iraki.py
# cli entrypoint wiring config → dataloader → experts/moderator → run (single case)
# keeps to yagni: minimal flags, fail fast, clear logs.

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from aggregator import WeightedMeanAggregator
from dataloader import DataLoader
from expert import Expert
from llm_backend import LLMBackend
from moderator import Moderator
from router import FullRouter, SparseRouter
from schema import load_qids

LOG = logging.getLogger("delphea_iraki")
_ID_RE = re.compile(r"(\d+)")


def _configure_logging(verbosity: int) -> None:
    """configure root logger with a simple, consistent format.

    args:
        verbosity: 0=warning, 1=info, 2+=debug
    """
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
    """normalize arbitrary patient ids to canonical 'iraki_case_<digits>'.

    accepts inputs such as:
      - 'iraki_case_123456'
      - 'P123456'
      - '123456'
    """
    if not x or not isinstance(x, str):
        raise ValueError("case id must be a non-empty string")
    if x.startswith("iraki_case_"):
        return x
    m = _ID_RE.search(x)
    if not m:
        raise ValueError(f"cannot parse numeric id from: {x!r}")
    return f"iraki_case_{m.group(1)}"


def _load_panel(panel_path: str) -> List[Dict[str, Any]]:
    """load and validate expert panel config.

    supports either:
      - list[ {id, specialty, persona?} ]
      - {"experts": [ {id, specialty, persona?}, ... ]}

    returns:
        list of expert dicts.
    """
    p = Path(panel_path)
    if not p.exists():
        raise FileNotFoundError(f"panel file not found: {panel_path}")
    try:
        cfg = json.loads(p.read_text())
    except Exception as e:
        raise ValueError(f"failed to parse panel json: {e}") from e

    experts = cfg.get("experts") if isinstance(cfg, dict) else cfg
    if not isinstance(experts, list) or not experts:
        raise ValueError("panel config must contain a non-empty list of experts")

    for i, e in enumerate(experts):
        if not isinstance(e, dict):
            raise ValueError(f"panel entry {i} must be an object")
        if not e.get("id") or not e.get("specialty"):
            raise ValueError(
                f"panel entry {i} missing required fields 'id' and/or 'specialty'"
            )
        # persona is optional; normalize to dict
        if "persona" in e and e["persona"] is None:
            e["persona"] = {}
        if "persona" not in e:
            e["persona"] = {}
    return experts


def _build_backend(
    endpoint_url: str, model_name: str, t_r1: float, t_r2: float, t_r3: float
) -> LLMBackend:
    """construct llm backend and apply round temperatures if supported."""
    backend = LLMBackend(endpoint_url=endpoint_url, model_name=model_name)
    # best-effort temperature wiring; keep silent on success, warn if unsupported
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
    """instantiate expert objects from panel config."""
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
    # verify unique ids early
    ids = [x.expert_id for x in experts]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate expert_id in panel: {ids}")
    return experts


def _select_router(name: str):
    """select router strategy by name."""
    n = (name or "sparse").lower()
    if n == "sparse":
        return SparseRouter()
    if n == "full":
        return FullRouter()
    raise ValueError(f"unknown router: {name!r} (use 'sparse' or 'full')")


def _fetch_case(loader: Any, case_id: str) -> Dict[str, Any]:
    """fetch a single case dict from the dataloader using a tolerant method probe.

    tries methods in order: get_case, load_case, fetch_case.
    """
    for fn_name in ("get_case", "load_case", "fetch_case"):
        if hasattr(loader, fn_name):
            fn = getattr(loader, fn_name)
            return fn(case_id)
    raise AttributeError(
        "dataloader must expose one of: .get_case(case_id), .load_case(case_id), .fetch_case(case_id)"
    )


def _write_json(path: str, obj: Any) -> None:
    """write json to path with pretty formatting."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def parse_args(argv: List[str]) -> argparse.Namespace:
    """parse command line flags."""
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

    # backend config
    ap.add_argument(
        "--endpoint-url", default=os.getenv("ENDPOINT_URL", "http://localhost:8000")
    )
    ap.add_argument(
        "--model-name", default=os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
    )

    # per-round temperatures (user asked for r1 and r2 explicitly; we keep r3 for completeness)
    ap.add_argument("--temperature-r1", type=float, default=0.2)
    ap.add_argument("--temperature-r2", type=float, default=0.6)
    ap.add_argument("--temperature-r3", type=float, default=0.2)

    # moderator behavior
    ap.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="per-expert retry attempts on validation failure",
    )

    # logging
    ap.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="increase verbosity (-v info, -vv debug)",
    )
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """main cli entrypoint."""
    args = parse_args(argv or sys.argv[1:])
    _configure_logging(args.verbose)

    LOG.info("starting delphea-iraki")
    LOG.info("questionnaire=%s panel=%s", args.questionnaire, args.panel)
    LOG.info("endpoint_url=%s model_name=%s", args.endpoint_url, args.model_name)

    # fail early on questionnaire path; moderator will load qids/rules from it
    qpath = Path(args.questionnaire)
    if not qpath.exists():
        raise FileNotFoundError(f"questionnaire file not found: {args.questionnaire}")

    # load qids only for logging visibility; moderator enforces correctness later
    try:
        qids = load_qids(args.questionnaire)
        LOG.info("questionnaire qids: %d", len(qids))
    except Exception as e:
        raise RuntimeError(
            f"failed to load questionnaire ids from {args.questionnaire}: {e}"
        ) from e

    # build backend and experts
    backend = _build_backend(
        endpoint_url=args.endpoint_url,
        model_name=args.model_name,
        t_r1=args.temperature_r1,
        t_r2=args.temperature_r2,
        t_r3=args.temperature_r3,
    )
    panel_cfg = _load_panel(args.panel)
    experts = _build_experts(panel_cfg, backend)
    LOG.info("constructed expert panel: %s", [e.expert_id for e in experts])

    # router + aggregator
    router = _select_router(args.router)
    aggregator = WeightedMeanAggregator()

    # moderator expects questionnaire_path (not qpath kwarg name)
    moderator = Moderator(
        experts=experts,
        questionnaire_path=args.questionnaire,
        router=router,
        aggregator=aggregator,
        logger=logging.getLogger("moderator"),
        max_retries=args.max_retries,
    )

    # dataloader should self-log its own 'loading data' messages; instantiate once
    loader = DataLoader()

    # normalize case id and fetch case dict
    case_id = _normalize_case_id(args.case)
    case = _fetch_case(loader, case_id)
    if not isinstance(case, dict) or not case:
        raise ValueError(f"case not found or invalid structure for id: {case_id}")

    LOG.info("[case %s] running r1 → debate → r3 → aggregate", case_id)
    report = moderator.run_case(case)

    # dump output
    if args.out:
        _write_json(args.out, report)
        LOG.info("wrote results to %s", args.out)
    else:
        # compact stdout summary for quick inspection
        summary = {
            "case_id": case_id,
            "consensus": report.get("consensus", {}),
            "debate_skipped": report.get("debate", {}).get("debate_skipped", None),
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
