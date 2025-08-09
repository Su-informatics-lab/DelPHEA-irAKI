# delphea_iraki.py
# cli entrypoint wiring config → dataloader → agents → run (single or batch)
# vllm-only client: require explicit --endpoint-url and --model-name. no env fallbacks.

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from aggregator import WeightedMeanAggregator
from dataloader import DataLoader
from expert import Expert
from moderator import Moderator
from router import FullRouter, SparseRouter
from schema import load_qids  # for early qid introspection
from vllm_backend import VLLMBackend

# -------------------- helpers: id parsing & dataloader glue --------------------

_ID_RE = re.compile(r"(\d+)")


def _to_case_id_from_patient_id(x: str | int) -> str:
    """normalize arbitrary patient_id inputs (e.g., 'P123', '123') to 'iraki_case_123'."""
    if isinstance(x, int):
        return f"iraki_case_{x}"
    s = str(x).strip()
    m = _ID_RE.search(s)
    if not m:
        raise ValueError(f"could not extract an integer patient_id from: {x!r}")
    return f"iraki_case_{int(m.group(1))}"


def _load_panel(panel_path: str) -> List[Dict[str, Any]]:
    """load expert panel entries and return the list of experts.

    supports either:
      {"expert_panel": {"experts": [...]}}  # legacy
    or
      {"experts": [...]}                    # preferred
    """
    p = Path(panel_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"panel json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "experts" in data and isinstance(data["experts"], list):
            return data["experts"]
        if "expert_panel" in data and isinstance(data["expert_panel"], dict):
            ep = data["expert_panel"].get("experts")
            if isinstance(ep, list):
                return ep
    raise ValueError(f"panel format must contain an experts list: {p}")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _case_from_loader(case_like: Any, loader: DataLoader) -> Tuple[str, Dict[str, Any]]:
    """accept either a patient_id (string/int) or a full case dict; return (case_id, case_dict)."""
    # full case dict path (already structured like DataLoader output)
    if isinstance(case_like, dict) and "clinical_notes" in case_like:
        cid = str(
            case_like.get("case_id")
            or case_like.get("person_id")
            or case_like.get("patient_id")
            or "case_001"
        )
        return cid, case_like

    # patient_id string/int path → use DataLoader
    if isinstance(case_like, (str, int)):
        case_id = _to_case_id_from_patient_id(case_like)
        case_dict = loader.load_patient_case(case_id)
        return case_id, case_dict

    # minimal dict with id fields (load via DataLoader)
    if isinstance(case_like, dict):
        any_id = (
            case_like.get("case_id")
            or case_like.get("person_id")
            or case_like.get("patient_id")
        )
        if not any_id:
            raise ValueError(
                "case dict must include one of: case_id, person_id, patient_id"
            )
        case_id = (
            any_id
            if str(any_id).startswith("iraki_case_")
            else _to_case_id_from_patient_id(any_id)
        )
        case_dict = loader.load_patient_case(case_id)
        return case_id, case_dict

    raise ValueError(f"unsupported case object type: {type(case_like)}")


def _iter_cases(
    case_arg: Optional[str], cases_path: Optional[str], loader: DataLoader
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """yield (case_id, case_dict) for single or batch inputs."""
    if cases_path:
        p = Path(cases_path)
        if not p.exists():
            raise FileNotFoundError(f"--cases path not found: {p}")
        if p.suffix.lower() == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        obj = line  # treat as a plain patient_id
                    yield _case_from_loader(obj, loader)
            return
        if p.suffix.lower() == ".json":
            obj = _read_json(p)
            seq = obj.get("cases") if isinstance(obj, dict) and "cases" in obj else obj
            if isinstance(seq, list):
                for item in seq:
                    yield _case_from_loader(item, loader)
                return
            yield _case_from_loader(obj, loader)
            return
        raise ValueError(f"--cases only supports .jsonl or .json, got {p.suffix}")

    # single-case mode
    if case_arg is None:
        raise ValueError("provide --case/--case-id (patient_id) or --cases (file).")
    p = Path(case_arg)
    if p.exists():
        obj = _read_json(p)
        yield _case_from_loader(obj, loader)
        return
    # try JSON first; if it fails, treat as patient_id string
    try:
        obj = json.loads(case_arg)
        yield _case_from_loader(obj, loader)
    except Exception:
        yield _case_from_loader(case_arg, loader)


# -------------------- main --------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DelPHEA-irAKI: run multi-agent Delphi consensus for irAKI"
    )

    # config: questionnaire/panel/router
    parser.add_argument(
        "-q",
        "--questionnaire",
        default=str((Path(__file__).parent / "questionnaire.json").resolve()),
        help="path to questionnaire json (default: questionnaire.json)",
    )
    parser.add_argument(
        "-p",
        "--panel",
        default=str((Path(__file__).parent / "panel.json").resolve()),
        help="path to expert panel json (default: panel.json)",
    )
    parser.add_argument(
        "--router", choices=["sparse", "full"], default="sparse", help="routing mode"
    )

    parser.add_argument(
        "--endpoint",
        dest="endpoint_url",
        default="http://localhost:8000",  # default: local vLLM
        help="openai-compatible base URL (no trailing /v1), e.g., http://localhost:8000",
    )

    parser.add_argument(
        "--model",
        dest="model_name",
        default="openai/gpt-oss-120b",  # default: oss model served in your cluster
        help="served model id as shown by /v1/models, e.g., openai/gpt-oss-120b",
    )

    # data loader
    parser.add_argument(
        "--data-dir", default="irAKI_data", help="root directory for patient data"
    )
    parser.add_argument(
        "--use-dummy-loader", action="store_true", help="use DataLoader dummy mode"
    )

    # input (choose one); support alias --case-id
    parser.add_argument(
        "--case",
        "--case-id",
        dest="case",
        default=None,
        help="patient_id string/int, JSON string, or path to .json (single case)",
    )
    parser.add_argument(
        "--cases", default=None, help="path to .jsonl or .json with list/cases object"
    )

    # output & logging
    parser.add_argument(
        "--outdir", default="out", help="directory to write one report per case"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity",
    )

    # optional: just print served models and exit
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="ping endpoint and print served models, then exit",
    )

    args = parser.parse_args()

    # sanity
    args.endpoint_url = re.sub(r"/v1/?$", "", args.endpoint_url.rstrip("/"))
    if not re.match(r"^https?://", args.endpoint_url):
        raise ValueError(
            f"--endpoint must start with http:// or https:// (got {args.endpoint_url!r})"
        )
    if not args.model_name.strip():
        raise ValueError("--model/--model-name cannot be empty")

    # logging config
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("delphea")
    log.info("starting delphea-iraki")
    log.info("questionnaire=%s panel=%s", args.questionnaire, args.panel)
    log.info("endpoint_url=%s model_name=%s", args.endpoint_url, args.model_name)

    # health check mode
    if args.health_check:
        import requests

        url = args.endpoint_url.rstrip("/") + "/v1/models"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        served = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        print(
            json.dumps(
                {"endpoint": args.endpoint_url, "served_models": served}, indent=2
            )
        )
        return

    # resolve and sanity check questionnaire/panel
    qpath = Path(args.questionnaire).expanduser().resolve()
    ppath = Path(args.panel).expanduser().resolve()
    if not qpath.exists():
        raise FileNotFoundError(f"questionnaire json not found: {qpath}")
    if not ppath.exists():
        raise FileNotFoundError(f"panel json not found: {ppath}")
    args.questionnaire = str(qpath)
    args.panel = str(ppath)

    # dataloader (fail fast if data_dir missing)
    loader = DataLoader(data_dir=args.data_dir, use_dummy=args.use_dummy_loader)
    if not loader.is_available():
        raise RuntimeError(
            "DataLoader is not available. check --data-dir to point at your dataset."
        )

    # backend
    backend = VLLMBackend(
        model=args.model_name, base_url=args.endpoint_url, api_key=None
    )

    # experts (reused across cases)
    experts = [
        Expert(
            expert_id=ex["id"],
            specialty=ex["specialty"],
            persona=ex,
            backend=backend,
        )
        for ex in _load_panel(args.panel)
    ]
    log.info("constructed expert panel: %s", [e.expert_id for e in experts])

    # router + moderator
    router = SparseRouter() if args.router == "sparse" else FullRouter()
    moderator = Moderator(
        experts=experts,
        questionnaire_path=args.questionnaire,
        router=router,
        aggregator=WeightedMeanAggregator(),
    )

    # prepare output dir if requested
    outdir: Optional[Path] = None
    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    # precompute qids for nicer logs
    qids = load_qids(args.questionnaire)
    log.debug("questionnaire qids: %s", qids)

    # run
    results: List[Dict[str, Any]] = []
    for idx, (case_id, case) in enumerate(
        _iter_cases(args.case, args.cases, loader), start=1
    ):
        log.info(
            "[case %s] round 1 start — fan-out %d experts × %d qids",
            case_id,
            len(experts),
            len(qids),
        )
        r1 = moderator.assess_round(1, case)
        log.info(
            "[case %s] round 1 done — received %d/%d assessments",
            case_id,
            len(r1),
            len(experts),
        )
        log.debug("[case %s] r1 experts: %s", case_id, [eid for eid, _ in r1])

        log.info("[case %s] debate planning & collection…", case_id)
        debate_ctx = moderator.detect_and_run_debates(r1, case)
        solicitations = sum(len(v) for v in debate_ctx.get("debate_plan", {}).values())
        log.info(
            "[case %s] debate done — %d solicitations across %d qids",
            case_id,
            solicitations,
            len(debate_ctx.get("debate_plan", {})),
        )

        log.info("[case %s] round 3 start — reassess all experts", case_id)
        r3 = moderator.assess_round(3, case, debate_ctx)
        log.info(
            "[case %s] round 3 done — received %d/%d assessments",
            case_id,
            len(r3),
            len(experts),
        )

        p_hat = getattr(
            consensus, "iraki_probability", getattr(consensus, "p_iraki", None)
        )
        log.info(
            "[case %s] aggregation done — p=%.3f, ci=%s",
            case_id,
            p_hat,
            consensus.ci_iraki,
        )
        consensus = moderator.aggregator.aggregate([a for _, a in r3])
        log.info(
            "[case %s] aggregation done — p=%.3f, ci=%s",
            case_id,
            consensus.p_iraki,
            consensus.ci_iraki,
        )

        report = {
            "round1": [(eid, a.model_dump()) for eid, a in r1],
            "debate": debate_ctx,
            "round3": [(eid, a.model_dump()) for eid, a in r3],
            "consensus": consensus.model_dump(),
        }
        report_with_id = {"case_id": case_id, **report}

        if outdir:
            out_path = outdir / f"report_{idx:03d}_{case_id}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(report_with_id, f, indent=2)
            log.info("[case %s] wrote %s", case_id, out_path)
        else:
            results.append(report_with_id)

    # print to stdout if not writing to files
    if not outdir:
        is_batch = bool(args.cases)
        print(json.dumps(results if is_batch else results[0], indent=2))


if __name__ == "__main__":
    main()
