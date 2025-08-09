# delphea_iraki.py
# cli entrypoint wiring config → dataloader → agents → run (single or batch)
# defaults to vllm backend; --case accepts a plain patient_id string for fast testing.

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from aggregator import WeightedMeanAggregator
from dataloader import DataLoader
from expert import Expert
from llm_backend import LLMBackend
from moderator import Moderator
from router import FullRouter, SparseRouter
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

    # backend: default vllm
    parser.add_argument("--backend", choices=["vllm", "dummy"], default="vllm")
    parser.add_argument(
        "--base-url",
        default=None,
        help="vllm base url (no /v1), e.g., http://localhost:8000",
    )
    parser.add_argument("--model", default=None, help="model name as served by vllm")
    parser.add_argument("--api-key", default=None, help="optional bearer token")

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

    # output
    parser.add_argument(
        "--outdir", default="out", help="directory to write one report per case"
    )

    args = parser.parse_args()

    # resolve and sanity check questionnaire/panel
    qpath = Path(args.questionnaire).expanduser().resolve()
    ppath = Path(args.panel).expanduser().resolve()
    if not qpath.exists():
        raise FileNotFoundError(f"questionnaire json not found: {qpath}")
    if not ppath.exists():
        raise FileNotFoundError(f"panel json not found: {ppath}")
    args.questionnaire = str(qpath)
    args.panel = str(ppath)

    # backend settings from env fallbacks
    if args.base_url is None:
        args.base_url = (
            os.getenv("VLLM_ENDPOINT")
            or os.getenv("VLLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "http://127.0.0.1:8000"
        )
    # normalize: users sometimes paste a base that already includes /v1
    args.base_url = re.sub(r"/v1/?$", "", args.base_url.rstrip("/"))

    if args.model is None:
        args.model = (
            os.getenv("OPENAI_MODEL")
            or os.getenv("VLLM_MODEL")
            or "openai/gpt-oss-120b"
        )

    # api key (if your vllm server requires it)
    if args.api_key is None:
        args.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY")

    # propagate to env for modules that read from environment
    os.environ["VLLM_BASE_URL"] = args.base_url
    os.environ["OPENAI_BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
        os.environ["VLLM_API_KEY"] = args.api_key
    os.environ["OPENAI_MODEL"] = args.model
    os.environ["VLLM_MODEL"] = args.model

    # fail fast if endpoint/model is wrong (only for vllm backend)
    if args.backend == "vllm":
        try:
            base = args.base_url.rstrip("/")
            r = requests.get(f"{base}/v1/models", timeout=3)
            r.raise_for_status()
            served = {
                m.get("id") for m in r.json().get("data", []) if isinstance(m, dict)
            }
            if args.model not in served:
                raise RuntimeError(
                    f"requested model '{args.model}' not served at {base}. "
                    f"available: {sorted(served) or 'none'}"
                )
        except Exception as e:
            print(f"[fatal] vllm endpoint/model check failed: {e}", file=sys.stderr)
            sys.exit(2)

    # dataloader (fail fast if data_dir missing and not using dummy)
    loader = DataLoader(data_dir=args.data_dir, use_dummy=args.use_dummy_loader)
    if not loader.is_available():
        raise RuntimeError(
            "DataLoader is not available. check --data-dir or pass --use-dummy-loader for quick tests."
        )

    # backend selection
    if args.backend == "vllm":
        backend = VLLMBackend(
            model=args.model, base_url=args.base_url, api_key=args.api_key
        )
    else:
        # debugging only
        from schema import load_qids

        class DummyBackend(LLMBackend):
            """minimal backend that echoes neutral-but-valid payloads."""

            def assess_round1(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
                qids = load_qids(expert_ctx["questionnaire_path"])
                return {
                    "scores": {q: 7 for q in qids},
                    "evidence": {q: "placeholder evidence" for q in qids},
                    "clinical_reasoning": "placeholder reasoning",
                    "p_iraki": 0.7,
                    "ci_iraki": [0.6, 0.8],
                    "confidence": 0.8,
                    "differential_diagnosis": ["ATIN", "ATN"],
                    "primary_diagnosis": "irAKI",
                }

            def assess_round3(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
                qids = load_qids(expert_ctx["questionnaire_path"])
                return {
                    "scores": {q: 8 for q in qids},
                    "evidence": {q: "updated evidence after debate" for q in qids},
                    "p_iraki": 0.75,
                    "ci_iraki": [0.65, 0.85],
                    "confidence": 0.85,
                    "changes_from_round1": {"overall": "nudged up after debate"},
                    "debate_influence": "convinced by nephrology/rheum arguments",
                    "verdict": True,
                    "final_diagnosis": "irAKI",
                    "confidence_in_verdict": 0.85,
                    "recommendations": [
                        "hold ICI",
                        "consider steroids",
                        "nephrology consult",
                    ],
                }

            def debate(self, expert_ctx: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "text": "constructive rebuttal text",
                    "citations": [],
                    "satisfied": True,
                }

        backend = DummyBackend()

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

    # run
    results: List[Dict[str, Any]] = []
    for idx, (case_id, case) in enumerate(
        _iter_cases(args.case, args.cases, loader), start=1
    ):
        report = moderator.run_case(case)
        report_with_id = {"case_id": case_id, **report}
        if outdir:
            out_path = outdir / f"report_{idx:03d}_{case_id}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(report_with_id, f, indent=2)
        else:
            results.append(report_with_id)

    # print to stdout if not writing to files
    if not outdir:
        is_batch = bool(args.cases)
        print(json.dumps(results if is_batch else results[0], indent=2))


if __name__ == "__main__":
    main()
