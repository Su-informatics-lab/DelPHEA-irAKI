# tests/test_postcheck_r1r3_vs_questionnaire.py
# post-run validation: round1/round3 per-expert files must match questionnaire QID order & set

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# env overrides for flexibility
OUT_ROOT = Path(os.environ.get("OUT_DIR", "out"))
QUESTIONNAIRE = Path(os.environ.get("QUESTIONNAIRE", "questionnaire_full.json"))

# per-question dict fields we validate for order + membership
PQ_FIELDS = ["scores", "rationale", "evidence", "q_confidence", "importance"]


def _load_questionnaire_qids(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"questionnaire file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # expected shape: {"questionnaire": {"questions": [{"id": "Q1"}, ...]}}
    qnode = (data.get("questionnaire") or {}).get("questions") or []
    qids = [
        str(q.get("id")).strip() for q in qnode if isinstance(q, dict) and q.get("id")
    ]
    if not qids:
        raise ValueError(f"no questions found in questionnaire: {path}")
    return qids


def _iter_round_files(root: Path, round_dir: str) -> Iterable[Path]:
    """yield all per-expert json files under out/<CASE_ID>/<round_dir>/*.json"""
    if not root.exists():
        return
    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        rdir = case_dir / round_dir
        if rdir.is_dir():
            for f in sorted(rdir.glob("*.json")):
                yield f


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _first_order_mismatch(a: List[str], b: List[str]) -> Tuple[int, str, str]:
    """return (idx, a[idx], b[idx]) where they first differ; (-1, '', '') if equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        # differing lengths but same prefix
        return (
            min(len(a), len(b)),
            a[min(len(a), len(b)) - 1] if a else "",
            b[min(len(a), len(b)) - 1] if b else "",
        )
    return -1, "", ""


def _check_per_question_fields(payload: Dict, expected_qids: List[str]) -> List[str]:
    errors: List[str] = []
    for field in PQ_FIELDS:
        if field not in payload:
            errors.append(f"missing field '{field}'")
            continue
        node = payload[field]
        if not isinstance(node, dict):
            errors.append(f"field '{field}' must be an object/dict")
            continue

        keys = list(node.keys())
        set_keys, set_expected = set(keys), set(expected_qids)
        if set_keys != set_expected:
            missing = sorted(set_expected - set_keys)
            extra = sorted(set_keys - set_expected)
            if missing:
                errors.append(f"field '{field}': missing qids: {missing}")
            if extra:
                errors.append(f"field '{field}': extra qids: {extra}")

        # order check (preserves json insertion order in python ≥3.7)
        if keys != expected_qids:
            i, got, exp = _first_order_mismatch(keys, expected_qids)
            errors.append(
                f"field '{field}': qid order mismatch at index {i}: got '{got}' vs expected '{exp}'"
            )

        # extra numeric sanity for importance
        if field == "importance" and set_keys == set_expected:
            total = 0
            bad = []
            for qid, v in node.items():
                try:
                    iv = int(v)
                except Exception:
                    bad.append(qid)
                    continue
                total += iv
                if iv < 0:
                    bad.append(qid)
            if bad:
                errors.append(f"field 'importance': non-integer or negative for {bad}")
            if total != 100:
                errors.append(f"field 'importance': sum != 100 (got {total})")

    # id fields are helpful for debugging
    if not payload.get("case_id"):
        errors.append("missing 'case_id'")
    if not payload.get("expert_id"):
        errors.append("missing 'expert_id'")

    return errors


def _run_round_checks(round_dir: str, expected_qids: List[str]) -> List[str]:
    errors: List[str] = []
    files = list(_iter_round_files(OUT_ROOT, round_dir))
    if not files:
        errors.append(f"no files found under {OUT_ROOT}/<CASE_ID>/{round_dir}/")
        return errors
    for f in files:
        try:
            payload = _read_json(f)
        except Exception as e:
            errors.append(f"{f}: not valid json: {e}")
            continue
        perr = _check_per_question_fields(payload, expected_qids)
        if perr:
            errors.append(f"{f}:\n  - " + "\n  - ".join(perr))
    return errors


def test_round1_order_matches_questionnaire():
    expected_qids = _load_questionnaire_qids(QUESTIONNAIRE)
    errors = _run_round_checks("round1", expected_qids)
    assert not errors, "\n".join(errors)


def test_round3_order_matches_questionnaire():
    expected_qids = _load_questionnaire_qids(QUESTIONNAIRE)
    errors = _run_round_checks("round3", expected_qids)
    assert not errors, "\n".join(errors)
