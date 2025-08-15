# test_postcheck_outputs.py
# post-run validation for debate transcripts in out/<CASE_ID>/debate/Q*.jsonl

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

QID_RE = re.compile(r"^Q(\d+)$")


def _read_jsonl(path: Path) -> List[Tuple[int, Dict]]:
    """read a jsonl file into (lineno, obj) tuples, skipping blank lines.

    raises:
        ValueError: if a non-blank line is not valid json or not a dict.
    """
    rows: List[Tuple[int, Dict]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {i} is not valid json: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: line {i} is not a json object (dict)")
            rows.append((i, obj))
    return rows


def _gather_debate_files(root: Path) -> List[Path]:
    """discover debate jsonl files under out/<CASE_ID>/debate/."""
    files: List[Path] = []
    if not root.exists():
        return files
    for case_dir in root.iterdir():
        if not case_dir.is_dir():
            continue
        debate_dir = case_dir / "debate"
        if not debate_dir.is_dir():
            continue
        files.extend(sorted(debate_dir.glob("Q*.jsonl")))
    return files


def _check_file(path: Path) -> List[str]:
    """validate one debate jsonl file for qid consistency."""
    errors: List[str] = []
    stem = path.stem  # e.g., "Q3"
    if not QID_RE.match(stem):
        errors.append(f"{path}: filename does not look like a QID (got '{stem}')")
        return errors

    try:
        rows = _read_jsonl(path)
    except ValueError as e:
        errors.append(str(e))
        return errors

    for lineno, obj in rows:
        qid = obj.get("qid")
        if qid is None:
            errors.append(f"{path}: line {lineno}: missing 'qid' field")
            continue
        if qid != stem:
            errors.append(
                f"{path}: line {lineno}: qid mismatch: file='{stem}' json='{qid}'"
            )
    return errors


def _run_checks(root: Path) -> List[str]:
    """run all checks and return a list of error strings (empty if ok)."""
    all_errors: List[str] = []
    files = _gather_debate_files(root)
    if not files:
        all_errors.append(f"no debate files found under {root}/<CASE_ID>/debate/")
        return all_errors
    for p in files:
        all_errors.extend(_check_file(p))
    return all_errors


# ---------------- pytest integration ----------------


def test_debate_qid_labels_consistent():
    """pytest entry: asserts that all debate rows have qid matching their filename."""
    out_dir = Path(os.environ.get("OUT_DIR", "out"))
    errors = _run_checks(out_dir)
    assert not errors, "\n".join(errors)


# ---------------- cli entrypoint --------------------


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="post-check debate transcript qids in out/<CASE_ID>/debate/"
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("OUT_DIR", "out"),
        help="root output folder (default: %(default)s or $OUT_DIR)",
    )
    args = parser.parse_args()
    root = Path(args.root)

    errors = _run_checks(root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"\nfailed: {len(errors)} error(s) found.", file=sys.stderr)
        return 1

    print(f"ok: debate qids consistent under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
