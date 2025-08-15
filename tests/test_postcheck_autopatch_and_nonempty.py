# tests/test_postcheck_autopatch_and_nonempty.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterator, Tuple

OUT = Path(os.environ.get("OUT_DIR", "out"))


def _iter_debate_files(root: Path) -> Iterator[Path]:
    for case_dir in root.iterdir():
        if not case_dir.is_dir():
            continue
        deb = case_dir / "debate"
        if deb.is_dir():
            yield from sorted(deb.glob("Q*.jsonl"))


def _iter_rows(path: Path) -> Iterator[Tuple[int, Dict]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                yield i, obj


def test_debate_files_nonempty():
    files = list(_iter_debate_files(OUT))
    assert files, f"no debate files found under {OUT}/<CASE_ID>/debate/"
    empties = [p for p in files if not any(_iter_rows(p))]
    assert not empties, "empty debate transcripts:\n" + "\n".join(
        str(p) for p in empties
    )


def test_no_qid_autopatched_flags():
    # fail if any row reports meta.qid_autopatched == True
    offenders = []
    for p in _iter_debate_files(OUT):
        for lineno, obj in _iter_rows(p):
            meta = obj.get("meta") or {}
            if meta.get("qid_autopatched") is True:
                offenders.append(f"{p}: line {lineno} has meta.qid_autopatched=True")
    assert not offenders, "autopatched qid found:\n" + "\n".join(offenders)
