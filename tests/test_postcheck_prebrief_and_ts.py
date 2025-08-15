# tests/test_postcheck_prebrief_and_ts.py
from __future__ import annotations

import json
import os
from pathlib import Path

OUT = Path(os.environ.get("OUT_DIR", "out"))


def test_moderator_prebrief_present_and_ts_everywhere():
    cases = [p for p in OUT.iterdir() if p.is_dir()]
    assert cases, f"no cases under {OUT}"
    missing = []
    nots = []
    for case in cases:
        deb = case / "debate"
        if not deb.is_dir():
            continue
        for f in deb.glob("Q*.jsonl"):
            with f.open() as fh:
                rows = [json.loads(ln) for ln in fh if ln.strip()]
            # first row should be moderator pre-brief
            if (
                not rows
                or rows[0].get("expert_id") != "moderator"
                or rows[0].get("turn_index") != -1
            ):
                missing.append(str(f))
            # every row must have ts
            for i, r in enumerate(rows, 1):
                if "ts" not in r:
                    nots.append(f"{f}: line {i}")
    assert not missing, "missing moderator pre-brief in:\n" + "\n".join(missing)
    assert not nots, "missing ts in:\n" + "\n".join(nots)
