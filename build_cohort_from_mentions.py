#!/usr/bin/env python3
"""
build_cohort_from_mentions.py
-----------------------------
Build a starter irAKI cohort from a mention index produced by scan_mentions.py.

Heuristic (configurable):
  - Identify patients (case_id) with at least one non-negated ICI mention AND
    at least one non-negated AKI mention (strict preferred; broad allowed via --aki-mode).
  - Optionally require that the first AKI mention occurs within N days of the first ICI mention.
  - Output:
      1) CSV with per-patient summary stats and flags
      2) Plaintext file listing candidate case_ids (one per line) for delphea_iraki.py

Usage:
  python build_cohort_from_mentions.py \
      --mentions out/mention_index.csv \
      --out-csv out/cohort_from_mentions.csv \
      --out-list out/cohort_case_ids.txt \
      --window-days 180 --aki-mode strict_preferred --require-nonnegated

Notes:
  - The mentions CSV must have the following columns (from scan_mentions.py):
      case_id, person_id, timestamp, match_type, term, negated, context
  - Timestamps are parsed with pandas.to_datetime(..., errors='coerce'). Rows with NaT are allowed
    but windowing by days will be skipped for those cases.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def load_mentions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"case_id", "person_id", "timestamp", "match_type", "term", "negated"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"mentions CSV is missing columns: {sorted(missing)}")
    # normalize
    df["case_id"] = df["case_id"].astype(str)
    df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce").astype("Int64")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["match_type"] = df["match_type"].astype(str)
    df["term"] = df["term"].astype(str).str.strip()
    # negated might be bool, str "True/False", or 0/1
    if df["negated"].dtype == bool:
        pass
    else:
        df["negated"] = (
            df["negated"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        )
    return df


def summarize_patient(
    df: pd.DataFrame,
    window_days: Optional[int],
    aki_mode: str,
    require_nonnegated: bool,
) -> pd.DataFrame:
    """
    Return one row per case_id with summary stats and selection flags.
    aki_mode:
      - 'strict_only'        : require AKI_strict mention(s) only
      - 'any'                : allow AKI_broad if strict absent
      - 'strict_preferred'   : prefer strict for dating; fallback to broad for dates/flags
    """
    # derive convenience flags
    df["is_ici"] = df["match_type"].eq("ICI")
    df["is_aki_strict"] = df["match_type"].eq("AKI_strict")
    df["is_aki_broad"] = df["match_type"].eq("AKI_broad")
    df["is_nonneg"] = ~df["negated"].astype(bool)

    # counts
    grp = df.groupby("case_id", as_index=False)
    counts = grp.agg(
        person_id=("person_id", "first"),
        n_notes=("timestamp", "count"),
        ici_hits=("is_ici", "sum"),
        ici_hits_nonneg=(
            "is_ici",
            lambda s: int((s & df.loc[s.index, "is_nonneg"]).sum()),
        ),
        aki_strict_hits=("is_aki_strict", "sum"),
        aki_strict_nonneg=(
            "is_aki_strict",
            lambda s: int((s & df.loc[s.index, "is_nonneg"]).sum()),
        ),
        aki_broad_hits=("is_aki_broad", "sum"),
        aki_broad_nonneg=(
            "is_aki_broad",
            lambda s: int((s & df.loc[s.index, "is_nonneg"]).sum()),
        ),
    )

    # earliest dates (non-negated by default if require_nonnegated else any)
    def _first_time(mask: pd.Series) -> pd.Timestamp | pd.NaT:
        idx = df.index[mask]
        if len(idx) == 0:
            return pd.NaT
        return df.loc[idx, "timestamp"].min()

    earliest_by_case: Dict[str, Dict[str, pd.Timestamp]] = {}
    for case_id, sub in df.groupby("case_id"):
        if require_nonnegated:
            ici_time = sub.loc[sub["is_ici"] & sub["is_nonneg"], "timestamp"].min()
            aki_strict_time = sub.loc[
                sub["is_aki_strict"] & sub["is_nonneg"], "timestamp"
            ].min()
            aki_broad_time = sub.loc[
                sub["is_aki_broad"] & sub["is_nonneg"], "timestamp"
            ].min()
        else:
            ici_time = sub.loc[sub["is_ici"], "timestamp"].min()
            aki_strict_time = sub.loc[sub["is_aki_strict"], "timestamp"].min()
            aki_broad_time = sub.loc[sub["is_aki_broad"], "timestamp"].min()
        earliest_by_case[case_id] = {
            "first_ici_time": ici_time,
            "first_aki_strict_time": aki_strict_time,
            "first_aki_broad_time": aki_broad_time,
        }

    earliest_df = pd.DataFrame.from_dict(earliest_by_case, orient="index").reset_index(
        names="case_id"
    )

    out = counts.merge(earliest_df, on="case_id", how="left")

    # decide final AKI flag/time based on mode
    def pick_aki_time(row) -> pd.Timestamp | pd.NaT:
        s, b = row["first_aki_strict_time"], row["first_aki_broad_time"]
        if aki_mode == "strict_only":
            return s
        if aki_mode == "any":
            return s if pd.notna(s) else b
        # strict_preferred
        return s if pd.notna(s) else b

    out["first_aki_time"] = out.apply(pick_aki_time, axis=1)
    out["has_ici"] = (
        out["ici_hits_nonneg"] if require_nonnegated else out["ici_hits"]
    ) > 0
    if aki_mode == "strict_only":
        base_hits = "aki_strict_nonneg" if require_nonnegated else "aki_strict_hits"
    elif aki_mode == "any":
        # any (nonnegated if required) across strict or broad
        if require_nonnegated:
            out["aki_any_nonneg"] = (
                out["aki_strict_nonneg"] + out["aki_broad_nonneg"]
            ) > 0
            base_hits = "aki_any_nonneg"
        else:
            out["aki_any_hits"] = (out["aki_strict_hits"] + out["aki_broad_hits"]) > 0
            base_hits = "aki_any_hits"
    else:  # strict_preferred
        if require_nonnegated:
            out["aki_any_nonneg"] = (
                out["aki_strict_nonneg"] + out["aki_broad_nonneg"]
            ) > 0
            base_hits = "aki_any_nonneg"
        else:
            out["aki_any_hits"] = (out["aki_strict_hits"] + out["aki_broad_hits"]) > 0
            base_hits = "aki_any_hits"

    out["has_aki"] = out[base_hits]

    # window filter (AKI within N days after ICI)
    if window_days is not None:

        def _days_after(row):
            ici, aki = row["first_ici_time"], row["first_aki_time"]
            if pd.isna(ici) or pd.isna(aki):
                return pd.NA
            return (aki - ici).days

        out["aki_after_ici_days"] = out.apply(_days_after, axis=1)
        out["within_window"] = out["aki_after_ici_days"].apply(
            lambda d: (pd.notna(d)) and (d >= 0) and (d <= int(window_days))
        )
    else:
        out["aki_after_ici_days"] = pd.NA
        out["within_window"] = True  # no windowing applied

    # evidence score (for sorting): strict nonneg*2 + broad nonneg + ici nonneg
    out["evidence_score"] = (
        2 * out.get("aki_strict_nonneg", 0)
        + 1 * out.get("aki_broad_nonneg", 0)
        + 1 * out.get("ici_hits_nonneg", 0)
    )

    # final candidate flag
    out["candidate"] = out["has_ici"] & out["has_aki"] & out["within_window"]

    # stable ordering for readability
    cols = [
        "case_id",
        "person_id",
        "has_ici",
        "has_aki",
        "candidate",
        "ici_hits",
        "ici_hits_nonneg",
        "aki_strict_hits",
        "aki_strict_nonneg",
        "aki_broad_hits",
        "aki_broad_nonneg",
        "first_ici_time",
        "first_aki_strict_time",
        "first_aki_broad_time",
        "first_aki_time",
        "aki_after_ici_days",
        "within_window",
        "evidence_score",
        "n_notes",
    ]
    # keep only existing columns (some conditional temp cols may not exist)
    cols = [c for c in cols if c in out.columns]
    out = out[cols].sort_values(
        ["candidate", "evidence_score", "case_id"], ascending=[False, False, True]
    )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Build a starter irAKI cohort from mention index."
    )
    ap.add_argument(
        "--mentions",
        required=True,
        help="path to mention_index.csv from scan_mentions.py",
    )
    ap.add_argument(
        "--out-csv",
        default="out/cohort_from_mentions.csv",
        help="where to write summary CSV",
    )
    ap.add_argument(
        "--out-list",
        default="out/cohort_case_ids.txt",
        help="where to write case_id list (one per line)",
    )
    ap.add_argument(
        "--window-days",
        type=int,
        default=180,
        help="AKI must occur within this many days after first ICI (default: 180)",
    )
    ap.add_argument(
        "--aki-mode",
        choices=["strict_only", "any", "strict_preferred"],
        default="strict_preferred",
        help="AKI evidence requirement (default: strict_preferred)",
    )
    ap.add_argument(
        "--require-nonnegated",
        action="store_true",
        help="only count non-negated mentions (recommended)",
    )
    args = ap.parse_args()

    mentions_path = Path(args.mentions)
    out_csv = Path(args.out_csv)
    out_list = Path(args.out_list)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_list.parent.mkdir(parents=True, exist_ok=True)

    df = load_mentions(str(mentions_path))
    cohort = summarize_patient(
        df,
        window_days=int(args.window_days) if args.window_days is not None else None,
        aki_mode=args.aki_mode,
        require_nonnegated=bool(args.require_nonnegated),
    )

    cohort.to_csv(out_csv, index=False)

    # write case list (candidates only)
    candidates = cohort.loc[cohort["candidate"] == True, "case_id"].tolist()
    with out_list.open("w", encoding="utf-8") as f:
        for cid in candidates:
            f.write(f"{cid}\n")

    print(
        json.dumps(
            {
                "mentions": str(mentions_path),
                "out_csv": str(out_csv),
                "out_list": str(out_list),
                "n_patients": int(cohort.shape[0]),
                "n_candidates": int(len(candidates)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
