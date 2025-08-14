#!/usr/bin/env python3
"""
scan_mentions.py
----------------
Heuristically scan clinical notes for Immune Checkpoint Inhibitors (ICIs) and AKI-related mentions.
Outputs a CSV: one row per match with patient (case_id), note metadata, match type (ICI|AKI), term, and a context snippet.

Usage examples:
  python scan_mentions.py --ici-csv ici_vocab.csv --dummy --out /tmp/mention_index.csv
  python scan_mentions.py --data-dir irAKI_data --ici-csv ici_vocab.csv --out out/mention_index.csv

Notes:
- If --dummy is provided, a single fixture patient is scanned (from DataLoader).
- Otherwise, the script loads the full cohort via DataLoader and scans all notes.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

# Import the project's DataLoader
from dataloader import DataLoader


def read_ici_terms(path: str) -> List[str]:
    """
    Read ICI vocabulary from a CSV (any column layout). Extract all unique non-empty strings.
    Adds common brand/generic synonyms automatically.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ICI vocabulary file not found: {path}")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    terms: List[str] = []
    for col in df.columns:
        terms.extend([str(x).strip() for x in df[col].tolist() if str(x).strip()])
    # normalize and unique
    terms = sorted({t.lower() for t in terms if t})
    # add common brands/generics/abbreviations (in case not provided)
    fallback = [
        "pembrolizumab",
        "nivolumab",
        "ipilimumab",
        "atezolizumab",
        "durvalumab",
        "avelumab",
        "cemiplimab",
        "keytruda",
        "opdivo",
        "yervoy",
        "tecentriq",
        "imfinzi",
        "bavencio",
        "libtayo",
        "checkpoint inhibitor",
        "immune checkpoint inhibitor",
        "pd-1",
        "pd1",
        "pd-l1",
        "pdl1",
        "ctla-4",
        "ctla4",
        "pembro",
        "ipi+nivo",
        "ipi",
        "nivo",
    ]
    for t in fallback:
        terms.append(t)
    return sorted({t for t in terms})


def build_regex_union(terms: Iterable[str]) -> re.Pattern:
    """
    Build a case-insensitive word-boundary regex union for a set of terms.
    Handles multi-word phrases and hyphens.
    """
    parts: List[str] = []
    for t in terms:
        t = re.escape(t)
        # allow flexible hyphens/spaces for things like PD-L1 / PD L1
        t = t.replace(r"\-", "[- ]")
        if " " in t:
            parts.append(rf"\b{t}\b")
        else:
            parts.append(rf"\b{t}\b")
    pattern = "|".join(parts)
    return re.compile(pattern, flags=re.IGNORECASE)


def aki_patterns() -> Tuple[re.Pattern, re.Pattern]:
    """
    Returns (strict, broad) AKI regex patterns.
      - strict: highly kidney-specific terms (low false positives)
      - broad: includes ARF/AKI abbreviations and creatinine rises (more recall, more noise)
    """
    strict_terms = [
        r"acute kidney injury",
        r"acute renal failure",
        r"acute tubular (necrosis|injury)",
        r"interstitial nephritis",
        r"acute interstitial nephritis",
        r"\bain\b",
        r"\batn\b",
    ]
    broad_terms = [
        r"\baki\b",
        r"\barf\b",  # ambiguous (resp vs renal)
        r"(rise|bump|increase|elevat\w+)\s+(in\s+)?(serum\s+)?(creatinine|cr)\b",
        r"\bcreatinine\b",
    ]
    strict = re.compile("|".join(strict_terms), flags=re.IGNORECASE)
    broad = re.compile("|".join(broad_terms), flags=re.IGNORECASE)
    return strict, broad


NEGATION_WINDOW = 6  # words before match to scan for negation
NEGATION_TRIGGERS = re.compile(
    r"\b(no|denies|without|not|rule\s+out|ruled\s+out|negative\s+for)\b", re.IGNORECASE
)


def is_negated(text: str, start_idx: int) -> bool:
    """
    Very simple negation scope: look back up to NEGATION_WINDOW words and check for triggers.
    """
    prefix = text[:start_idx]
    # take last N words region
    tokens = re.findall(r"\w+|\S", prefix)
    last_words = tokens[-(NEGATION_WINDOW * 2) :]  # rough slice
    window = " ".join(last_words[-(NEGATION_WINDOW * 2) :])
    return NEGATION_TRIGGERS.search(window) is not None


def iter_matches(text: str, pat: re.Pattern, mtype: str) -> Iterable[Dict]:
    """
    Yield match dicts for a given regex pattern on text.
    """
    if not text:
        return
    for m in pat.finditer(text):
        s, e = m.start(), m.end()
        snippet_start = max(0, s - 80)
        snippet_end = min(len(text), e + 80)
        snippet = text[snippet_start:snippet_end].replace("\n", " ").strip()
        yield {
            "match_type": mtype,
            "term": m.group(0),
            "char_start": s,
            "char_end": e,
            "context": snippet,
            "negated": is_negated(text, s),
        }


def scan_note(
    note_text: str, ici_pat: re.Pattern, aki_strict: re.Pattern, aki_broad: re.Pattern
) -> List[Dict]:
    """
    Scan a single note's text and return a list of match dicts.
    """
    out: List[Dict] = []
    # ICIs
    out.extend(iter_matches(note_text, ici_pat, "ICI"))
    # AKI strict
    out.extend(iter_matches(note_text, aki_strict, "AKI_strict"))
    # AKI broad
    out.extend(iter_matches(note_text, aki_broad, "AKI_broad"))
    return out


def gather_notes_dummy(dl: DataLoader) -> Iterable[Tuple[str, Dict]]:
    case = dl.load_patient_case("iraki_case_001")
    for i, n in enumerate(case.get("clinical_notes", [])):
        yield case["case_id"], {
            "note_index": i,
            "timestamp": n.get("timestamp"),
            "service": n.get("service"),
            "encounter_id": n.get("encounter_id"),
            "text": n.get("text", "") or "",
            "person_id": case.get("patient_info", {}).get("person_id"),
        }


def gather_notes_full(dl: DataLoader) -> Iterable[Tuple[str, Dict]]:
    """
    Iterate all notes across all patients using the loader's dataframes directly
    (faster than per-patient roundtrips).
    """
    df = dl.clinical_notes_df
    if df is None or df.empty:
        return
    # normalize columns
    cols = {c.lower(): c for c in df.columns}
    pid_col = cols.get("obfuscated_global_person_id", "OBFUSCATED_GLOBAL_PERSON_ID")
    text_col = cols.get("report_text", "REPORT_TEXT")
    time_col = cols.get("physiologic_time", "PHYSIOLOGIC_TIME")
    svc_col = cols.get("service_name", "SERVICE_NAME")
    enc_col = cols.get("encounter_id", "ENCOUNTER_ID")

    # ensure types
    for _, row in df.sort_values(time_col).iterrows():
        person_id = int(row[pid_col])
        case_id = f"iraki_case_{person_id}"
        ts = row.get(time_col)
        ts = (
            ts.isoformat()
            if hasattr(ts, "isoformat")
            else (str(ts) if pd.notna(ts) else None)
        )
        yield case_id, {
            "note_index": None,  # unknown in bulk mode
            "timestamp": ts,
            "service": str(row.get(svc_col, "")),
            "encounter_id": str(row.get(enc_col, "")),
            "text": str(row.get(text_col, "") or ""),
            "person_id": person_id,
        }


def main():
    ap = argparse.ArgumentParser(
        description="Scan notes for ICI and AKI mentions and output a CSV index."
    )
    ap.add_argument(
        "--data-dir",
        default="irAKI_data",
        help="path to dataset root (default: irAKI_data)",
    )
    ap.add_argument(
        "--ici-csv", required=True, help="CSV with ICI vocabulary (any column layout)"
    )
    ap.add_argument("--out", default="out/mention_index.csv", help="output CSV path")
    ap.add_argument("--dummy", action="store_true", help="use dummy patient only")
    args = ap.parse_args()

    # prepare output path
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # load vocabulary & build regex
    ici_terms = read_ici_terms(args.ici_csv)
    ici_pat = build_regex_union(ici_terms)
    aki_strict, aki_broad = aki_patterns()

    # init loader
    dl = DataLoader(data_dir=args.data_dir, use_dummy=args.dummy)
    if not dl.is_available():
        raise SystemExit(
            "DataLoader is not available; check --data-dir or use --dummy for a quick test."
        )

    notes_iter = gather_notes_dummy(dl) if args.dummy else gather_notes_full(dl)

    # write CSV
    fields = [
        "case_id",
        "person_id",
        "timestamp",
        "service",
        "encounter_id",
        "note_index",
        "match_type",
        "term",
        "negated",
        "char_start",
        "char_end",
        "context",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        total_notes = 0
        total_hits = 0
        for case_id, note in notes_iter:
            total_notes += 1
            text = note["text"]
            hits = scan_note(text, ici_pat, aki_strict, aki_broad)
            for h in hits:
                row = {
                    "case_id": case_id,
                    "person_id": note.get("person_id"),
                    "timestamp": note.get("timestamp"),
                    "service": note.get("service"),
                    "encounter_id": note.get("encounter_id"),
                    "note_index": note.get("note_index"),
                    **h,
                }
                w.writerow(row)
                total_hits += 1

    print(
        json.dumps(
            {
                "out": str(out_path),
                "notes_scanned": total_notes,
                "mentions_found": total_hits,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
