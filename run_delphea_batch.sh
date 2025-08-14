#!/usr/bin/env bash
# run_delphea_batch.sh
# Batch-run DelPHEA-irAKI over a list of case_ids (one per line).
# Minimal deps: bash, python. Optional: bash≥4.3 for wait -n (for -j >1).

set -euo pipefail

# defaults (can be overridden by flags)
CASES_FILE=""
QUESTIONNAIRE="questionnaire_full.json"
PANEL="panel.json"
ROUTER="sparse"
ENDPOINT_URL="${ENDPOINT_URL:-http://localhost:8000}"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-120b}"
OUT_DIR="out"
JOBS=1
SKIP_EXISTING=0
VERBOSE=1

# prompt budget & temps (mirror delphea_iraki.py)
CTX_WINDOW=102400
MAX_NOTES=1024
NOTE_CHAR_CAP=10240
TOTAL_CHARS_CAP=600000
T1=0.3
T2=0.6
T3=0.3

usage() {
  cat <<USAGE
Usage: $0 -f CASES_FILE [options]

Required:
  -f FILE                Text file with one case_id per line (e.g., iraki_case_123...)

Options:
  -q FILE                Questionnaire path (default: ${QUESTIONNAIRE})
  -p FILE                Panel path (default: ${PANEL})
  -r sparse|full         Router (default: ${ROUTER})
  -e URL                 Endpoint URL (default: ${ENDPOINT_URL})
  -m MODEL               Model name (default: ${MODEL_NAME})
  -o DIR                 Output base dir (default: ${OUT_DIR})
  -j N                   Parallel jobs (default: ${JOBS})
  -s                     Skip cases that already have summary/report (default: off)
  -v                     Verbose (-v)  (use -vv by setting -v twice)
  --ctx-window N         Context window tokens (default: ${CTX_WINDOW})
  --max-notes N          Max notes passed to prompts (default: ${MAX_NOTES})
  --note-char-cap N      Per-note char cap (default: ${NOTE_CHAR_CAP})
  --total-chars-cap N    Total chars cap (default: ${TOTAL_CHARS_CAP})
  --t1 F                 Temperature round1 (default: ${T1})
  --t2 F                 Temperature debate  (default: ${T2})
  --t3 F                 Temperature round3 (default: ${T3})

Example:
  $0 -f cohort_case_ids.txt -j 2 -s -e "\$ENDPOINT_URL" -m "\$MODEL_NAME"
USAGE
  exit 1
}

# parse flags (supports a few long options)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f) CASES_FILE="$2"; shift 2 ;;
    -q) QUESTIONNAIRE="$2"; shift 2 ;;
    -p) PANEL="$2"; shift 2 ;;
    -r) ROUTER="$2"; shift 2 ;;
    -e) ENDPOINT_URL="$2"; shift 2 ;;
    -m) MODEL_NAME="$2"; shift 2 ;;
    -o) OUT_DIR="$2"; shift 2 ;;
    -j) JOBS="$2"; shift 2 ;;
    -s) SKIP_EXISTING=1; shift ;;
    -v) VERBOSE=$((VERBOSE+1)); shift ;;
    --ctx-window) CTX_WINDOW="$2"; shift 2 ;;
    --max-notes) MAX_NOTES="$2"; shift 2 ;;
    --note-char-cap) NOTE_CHAR_CAP="$2"; shift 2 ;;
    --total-chars-cap) TOTAL_CHARS_CAP="$2"; shift 2 ;;
    --t1) T1="$2"; shift 2 ;;
    --t2) T2="$2"; shift 2 ;;
    --t3) T3="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

[[ -z "$CASES_FILE" ]] && { echo "ERROR: -f CASES_FILE is required"; usage; }
[[ ! -f "$CASES_FILE" ]] && { echo "ERROR: cases file not found: $CASES_FILE"; exit 2; }

mkdir -p "$OUT_DIR"

run_one() {
  local cid="$1"

  # skip if requested and already exists
  if [[ "$SKIP_EXISTING" -eq 1 ]]; then
    if [[ -f "$OUT_DIR/$cid/summary.json" || -f "$OUT_DIR/$cid/report.json" ]]; then
      echo "[skip] $cid"
      return 0
    fi
  fi

  echo "[run ] $cid"

  # build args for delphea_iraki.py
  python delphea_iraki.py \
    --case "$cid" \
    --q "$QUESTIONNAIRE" \
    --panel "$PANEL" \
    --router "$ROUTER" \
    --endpoint-url "$ENDPOINT_URL" \
    --model-name "$MODEL_NAME" \
    --temperature-r1 "$T1" \
    --temperature-r2 "$T2" \
    --temperature-r3 "$T3" \
    --ctx-window "$CTX_WINDOW" \
    --max-notes "$MAX_NOTES" \
    --note-char-cap "$NOTE_CHAR_CAP" \
    --total-chars-cap "$TOTAL_CHARS_CAP" \
    --out-dir "$OUT_DIR" \
    $( [[ $VERBOSE -ge 2 ]] && echo "-vv" || echo "-v" ) \
    >"$OUT_DIR/$cid/run.stdout.log" 2>"$OUT_DIR/$cid/run.stderr.log" || {
      echo "[fail] $cid (see $OUT_DIR/$cid/run.stderr.log)"
      return 1
    }

  echo "[done] $cid"
}

# semaphore for parallelism
pids=()
active=0
max_jobs="${JOBS}"

# read case IDs (ignore blank lines / comments)
while IFS= read -r line || [[ -n "$line" ]]; do
  cid="$(echo "$line" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  [[ -z "$cid" ]] && continue
  [[ "$cid" =~ ^# ]] && continue

  if (( max_jobs > 1 )); then
    run_one "$cid" &
    pids+=("$!")
    active=$((active+1))
    # throttle if too many active jobs
    if (( active >= max_jobs )); then
      # requires bash >= 4.3
      if command -v bash >/dev/null 2>&1 && [[ "${BASH_VERSINFO[0]:-0}" -ge 4 ]]; then
        wait -n
      else
        # fallback: wait for all then reset
        wait
        active=0
        pids=()
      fi
      active=$((active-1))
    fi
  else
    run_one "$cid"
  fi

done < "$CASES_FILE"

# wait for remaining background jobs
wait || true

# optional: write a tiny batch summary CSV (case_id, verdict, p_iraki, status)
python - <<'PY'
import json, csv, sys, os
from pathlib import Path

out_dir = os.environ.get("OUT_DIR", "out")
rows = []
for case_path in Path(out_dir).iterdir():
    if not case_path.is_dir(): continue
    cid = case_path.name
    summ = case_path / "summary.json"
    rep = case_path / "report.json"
    row = {"case_id": cid, "status": "missing", "verdict": None, "p_iraki": None, "decision_threshold": None, "debate_skipped": None}
    try:
        data = json.loads(summ.read_text()) if summ.exists() else (json.loads(rep.read_text()) if rep.exists() else {})
        cons = data.get("consensus", {})
        row.update({
            "status": "ok" if data else "missing",
            "verdict": cons.get("verdict"),
            "p_iraki": cons.get("p_iraki"),
            "decision_threshold": cons.get("decision_threshold"),
            "debate_skipped": data.get("debate_skipped"),
        })
    except Exception as e:
        row["status"] = "error"
    rows.append(row)

rows.sort(key=lambda r: r["case_id"])
with open(os.path.join(out_dir, "batch_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["case_id","status","verdict","p_iraki","decision_threshold","debate_skipped"])
    w.writeheader()
    for r in rows: w.writerow(r)
print(f"wrote {out_dir}/batch_summary.csv ({len(rows)} cases)")
PY
