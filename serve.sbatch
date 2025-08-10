#!/bin/bash -l
#SBATCH --job-name=gptoss_120b
#SBATCH --partition=nextgen-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --signal=USR1@120

set -euo pipefail

# --- config ---
MODEL="${MODEL:-openai/gpt-oss-120b}"   # use 120b on 2×h100
PORT="${PORT:-8000}"
TP="${TP:-2}"                           # tensor parallel size = num gpus
HF_HOME="${HF_HOME:-$HOME/hf-cache}"
IMG="$HOME/containers/vllm-gptoss.sif"

mkdir -p "$HF_HOME" logs

echo "host: $(hostname)"
echo "model: $MODEL"
echo "port:  $PORT"
echo "tp:    $TP"

# optional: make vllm multiproc explicit (helps inside apptainer sometimes)
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# graceful preemption handling (prints a note, then exits cleanly)
trap 'echo "[signal] got USR1, shutting down soon..."; sleep 2; exit 0' USR1

# apptainer will bind host nvidia libs with --nv
apptainer run --nv \
  --bind "$HF_HOME:/root/.cache/huggingface" \
  "$IMG" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --download-dir /root/.cache/huggingface \
  --gpu-memory-utilization 0.95 \
  --max-model-len 131072 \
  --api-key ""
