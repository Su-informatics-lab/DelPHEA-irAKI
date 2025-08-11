#!/bin/bash
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
MODEL="${MODEL:-openai/gpt-oss-120b}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
HF_HOME="${HF_HOME:-$HOME/hf-cache}"
IMG="${IMG:-$HOME/containers/vllm-gptoss.sif}"

mkdir -p "$HF_HOME" logs

export VLLM_WORKER_MULTIPROC_METHOD=spawn
trap 'echo "[signal] USR1 received, exiting"; sleep 1; exit 0' USR1

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
