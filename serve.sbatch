#!/bin/bash
#SBATCH --job-name=gptoss_server
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

# caches & image
export HF_HOME="$HOME/hf-cache"
export IMG_FILE="$HOME/containers/vllm-gptoss.sif"   # keep this SIF pinned; do NOT overwrite it
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$HF_HOME" "$JOB_SCRATCH" logs

# keep v1, let vLLM pick FA3 on H100; don't force any backend
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
unset FLASHINFER_DISABLE_JIT

# TP across the two GPUs
export TENSOR_PARALLEL=${SLURM_JOB_GPUS_PER_NODE:-2}

# optional: quick GPU info in logs
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || true

# IMPORTANT: no host CUDA bind; --nv passes the driver through
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  "$IMG_FILE" \
  python3 -u -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port 8000 \
    --async-scheduling
