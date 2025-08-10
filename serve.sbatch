#!/bin/bash
#SBATCH --job-name=gptoss_20b_v1_fa3
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

# --- host paths ---
export HF_HOME="$HOME/hf-cache"
export IMG_FILE="$HOME/containers/vllm-gptoss.sif"
mkdir -p "$HF_HOME" logs

# per-job scratch for caches + tmp
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$JOB_SCRATCH"/{tmp,.cache,torch_extensions,triton}

# --- GPU / NCCL ---
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=1

# --- vLLM on H100: V1 + FA3, no FlashInfer ---
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_DISABLE_FLASHINFER=1

# Multiprocessing + logging (get inner stacktraces)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=DEBUG

# Ensure JIT + triton caches are writable & large
export XDG_CACHE_HOME="$JOB_SCRATCH/.cache"
export TORCH_EXTENSIONS_DIR="$JOB_SCRATCH/torch_extensions"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export TMPDIR="$JOB_SCRATCH/tmp"

# H100 arch + saner CUDA allocator
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True

# Tensor-parallel (start with TP=1 to isolate causes)
export TENSOR_PARALLEL=1

echo "============================================"
echo "Host:      $(hostname)"
echo "JobID:     $SLURM_JOB_ID"
echo "GPUs:      $(echo $CUDA_VISIBLE_DEVICES)"
echo "Container: $(basename "$IMG_FILE")"
echo "Scratch:   $JOB_SCRATCH"
echo "============================================"

# Quick preflight inside the container (logs show up in .err/.out)
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  "$IMG_FILE" \
  python3 - <<'PY'
import os, torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("cc:", torch.cuda.get_device_capability(0))
print("XDG_CACHE_HOME=", os.getenv("XDG_CACHE_HOME"))
print("TORCH_EXTENSIONS_DIR=", os.getenv("TORCH_EXTENSIONS_DIR"))
print("TRITON_CACHE_DIR=", os.getenv("TRITON_CACHE_DIR"))
PY

# Launch vLLM OpenAI server (start with 20B; move to 120B after it’s stable)
echo "Starting vLLM (gpt-oss-20b, V1+FA3, TP=$TENSOR_PARALLEL) ..."
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  --env VLLM_USE_V1="$VLLM_USE_V1" \
  --env VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND" \
  --env VLLM_DISABLE_FLASHINFER="$VLLM_DISABLE_FLASHINFER" \
  --env VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD" \
  --env VLLM_LOGGING_LEVEL="$VLLM_LOGGING_LEVEL" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  --env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  "$IMG_FILE" \
  vllm serve openai/gpt-oss-20b \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --host 0.0.0.0 --port 8000 \
    --disable-log-requests

echo "Server exited at $(date)"
