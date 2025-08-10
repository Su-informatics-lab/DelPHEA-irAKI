#!/bin/bash
#SBATCH --job-name=gptoss_20b_hotpatch
#SBATCH --partition=nextgen-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

# Paths
export HF_HOME="$HOME/hf-cache"
export IMG_FILE="$HOME/containers/vllm-gptoss.sif"
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$HF_HOME" "$JOB_SCRATCH"/{tmp,pydeps,.cache,torch_extensions,triton} logs

# GPUs / NCCL
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=1

# vLLM on H100: use V1 + FA3, no FlashInfer, spawn workers
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_DISABLE_FLASHINFER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=INFO

# JIT / cache dirs
export XDG_CACHE_HOME="$JOB_SCRATCH/.cache"
export TORCH_EXTENSIONS_DIR="$JOB_SCRATCH/torch_extensions"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export TMPDIR="$JOB_SCRATCH/tmp"

# H100 arch + allocator
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True

# Tensor parallel: start with 1 while stabilizing
export TENSOR_PARALLEL=1

echo "== Preflight =="
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  GPUs: $CUDA_VISIBLE_DEVICES"
echo "Container: $(basename "$IMG_FILE")"

# --- Hotpatch NumPy/Numba into a writable target, then prefer it via PYTHONPATH ---
# NumPy < 2.3 (2.2.x) + Numba 0.61.2 (supports NumPy 2.2)
apptainer exec --nv \
  --bind "$JOB_SCRATCH":/mnt/job \
  "$IMG_FILE" \
  python3 - <<'PY'
import sys, subprocess, os
tgt = "/mnt/job/pydeps"
os.makedirs(tgt, exist_ok=True)
# Pin versions known to work together
pkgs = ["numpy<2.3", "numba==0.61.2"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", "--target", tgt, *pkgs])
print("Installed patched packages into", tgt)
PY

# Make sure Python prefers our patched site first
export PYTHONPATH="$JOB_SCRATCH/pydeps:$PYTHONPATH"

# Quick sanity check of versions inside the container
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  --bind "$JOB_SCRATCH":/mnt/job \
  --env PYTHONPATH="$PYTHONPATH" \
  "$IMG_FILE" \
  python3 - <<'PY'
import numpy, numba, torch, os
print("numpy:", numpy.__version__, "numba:", numba.__version__)
print("torch:", getattr(torch, "__version__", None))
print("CUDA avail:", torch.cuda.is_available())
PY

echo "== Launching vLLM (20B, TP=$TENSOR_PARALLEL) =="

apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  --bind "$JOB_SCRATCH":/mnt/job \
  --env PYTHONPATH="$PYTHONPATH" \
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
    --host 0.0.0.0 --port 8000 \
    --disable-log-requests

echo "Server exited at $(date)"
