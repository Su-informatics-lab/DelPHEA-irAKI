#!/bin/bash
#SBATCH --job-name=gptoss_diag
#SBATCH --partition=nextgen-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2            # node has 2; we'll start with TP=1 anyway
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

# Paths
export HF_HOME="$HOME/hf-cache"
export IMG_FILE="$HOME/containers/vllm-gptoss.sif"
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$HF_HOME" "$JOB_SCRATCH" logs

# Model & conservative defaults
MODEL="${MODEL:-openai/gpt-oss-20b}"   # start small; switch to 120b after it boots once
TP="${TP:-1}"                          # single-GPU to bypass NCCL while we debug
GMU="${GMU:-0.80}"                     # make prealloc gentler
MAX_LEN="${MAX_LEN:-8192}"             # small context for first boot
PORT="${PORT:-8000}"

echo "== GPU =="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || true

# Run inside the image. Do NOT bind host CUDA; let the image provide user-space CUDA.
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  \
  --env CUDA_VISIBLE_DEVICES=0 \              # single GPU, avoids NCCL paths
  --env VLLM_USE_V1=1 \                       # v1 engine (required for GPT-OSS path)
  --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \   # Hopper path
  --env VLLM_FLASH_ATTN_VERSION=3 \           # FA3 on H100
  --env VLLM_DISABLE_FLASHINFER=1 \           # avoid FlashInfer (Blackwell kernels)
  --env NCCL_IB_DISABLE=1 \                   # keep IB off for now
  --env TORCH_CUDA_ARCH_LIST=9.0 \            # H100
  --env CUDA_LAUNCH_BLOCKING=1 \              # better stacktraces
  --env VLLM_LOGGING_LEVEL=DEBUG \            # get the inner worker error
  "$IMG_FILE" \
  python3 -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAX_LEN" \
    --enforce-eager \
    --async-scheduling \
    --host 0.0.0.0 --port "$PORT"
