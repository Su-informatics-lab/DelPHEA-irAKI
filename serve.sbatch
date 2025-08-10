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
  --env VLLM_USE_V1=1 \
  --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  --env VLLM_FLASH_ATTN_VERSION=3 \
  --env VLLM_DISABLE_FLASHINFER=1 \
  --env NCCL_IB_DISABLE=1 \
  --env TORCH_CUDA_ARCH_LIST=9.0 \
  --env VLLM_CONFIGURE_LOGGING=1 \
  --env VLLM_LOGGING_LEVEL=DEBUG \
  "$IMG_FILE" python3 -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAX_LEN" \
    --async-scheduling \
    --host 0.0.0.0 --port 8000

