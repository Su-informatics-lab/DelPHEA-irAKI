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
export IMG_FILE="$HOME/containers/vllm-gptoss.sif"   # keep this pinned
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$HF_HOME" "$JOB_SCRATCH" logs

# simple, robust defaults
MODEL="${MODEL:-openai/gpt-oss-20b}"      # try 20B first; switch to 120B after it boots
TP="${SLURM_JOB_GPUS_PER_NODE:-2}"
GMU="${GMU:-0.90}"
MAX_LEN="${MAX_LEN:-16384}"

# quick gpu info
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || true

# IMPORTANT: no host CUDA bind; let the image supply user-space CUDA
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  \
  --env VLLM_USE_V1=1 \
  --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  --env VLLM_FLASH_ATTN_VERSION=3 \
  --env VLLM_DISABLE_FLASHINFER=1 \
  --env NCCL_IB_DISABLE=1 \
  --env TORCH_CUDA_ARCH_LIST=9.0 \
  --env VLLM_CONFIGURE_LOGGING=1 \
  --env VLLM_LOGGING_LEVEL=INFO \
  "$IMG_FILE" \
  python3 -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAX_LEN" \
    --async-scheduling \
    --host 0.0.0.0 --port 8000
