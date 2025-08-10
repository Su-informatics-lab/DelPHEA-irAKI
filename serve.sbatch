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
export IMG_FILE="$HOME/containers/vllm-gptoss.sif"
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$HF_HOME" "$JOB_SCRATCH" logs

# vLLM v1 + FA3 on Hopper. Do not force Triton. Disable FlashInfer.
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_FLASH_ATTN_VERSION=3
export VLLM_DISABLE_FLASHINFER=1
export TORCH_CUDA_ARCH_LIST="9.0"      # H100
export CUDA_VISIBLE_DEVICES=0,1

# smaller context for first boot; raise later after it’s stable
MAX_LEN="${MAX_LEN:-16384}"
TP="${SLURM_JOB_GPUS_PER_NODE:-2}"

# optional: make startup more chatty while we verify
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_LEVEL=INFO
# export NCCL_DEBUG=INFO   # uncomment if it still fails to start

# no host CUDA bind; let the image provide user-space CUDA
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || true

apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  "$IMG_FILE" \
  python3 -u -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$MAX_LEN" \
    --async-scheduling \
    --host 0.0.0.0 --port 8000
