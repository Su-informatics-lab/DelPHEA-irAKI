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

# host-side caches & images
export HF_HOME="$HOME/hf-cache"
export IMG_DIR="$HOME/containers"
export IMG_FILE="$IMG_DIR/vllm-gptoss.sif"

mkdir -p "$HF_HOME" "$IMG_DIR" logs

# per-job temporary area (inside project folder)
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$JOB_SCRATCH"

# CUDA tool-chain inside container
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# ===========================================
# CRITICAL FIX: Force v0 engine to avoid v1 multiprocessing error
# ===========================================
export VLLM_USE_V1=0
export TORCH_CUDA_ARCH_LIST="9.0"  # H100 architecture

# Additional stability settings
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=1

# Optional debugging (comment out after it works)
# export VLLM_LOGGING_LEVEL=DEBUG
# export FLASHINFER_DISABLE_JIT=1  # only for debug

# launch vLLM inside Apptainer container
export TENSOR_PARALLEL=${SLURM_JOB_GPUS_PER_NODE:-2}

echo "========================================="
echo "vLLM Server Starting"
echo "Model: openai/gpt-oss-120b"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Engine: v0 (VLLM_USE_V1=$VLLM_USE_V1)"
echo "========================================="

apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind /mnt/shared/moduleapps/eb/CUDA/12.3.0:/mnt/shared/moduleapps/eb/CUDA/12.3.0 \
  --bind "$JOB_SCRATCH":/tmp \
  --env CUDA_HOME="$CUDA_HOME" \
  --env PATH="$PATH" \
  --env VLLM_USE_V1="$VLLM_USE_V1" \
  --env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  --env VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  "$IMG_FILE" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --disable-log-requests

echo "Server terminated at $(date)"