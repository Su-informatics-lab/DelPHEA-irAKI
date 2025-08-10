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

# per-job temporary area
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$JOB_SCRATCH"

# CUDA setup
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# ============================================
# OPTIMAL H100 CONFIGURATION WITH TensorRT-LLM
# ============================================

# Force v0 engine
export VLLM_USE_V1=0

# TensorRT-LLM attention - BEST for H100 (Hopper)
export VLLM_USE_TRTLLM_ATTENTION=1
export VLLM_USE_TRTLLM_DECODE_ATTENTION=1
export VLLM_USE_TRTLLM_CONTEXT_ATTENTION=1

# MoE optimization for H100
# Use BF16 for accuracy (recommended by vLLM docs)
export VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1
# Alternative for faster but slightly less accurate:
# export VLLM_USE_FLASHINFER_MXFP4_MOE=1

# H100 (Hopper SM_90) architecture
export TORCH_CUDA_ARCH_LIST="9.0"

# GPU selection
export CUDA_VISIBLE_DEVICES=0,1

# Network optimization
export NCCL_IB_DISABLE=1

echo "========================================="
echo "vLLM GPT-OSS-120B Server (OPTIMAL H100)"
echo "GPU: 2x H100 80GB (Hopper SM_90)"
echo "Attention: TensorRT-LLM (H100 optimized)"
echo "MoE: FLASHINFER MXFP4 with BF16"
echo "Features: Async scheduling enabled"
echo "========================================="

export TENSOR_PARALLEL=${SLURM_JOB_GPUS_PER_NODE:-2}

apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind /mnt/shared/moduleapps/eb/CUDA/12.3.0:/mnt/shared/moduleapps/eb/CUDA/12.3.0 \
  --bind "$JOB_SCRATCH":/tmp \
  --env CUDA_HOME="$CUDA_HOME" \
  --env PATH="$PATH" \
  --env VLLM_USE_V1="$VLLM_USE_V1" \
  --env VLLM_USE_TRTLLM_ATTENTION="$VLLM_USE_TRTLLM_ATTENTION" \
  --env VLLM_USE_TRTLLM_DECODE_ATTENTION="$VLLM_USE_TRTLLM_DECODE_ATTENTION" \
  --env VLLM_USE_TRTLLM_CONTEXT_ATTENTION="$VLLM_USE_TRTLLM_CONTEXT_ATTENTION" \
  --env VLLM_USE_FLASHINFER_MXFP4_BF16_MOE="$VLLM_USE_FLASHINFER_MXFP4_BF16_MOE" \
  --env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  "$IMG_FILE" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --trust-remote-code \
    --disable-log-requests

echo "========================================="
echo "Server terminated at $(date)"
echo "========================================="