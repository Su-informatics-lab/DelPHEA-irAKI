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

# ============================================
# Based on Official vLLM Documentation
# Source: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html
# Optimized for H100 (Hopper) GPUs
# ============================================

# Paths
export HF_HOME="$HOME/hf-cache"
export IMG_FILE="$HOME/containers/vllm-gptoss.sif"
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"

# Create directories
mkdir -p "$HF_HOME" logs "$JOB_SCRATCH"

# CUDA configuration
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1

# H100 specific settings from vLLM docs
export TORCH_CUDA_ARCH_LIST="9.0"  # H100 SM_90

# Network optimization
export NCCL_IB_DISABLE=1

# Tensor parallel (vLLM recommends TP=2 for H100 as best performance tradeoff)
export TENSOR_PARALLEL=2

# ============================================
# Official vLLM Environment Settings
# ============================================
# From vLLM docs: H100 doesn't need special attention backend
# TRITON_ATTN_VLLM_V1 is only for Ampere (A100) GPUs
# H100 uses optimized FlashAttention 3 kernels automatically

# If you encounter issues, uncomment this fallback:
# export VLLM_ATTENTION_BACKEND="TRITON_ATTN_VLLM_V1"

# ============================================
# Diagnostics
# ============================================
echo "========================================="
echo "vLLM GPT-OSS-120B Server (Official Config)"
echo "========================================="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs: 2 x H100 80GB"
echo "Tensor Parallel: 2 (recommended for H100)"
echo "Container: $(basename $IMG_FILE)"
echo "========================================="

# Show GPU info
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ============================================
# Launch Server - Official vLLM Parameters
# From: vLLM docs and OpenAI cookbook
# ============================================
echo "Starting vLLM server at $(date)..."

# Since 'vllm serve' CLI isn't available in container,
# use Python module with exact same parameters as official docs
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind /mnt/shared/moduleapps/eb/CUDA/12.3.0:/mnt/shared/moduleapps/eb/CUDA/12.3.0 \
  --bind "$JOB_SCRATCH":/tmp \
  --env CUDA_HOME="$CUDA_HOME" \
  --env PATH="$PATH" \
  --env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  "$IMG_FILE" \
  python3 -u -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 2 \
    --async-scheduling \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code

# ============================================
# Alternative with all recommended flags
# ============================================
# For production, you can add these official parameters:
#
#   --max-model-len 131072 \              # Full 128K context
#   --max-num-batched-tokens 10240 \      # Batch optimization
#   --max-num-seqs 128 \                  # Max concurrent sequences
#   --enable-prefix-caching \              # Performance optimization
#   --disable-log-requests                # Reduce logging overhead

echo "========================================="
echo "Server terminated at $(date)"
echo "========================================="