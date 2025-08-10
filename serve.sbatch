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
# Fixed for Container Missing vllm CLI
# Using Python Module Instead
# ============================================

# Paths
export HF_HOME="$HOME/hf-cache"
export IMG_FILE="$HOME/containers/vllm-gptoss.sif"
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"

# Create directories
mkdir -p "$HF_HOME" logs "$JOB_SCRATCH"

# CUDA configuration for H100
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_CUDA_ARCH_LIST="9.0"  # H100 SM_90

# Network optimization
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL="NVL"

# Tensor parallel
export TENSOR_PARALLEL=${SLURM_JOB_GPUS_PER_NODE:-2}

# IMPORTANT: Fallback for attention backend
# The new container should auto-detect, but add this if needed
export VLLM_ATTENTION_BACKEND="TRITON_ATTN_VLLM_V1"

# Force v0 engine for stability
export VLLM_USE_V1=0

# ============================================
# Diagnostics
# ============================================
echo "========================================="
echo "vLLM GPT-OSS-120B Server Configuration"
echo "========================================="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs: ${TENSOR_PARALLEL} x H100"
echo "Container: $(basename $IMG_FILE)"
echo "========================================="

# Show GPU info
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ============================================
# Launch Using Python Module (WORKING METHOD)
# ============================================
echo "Starting vLLM server at $(date)..."

apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind /mnt/shared/moduleapps/eb/CUDA/12.3.0:/mnt/shared/moduleapps/eb/CUDA/12.3.0 \
  --bind "$JOB_SCRATCH":/tmp \
  --env CUDA_HOME="$CUDA_HOME" \
  --env PATH="$PATH" \
  --env VLLM_USE_V1="$VLLM_USE_V1" \
  --env VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND" \
  --env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  --env NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
  "$IMG_FILE" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --async-scheduling \
    --disable-log-requests

echo "========================================="
echo "Server terminated at $(date)"
echo "========================================="