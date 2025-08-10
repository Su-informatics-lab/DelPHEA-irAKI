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

# CUDA tool-chain setup
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# CRITICAL: Set TORCH_CUDA_ARCH_LIST for H100
export TORCH_CUDA_ARCH_LIST="9.0"

# CRITICAL: Disable v1 engine and use stable v0
export VLLM_USE_V1=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Additional stability settings
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

# logging
export VLLM_LOGGING_LEVEL=INFO

# diagnostics
echo "========================================="
echo "vLLM GPT-OSS-120B Server Configuration"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "VLLM_USE_V1: ${VLLM_USE_V1}"
echo "========================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv

# launch vLLM with v0 engine
export TENSOR_PARALLEL=${SLURM_JOB_GPUS_PER_NODE:-2}

apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind /mnt/shared/moduleapps/eb/CUDA/12.3.0:/mnt/shared/moduleapps/eb/CUDA/12.3.0 \
  --bind "$JOB_SCRATCH":/tmp \
  --env CUDA_HOME="$CUDA_HOME" \
  --env PATH="$PATH" \
  --env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  --env VLLM_USE_V1="$VLLM_USE_V1" \
  --env VLLM_ENABLE_V1_MULTIPROCESSING="$VLLM_ENABLE_V1_MULTIPROCESSING" \
  --env VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND" \
  --env VLLM_LOGGING_LEVEL="$VLLM_LOGGING_LEVEL" \
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