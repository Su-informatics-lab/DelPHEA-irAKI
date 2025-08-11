#!/bin/bash
#SBATCH --job-name=gptoss_120b
#SBATCH --partition=nextgen-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --signal=USR1@120

set -euo pipefail

# create directories
mkdir -p logs
mkdir -p cache/hf
mkdir -p cache/models
mkdir -p tmp/$SLURM_JOB_ID

# print basic info
echo "Starting GPT-OSS-120B on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Time: $(date)"
nvidia-smi

# Set environment variables
export HF_HOME=$(pwd)/cache/hf
export JOB_SCRATCH=$(pwd)/tmp/$SLURM_JOB_ID

# CUDA setup for container
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# CRITICAL: Use V0 engine and set attention backend for H100
export VLLM_USE_V1=0

# Use Triton attention backend for attention sinks support without FA3
export VLLM_ATTENTION_BACKEND=TRITON
export VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1

# Alternative: If Triton doesn't work, disable sinks entirely (uncomment below)
# export VLLM_FLASH_ATTN_VERSION=2
# Add to command: --hf-overrides '{"sink_token_len": 0, "use_sliding_window": false}'

# Tensor parallelism based on GPU count
export TENSOR_PARALLEL=${SLURM_JOB_GPUS_PER_NODE:-2}

# Run vLLM using Python module directly (more control over startup)
apptainer exec --nv \
    --bind $(pwd)/cache/hf:/root/.cache/huggingface \
    --bind $(pwd)/cache/models:/workspace/models \
    --bind $JOB_SCRATCH:/tmp \
    --env HF_HOME=/root/.cache/huggingface \
    --env CUDA_HOME=$CUDA_HOME \
    --env PATH=$PATH \
    --env VLLM_USE_V1=0 \
    --env VLLM_ATTENTION_BACKEND=TRITON \
    --env VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1 \
    gpt-oss-120b.sif \
    python3 -m vllm.entrypoints.openai.api_server \
        --model openai/gpt-oss-120b \
        --tensor-parallel-size $TENSOR_PARALLEL \
        --gpu-memory-utilization 0.90 \
        --max-model-len 8192 \
        --host 0.0.0.0 \
        --port 8000 \
        --download-dir /workspace/models

echo "Job finished at $(date)"