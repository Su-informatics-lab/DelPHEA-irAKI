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

# create local directories for cache
mkdir -p logs
mkdir -p cache/hf
mkdir -p cache/models

# print basic info
echo "Starting GPT-OSS-120B on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Time: $(date)"
nvidia-smi

# CRITICAL: Use Triton attention backend for GPT-OSS attention sinks support
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1

# Disable V1 engine
export VLLM_USE_V1=0

# Set cache location
export HF_HOME=$(pwd)/cache/hf

# run vllm server with bind mounts for cache
apptainer run --nv \
    --bind $(pwd)/cache/hf:/workspace/hf_cache \
    --bind $(pwd)/cache/models:/workspace/models \
    --env HF_HOME=/workspace/hf_cache \
    --env VLLM_USE_V1=0 \
    --env VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 \
    gpt-oss-120b.sif \
    vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --async-scheduling \
    --gpu-memory-utilization 0.90 \
    --download-dir /workspace/models

echo "Job finished at $(date)"