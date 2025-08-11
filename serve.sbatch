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
mkdir -p $HOME/hf-cache
mkdir -p tmp/$SLURM_JOB_ID

# print basic info
echo "Starting GPT-OSS-120B on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Time: $(date)"
nvidia-smi

# Set paths
export HF_CACHE_DIR=$HOME/hf-cache
export JOB_SCRATCH=$(pwd)/tmp/$SLURM_JOB_ID

# CUDA setup for container
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# CRITICAL: Use V0 engine
export VLLM_USE_V1=0

# Use Triton attention backend for attention sinks support without FA3
export VLLM_ATTENTION_BACKEND=TRITON
export VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1

# Tensor parallelism based on GPU count
export TENSOR_PARALLEL=${SLURM_JOB_GPUS_PER_NODE:-2}

# Run vLLM - explicitly set HF_HOME and TRANSFORMERS_CACHE inside container
apptainer exec --nv \
    --bind $HF_CACHE_DIR:/root/.cache/huggingface \
    --bind $JOB_SCRATCH:/tmp \
    --env HF_HOME=/root/.cache/huggingface \
    --env TRANSFORMERS_CACHE=/root/.cache/huggingface \
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
        --port 8000

echo "Job finished at $(date)"