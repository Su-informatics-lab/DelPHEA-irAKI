#!/bin/bash
#SBATCH --job-name=gptoss-vllm          # shown by squeue
#SBATCH --partition=nextgen-gpu
#SBATCH --nodes=1                       # stay on a single host
#SBATCH --gres=gpu:h100:2               # 2× H100
#SBATCH --cpus-per-gpu=4                # 8 logical cores total
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1                      # one task → one Apptainer process
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --signal=USR1@120               # 2-min pre-emtion notice

set -euo pipefail                       # safer bash

# paths
export HF_HOME="$HOME/hf-cache"         # HuggingFace cache (host)
export IMG_DIR="$HOME/containers"
export IMG_FILE="$IMG_DIR/vllm-gptoss.sif"

# per-job scratch for Flash-infer / torch.compile caches
export JOB_SCRATCH="/scratch/$USER/vllm-cache/$SLURM_JOB_ID"
mkdir -p "$HF_HOME" "$JOB_SCRATCH"

# CUDA tool-chain inside the container
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# (optional) Skip Flash-infer JIT if you don’t want compilation
# export FLASHINFER_DISABLE_JIT=1

# launch
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind /mnt/shared/moduleapps/eb/CUDA/12.3.0:/mnt/shared/moduleapps/eb/CUDA/12.3.0 \
  --bind "$JOB_SCRATCH":/tmp \
  --env CUDA_HOME \
  --env PATH \
  "$IMG_FILE" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size "$SLURM_GPUS" \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port 8000
