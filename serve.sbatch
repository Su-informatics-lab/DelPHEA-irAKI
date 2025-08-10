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

# paths
export HF_HOME="$HOME/hf-cache"
export IMG_FILE="$HOME/containers/vllm-gptoss.sif"
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$HF_HOME" logs "$JOB_SCRATCH"

# gpu env
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=1
export TORCH_CUDA_ARCH_LIST="9.0"    # h100 (sm_90)

# vllm: force v1 and a conservative attention backend (avoids FA3 issues)
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_V2

# helpful logging so we actually see the worker error if any
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_LEVEL=DEBUG

# tensor parallel across the two gpus
export TENSOR_PARALLEL=2

echo "========================================="
echo "vLLM gpt-oss-120b on $(hostname)  job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "========================================="

# quick preflight inside container: show versions & cuda
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  "$IMG_FILE" \
  python3 - << 'PY'
import torch, vllm, sys
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("vllm:", vllm.__version__)
PY

# serve (use the official vLLM CLI path)
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  "$IMG_FILE" \
  vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling

echo "server terminated at $(date)"
