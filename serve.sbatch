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

# cuda & gpu
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=1
export TORCH_CUDA_ARCH_LIST="9.0"   # hopper (h100)

# ---- vllm engine/attention fixes ----
# force v1 engine; v0 is what threw the "Invalid attention backend" error.
export VLLM_USE_V1=1
# on h100, v1 + flash_attn (fa3) is the intended path.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# tp=2 on 2×H100
export TENSOR_PARALLEL=2

echo "========================================="
echo "vLLM gpt-oss-120b server"
echo "node: $(hostname)"
echo "job : ${SLURM_JOB_ID}"
echo "gpus: ${CUDA_VISIBLE_DEVICES} (TP=${TENSOR_PARALLEL})"
echo "img : $(basename "$IMG_FILE")"
echo "========================================="

# quick gpu info
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# launch
# note: we do NOT bind host CUDA into the container; --nv handles driver libs
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  "$IMG_FILE" \
  python3 -u -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --async-scheduling \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code

echo "========================================="
echo "server terminated at $(date)"
echo "========================================="
