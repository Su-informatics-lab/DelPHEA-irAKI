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
# use the freshest image; gptoss tag may lag on FA3 fixes
export IMG_FILE="$HOME/containers/vllm-latest.sif"
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$HF_HOME" logs "$JOB_SCRATCH"

# gpu env
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=1
export TORCH_CUDA_ARCH_LIST="9.0"   # h100

# vllm v1 + fa3: don't force a backend; v1 on hopper selects flash-attn 3
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
export VLLM_DISABLE_FLASHINFER=1
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_LEVEL=INFO

# tp across 2 gpus
export TENSOR_PARALLEL=2

echo "=== vllm gpt-oss-120b on $(hostname) job=$SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# quick preflight inside container
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  "$IMG_FILE" \
  python3 - << 'PY'
import torch, vllm
print("torch:", torch.__version__)
print("cuda ok:", torch.cuda.is_available(), "ngpus:", torch.cuda.device_count())
print("vllm:", vllm.__version__)
PY

# serve (v1 chooses fa3 on hopper)
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  "$IMG_FILE" \
  vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling

echo "server terminated at $(date)"
