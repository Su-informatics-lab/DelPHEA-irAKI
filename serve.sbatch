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
export IMG_FILE="$HOME/containers/vllm-gptoss.sif"   # use the GPT-OSS image
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$HF_HOME" logs "$JOB_SCRATCH"

# gpu env
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=1
export TORCH_CUDA_ARCH_LIST="9.0"   # H100 (sm_90)

# vllm v1 on hopper selects FA3 automatically; do NOT force Triton
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND
export VLLM_DISABLE_FLASHINFER=1
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_LEVEL=INFO

# tensor parallel
export TENSOR_PARALLEL=2

echo "=== vLLM gpt-oss-120b on $(hostname) job=$SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# --- sanity: print versions and confirm transformers sees 'gpt_oss'
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  "$IMG_FILE" \
  python3 - << 'PY'
import importlib, sys
import torch, vllm
print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
try:
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    print("transformers:", transformers.__version__)
    print("has gpt_oss in CONFIG_MAPPING_NAMES:",
          any("gpt_oss" in k for k in CONFIG_MAPPING_NAMES.keys()))
except Exception as e:
    print("transformers import failed:", e, file=sys.stderr)
    sys.exit(1)
PY

# serve (keep context modest for first boot; raise later)
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
