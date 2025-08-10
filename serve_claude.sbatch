#!/bin/bash
#SBATCH --job-name=gptoss_120b
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

# ---------- paths & caches (host) ----------
export HF_HOME="$HOME/hf-cache"
export IMG_DIR="$HOME/containers"
export IMG_FILE="$IMG_DIR/vllm-gptoss.sif"
mkdir -p "$HF_HOME" "$IMG_DIR" logs

export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/${SLURM_JOB_ID}"
export FLASHINFER_CACHE="$HOME/flashinfer-cache"
mkdir -p "$JOB_SCRATCH" "$FLASHINFER_CACHE"

# ---------- bind a CUDA toolchain ----------
HOST_CUDA="/mnt/shared/moduleapps/eb/CUDA/12.3.0"
if [[ ! -d "$HOST_CUDA" ]]; then
  echo "WARN: $HOST_CUDA not found on $(hostname). Falling back to 12.2.0"
  HOST_CUDA="/mnt/shared/moduleapps/eb/CUDA/12.2.0"
fi

# ---------- container env (clean, minimal) ----------
APPT_ENV=(
  --env HF_HOME="/root/.cache/huggingface"
  --env CUDA_HOME="${HOST_CUDA}"
  --env PATH="${HOST_CUDA}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  --env TORCH_CUDA_ARCH_LIST="90"
  --env NCCL_ASYNC_ERROR_HANDLING=1
  --env NCCL_DEBUG=WARN
  --env CUDA_DEVICE_MAX_CONNECTIONS=1
  --env PYTHONUNBUFFERED=1
  # stop shell from sourcing host module hooks
  --env ENV=
  --env BASH_ENV=

  # ===== CRITICAL OSS MODEL FIXES =====
  # Option 1: Use Triton attention backend (most reliable for H100)
  --env VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1

  # Option 2: If you want to try flashinfer (less stable), comment above and uncomment below:
  # --env VLLM_USE_FLASHINFER_MXFP4_MOE=1
  # --env VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1

  # Disable problematic features for initial stability
  --env VLLM_DISABLE_FLASHINFER=1
  --env VLLM_V1_SPEC_DECODE_NGRAM=0

  # Additional stability settings
  --env VLLM_USE_V1=1
  --env VLLM_ENABLE_V1_MULTIPROCESSING=1
  --env HF_HUB_OFFLINE=0
  --env TRANSFORMERS_OFFLINE=0
)

echo "== Summary =="
echo "Node: $(hostname)"
echo "CUDA bind: ${HOST_CUDA}"
echo "GPUs: ${SLURM_JOB_GPUS_PER_NODE:-2}"
echo "HF cache: ${HF_HOME}"
echo "Scratch: ${JOB_SCRATCH}"
echo "================"

# Optional quick CUDA check inside image
apptainer exec --nv --cleanenv \
  --bind "${HOST_CUDA}:${HOST_CUDA}" \
  "${APPT_ENV[@]}" \
  "$IMG_FILE" bash -lc 'nvidia-smi || true'

# ---------- run vLLM OpenAI server ----------
TENSOR_PARALLEL="${SLURM_JOB_GPUS_PER_NODE:-2}"

# CRITICAL: Different settings for TP=1 vs TP=2
if [[ "$TENSOR_PARALLEL" == "1" ]]; then
  # For single GPU, need higher memory utilization and lower batch tokens
  GPU_MEM_UTIL="0.95"
  MAX_BATCH_TOKENS="512"
  MAX_MODEL_LEN="32768"
else
  # For TP=2 on H100, must use lower memory utilization to avoid OOM
  GPU_MEM_UTIL="0.90"
  MAX_BATCH_TOKENS="1024"
  MAX_MODEL_LEN="131072"
fi

apptainer exec --nv --cleanenv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$FLASHINFER_CACHE":/root/.cache/flashinfer \
  --bind "${HOST_CUDA}:${HOST_CUDA}" \
  --bind "$JOB_SCRATCH":/tmp \
  "${APPT_ENV[@]}" \
  "$IMG_FILE" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-num-batched-tokens "${MAX_BATCH_TOKENS}" \
    --async-scheduling \
    --host 0.0.0.0 --port 8000 \
    --uvicorn-log-level info \
    --disable-uvicorn-access-log \
    --trust-remote-code