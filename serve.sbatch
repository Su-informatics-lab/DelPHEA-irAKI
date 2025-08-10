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

# ---------- paths & caches ----------
export HF_HOME="$HOME/hf-cache"
export IMG_DIR="$HOME/containers"
export IMG_FILE="$IMG_DIR/vllm-gptoss.sif"

mkdir -p "$HF_HOME" "$IMG_DIR" logs

# per-job temp dir & persistent JIT cache for flashinfer
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/${SLURM_JOB_ID}"
export FLASHINFER_CACHE="$HOME/flashinfer-cache"
mkdir -p "$JOB_SCRATCH" "$FLASHINFER_CACHE"

# ---------- CUDA toolchain bound into container ----------
# We bind the system CUDA 12.3 toolchain so nvcc exists inside the container.
export HOST_CUDA="/mnt/shared/moduleapps/eb/CUDA/12.3.0"
if [[ ! -x "${HOST_CUDA}/bin/nvcc" ]]; then
  echo "FATAL: ${HOST_CUDA}/bin/nvcc not found on this node."; exit 1
fi

# limit JIT archs to H100 (sm_90) to speed compilation
export TORCH_CUDA_ARCH_LIST=90

# ---------- vLLM / NCCL environment ----------
# (Keep FlashInfer enabled; we want it to JIT once and cache.)
# If you ever want to force-disable FlashInfer sampling:
# export VLLM_DISABLE_FLASHINFER=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1

# tensor parallel = number of visible GPUs (default 2)
TENSOR_PARALLEL="${SLURM_JOB_GPUS_PER_NODE:-2}"

echo "== Environment summary =="
echo "Node: $(hostname)"
echo "GPUs: ${TENSOR_PARALLEL}"
echo "CUDA bind: ${HOST_CUDA}"
echo "HF cache: ${HF_HOME}"
echo "FlashInfer cache: ${FLASHINFER_CACHE}"
echo "-----------------------------------------"

# quick sanity: ensure nvcc resolves INSIDE the container
apptainer exec --nv \
  --bind "${HOST_CUDA}:${HOST_CUDA}" \
  --env CUDA_HOME="${HOST_CUDA}" \
  --env PATH="${HOST_CUDA}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "$IMG_FILE" bash -lc 'which nvcc && nvcc --version || true'

# ---------- launch vLLM OpenAI server ----------
# Binds:
#  - HuggingFace cache (persist model files)
#  - FlashInfer cache (persist JIT .so files)
#  - Job scratch as /tmp
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$FLASHINFER_CACHE":/root/.cache/flashinfer \
  --bind "${HOST_CUDA}:${HOST_CUDA}" \
  --bind "$JOB_SCRATCH":/tmp \
  --env CUDA_HOME="${HOST_CUDA}" \
  --env PATH="${HOST_CUDA}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  "$IMG_FILE" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port 8000
