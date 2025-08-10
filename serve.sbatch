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

# --------- paths & caches (host) ----------
export HF_HOME="$HOME/hf-cache"
export IMG_DIR="$HOME/containers"
export IMG_FILE="$IMG_DIR/vllm-gptoss.sif"

mkdir -p "$HF_HOME" "$IMG_DIR" logs

# per-job scratch and persistent JIT cache dirs
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/${SLURM_JOB_ID}"
export FLASHINFER_CACHE="$HOME/flashinfer-cache"
mkdir -p "$JOB_SCRATCH" "$FLASHINFER_CACHE"

# --------- CUDA toolchain to bind ----------
# You can switch this to 12.2.0 if a node doesn’t have 12.3.0.
HOST_CUDA="/mnt/shared/moduleapps/eb/CUDA/12.3.0"
if [[ ! -d "$HOST_CUDA" ]]; then
  echo "WARN: $HOST_CUDA not found on $(hostname). Trying 12.2.0..."
  HOST_CUDA="/mnt/shared/moduleapps/eb/CUDA/12.2.0"
fi

# --------- vLLM / runtime env (inside container) ----------
# Clean, minimal env to avoid tclsh/module pollution
# Disable FlashInfer (no nvcc JIT), and disable Numba n-gram proposer path.
APPT_ENV=(
  --env HF_HOME="/root/.cache/huggingface"
  --env CUDA_HOME="${HOST_CUDA}"
  --env PATH="${HOST_CUDA}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  --env TORCH_CUDA_ARCH_LIST="90"
  --env NCCL_ASYNC_ERROR_HANDLING=1
  --env NCCL_DEBUG=WARN
  --env CUDA_DEVICE_MAX_CONNECTIONS=1
  --env PYTHONUNBUFFERED=1
  --env VLLM_DISABLE_FLASHINFER=1
  --env VLLM_V1_SPEC_DECODE_NGRAM=0
  # Optional: better error messages while stabilizing (slower)
  # --env CUDA_LAUNCH_BLOCKING=1
)

echo "== Summary =="
echo "Node: $(hostname)"
echo "CUDA bind path: ${HOST_CUDA}"
echo "GPUs requested: ${SLURM_JOB_GPUS_PER_NODE:-2}"
echo "HF cache: ${HF_HOME}"
echo "FlashInfer cache (not used when disabled): ${FLASHINFER_CACHE}"
echo "Scratch: ${JOB_SCRATCH}"
echo "=============================="

# (Optional) quick sanity check of CUDA in container
apptainer exec --nv --cleanenv \
  --bind "${HOST_CUDA}:${HOST_CUDA}" \
  "${APPT_ENV[@]}" \
  "$IMG_FILE" bash -lc 'nvidia-smi || true'

# --------- run vLLM OpenAI server ----------
TENSOR_PARALLEL="${SLURM_JOB_GPUS_PER_NODE:-2}"

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
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port 8000 \
    --log-level DEBUG
