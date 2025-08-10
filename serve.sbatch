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

# job-local scratch (good place for temp files)
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/${SLURM_JOB_ID}"
mkdir -p "$JOB_SCRATCH"

# ---------- pull official vLLM gptoss image if missing ----------
if [[ ! -f "$IMG_FILE" ]]; then
  echo "pulling vllm/vllm-openai:gptoss into $IMG_FILE ..."
  apptainer pull "$IMG_FILE" docker://vllm/vllm-openai:gptoss
fi

# ---------- container env (clean, minimal) ----------
# IMPORTANT: do NOT bind or override host CUDA into the container;
# the image already bundles a matching CUDA/Torch/Triton stack for gpt-oss.
APPT_ENV=(
  --env HF_HOME="/root/.cache/huggingface"
  --env TRANSFORMERS_CACHE="/root/.cache/huggingface"
  --env HUGGINGFACE_HUB_CACHE="/root/.cache/huggingface"
  --env PYTHONUNBUFFERED=1

  # stabilize vLLM startup on Hopper (we'll use FA3 + Triton MoE kernels)
  --env VLLM_DISABLE_FLASHINFER=1

  # prevent sourcing user login files that call 'module'
  --env BASH_ENV=/dev/null
)

# ---------- summary ----------
TP="${SLURM_GPUS_ON_NODE:-${SLURM_JOB_GPUS_PER_NODE:-2}}"
echo "== Summary =="
echo "Node: $(hostname)"
echo "GPUs requested: ${TP}"
echo "HF cache: ${HF_HOME}"
echo "Scratch: ${JOB_SCRATCH}"
echo "Image: ${IMG_FILE}"
echo "================"

# optional, quick CUDA presence check (no login shell: avoid ~/.modules)
apptainer exec --nv --cleanenv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  "${APPT_ENV[@]}" \
  "$IMG_FILE" bash -c 'nvidia-smi || true'

# ---------- run vLLM OpenAI server ----------
# Notes:
# - use `vllm serve` (official entrypoint for OpenAI-compatible server)
# - gpt-oss support requires the prerelease stack shipped in this image
# - 120B runs on H100 with MoE weights in MXFP4 and attention in bf16
#   per OpenAI/vLLM guidance. Max context 131072.
#   refs: vLLM blog + OpenAI cookbook + HF card.
#   (We don’t pin dtype; vLLM auto-selects for H100.)
apptainer exec --nv --cleanenv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  "${APPT_ENV[@]}" \
  "$IMG_FILE" \
  vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size "${TP}" \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8000 \
    --uvicorn-log-level info \
    --disable-uvicorn-access-log
