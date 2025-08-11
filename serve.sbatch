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

# ---------- dirs ----------
mkdir -p logs
export HF_HOME="$HOME/hf-cache"
mkdir -p "$HF_HOME"
export JOB_SCRATCH="$(pwd)/tmp/${SLURM_JOB_ID}"
mkdir -p "$JOB_SCRATCH"

# ---------- basic info ----------
echo "starting gpt-oss-120b on $(hostname)"
echo "job id: ${SLURM_JOB_ID}"
date
nvidia-smi || true

# ---------- tensor parallel auto (fallback to 2 if unknown) ----------
TP="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "${TP}" ]]; then
  TP="$(nvidia-smi -L | wc -l || echo 2)"
fi

# ---------- image (official vLLM gpt-oss build with FA3 + cu128) ----------
# if you can't pull at runtime, pre-pull with:
#   apptainer pull gptoss.sif docker://vllm/vllm-openai:gptoss
IMAGE="${IMAGE:-docker://vllm/vllm-openai:gptoss}"

# ---------- env (H100 path uses defaults; no legacy TRITON/V0 flags) ----------
# keep caches clean and silence TRANSFORMERS_CACHE deprecation
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# known limitation: tp1 on H100 can OOM unless we cap batched tokens (vLLM docs)
EXTRA_ARGS=()
if [[ "${TP}" == "1" ]]; then
  EXTRA_ARGS+=(--gpu-memory-utilization 0.95 --max-num-batched-tokens 1024)
fi

# optional: async scheduling improves throughput (not compatible with structured output)
EXTRA_ARGS+=(--async-scheduling)

# ---------- run ----------
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  "$IMAGE" \
  python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size "${TP}" \
    --host 0.0.0.0 \
    --port 8000 \
    "${EXTRA_ARGS[@]}"

echo "job finished at $(date)"
