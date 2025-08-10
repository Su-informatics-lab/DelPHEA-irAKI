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

# caches & image
export HF_HOME="$HOME/hf-cache"
export IMG_FILE="$HOME/containers/vllm-gptoss.sif"
mkdir -p "$HF_HOME" logs

# per-job scratch
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/${SLURM_JOB_ID}"
mkdir -p "$JOB_SCRATCH"

# log levels (server via uvicorn flag; engine via env)
export VLLM_LOGGING_LEVEL=DEBUG

# pick tensor-parallel from allocated GPUs
export TENSOR_PARALLEL="${SLURM_JOB_GPUS_PER_NODE:-2}"

echo "== sanity =="
nvidia-smi || true
apptainer exec --nv "$IMG_FILE" vllm --version || true
apptainer exec --nv "$IMG_FILE" python -c "import vllm,sys;print('vllm',vllm.__version__)" || true

# launch vLLM (no invalid flags)
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  --env VLLM_LOGGING_LEVEL="$VLLM_LOGGING_LEVEL" \
  "$IMG_FILE" \
  vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16 \
    --host 0.0.0.0 --port 8000 \
    --uvicorn-log-level debug --disable-uvicorn-access-log
