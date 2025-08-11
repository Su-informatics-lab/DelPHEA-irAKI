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

# host-side caches & image
export HF_HOME="$HOME/hf-cache"
export IMG_DIR="$HOME/containers"
export IMG_FILE="$IMG_DIR/vllm-gptoss.sif"
mkdir -p "$HF_HOME" "$IMG_DIR" logs

# per-job scratch inside project
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$JOB_SCRATCH"

# cuda toolchain path exposed inside container
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

# stability: avoid v1 multiprocess + JIT flakiness on HPC
export VLLM_ENABLE_V1_MULTIPROCESSING=0     # force no v1 multiprocessing
export FLASHINFER_DISABLE_JIT=1             # skip flashinfer JIT (more deterministic)
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1  # quiet hub warnings
ulimit -l unlimited || true                 # allow mlock if permitted

# tensor-parallel from slurm (fallback to gpu count)
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  TENSOR_PARALLEL="$SLURM_GPUS_ON_NODE"
elif [[ -n "${SLURM_JOB_GPUS_PER_NODE:-}" ]]; then
  TENSOR_PARALLEL="$SLURM_JOB_GPUS_PER_NODE"
else
  TENSOR_PARALLEL="$(nvidia-smi -L | wc -l || echo 2)"
fi

echo "starting gpt-oss-120b | job ${SLURM_JOB_ID} on $(hostname)"
date
nvidia-smi || true

# launch vLLM inside apptainer (python3 in container)
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind /mnt/shared/moduleapps/eb/CUDA/12.3.0:/mnt/shared/moduleapps/eb/CUDA/12.3.0 \
  --bind "$JOB_SCRATCH":/tmp \
  --env CUDA_HOME="$CUDA_HOME" \
  --env PATH="$PATH" \
  "$IMG_FILE" \
  bash -c "pip install 'numpy<2.3' && \
  python3 -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size \"$TENSOR_PARALLEL\" \
    --distributed-executor-backend uni \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --max-num-batched-tokens 1024 \
    --host 0.0.0.0 --port 8000"

echo "job finished at $(date)"