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

# per-job scratch
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$JOB_SCRATCH"

# cuda toolchain path exposed inside container
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

# tensor-parallel from slurm (fallback to gpu count)
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  TENSOR_PARALLEL="$SLURM_GPUS_ON_NODE"
elif [[ -n "${SLURM_JOB_GPUS_PER_NODE:-}" ]]; then
  TENSOR_PARALLEL="$SLURM_JOB_GPUS_PER_NODE"
else
  TENSOR_PARALLEL="$(nvidia-smi -L | wc -l || echo 2)"
fi

# pick a localhost port for torch rendezvous
MASTER_PORT="$(shuf -i 20000-29999 -n 1)"

echo "starting gpt-oss-120b | job ${SLURM_JOB_ID} on $(hostname) | TP=${TENSOR_PARALLEL} | MASTER_PORT=${MASTER_PORT}"
date
nvidia-smi || true

# launch inside apptainer:
# - pin torch dist to localhost (no IB)
# - fix numpy for numba (numpy<2.3)
# - run vLLM v1 single-process executor with TP=2
apptainer exec --nv --cleanenv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind /mnt/shared/moduleapps/eb/CUDA/12.3.0:/mnt/shared/moduleapps/eb/CUDA/12.3.0 \
  --bind "$JOB_SCRATCH":/tmp \
  --env CUDA_HOME="$CUDA_HOME" \
  --env PATH="$PATH" \
  --env BASH_ENV=/dev/null \
  "$IMG_FILE" \
  bash -lc '
    set -euo pipefail
    # fix numba/numPy
    PYDEPS=/tmp/pydeps
    mkdir -p "$PYDEPS"
    python3 -m pip install --no-cache-dir --target "$PYDEPS" "numpy<2.3" >/tmp/pip.out 2>&1 || cat /tmp/pip.out
    export PYTHONPATH="$PYDEPS:${PYTHONPATH-}"

    # torch/vllm distributed on localhost only
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT='"$MASTER_PORT"'
    export GLOO_SOCKET_IFNAME=lo
    export NCCL_SOCKET_IFNAME=lo
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_SHM_DISABLE=0
    export NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_DISTRIBUTED_DEBUG=OFF
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
    export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

    python3 -m vllm.entrypoints.openai.api_server \
      --model openai/gpt-oss-120b \
      --tensor-parallel-size '"$TENSOR_PARALLEL"' \
      --distributed-executor-backend uni \
      --gpu-memory-utilization 0.85 \
      --max-model-len 8192 \
      --max-num-batched-tokens 1024 \
      --host 0.0.0.0 --port 8000
  '

echo "job finished at $(date)"
