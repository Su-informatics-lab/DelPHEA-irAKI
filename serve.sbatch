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

# host-side caches & images
export HF_HOME="$HOME/hf-cache"
export IMG_DIR="$HOME/containers"
export IMG_FILE="$IMG_DIR/vllm-gptoss.sif"

mkdir -p "$HF_HOME" "$IMG_DIR" logs

# per-job scratch
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/${SLURM_JOB_ID}"
mkdir -p "$JOB_SCRATCH"

# point to host cuda toolkit that we bind into the container
export HOST_CUDA="/mnt/shared/moduleapps/eb/CUDA/12.3.0"
export CUDA_HOME="$HOST_CUDA"
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# h100 arch for any cpp_extension builds (sm_90)
export TORCH_CUDA_ARCH_LIST="90"

# avoid runtime kernel compilation surprises
export FLASHINFER_DISABLE_JIT=1

# nccl: prefer on-node (nvlink) comms; avoid ib oddities in containers
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# vllm logging verbosity
export VLLM_LOGGING_LEVEL=DEBUG

# tensor parallel based on slurm allocation (fallback 2)
export TENSOR_PARALLEL="${SLURM_JOB_GPUS_PER_NODE:-2}"

# quick env sanity in the job logs
echo "== cuda =="
nvidia-smi || true
echo "CUDA_HOME=$CUDA_HOME"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "PATH=$PATH"

# launch vllm inside apptainer
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$HOST_CUDA":"$HOST_CUDA" \
  --bind "$JOB_SCRATCH":/tmp \
  --env CUDA_HOME="$CUDA_HOME" \
  --env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  --env PATH="$PATH" \
  --env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  --env FLASHINFER_DISABLE_JIT="$FLASHINFER_DISABLE_JIT" \
  --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  --env NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
  --env NCCL_SHM_DISABLE="$NCCL_SHM_DISABLE" \
  --env VLLM_LOGGING_LEVEL="$VLLM_LOGGING_LEVEL" \
  "$IMG_FILE" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16 \
    --host 0.0.0.0 --port 8000 \
    --engine-log-level debug --log-level debug
