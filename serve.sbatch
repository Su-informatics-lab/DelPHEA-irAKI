#!/bin/bash
#SBATCH --job-name=gptoss_server
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

# caches & image
export HF_HOME="$HOME/hf-cache"
export IMG_FILE="$HOME/containers/vllm-gptoss.sif"   # pull from: apptainer pull $IMG_FILE docker://vllm/vllm-openai:gptoss
mkdir -p "$HF_HOME" logs

# per-job scratch
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/${SLURM_JOB_ID}"
mkdir -p "$JOB_SCRATCH"

# vllm logging (engine) + uvicorn verbosity
export VLLM_LOGGING_LEVEL=DEBUG

# hopper-friendly settings: no extra JIT, target sm_90
export TORCH_CUDA_ARCH_LIST="90"
export FLASHINFER_DISABLE_JIT=1
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"   # Hopper uses FA3 per vLLM’s guidance

# networking quirks often seen on HPC
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# infer TP from allocation
export TENSOR_PARALLEL="${SLURM_JOB_GPUS_PER_NODE:-2}"

echo "== nvidia-smi =="
nvidia-smi || true
echo "VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL"

# launch vLLM (supported CLI)
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  --env VLLM_LOGGING_LEVEL="$VLLM_LOGGING_LEVEL" \
  --env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  --env FLASHINFER_DISABLE_JIT="$FLASHINFER_DISABLE_JIT" \
  --env VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND" \
  --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  --env NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
  --env NCCL_SHM_DISABLE="$NCCL_SHM_DISABLE" \
  "$IMG_FILE" \
  vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization 0.88 \
    --max-num-seqs 16 \
    --max-parallel-loading-workers 1 \
    --host 0.0.0.0 --port 8000 \
    --uvicorn-log-level debug --disable-uvicorn-access-log
