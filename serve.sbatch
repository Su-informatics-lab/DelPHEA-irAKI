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

# --- Paths / dirs ------------------------------------------------------------
export HF_HOME="$HOME/hf-cache"
export IMG_DIR="$HOME/containers"
export IMG_FILE="$IMG_DIR/vllm-gptoss.sif"   # we’ll (re)pull this tag below
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$HF_HOME" "$IMG_DIR" "$JOB_SCRATCH" logs

# --- Pull/pin the gptoss image (works for GPT-OSS) --------------------------
# The :gptoss tag is the one vLLM publishes for GPT-OSS support.
# (Avoid :latest; it may lag transformers needed for gpt_oss.)
if [ ! -f "$IMG_FILE" ]; then
  module purge >/dev/null 2>&1 || true
  module load Apptainer >/dev/null 2>&1 || true
  apptainer pull "$IMG_FILE" docker://vllm/vllm-openai:gptoss
fi

# --- GPU / NCCL env ----------------------------------------------------------
# Don’t bind host CUDA into the container. Let --nv provide driver/runtime.
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=1

# Hopper arch to avoid slow JIT warnings in Triton/FA
export TORCH_CUDA_ARCH_LIST="9.0"

# --- vLLM engine & attention backend (H100 path) -----------------------------
# On Hopper, V1 + FlashAttention-3 is the recommended path.
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# Optional: crank logging if you still get “Failed core proc(s): {}”.
# export VLLM_LOGGING_LEVEL=DEBUG

# --- Big /dev/shm inside container ------------------------------------------
# Apptainer doesn’t have --shm-size; bind host /dev/shm to get large shared mem.
# (vLLM docs: ensure large /dev/shm or use host IPC on Docker.)
# We also bind a job-local tmp for any large temp files.
BIND_OPTS=(
  --bind "$HF_HOME":/root/.cache/huggingface
  --bind /dev/shm:/dev/shm
  --bind "$JOB_SCRATCH":/tmp
)

# --- Print quick diagnostics -------------------------------------------------
echo "======== Diagnostics ========"
echo "Host: $(hostname)"
echo "Job:  $SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "VLLM_USE_V1=$VLLM_USE_V1  VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "============================="

# --- Choose TP from allocated GPUs ------------------------------------------
TP=${SLURM_GPUS_ON_NODE:-2}

# --- Start with 20B to verify the stack, then you can bump to 120B ----------
MODEL="${MODEL:-openai/gpt-oss-20b}"   # change to openai/gpt-oss-120b after verify

# --- Launch ------------------------------------------------------------------
# Important: do NOT bind host CUDA paths; --nv is enough and avoids ABI mismatches.
apptainer exec --nv "${BIND_OPTS[@]}" "$IMG_FILE" \
  vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --disable-log-requests

# Notes:
#  - If you still see “Engine core initialization failed”, toggle:
#      * export VLLM_USE_V1=0          # fall back to vLLM v0 executor
#      * or:    VLLM_ATTENTION_BACKEND=TRITON_ATTN  # try Triton backend
#  - After the 20B run is stable, set MODEL=openai/gpt-oss-120b and re-submit.
