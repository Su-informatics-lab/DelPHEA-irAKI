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

# ============================================
# Optimized for 2x H100 with Latest vLLM
# Container: vllm-gptoss.sif (0.10.2.dev2)
# ============================================

# Paths
export HF_HOME="$HOME/hf-cache"
export IMG_FILE="$HOME/containers/vllm-gptoss.sif"
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"

# Create directories
mkdir -p "$HF_HOME" logs "$JOB_SCRATCH"

# CUDA configuration for H100
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_CUDA_ARCH_LIST="9.0"  # H100 SM_90

# Network optimization for HPC
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL="NVL"  # H100 NVLink optimization

# Tensor parallel from SLURM allocation
export TENSOR_PARALLEL=${SLURM_JOB_GPUS_PER_NODE:-2}

# ============================================
# Fallback Settings (if needed)
# ============================================
# Uncomment if you get attention backend errors:
# export VLLM_ATTENTION_BACKEND="TRITON_ATTN_VLLM_V1"

# For maximum performance on H100 (experimental):
# export VLLM_USE_TRTLLM_ATTENTION=1
# export VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1

# ============================================
# Diagnostics
# ============================================
echo "========================================="
echo "vLLM GPT-OSS-120B Server Configuration"
echo "========================================="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs: ${TENSOR_PARALLEL} x H100 80GB HBM3"
echo "Container: vllm-gptoss.sif (v0.10.2.dev2)"
echo "Model: openai/gpt-oss-120b"
echo "========================================="

# Show GPU information
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
echo "========================================="

# Show container info
echo "Container details:"
apptainer exec "$IMG_FILE" python3 -c "
import vllm, torch
print(f'vLLM: {vllm.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA in container: {torch.cuda.is_available()}')
" 2>/dev/null || echo "Container info unavailable"
echo "========================================="

# ============================================
# Launch vLLM Server (Official Method)
# ============================================
echo "Starting vLLM server at $(date)..."

apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind /mnt/shared/moduleapps/eb/CUDA/12.3.0:/mnt/shared/moduleapps/eb/CUDA/12.3.0 \
  --bind "$JOB_SCRATCH":/tmp \
  --env CUDA_HOME="$CUDA_HOME" \
  --env PATH="$PATH" \
  --env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  --env NCCL_P2P_LEVEL="$NCCL_P2P_LEVEL" \
  "$IMG_FILE" \
  vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --max-num-seqs 256 \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --enable-prefix-caching \
    --disable-log-requests

# ============================================
# Alternative if 'vllm serve' doesn't work:
# ============================================
# Comment out the above and uncomment below:
#
# apptainer exec --nv \
#   --bind "$HF_HOME":/root/.cache/huggingface \
#   --bind /mnt/shared/moduleapps/eb/CUDA/12.3.0:/mnt/shared/moduleapps/eb/CUDA/12.3.0 \
#   --bind "$JOB_SCRATCH":/tmp \
#   --env CUDA_HOME="$CUDA_HOME" \
#   --env PATH="$PATH" \
#   --env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
#   --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
#   --env NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
#   "$IMG_FILE" \
#   python3 -m vllm.entrypoints.openai.api_server \
#     --model openai/gpt-oss-120b \
#     --tensor-parallel-size "$TENSOR_PARALLEL" \
#     --gpu-memory-utilization 0.90 \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --async-scheduling \
#     --trust-remote-code

echo "========================================="
echo "Server terminated at $(date)"
echo "Exit code: $?"
echo "========================================="