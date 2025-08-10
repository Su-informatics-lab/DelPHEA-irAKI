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

set -euo pipefail

# -------------------- paths --------------------
export HF_HOME="$HOME/hf-cache"
export IMG_DIR="$HOME/containers"
export IMG_FILE="$IMG_DIR/vllm-gptoss.sif"
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$HF_HOME" "$IMG_DIR" "$JOB_SCRATCH"/{tmp,pydeps,.cache,torch_extensions,triton} logs

# Pull image if missing
if [ ! -f "$IMG_FILE" ]; then
  module purge >/dev/null 2>&1 || true
  module load Apptainer >/dev/null 2>&1 || true
  apptainer pull "$IMG_FILE" docker://vllm/vllm-openai:gptoss
fi

# -------------------- GPUs/NCCL --------------------
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_DISABLE=1
unset TORCH_NCCL_AVOID_RECORD_STREAMS   # silence deprecation warning

# -------------------- H100-safe vLLM --------------------
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_DISABLE_FLASHINFER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=INFO

# -------------------- caches/JIT --------------------
export XDG_CACHE_HOME="$JOB_SCRATCH/.cache"
export TORCH_EXTENSIONS_DIR="$JOB_SCRATCH/torch_extensions"
export TRITON_CACHE_DIR="$JOB_SCRATCH/triton"
export TMPDIR="$JOB_SCRATCH/tmp"

# H100 arch + allocator
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True

# tensor parallel size
TP=${SLURM_GPUS_ON_NODE:-2}

echo "==== Preflight ===="
echo "Host: $(hostname)  Job: $SLURM_JOB_ID  GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# -------------------- hotpatch numpy/numba --------------------
# Numba supports up to NumPy 2.2; some container revs ship NumPy 2.3+
# Install compatible pair into a writable dir and prepend to PYTHONPATH.
apptainer exec --nv \
  --bind "$JOB_SCRATCH":/mnt/job \
  "$IMG_FILE" \
  python3 - <<'PY'
import sys, subprocess, os
tgt = "/mnt/job/pydeps"
os.makedirs(tgt, exist_ok=True)
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "--no-cache-dir", "--upgrade", "--target", tgt,
                       "numpy<2.3", "numba==0.61.2"])
print("Patched: numpy<2.3 & numba==0.61.2 ->", tgt)
PY

export PYTHONPATH="$JOB_SCRATCH/pydeps:$PYTHONPATH"

# quick sanity print
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$JOB_SCRATCH":/tmp \
  --bind "$JOB_SCRATCH":/mnt/job \
  --env PYTHONPATH="$PYTHONPATH" \
  "$IMG_FILE" \
  python3 - <<'PY'
import numpy, numba, torch
print("numpy", numpy.__version__, "| numba", numba.__version__, "| torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY

# -------------------- launch vLLM (OSS 120B) --------------------
MODEL="openai/gpt-oss-120b"
MAX_LEN="${MAX_LEN:-8192}"            # bump later if needed
GMU="${GMU:-0.90}"

echo "==== Launching vLLM: $MODEL  TP=$TP  MAX_LEN=$MAX_LEN  GMU=$GMU ===="

# Bind only what we need; let --nv handle driver/runtime
apptainer exec --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind /dev/shm:/dev/shm \
  --bind "$JOB_SCRATCH":/tmp \
  --bind "$JOB_SCRATCH":/mnt/job \
  --env PYTHONPATH="$PYTHONPATH" \
  --env VLLM_USE_V1="$VLLM_USE_V1" \
  --env VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND" \
  --env VLLM_DISABLE_FLASHINFER="$VLLM_DISABLE_FLASHINFER" \
  --env VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD" \
  --env VLLM_LOGGING_LEVEL="$VLLM_LOGGING_LEVEL" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  --env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  "$IMG_FILE" vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAX_LEN" \
    --host 0.0.0.0 --port 8000 \
    --disable-log-requests

echo "Server exited at $(date)"
