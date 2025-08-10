#!/bin/bash
#SBATCH --job-name=gptoss_native
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

# ---------- Modules (no host CUDA module needed) ----------
module purge
module load GCCcore/13.3.0
module load Python/3.12.3-GCCcore-13.3.0

# ---------- Paths ----------
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" logs

# Persistent venv
export VENV_DIR="${VENV_DIR:-$HOME/.venvs/gptoss-cu128-py312}"
mkdir -p "$(dirname "$VENV_DIR")"

# Job scratch
export JOB_SCRATCH="${JOB_SCRATCH:-$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID}"
mkdir -p "$JOB_SCRATCH"

# ---------- Exact Torch nightly required by vLLM gptoss ----------
# As of Aug 10, 2025 the wheel pins Torch to this build:
export TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
export TORCH_VER="2.9.0.dev20250804+cu128"   # must match vllm==0.10.1+gptoss

# ---------- Bootstrap venv (idempotent, fast on re-run) ----------
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install -U pip uv
fi

source "$VENV_DIR/bin/activate"

# Ensure pinned Torch is installed first (avoid solver conflicts)
if ! python - <<PY | grep -q "$TORCH_VER"; then
import sys
try:
    import torch
    print(getattr(torch, "__version__", ""))
except Exception:
    print("none")
PY
then
  uv pip uninstall -y torch || true
  uv pip install --pre "torch==$TORCH_VER" --index-url "$TORCH_INDEX"
fi

# Install vLLM GPT-OSS wheel and deps
uv pip install --pre "vllm==0.10.1+gptoss" \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url "$TORCH_INDEX" \
  --index-strategy unsafe-best-match

# Newer transformers works best with gpt-oss configs
python -m pip install -U "transformers>=4.46.0"

# ---------- Runtime env (H100 + V1 + FA3) ----------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_USE_V1=1
export VLLM_DISABLE_FLASHINFER=1
# If you want to be explicit:
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export VLLM_FLASH_ATTN_VERSION=3
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# ---------- Tunables (overridable with sbatch --export=...) ----------
export MODEL="${MODEL:-openai/gpt-oss-20b}"   # switch to openai/gpt-oss-120b after first boot
export PORT="${PORT:-8000}"
export GMU="${GMU:-0.90}"
GPU_COUNT=$(nvidia-smi -L | wc -l | awk '{print $1}')
export TP="${TP:-$GPU_COUNT}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# ---------- Preflight ----------
echo "=== gpt-oss serve (native) ==="
echo "node: $(hostname) | job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
python - <<'PY'
import torch, vllm, transformers
print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
print("has gpt_oss config:", any("gpt_oss" in k for k in CONFIG_MAPPING_NAMES.keys()))
PY

trap 'echo "[serve] received USR1 at $(date); exiting"; exit 0' USR1

# ---------- Serve ----------
echo "model=$MODEL tp=$TP gmu=$GMU max_len=$MAX_MODEL_LEN port=$PORT"
vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GMU" \
  --max-model-len "$MAX_MODEL_LEN" \
  --async-scheduling \
  --host 0.0.0.0 --port "$PORT"
