#!/bin/bash
#SBATCH --job-name=gptoss_auto
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

# -------------------- config you can tweak --------------------
# Model: start with 20B to validate; flip to 120B once stable
export MODEL="${MODEL:-openai/gpt-oss-20b}"
export PORT="${PORT:-8000}"
export GMU="${GMU:-0.90}"                 # gpu-memory-utilization
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export VENV_DIR="${VENV_DIR:-$HOME/.venvs/gptoss-cu128-py312}"
export IMG_FILE="${IMG_FILE:-$HOME/containers/vllm-gptoss.sif}"

# Torch nightly pinned to match vllm==0.10.1+gptoss
export TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
export TORCH_VER="2.9.0.dev20250804+cu128"

mkdir -p "$HF_HOME" logs

# -------------------- basic hardware info --------------------
echo "=== host info ==="
echo "node: $(hostname) | job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || true
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}')
GPU_COUNT=${GPU_COUNT:-2}
export TP="${TP:-$GPU_COUNT}"

# -------------------- detect GLIBC --------------------
glibc_str="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
if [[ -z "$glibc_str" ]]; then
  glibc_str="$(ldd --version 2>&1 | head -n1 || true)"
fi
echo "GLIBC reported: ${glibc_str:-unknown}"

# parse major.minor
glibc_ver="$(echo "$glibc_str" | grep -Eo '[0-9]+\.[0-9]+' | head -n1 || true)"
use_native=0
if [[ -n "$glibc_ver" ]]; then
  maj="${glibc_ver%%.*}"
  min="${glibc_ver##*.}"
  if (( maj > 2 || (maj == 2 && min >= 32) )); then
    use_native=1
  fi
fi

# -------------------- common runtime env --------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_USE_V1=1
export VLLM_DISABLE_FLASHINFER=1
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

trap 'echo "[serve] received USR1 at $(date); exiting cleanly"; exit 0' USR1

if (( use_native )); then
  echo "=== path: NATIVE (GLIBC >= 2.32) ==="

  # modules (no host CUDA module needed)
  module purge
  module load GCCcore/13.3.0
  module load Python/3.12.3-GCCcore-13.3.0

  # persistent venv
  mkdir -p "$(dirname "$VENV_DIR")"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install -U pip uv
  fi
  source "$VENV_DIR/bin/activate"

  # ensure pinned torch is present (force-reinstall is safer than uninstall)
  if ! python - <<PY | grep -q "$TORCH_VER"; then
import sys
try:
    import torch
    print(getattr(torch, "__version__", "none"))
except Exception:
    print("none")
PY
  then
    uv pip install --pre "torch==$TORCH_VER" --force-reinstall --index-url "$TORCH_INDEX"
  fi

  # install vLLM GPT-OSS wheel & deps
  uv pip install --pre "vllm==0.10.1+gptoss" \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url "$TORCH_INDEX" \
    --index-strategy unsafe-best-match
  python -m pip install -U "transformers>=4.46.0"

  echo "=== versions (native) ==="
  python - <<'PY'
import torch, vllm, transformers
print("torch:", torch.__version__, "cuda:", getattr(torch.version, "cuda", "?"))
print("vllm :", vllm.__version__)
print("trxf :", transformers.__version__)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
print("has gpt_oss config:", any("gpt_oss" in k for k in CONFIG_MAPPING_NAMES.keys()))
PY

  echo "model=$MODEL tp=$TP gmu=$GMU max_len=$MAX_MODEL_LEN port=$PORT"
  exec vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAX_MODEL_LEN" \
    --async-scheduling \
    --host 0.0.0.0 --port "$PORT"

else
  echo "=== path: CONTAINER (GLIBC < 2.32) ==="
  echo "Using image: $IMG_FILE"
  mkdir -p "$HOME/containers" "$HF_HOME"

  # Launch vLLM GPT-OSS in container (V1 + FA3 on H100). Do NOT bind host CUDA.
  apptainer exec --nv \
    --bind "$HF_HOME":/root/.cache/huggingface \
    "$IMG_FILE" \
    vllm serve openai/gpt-oss-20b \
      --tensor-parallel-size "$TP" \
      --gpu-memory-utilization "$GMU" \
      --max-model-len "$MAX_MODEL_LEN" \
      --async-scheduling \
      --host 0.0.0.0 --port "$PORT"
fi
