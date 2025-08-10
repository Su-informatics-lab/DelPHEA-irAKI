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

# usage:
#   sbatch serve.sbatch
#   sbatch --export=MODEL=openai/gpt-oss-120b,PORT=8001,TP=2,MAX_MODEL_LEN=32768 serve.sbatch

set -euo pipefail

# -------------------- modules (no cuda module required) --------------------
module purge
module load GCCcore/13.3.0
module load Python/3.12.3-GCCcore-13.3.0

# -------------------- paths & caches --------------------
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" logs

# persistent venv to avoid reinstall every job
export VENV_DIR="${VENV_DIR:-$HOME/.venvs/gptoss-cu128-py312}"
mkdir -p "$(dirname "$VENV_DIR")"

# job scratch
export JOB_SCRATCH="${JOB_SCRATCH:-$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID}"
mkdir -p "$JOB_SCRATCH"

# -------------------- venv bootstrap (idempotent) --------------------
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install -U pip uv
  # install torch nightly cu128 + vllm gpt-oss + recent transformers
  "$VENV_DIR/bin/uv" pip install --pre "torch>=2.5.0.dev" \
      --index-url https://download.pytorch.org/whl/nightly/cu128
  "$VENV_DIR/bin/uv" pip install --pre "vllm==0.10.1+gptoss" \
      --extra-index-url https://wheels.vllm.ai/gpt-oss/
  "$VENV_DIR/bin/python" -m pip install -U "transformers>=4.46.0"
  touch "$VENV_DIR/.setup-complete"
fi

# activate venv
source "$VENV_DIR/bin/activate"

# -------------------- runtime env (h100 + v1 + fa3) --------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"   # slurm usually sets this; keep explicit
export VLLM_USE_V1=1
export VLLM_DISABLE_FLASHINFER=1                             # stay on flash-attn path
# leave backend selection to vllm v1 on hopper (fa3). uncomment to force:
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export VLLM_FLASH_ATTN_VERSION=3

# quieter logs once stable; during bring-up, keep info
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# nccl tweaks are optional; uncomment if ib issues arise
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=WARN

# -------------------- parameters (overridable via --export) --------------------
# start with 20b to verify the stack, then switch to 120b
export MODEL="${MODEL:-openai/gpt-oss-20b}"
export PORT="${PORT:-8000}"
export GMU="${GMU:-0.90}"                       # gpu memory utilization
# auto-detect gpus for tp if not provided
GPU_COUNT=$(nvidia-smi -L | wc -l | awk '{print $1}')
export TP="${TP:-$GPU_COUNT}"
# conservative first boot; raise after verifying
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

# -------------------- preflight: show versions & driver --------------------
echo "=== gpt-oss serve (native) ==="
echo "node: $(hostname) | job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
python - <<'PY'
import torch, vllm, transformers
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpus:", torch.cuda.device_count())
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
print("has gpt_oss config:", any("gpt_oss" in k for k in CONFIG_MAPPING_NAMES.keys()))
PY

# graceful shutdown on preemption
trap 'echo "[serve] received usr1, exiting gracefully at $(date)"; exit 0' USR1

# -------------------- serve --------------------
echo "model=$MODEL tp=$TP gmu=$GMU max_len=$MAX_MODEL_LEN port=$PORT"
vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GMU" \
  --max-model-len "$MAX_MODEL_LEN" \
  --async-scheduling \
  --host 0.0.0.0 --port "$PORT"
