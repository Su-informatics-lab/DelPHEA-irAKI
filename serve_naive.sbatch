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
set -euo pipefail

module purge
module load GCCcore/13.3.0
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

# venv in job scratch to avoid conflicts
export JOB_SCRATCH="$HOME/DelPHEA-irAKI/tmp/$SLURM_JOB_ID"
mkdir -p "$JOB_SCRATCH"
python -m venv "$JOB_SCRATCH/.venv" && source "$JOB_SCRATCH/.venv/bin/activate"
python -m pip install -U pip uv

# install the gpt-oss build of vllm + torch cu128 (official recipe)
uv pip install --pre "vllm==0.10.1+gptoss" \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match

# optional: confirm torch + vllm
python - <<'PY'
import torch, vllm
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
print("vllm:", vllm.__version__)
PY

# h100 + v1 + fa3
export VLLM_USE_V1=1
export VLLM_DISABLE_FLASHINFER=1
export CUDA_VISIBLE_DEVICES=0,1

# start smaller first to verify stack; then switch to 120b
vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --async-scheduling \
  --host 0.0.0.0 --port 8000
