#!/bin/bash
#SBATCH --job-name=vllm-oss120b
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -e                       # fail fast

export PYTHONNOUSERSITE=1    # never import from ~/.local/…

module purge
module restore py312         # loads GCC 13.3 + Python 3.12 + CUDA 12.3

# --- ABSOLUTE path so activation never fails -------------
source /home/g91p721/DelPHEA-irAKI/.venv-py312/bin/activate

# optional sanity check
python - <<'PY'
import vllm, sys, platform, os
print("Using vllm from:", vllm.__file__)
print("glibc:", platform.libc_ver())
PY

# --- launch ------------------------------------------------
python -m vllm.entrypoints.openai.api_server \
       --model openai/gpt-oss-120b \
       --host 0.0.0.0 \
       --port 8000 \
       --tensor-parallel-size 2 \
       --gpu-memory-utilization 0.90 \
       --trust-remote-code
