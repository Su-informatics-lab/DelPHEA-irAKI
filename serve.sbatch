#!/bin/bash -l
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
#SBATCH --signal=USR1@120
#SBATCH --export=NONE

set -euo pipefail

# --- modules: clean, drop the bad path, and restore py312 ---------------------
module unuse /etc/scl/modulefiles 2>/dev/null || true   # silence the spurious warning
module purge
module restore py312
# if apptainer is a module on tempest, uncomment the next line
# module load Apptainer
module -t list

# --- config -------------------------------------------------------------------
MODEL="${MODEL:-openai/gpt-oss-120b}"    # 120b on 2×h100
PORT="${PORT:-8000}"
TP="${TP:-2}"
HF_HOME="${HF_HOME:-$HOME/hf-cache}"
IMG="$HOME/containers/vllm-gptoss.sif"

mkdir -p "$HF_HOME" logs

# keep threads tame to avoid oversubscription
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

trap 'echo "[signal] USR1 received, exiting"; sleep 2; exit 0' USR1

# --- run vllm openai server inside apptainer ----------------------------------
apptainer run --nv \
  --bind "$HF_HOME:/root/.cache/hf" \
  "$IMG" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --download-dir /root/.cache/hf \
  --gpu-memory-utilization 0.95 \
  --max-model-len 131072 \
  --api-key ""
