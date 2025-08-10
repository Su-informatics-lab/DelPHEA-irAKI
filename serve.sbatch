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
#SBATCH --export=NONE                 # avoid inheriting your local venv/path

set -euo pipefail

# --- modules: restore your py312 collection -----------------------------------
module purge                          # clear any defaults that slurm might add
module restore py312                  # your saved module set
module list                           # debug: show what's loaded

# --- config -------------------------------------------------------------------
MODEL="${MODEL:-openai/gpt-oss-120b}" # using 120b on 2×h100
PORT="${PORT:-8000}"
TP="${TP:-2}"                         # tensor parallel size = gpus
HF_HOME="${HF_HOME:-$HOME/hf-cache}"
IMG="$HOME/containers/vllm-gptoss.sif"

mkdir -p "$HF_HOME" logs

echo "host: $(hostname)"
echo "model: $MODEL"
echo "port:  $PORT"
echo "tp:    $TP"

# optional: this can help vllm workers under apptainer
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# graceful preemption / end signal
trap 'echo "[signal] USR1 received, exiting"; sleep 2; exit 0' USR1

# --- run vllm openai server inside apptainer ----------------------------------
apptainer run --nv \
  --bind "$HF_HOME:/root/.cache/huggingface" \
  "$IMG" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --download-dir /root/.cache/huggingface \
  --gpu-memory-utilization 0.95 \
  --max-model-len 131072 \
  --api-key ""
