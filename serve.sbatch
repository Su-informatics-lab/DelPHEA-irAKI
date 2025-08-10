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
# comment out the next line for now so HOME survives; re-add later if you want
# #SBATCH --export=NONE

set -euo pipefail

# modules
module purge
module restore py312 || true
# if apptainer is a module on tempest, uncomment:
# module load Apptainer
module -t list || true

# make sure HOME is set even if --export=NONE is used
if [ -z "${HOME:-}" ]; then
  HOME="$(getent passwd "$USER" | cut -d: -f6 || echo "/home/$USER")"
  export HOME
fi

# config
MODEL="${MODEL:-openai/gpt-oss-120b}"
TP="${TP:-${SLURM_GPUS_ON_NODE:-2}}"
PORT="${PORT:-8000}"
HF_HOME="${HF_HOME:-$HOME/hf-cache}"
IMG="${IMG:-$HOME/containers/vllm-gptoss.sif}"

# dirs
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$SUBMIT_DIR/logs" "$HF_HOME"

# debug banner
echo "node=$(hostname)"
echo "user=$USER home=$HOME"
echo "img=$IMG"
echo "model=$MODEL tp=$TP port=$PORT"
which apptainer || { echo "apptainer not found on PATH"; exit 127; }
nvidia-smi || true

# nicer worker spawn inside apptainer
export VLLM_WORKER_MULTIPROC_METHOD=spawn
trap 'echo "[signal] USR1 received, exiting"; sleep 1; exit 0' USR1

# run server
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
