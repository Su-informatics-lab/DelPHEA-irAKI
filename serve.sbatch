#!/bin/bash
#SBATCH --job-name=gptoss-vllm
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

export HF_HOME=$HOME/hf-cache
export IMG_DIR=$HOME/containers

apptainer exec --nv \
  --bind "${HOME}/hf-cache":/root/.cache/huggingface \
  "${HOME}/containers/vllm-gptoss.sif" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port 8000