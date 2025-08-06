#!/bin/sh
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

module restore py312
source .venv-py312/bin/activate

python -m vllm.entrypoints.openai.api_server \
       --model openai/gpt-oss-120b \
       --host 0.0.0.0 \
       --port 8000 \
       --tensor-parallel-size 2 \
       --gpu-memory-utilization 0.90 \
       --trust-remote-code
