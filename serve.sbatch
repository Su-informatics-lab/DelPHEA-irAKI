#!/bin/bash
#SBATCH --job-name=gptoss_120b
#SBATCH --partition=nextgen-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --signal=USR1@120

set -euo pipefail

#############################################
# Environment Setup
#############################################

echo "=========================================="
echo "DelPHEA-irAKI GPT-OSS-120B Deployment"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES:-0,1}"
echo "=========================================="

# create necessary directories
mkdir -p logs
mkdir -p ${SCRATCH}/gptoss_cache/models
mkdir -p ${SCRATCH}/gptoss_cache/hf_cache
mkdir -p ${SCRATCH}/gptoss_logs

# load required modules
module purge
module load apptainer
module load cuda/12.8

# set environment variables
export HF_HOME=${SCRATCH}/gptoss_cache/hf_cache
export TRANSFORMERS_CACHE=${SCRATCH}/gptoss_cache/hf_cache
export HF_HUB_CACHE=${SCRATCH}/gptoss_cache/hf_cache
export CUDA_VISIBLE_DEVICES=0,1
export OPENAI_API_KEY="delphea-iraki-${SLURM_JOB_ID}"

# vLLM optimizations for GPT-OSS
export VLLM_USE_TRTLLM_ATTENTION=1
export VLLM_USE_TRTLLM_DECODE_ATTENTION=1
export VLLM_USE_TRTLLM_CONTEXT_ATTENTION=1
export VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1  # bf16 for accuracy

# log GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=========================================="

#############################################
# Signal Handling for Graceful Shutdown
#############################################

# function to handle shutdown signal
cleanup() {
    echo "$(date): Received shutdown signal, saving state..."

    # send SIGTERM to vLLM process
    if [ ! -z "${VLLM_PID:-}" ]; then
        kill -TERM ${VLLM_PID} 2>/dev/null || true

        # wait up to 60 seconds for graceful shutdown
        for i in {1..60}; do
            if ! kill -0 ${VLLM_PID} 2>/dev/null; then
                echo "$(date): vLLM server shut down gracefully"
                break
            fi
            sleep 1
        done

        # force kill if still running
        kill -9 ${VLLM_PID} 2>/dev/null || true
    fi

    # save logs
    cp logs/${SLURM_JOB_ID}.out ${SCRATCH}/gptoss_logs/job_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).out || true

    echo "$(date): Cleanup complete"
    exit 0
}

# trap signals for graceful shutdown
trap cleanup TERM USR1 INT

#############################################
# Health Check Function
#############################################

health_check() {
    local max_retries=10
    local retry_count=0

    echo "$(date): Waiting for vLLM server to be ready..."

    while [ $retry_count -lt $max_retries ]; do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health | grep -q "200"; then
            echo "$(date): vLLM server is healthy and ready!"

            # test the model endpoint
            curl -s http://localhost:8000/v1/models | python3 -m json.tool
            return 0
        fi

        retry_count=$((retry_count + 1))
        echo "$(date): Health check attempt $retry_count/$max_retries..."
        sleep 30
    done

    echo "$(date): ERROR - vLLM server failed to start after $max_retries attempts"
    return 1
}

#############################################
# Main vLLM Server Launch
#############################################

echo "$(date): Starting vLLM server with GPT-OSS-120B..."
echo "API Key for this session: ${OPENAI_API_KEY}"
echo "=========================================="

# launch vLLM in the container with optimal settings for 2x H100
apptainer run --nv \
    --bind ${SCRATCH}/gptoss_cache/models:/workspace/models \
    --bind ${SCRATCH}/gptoss_cache/hf_cache:/workspace/hf_cache \
    --bind ${SCRATCH}/gptoss_logs:/workspace/logs \
    --env HF_HOME=/workspace/hf_cache \
    --env TRANSFORMERS_CACHE=/workspace/hf_cache \
    --env CUDA_VISIBLE_DEVICES=0,1 \
    --env OPENAI_API_KEY=${OPENAI_API_KEY} \
    --env VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1 \
    gpt-oss-120b.sif \
    vllm serve openai/gpt-oss-120b \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 2 \
        --async-scheduling \
        --max-model-len 65536 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 64 \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --swap-space 16 \
        --download-dir /workspace/models \
        --served-model-name "gpt-oss-120b-delphea" \
        --chat-template /workspace/harmony_template.jinja2 \
        --trust-remote-code \
        --disable-log-stats \
        --log-file /workspace/logs/vllm_${SLURM_JOB_ID}.log \
        2>&1 | tee -a ${SCRATCH}/gptoss_logs/vllm_${SLURM_JOB_ID}_live.log &

# capture the PID for signal handling
VLLM_PID=$!
echo "$(date): vLLM server started with PID: ${VLLM_PID}"

# wait for server to be ready
sleep 60
health_check

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "GPT-OSS-120B Server Successfully Started!"
    echo "=========================================="
    echo "Access the server at:"
    echo "  Internal: http://$(hostname):8000"
    echo "  API Endpoint: http://$(hostname):8000/v1"
    echo ""
    echo "Test with:"
    echo "  curl http://$(hostname):8000/v1/models"
    echo ""
    echo "Python client example:"
    echo "  from openai import OpenAI"
    echo "  client = OpenAI("
    echo "      base_url='http://$(hostname):8000/v1',"
    echo "      api_key='${OPENAI_API_KEY}'"
    echo "  )"
    echo "=========================================="

    # create a test script for users
    cat > test_gptoss_${SLURM_JOB_ID}.py << EOF
#!/usr/bin/env python3
"""Test script for GPT-OSS-120B deployment - Job ${SLURM_JOB_ID}"""

from openai import OpenAI
import time

client = OpenAI(
    base_url="http://$(hostname):8000/v1",
    api_key="${OPENAI_API_KEY}"
)

# test irAKI assessment
print("Testing GPT-OSS-120B for irAKI assessment...")
start = time.time()

response = client.chat.completions.create(
    model="gpt-oss-120b-delphea",
    messages=[
        {"role": "system", "content": "You are an expert nephrologist."},
        {"role": "user", "content": "What are the key diagnostic criteria for immune-related AKI in patients receiving checkpoint inhibitors?"}
    ],
    temperature=0.1,
    max_tokens=500
)

elapsed = time.time() - start
print(f"Response received in {elapsed:.2f} seconds")
print("-" * 40)
print(response.choices[0].message.content)
print("-" * 40)
print(f"Tokens: {response.usage.total_tokens}")
EOF

    chmod +x test_gptoss_${SLURM_JOB_ID}.py
    echo "Test script created: test_gptoss_${SLURM_JOB_ID}.py"

    # monitor server health periodically
    while true; do
        # check if vLLM process is still running
        if ! kill -0 ${VLLM_PID} 2>/dev/null; then
            echo "$(date): ERROR - vLLM process terminated unexpectedly"
            exit 1
        fi

        # periodic health check every 5 minutes
        if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health | grep -q "200"; then
            echo "$(date): WARNING - Health check failed"
        fi

        # log GPU utilization every 5 minutes
        echo "$(date): GPU Status:"
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu --format=csv

        # wait for the main process or handle signals
        sleep 300 &
        wait $!
    done

else
    echo "$(date): ERROR - Server failed to start properly"
    cleanup
    exit 1
fi

# wait for the vLLM process
wait ${VLLM_PID}
VLLM_EXIT_CODE=$?

echo "$(date): vLLM server exited with code: ${VLLM_EXIT_CODE}"
echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: ${SECONDS} seconds"
echo "=========================================="

# cleanup on exit
cleanup