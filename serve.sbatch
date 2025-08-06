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

module load cuda/12.3.0
module load Python/3.10.8-GCCcore-12.2.0
source .venv/bin/activate

python -m vllm.entrypoints.openai.api_server \
       --model openai/gpt-oss-120b \
       --host 0.0.0.0 \
       --port 8000 \
       --tensor-parallel-size 2 \
       --gpu-memory-utilization 0.90 \
       --trust-remote-code
