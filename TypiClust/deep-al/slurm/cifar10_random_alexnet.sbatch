#!/bin/bash
#SBATCH --job-name=c10_random_alex
#SBATCH --output=c10_random_alexnet.out
#SBATCH --error=c10_random_alexnet.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --mail-type=END
#SBATCH --mail-user=s219110279@deakin.edu.au

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$TOOLS"
nvidia-smi -L
nvidia-smi -q -d ECC
for seed in 1 2 3 4 5; do
  run_train_al alexnet random "$seed" "random_${seed}_50b"
done
