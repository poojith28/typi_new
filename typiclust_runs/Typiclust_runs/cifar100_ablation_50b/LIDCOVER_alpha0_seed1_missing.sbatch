#!/bin/bash

#SBATCH --job-name=abl_a0_r50_s1
#SBATCH --output=abl_a0_r50_s1_%j.out
#SBATCH --error=abl_a0_r50_s1_%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=batch-short
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --mail-type=END
#SBATCH --mail-user=s219110279@deakin.edu.au

module purge
module load Anaconda3
source activate
conda activate typiclust

cd /scratch/s219110279/TypiClust/deep-al/tools

export TYPI_FEATURES_ROOT=/vast/s219110279/results/results

nvidia-smi -L
nvidia-smi -q -d ECC

python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET50.yaml \
    --al=idprobcover \
    --exp-name=LIDCOVER_alpha0_1_50b \
    --budget=50 \
    --initial_size=50 \
    --initial_delta=0.25 \
    --idpc_alpha=0.0 \
    --idpc_mode=high_id_more_centers \
    --idpc_k_id=50 \
    --idpc_k_knn=50 \
    --idpc_cache_root=/scratch/s219110279/idpc_cache/resnet50 \
    --seed=1
