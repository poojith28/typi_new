#!/bin/bash

#SBATCH --job-name=alex_bud_coreset500
#SBATCH --output=alex_bud_coreset500.out
#SBATCH --error=alex_bud_coreset500.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=1-05:00:00
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
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/ALEXNET.yaml --al=coreset --exp-name=coreset_1_500b --budget=500 --initial_size=500 --seed 1
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/ALEXNET.yaml --al=coreset --exp-name=coreset_2_500b --budget=500 --initial_size=500 --seed 2
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/ALEXNET.yaml --al=coreset --exp-name=coreset_3_500b --budget=500 --initial_size=500 --seed 3
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/ALEXNET.yaml --al=coreset --exp-name=coreset_4_500b --budget=500 --initial_size=500 --seed 4
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/ALEXNET.yaml --al=coreset --exp-name=coreset_5_500b --budget=500 --initial_size=500 --seed 5
