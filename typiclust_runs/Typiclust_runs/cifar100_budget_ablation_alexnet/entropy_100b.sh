#!/bin/bash

#SBATCH --job-name=alex_bud_entropy100
#SBATCH --output=alex_bud_entropy100.out
#SBATCH --error=alex_bud_entropy100.err
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
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/ALEXNET.yaml --al=entropy --exp-name=entropy_1_100b --budget=100 --initial_size=100 --seed 1
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/ALEXNET.yaml --al=entropy --exp-name=entropy_2_100b --budget=100 --initial_size=100 --seed 2
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/ALEXNET.yaml --al=entropy --exp-name=entropy_3_100b --budget=100 --initial_size=100 --seed 3
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/ALEXNET.yaml --al=entropy --exp-name=entropy_4_100b --budget=100 --initial_size=100 --seed 4
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/ALEXNET.yaml --al=entropy --exp-name=entropy_5_100b --budget=100 --initial_size=100 --seed 5
