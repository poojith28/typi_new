#!/bin/bash

#SBATCH --job-name=c100_dbal
#SBATCH --output=c100_dbal.out
#SBATCH --error=c100_dbal.err
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
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET50.yaml --al=dbal --exp-name=dbal_1_50b --budget=50 --initial_size=50 --seed 1
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET50.yaml --al=dbal --exp-name=dbal_2_50b --budget=50 --initial_size=50 --seed 2
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET50.yaml --al=dbal --exp-name=dbal_3_50b --budget=50 --initial_size=50 --seed 3
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET50.yaml --al=dbal --exp-name=dbal_4_50b --budget=50 --initial_size=50 --seed 4
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET50.yaml --al=dbal --exp-name=dbal_5_50b --budget=50 --initial_size=50 --seed 5
