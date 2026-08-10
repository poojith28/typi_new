#!/bin/bash

#SBATCH --job-name=bud_prob100
#SBATCH --output=bud_probcover100.out
#SBATCH --error=bud_probcover100.err
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
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET50.yaml --al=probcover --exp-name=probcover_1_100b --budget=100 --initial_size=100 --initial_delta=0.25 --seed 1
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET50.yaml --al=probcover --exp-name=probcover_2_100b --budget=100 --initial_size=100 --initial_delta=0.25 --seed 2
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET50.yaml --al=probcover --exp-name=probcover_3_100b --budget=100 --initial_size=100 --initial_delta=0.25 --seed 3
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET50.yaml --al=probcover --exp-name=probcover_4_100b --budget=100 --initial_size=100 --initial_delta=0.25 --seed 4
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET50.yaml --al=probcover --exp-name=probcover_5_100b --budget=100 --initial_size=100 --initial_delta=0.25 --seed 5
