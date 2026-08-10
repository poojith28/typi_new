#!/bin/bash

#SBATCH --job-name=c100_probcover
#SBATCH --output=c100_probcover.out
#SBATCH --error=c100_probcover.err
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
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/RESNET50.yaml --al=probcover --exp-name=probcover_1_50b --budget=50 --initial_size=0 --seed 1
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/RESNET50.yaml --al=probcover --exp-name=probcover_2_50b --budget=50 --initial_size=0 --seed 2
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/RESNET50.yaml --al=probcover --exp-name=probcover_3_50b --budget=50 --initial_size=0 --seed 3
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/RESNET50.yaml --al=probcover --exp-name=probcover_4_50b --budget=50 --initial_size=0 --seed 4
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/RESNET50.yaml --al=probcover --exp-name=probcover_5_50b --budget=50 --initial_size=0 --seed 5
