#!/bin/bash

#SBATCH --job-name=alex_c100_varcover
#SBATCH --output=alex_c100_distance_variance_cover.out
#SBATCH --error=alex_c100_distance_variance_cover.err
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
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/ALEXNET.yaml --al=distance_variance_cover --exp-name=distance_variance_cover_1_50b --budget=50 --initial_size=0 --initial_delta=0.25 --arc_alpha=1.0 --arc_k_signal=50 --arc_k_knn=50 --arc_cache_root=/scratch/s219110279/idpc_cache/alexnet/adaptive_cover --seed 1
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/ALEXNET.yaml --al=distance_variance_cover --exp-name=distance_variance_cover_2_50b --budget=50 --initial_size=0 --initial_delta=0.25 --arc_alpha=1.0 --arc_k_signal=50 --arc_k_knn=50 --arc_cache_root=/scratch/s219110279/idpc_cache/alexnet/adaptive_cover --seed 2
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/ALEXNET.yaml --al=distance_variance_cover --exp-name=distance_variance_cover_3_50b --budget=50 --initial_size=0 --initial_delta=0.25 --arc_alpha=1.0 --arc_k_signal=50 --arc_k_knn=50 --arc_cache_root=/scratch/s219110279/idpc_cache/alexnet/adaptive_cover --seed 3
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/ALEXNET.yaml --al=distance_variance_cover --exp-name=distance_variance_cover_4_50b --budget=50 --initial_size=0 --initial_delta=0.25 --arc_alpha=1.0 --arc_k_signal=50 --arc_k_knn=50 --arc_cache_root=/scratch/s219110279/idpc_cache/alexnet/adaptive_cover --seed 4
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/ALEXNET.yaml --al=distance_variance_cover --exp-name=distance_variance_cover_5_50b --budget=50 --initial_size=0 --initial_delta=0.25 --arc_alpha=1.0 --arc_k_signal=50 --arc_k_knn=50 --arc_cache_root=/scratch/s219110279/idpc_cache/alexnet/adaptive_cover --seed 5
