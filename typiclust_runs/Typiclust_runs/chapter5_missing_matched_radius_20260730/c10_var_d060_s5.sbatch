#!/bin/bash
#SBATCH --job-name=c10_var_d060_s5
#SBATCH --output=/vast/s219110279/typiclust_runs/Typiclust_runs/chapter5_missing_matched_radius_20260730/c10_var_d060_s5_%j.out
#SBATCH --error=/vast/s219110279/typiclust_runs/Typiclust_runs/chapter5_missing_matched_radius_20260730/c10_var_d060_s5_%j.err
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
python train_al.py   --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar10/al/RESNET18.yaml   --al=distance_variance_cover   --exp-name=cifar10_distance_variance_cover_d060_ch5_20260730_5_50b   --budget=50   --initial_size=50   --initial_delta=0.6   --arc_alpha=1.0 --arc_k_signal=50 --arc_k_knn=50 --arc_cache_root=/scratch/s219110279/idpc_cache/adaptive_cover   --seed=5
