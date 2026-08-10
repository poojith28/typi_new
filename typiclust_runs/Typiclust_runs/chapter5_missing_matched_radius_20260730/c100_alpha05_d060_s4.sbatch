#!/bin/bash
#SBATCH --job-name=c100_alpha05_d060_s4
#SBATCH --output=/vast/s219110279/typiclust_runs/Typiclust_runs/chapter5_missing_matched_radius_20260730/c100_alpha05_d060_s4_%j.out
#SBATCH --error=/vast/s219110279/typiclust_runs/Typiclust_runs/chapter5_missing_matched_radius_20260730/c100_alpha05_d060_s4_%j.err
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
python train_al.py   --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET18.yaml   --al=idprobcover   --exp-name=LIDCOVER_alpha05_d060_ch5_20260730_4_50b   --budget=50   --initial_size=50   --initial_delta=0.6   --idpc_alpha=0.5 --idpc_mode=high_id_more_centers --idpc_k_id=50 --idpc_k_knn=50 --idpc_cache_root=/scratch/s219110279/idpc_cache   --seed=4
