#!/bin/bash

#SBATCH --job-name=alex_c100_LIDCOVER
#SBATCH --output=alex_c100_LIDCOVER.out
#SBATCH --error=alex_c100_LIDCOVER.err
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
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar10/al/ALEXNET.yaml --al=idprobcover --exp-name=LIDCOVER_1_50b --budget=50 --initial_size=0 --initial_delta=0.25 --idpc_alpha=1.0 --idpc_mode=high_id_more_centers --idpc_k_id=50 --idpc_k_knn=50 --idpc_cache_root=/scratch/s219110279/idpc_cache/alexnet --seed 1
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar10/al/ALEXNET.yaml --al=idprobcover --exp-name=LIDCOVER_2_50b --budget=50 --initial_size=0 --initial_delta=0.25 --idpc_alpha=1.0 --idpc_mode=high_id_more_centers --idpc_k_id=50 --idpc_k_knn=50 --idpc_cache_root=/scratch/s219110279/idpc_cache/alexnet --seed 2
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar10/al/ALEXNET.yaml --al=idprobcover --exp-name=LIDCOVER_3_50b --budget=50 --initial_size=0 --initial_delta=0.25 --idpc_alpha=1.0 --idpc_mode=high_id_more_centers --idpc_k_id=50 --idpc_k_knn=50 --idpc_cache_root=/scratch/s219110279/idpc_cache/alexnet --seed 3
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar10/al/ALEXNET.yaml --al=idprobcover --exp-name=LIDCOVER_4_50b --budget=50 --initial_size=0 --initial_delta=0.25 --idpc_alpha=1.0 --idpc_mode=high_id_more_centers --idpc_k_id=50 --idpc_k_knn=50 --idpc_cache_root=/scratch/s219110279/idpc_cache/alexnet --seed 4
python train_al.py \
    --cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar10/al/ALEXNET.yaml --al=idprobcover --exp-name=LIDCOVER_5_50b --budget=50 --initial_size=0 --initial_delta=0.25 --idpc_alpha=1.0 --idpc_mode=high_id_more_centers --idpc_k_id=50 --idpc_k_knn=50 --idpc_cache_root=/scratch/s219110279/idpc_cache/alexnet --seed 5
