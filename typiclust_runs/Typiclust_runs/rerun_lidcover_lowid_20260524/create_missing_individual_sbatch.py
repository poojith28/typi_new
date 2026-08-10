#!/usr/bin/env python3
"""Create one SLURM job per missing corrected LIDCover seed."""
from pathlib import Path

OUT = Path("/vast/s219110279/typiclust_runs/Typiclust_runs/rerun_lidcover_lowid_20260524")

JOBS = []
for seed in range(1, 6):
    JOBS.append({
        "tag": f"tiny_lidcorrect_s{seed}",
        "cfg": "/scratch/s219110279/TypiClust/deep-al/configs/tinyimagenet/al/RESNET18.yaml",
        "exp": f"LIDCOVER_rerunlowid20260524_{seed}_50b",
        "mode": "high_id_more_centers",
        "seed": seed,
    })
for seed in range(1, 6):
    JOBS.append({
        "tag": f"c100_lowidcorrect_s{seed}",
        "cfg": "/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET18.yaml",
        "exp": f"LIDCOVER_lowID_rerunlowid20260524_{seed}_50b",
        "mode": "low_id_more_centers",
        "seed": seed,
    })

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={tag}
#SBATCH --output={out}/{tag}_%j.out
#SBATCH --error={out}/{tag}_%j.err
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
conda activate AutoAL

cd /scratch/s219110279/TypiClust/deep-al/tools

nvidia-smi -L
nvidia-smi -q -d ECC

python train_al.py \\
    --cfg={cfg} \\
    --al=idprobcover \\
    --exp-name={exp} \\
    --budget=50 \\
    --initial_size=50 \\
    --initial_delta=0.25 \\
    --idpc_alpha=1.0 \\
    --idpc_mode={mode} \\
    --idpc_k_id=50 \\
    --idpc_k_knn=50 \\
    --idpc_cache_root=/scratch/s219110279/idpc_cache \\
    --seed={seed}
"""

OUT.mkdir(parents=True, exist_ok=True)
for job in JOBS:
    path = OUT / f"{job['tag']}.sbatch"
    path.write_text(TEMPLATE.format(out=OUT, **job))
    print(path)
