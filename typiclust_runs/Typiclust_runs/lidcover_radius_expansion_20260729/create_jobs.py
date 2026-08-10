#!/usr/bin/env python3
"""Create one SLURM job per seed for expanded radius controls."""
from pathlib import Path

OUT = Path("/vast/s219110279/typiclust_runs/Typiclust_runs/lidcover_radius_expansion_20260729")
CFG = "/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET18.yaml"

jobs = []
for delta in (0.10, 0.40, 0.50, 0.60):
    tag = f"d{int(delta * 100):03d}"
    for seed in range(1, 6):
        jobs.append({
            "job": f"lid_{tag}_s{seed}",
            "method": "idprobcover",
            "exp": f"LIDCOVER_{tag}_rerunlowid20260524_{seed}_50b",
            "delta": delta,
            "initial_size": 50,
            "seed": seed,
            "extra": (
                "--idpc_alpha=1.0 "
                "--idpc_mode=high_id_more_centers "
                "--idpc_k_id=50 --idpc_k_knn=50 "
                "--idpc_cache_root=/scratch/s219110279/idpc_cache"
            ),
        })

for seed in range(1, 6):
    jobs.append({
        "job": f"prob_d025_s{seed}",
        "method": "probcover",
        "exp": f"probcover_d025_{seed}_50b",
        "delta": 0.25,
        "initial_size": 0,
        "seed": seed,
        "extra": "",
    })

template = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --output={out}/{job}_%j.out
#SBATCH --error={out}/{job}_%j.err
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

python train_al.py \\
    --cfg={cfg} \\
    --al={method} \\
    --exp-name={exp} \\
    --budget=50 \\
    --initial_size={initial_size} \\
    --initial_delta={delta} \\
    {extra} \\
    --seed={seed}
"""

OUT.mkdir(parents=True, exist_ok=True)
for item in jobs:
    path = OUT / f"{item['job']}.sbatch"
    path.write_text(template.format(out=OUT, cfg=CFG, **item))
    print(path)
