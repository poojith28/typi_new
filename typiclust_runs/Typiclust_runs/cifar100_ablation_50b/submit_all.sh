#!/bin/bash

set -euo pipefail

cd /vast/s219110279/typiclust_runs/Typiclust_runs/cifar100_ablation_50b

for f in LIDCOVER_alpha0.sh LIDCOVER_alpha05.sh LIDCOVER_alpha1.sh LIDCOVER_alpha2.sh LIDCOVER_lowID.sh LIDCOVER_k20.sh LIDCOVER_k75.sh LIDCOVER_d020.sh LIDCOVER_d030.sh LIDCOVER_tie_random.sh; do
    sbatch "$f"
done
