#!/bin/bash

set -euo pipefail

cd /vast/s219110279/typiclust_runs/Typiclust_runs/cifar100_budget_ablation_alexnet

for f in random_100b.sh entropy_100b.sh coreset_100b.sh probcover_100b.sh LIDCOVER_100b.sh random_500b.sh entropy_500b.sh coreset_500b.sh probcover_500b.sh LIDCOVER_500b.sh; do
    sbatch "$f"
done
