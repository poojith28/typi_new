#!/bin/bash
set -euo pipefail

cd /vast/s219110279/typiclust_runs/Typiclust_runs/topocover_d070_50b_20260702

for f in *.sbatch; do
  echo "Submitting $f"
  sbatch "$f"
done
