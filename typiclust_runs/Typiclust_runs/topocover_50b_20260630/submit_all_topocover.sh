#!/bin/bash
set -euo pipefail
cd /vast/s219110279/typiclust_runs/Typiclust_runs/topocover_50b_20260630
for f in *.sbatch; do
  echo "Submitting $f"
  sbatch "$f"
done
