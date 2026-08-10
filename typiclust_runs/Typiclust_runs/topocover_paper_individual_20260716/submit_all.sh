#!/bin/bash
set -euo pipefail
cd /vast/s219110279/typiclust_runs/Typiclust_runs/topocover_paper_individual_20260716
tail -n +2 manifest.tsv | while IFS=$'\t' read -r kind dataset backbone seed delta budget experiment script; do
  sbatch --parsable "${script}"
done

