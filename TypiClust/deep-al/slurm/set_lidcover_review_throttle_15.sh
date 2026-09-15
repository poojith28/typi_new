#!/usr/bin/env bash

# Update the already-submitted ICONIP reviewer arrays. Editing #SBATCH lines
# affects only future submissions; this updates the six live array records.

set -euo pipefail

JOB_IDS=(3483301 3483302 3483303 3483304 3483305 3483306)
for job_id in "${JOB_IDS[@]}"; do
  scontrol update ArrayTaskThrottle=15 JobId="$job_id"
  echo "job $job_id -> ArrayTaskThrottle=15"
done

squeue -j 3483301,3483302,3483303,3483304,3483305,3483306 \
  -o '%.24i %.24j %.10T %.12M %R'
