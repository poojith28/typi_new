#!/usr/bin/env bash

# Remove the afterok dependencies from the five arrays submitted on 2026-08-19.
# Safe to rerun: setting an empty dependency list is idempotent.

set -euo pipefail

JOB_IDS=(3483350 3483351 3483352 3483353 3483354)
for job_id in "${JOB_IDS[@]}"; do
  scontrol update JobId="$job_id" Dependency=
  echo "job $job_id released"
done

squeue -j 3483350,3483351,3483352,3483353,3483354 \
  -o '%.24i %.24j %.10T %.12M %R'
