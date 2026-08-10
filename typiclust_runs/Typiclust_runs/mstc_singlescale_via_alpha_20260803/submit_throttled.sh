#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
jid=$(sbatch --parsable submit_throttled_array.sbatch)
echo "$jid" | tee throttled_array_jobid.txt
echo "Submitted array $jid with throttle 20/160 (next starts when one completes OR fails)"
squeue -u "$USER" | head -20
