#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
: > submitted_job_ids.txt
tail -n +2 manifest.tsv | while IFS=$'\t' read -r kind dataset backbone seed dc df alpha budget experiment script; do
  jid=$(sbatch --parsable "${script}")
  echo "${jid}\t${script}\t${experiment}" | tee -a submitted_job_ids.txt
done
echo "submitted $(wc -l < submitted_job_ids.txt) jobs"
