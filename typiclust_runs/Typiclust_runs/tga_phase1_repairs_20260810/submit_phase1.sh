#!/bin/bash
set -euo pipefail

campaign_root=/vast/s219110279/typiclust_runs/Typiclust_runs/tga_phase1_repairs_20260810
jobs_file=${campaign_root}/jobs.tsv
array_file=${campaign_root}/phase1_array.sbatch
job_id_file=${campaign_root}/submitted_job_id.txt
attempt_log=${campaign_root}/submission_attempts.log
lock_file=${campaign_root}/submission.lock

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "ERROR: another Phase-1 submission process holds ${lock_file}" >&2
  exit 73
fi

timestamp=$(date --iso-8601=seconds)
echo "[${timestamp}] Phase-1 submission preflight" | tee -a "${attempt_log}"

if [[ -s "${job_id_file}" ]]; then
  echo "ERROR: submission already recorded: $(cat "${job_id_file}")" | tee -a "${attempt_log}" >&2
  exit 74
fi

echo "complete required concrete matrix: 165"
echo "already COMPLETE: 146"
echo "missing/invalid manifest rows: 19 (PARTIAL=4, MISSING=15)"
echo "phase-1 experiments to launch:"
tail -n +2 "${jobs_file}" | cut -f2-8
echo "total phase-1 jobs: 4"
echo "array concurrency: --array=0-3%20"
echo "collective MAX_CONCURRENT_RUNS=20"

while IFS=$'\t' read -r task_id experiment_id dataset backbone seed alpha delta_coarse delta_fine active_control preserved_source; do
  [[ "${task_id}" == "task_id" ]] && continue
  target=/vast/s219110279/TypiClust/output/${dataset}/${backbone}/${experiment_id}
  if [[ -e "${target}" ]]; then
    echo "ERROR: target already exists; re-audit required: ${target}" | tee -a "${attempt_log}" >&2
    exit 75
  fi
done < "${jobs_file}"

queue_snapshot=$(mktemp)
trap 'rm -f "${queue_snapshot}"' EXIT
if ! timeout 30s squeue -h -u s219110279 -o '%i|%T|%j|%o' >"${queue_snapshot}"; then
  echo "BLOCKED: Slurm queue state is unverifiable; nothing submitted." | tee -a "${attempt_log}" >&2
  exit 69
fi

if grep -Eiq 'tga_p1_repair|phase1_array\.sbatch|AUDIT_20260810_MSTC_(COARSE|FINE)' "${queue_snapshot}"; then
  echo "BLOCKED: a matching Phase-1 job is already RUNNING/PENDING; nothing submitted." | tee -a "${attempt_log}" >&2
  cat "${queue_snapshot}"
  exit 76
fi

submission=$(sbatch --parsable "${array_file}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: unexpected sbatch response: ${submission}" | tee -a "${attempt_log}" >&2
  exit 77
fi

printf '%s\n' "${job_id}" > "${job_id_file}"
echo "[${timestamp}] submitted Phase-1 array job ${job_id}; tasks 0-3; cap 20" | tee -a "${attempt_log}"
echo "SUBMITTED_JOB_ID=${job_id}"
