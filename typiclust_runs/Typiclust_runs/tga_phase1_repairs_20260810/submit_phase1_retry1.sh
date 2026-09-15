#!/bin/bash
set -euo pipefail

campaign_root=/vast/s219110279/typiclust_runs/Typiclust_runs/tga_phase1_repairs_20260810
jobs_file=${campaign_root}/jobs.tsv
array_file=${campaign_root}/phase1_array.sbatch
original_job_id_file=${campaign_root}/submitted_job_id.txt
retry_job_id_file=${campaign_root}/retry1_job_id.txt
attempt_log=${campaign_root}/submission_attempts.log
lock_file=${campaign_root}/submission.lock

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "ERROR: another Phase-1 submission process holds ${lock_file}" >&2
  exit 73
fi

if [[ "$(cat "${original_job_id_file}" 2>/dev/null || true)" != "3473736" ]]; then
  echo "ERROR: expected failed original job 3473736 is not recorded" >&2
  exit 74
fi
if [[ -s "${retry_job_id_file}" ]]; then
  echo "ERROR: the single allowed retry is already recorded: $(cat "${retry_job_id_file}")" >&2
  exit 74
fi

for task_id in 0 1 2 3; do
  err=/vast/s219110279/TypiClust/output/_logs/tga_p1_repair_3473736_${task_id}.err
  if [[ ! -f "${err}" ]] || ! grep -q 'MKL_INTERFACE_LAYER: unbound variable' "${err}"; then
    echo "ERROR: task ${task_id} does not have the verified transient Conda failure" >&2
    exit 75
  fi
done

while IFS=$'\t' read -r task_id experiment_id dataset backbone seed alpha delta_coarse delta_fine active_control preserved_source; do
  [[ "${task_id}" == "task_id" ]] && continue
  target=/vast/s219110279/TypiClust/output/${dataset}/${backbone}/${experiment_id}
  if [[ -e "${target}" ]]; then
    echo "ERROR: target exists after failed attempt; re-audit instead of retrying: ${target}" >&2
    exit 76
  fi
done < "${jobs_file}"

echo "verified failure classification: transient Conda activation wrapper error"
echo "scientific implementation/config changes: none"
echo "retry count: 1 of 1"
echo "experiments: 4"
echo "array concurrency: --array=0-3%20"

queue_snapshot=$(mktemp)
trap 'rm -f "${queue_snapshot}"' EXIT
if ! timeout 30s squeue -h -u s219110279 -o '%i|%T|%j|%o' >"${queue_snapshot}"; then
  echo "BLOCKED: Slurm queue state is unverifiable; retry not submitted." | tee -a "${attempt_log}" >&2
  exit 69
fi
if grep -Eiq 'tga_p1_repair|phase1_array\.sbatch|AUDIT_20260810_MSTC_(COARSE|FINE)' "${queue_snapshot}"; then
  echo "BLOCKED: matching Phase-1 work is already RUNNING/PENDING; retry not submitted." | tee -a "${attempt_log}" >&2
  cat "${queue_snapshot}"
  exit 77
fi

submission=$(sbatch --parsable "${array_file}")
job_id=${submission%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: unexpected sbatch response: ${submission}" >&2
  exit 78
fi
printf '%s\n' "${job_id}" > "${retry_job_id_file}"
echo "[$(date --iso-8601=seconds)] submitted single Phase-1 retry array ${job_id} for failed job 3473736" | tee -a "${attempt_log}"
echo "RETRY_JOB_ID=${job_id}"
