#!/bin/bash
set -eo pipefail
cd /vast/s219110279/typiclust_runs/Typiclust_runs/mstc_pilot_20260727

echo "Submitting cifar10_mstc_resnet18_c070_f050_50b.sbatch"
jid=$(sbatch --parsable cifar10_mstc_resnet18_c070_f050_50b.sbatch)
echo "$jid cifar10_mstc_resnet18_c070_f050_50b.sbatch" | tee -a submitted_job_ids.txt
echo "Submitting cifar100_mstc_resnet18_c070_f050_50b.sbatch"
jid=$(sbatch --parsable cifar100_mstc_resnet18_c070_f050_50b.sbatch)
echo "$jid cifar100_mstc_resnet18_c070_f050_50b.sbatch" | tee -a submitted_job_ids.txt
