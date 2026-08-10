#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
: > submitted_01_coarse_only.txt

jid=$(sbatch --parsable "coarse_only_c10_r18_c070_f050_a100_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c10_r18_c070_f050_a100_s1_b050.sbatch" "MSTC_C070_F050_A100_1_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c10_r18_c070_f050_a100_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c10_r18_c070_f050_a100_s2_b050.sbatch" "MSTC_C070_F050_A100_2_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c10_r18_c070_f050_a100_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c10_r18_c070_f050_a100_s3_b050.sbatch" "MSTC_C070_F050_A100_3_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c10_r18_c070_f050_a100_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c10_r18_c070_f050_a100_s4_b050.sbatch" "MSTC_C070_F050_A100_4_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c10_r18_c070_f050_a100_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c10_r18_c070_f050_a100_s5_b050.sbatch" "MSTC_C070_F050_A100_5_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c100_r18_c070_f050_a100_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c100_r18_c070_f050_a100_s1_b050.sbatch" "MSTC_C070_F050_A100_1_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c100_r18_c070_f050_a100_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c100_r18_c070_f050_a100_s2_b050.sbatch" "MSTC_C070_F050_A100_2_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c100_r18_c070_f050_a100_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c100_r18_c070_f050_a100_s3_b050.sbatch" "MSTC_C070_F050_A100_3_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c100_r18_c070_f050_a100_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c100_r18_c070_f050_a100_s4_b050.sbatch" "MSTC_C070_F050_A100_4_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c100_r18_c070_f050_a100_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c100_r18_c070_f050_a100_s5_b050.sbatch" "MSTC_C070_F050_A100_5_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_tin_r18_c070_f050_a100_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_tin_r18_c070_f050_a100_s1_b050.sbatch" "MSTC_C070_F050_A100_1_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_tin_r18_c070_f050_a100_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_tin_r18_c070_f050_a100_s2_b050.sbatch" "MSTC_C070_F050_A100_2_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_tin_r18_c070_f050_a100_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_tin_r18_c070_f050_a100_s3_b050.sbatch" "MSTC_C070_F050_A100_3_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_tin_r18_c070_f050_a100_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_tin_r18_c070_f050_a100_s4_b050.sbatch" "MSTC_C070_F050_A100_4_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_tin_r18_c070_f050_a100_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_tin_r18_c070_f050_a100_s5_b050.sbatch" "MSTC_C070_F050_A100_5_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c100_r50_c070_f050_a100_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c100_r50_c070_f050_a100_s1_b050.sbatch" "MSTC_C070_F050_A100_1_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c100_r50_c070_f050_a100_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c100_r50_c070_f050_a100_s2_b050.sbatch" "MSTC_C070_F050_A100_2_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c100_r50_c070_f050_a100_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c100_r50_c070_f050_a100_s3_b050.sbatch" "MSTC_C070_F050_A100_3_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c100_r50_c070_f050_a100_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c100_r50_c070_f050_a100_s4_b050.sbatch" "MSTC_C070_F050_A100_4_50b" | tee -a submitted_01_coarse_only.txt
jid=$(sbatch --parsable "coarse_only_c100_r50_c070_f050_a100_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "coarse_only_c100_r50_c070_f050_a100_s5_b050.sbatch" "MSTC_C070_F050_A100_5_50b" | tee -a submitted_01_coarse_only.txt

echo "submitted $(wc -l < submitted_01_coarse_only.txt) jobs from 01_coarse_only"
