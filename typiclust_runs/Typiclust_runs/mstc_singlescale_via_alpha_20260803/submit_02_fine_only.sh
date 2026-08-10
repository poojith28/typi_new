#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
: > submitted_02_fine_only.txt

jid=$(sbatch --parsable "fine_only_c10_r18_c070_f050_a000_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c10_r18_c070_f050_a000_s1_b050.sbatch" "MSTC_C070_F050_A000_1_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c10_r18_c070_f050_a000_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c10_r18_c070_f050_a000_s2_b050.sbatch" "MSTC_C070_F050_A000_2_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c10_r18_c070_f050_a000_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c10_r18_c070_f050_a000_s3_b050.sbatch" "MSTC_C070_F050_A000_3_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c10_r18_c070_f050_a000_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c10_r18_c070_f050_a000_s4_b050.sbatch" "MSTC_C070_F050_A000_4_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c10_r18_c070_f050_a000_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c10_r18_c070_f050_a000_s5_b050.sbatch" "MSTC_C070_F050_A000_5_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c100_r18_c070_f050_a000_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c100_r18_c070_f050_a000_s1_b050.sbatch" "MSTC_C070_F050_A000_1_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c100_r18_c070_f050_a000_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c100_r18_c070_f050_a000_s2_b050.sbatch" "MSTC_C070_F050_A000_2_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c100_r18_c070_f050_a000_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c100_r18_c070_f050_a000_s3_b050.sbatch" "MSTC_C070_F050_A000_3_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c100_r18_c070_f050_a000_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c100_r18_c070_f050_a000_s4_b050.sbatch" "MSTC_C070_F050_A000_4_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c100_r18_c070_f050_a000_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c100_r18_c070_f050_a000_s5_b050.sbatch" "MSTC_C070_F050_A000_5_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_tin_r18_c070_f050_a000_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_tin_r18_c070_f050_a000_s1_b050.sbatch" "MSTC_C070_F050_A000_1_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_tin_r18_c070_f050_a000_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_tin_r18_c070_f050_a000_s2_b050.sbatch" "MSTC_C070_F050_A000_2_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_tin_r18_c070_f050_a000_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_tin_r18_c070_f050_a000_s3_b050.sbatch" "MSTC_C070_F050_A000_3_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_tin_r18_c070_f050_a000_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_tin_r18_c070_f050_a000_s4_b050.sbatch" "MSTC_C070_F050_A000_4_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_tin_r18_c070_f050_a000_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_tin_r18_c070_f050_a000_s5_b050.sbatch" "MSTC_C070_F050_A000_5_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c100_r50_c070_f050_a000_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c100_r50_c070_f050_a000_s1_b050.sbatch" "MSTC_C070_F050_A000_1_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c100_r50_c070_f050_a000_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c100_r50_c070_f050_a000_s2_b050.sbatch" "MSTC_C070_F050_A000_2_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c100_r50_c070_f050_a000_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c100_r50_c070_f050_a000_s3_b050.sbatch" "MSTC_C070_F050_A000_3_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c100_r50_c070_f050_a000_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c100_r50_c070_f050_a000_s4_b050.sbatch" "MSTC_C070_F050_A000_4_50b" | tee -a submitted_02_fine_only.txt
jid=$(sbatch --parsable "fine_only_c100_r50_c070_f050_a000_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "fine_only_c100_r50_c070_f050_a000_s5_b050.sbatch" "MSTC_C070_F050_A000_5_50b" | tee -a submitted_02_fine_only.txt

echo "submitted $(wc -l < submitted_02_fine_only.txt) jobs from 02_fine_only"
