#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
: > submitted_03_scale_coarse.txt

jid=$(sbatch --parsable "scale_coarse_c10_r18_c060_f040_a100_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c10_r18_c060_f040_a100_s1_b050.sbatch" "MSTC_C060_F040_A100_1_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c10_r18_c060_f040_a100_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c10_r18_c060_f040_a100_s2_b050.sbatch" "MSTC_C060_F040_A100_2_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c10_r18_c060_f040_a100_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c10_r18_c060_f040_a100_s3_b050.sbatch" "MSTC_C060_F040_A100_3_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c10_r18_c060_f040_a100_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c10_r18_c060_f040_a100_s4_b050.sbatch" "MSTC_C060_F040_A100_4_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c10_r18_c060_f040_a100_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c10_r18_c060_f040_a100_s5_b050.sbatch" "MSTC_C060_F040_A100_5_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c10_r18_c080_f060_a100_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c10_r18_c080_f060_a100_s1_b050.sbatch" "MSTC_C080_F060_A100_1_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c10_r18_c080_f060_a100_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c10_r18_c080_f060_a100_s2_b050.sbatch" "MSTC_C080_F060_A100_2_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c10_r18_c080_f060_a100_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c10_r18_c080_f060_a100_s3_b050.sbatch" "MSTC_C080_F060_A100_3_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c10_r18_c080_f060_a100_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c10_r18_c080_f060_a100_s4_b050.sbatch" "MSTC_C080_F060_A100_4_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c10_r18_c080_f060_a100_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c10_r18_c080_f060_a100_s5_b050.sbatch" "MSTC_C080_F060_A100_5_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c100_r18_c060_f040_a100_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c100_r18_c060_f040_a100_s1_b050.sbatch" "MSTC_C060_F040_A100_1_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c100_r18_c060_f040_a100_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c100_r18_c060_f040_a100_s2_b050.sbatch" "MSTC_C060_F040_A100_2_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c100_r18_c060_f040_a100_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c100_r18_c060_f040_a100_s3_b050.sbatch" "MSTC_C060_F040_A100_3_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c100_r18_c060_f040_a100_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c100_r18_c060_f040_a100_s4_b050.sbatch" "MSTC_C060_F040_A100_4_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c100_r18_c060_f040_a100_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c100_r18_c060_f040_a100_s5_b050.sbatch" "MSTC_C060_F040_A100_5_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c100_r18_c080_f060_a100_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c100_r18_c080_f060_a100_s1_b050.sbatch" "MSTC_C080_F060_A100_1_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c100_r18_c080_f060_a100_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c100_r18_c080_f060_a100_s2_b050.sbatch" "MSTC_C080_F060_A100_2_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c100_r18_c080_f060_a100_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c100_r18_c080_f060_a100_s3_b050.sbatch" "MSTC_C080_F060_A100_3_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c100_r18_c080_f060_a100_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c100_r18_c080_f060_a100_s4_b050.sbatch" "MSTC_C080_F060_A100_4_50b" | tee -a submitted_03_scale_coarse.txt
jid=$(sbatch --parsable "scale_coarse_c100_r18_c080_f060_a100_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_coarse_c100_r18_c080_f060_a100_s5_b050.sbatch" "MSTC_C080_F060_A100_5_50b" | tee -a submitted_03_scale_coarse.txt

echo "submitted $(wc -l < submitted_03_scale_coarse.txt) jobs from 03_scale_coarse"
