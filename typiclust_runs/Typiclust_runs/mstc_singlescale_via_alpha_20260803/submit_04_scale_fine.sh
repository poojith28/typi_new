#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
: > submitted_04_scale_fine.txt

jid=$(sbatch --parsable "scale_fine_c10_r18_c060_f040_a000_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c10_r18_c060_f040_a000_s1_b050.sbatch" "MSTC_C060_F040_A000_1_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c10_r18_c060_f040_a000_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c10_r18_c060_f040_a000_s2_b050.sbatch" "MSTC_C060_F040_A000_2_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c10_r18_c060_f040_a000_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c10_r18_c060_f040_a000_s3_b050.sbatch" "MSTC_C060_F040_A000_3_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c10_r18_c060_f040_a000_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c10_r18_c060_f040_a000_s4_b050.sbatch" "MSTC_C060_F040_A000_4_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c10_r18_c060_f040_a000_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c10_r18_c060_f040_a000_s5_b050.sbatch" "MSTC_C060_F040_A000_5_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c10_r18_c080_f060_a000_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c10_r18_c080_f060_a000_s1_b050.sbatch" "MSTC_C080_F060_A000_1_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c10_r18_c080_f060_a000_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c10_r18_c080_f060_a000_s2_b050.sbatch" "MSTC_C080_F060_A000_2_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c10_r18_c080_f060_a000_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c10_r18_c080_f060_a000_s3_b050.sbatch" "MSTC_C080_F060_A000_3_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c10_r18_c080_f060_a000_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c10_r18_c080_f060_a000_s4_b050.sbatch" "MSTC_C080_F060_A000_4_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c10_r18_c080_f060_a000_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c10_r18_c080_f060_a000_s5_b050.sbatch" "MSTC_C080_F060_A000_5_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c100_r18_c060_f040_a000_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c100_r18_c060_f040_a000_s1_b050.sbatch" "MSTC_C060_F040_A000_1_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c100_r18_c060_f040_a000_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c100_r18_c060_f040_a000_s2_b050.sbatch" "MSTC_C060_F040_A000_2_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c100_r18_c060_f040_a000_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c100_r18_c060_f040_a000_s3_b050.sbatch" "MSTC_C060_F040_A000_3_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c100_r18_c060_f040_a000_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c100_r18_c060_f040_a000_s4_b050.sbatch" "MSTC_C060_F040_A000_4_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c100_r18_c060_f040_a000_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c100_r18_c060_f040_a000_s5_b050.sbatch" "MSTC_C060_F040_A000_5_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c100_r18_c080_f060_a000_s1_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c100_r18_c080_f060_a000_s1_b050.sbatch" "MSTC_C080_F060_A000_1_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c100_r18_c080_f060_a000_s2_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c100_r18_c080_f060_a000_s2_b050.sbatch" "MSTC_C080_F060_A000_2_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c100_r18_c080_f060_a000_s3_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c100_r18_c080_f060_a000_s3_b050.sbatch" "MSTC_C080_F060_A000_3_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c100_r18_c080_f060_a000_s4_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c100_r18_c080_f060_a000_s4_b050.sbatch" "MSTC_C080_F060_A000_4_50b" | tee -a submitted_04_scale_fine.txt
jid=$(sbatch --parsable "scale_fine_c100_r18_c080_f060_a000_s5_b050.sbatch")
printf "%s\t%s\t%s\n" "$jid" "scale_fine_c100_r18_c080_f060_a000_s5_b050.sbatch" "MSTC_C080_F060_A000_5_50b" | tee -a submitted_04_scale_fine.txt

echo "submitted $(wc -l < submitted_04_scale_fine.txt) jobs from 04_scale_fine"
