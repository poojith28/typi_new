#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

bash submit_01_coarse_only.sh
bash submit_02_fine_only.sh
bash submit_03_scale_coarse.sh
bash submit_04_scale_fine.sh
