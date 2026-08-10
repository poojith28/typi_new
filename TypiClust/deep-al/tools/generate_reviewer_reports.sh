#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/s219110279/TypiClust/deep-al"
PYTHON_BIN="${PYTHON_BIN:-python}"
ANALYZE_SCRIPT="$ROOT/tools/analyze_reviewer_experiments.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/s219110279/TypiClust/output}"
REPORT_ROOT="${REPORT_ROOT:-/scratch/s219110279/TypiClust/reviewer_reports}"

cd "$ROOT"

"$PYTHON_BIN" "$ANALYZE_SCRIPT" \
  --output_root "$OUTPUT_ROOT" \
  --dataset CIFAR100 \
  --model resnet18 \
  --report_dir "$REPORT_ROOT/cifar100_resnet18"

"$PYTHON_BIN" "$ANALYZE_SCRIPT" \
  --output_root "$OUTPUT_ROOT" \
  --dataset TINYIMAGENET \
  --model resnet18 \
  --report_dir "$REPORT_ROOT/tinyimagenet_resnet18"

"$PYTHON_BIN" "$ANALYZE_SCRIPT" \
  --output_root "$OUTPUT_ROOT" \
  --dataset CIFAR10 \
  --model resnet18 \
  --report_dir "$REPORT_ROOT/cifar10_resnet18"
