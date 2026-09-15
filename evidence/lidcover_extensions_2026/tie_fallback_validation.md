# Tie/fallback factorial pre-launch validation

Status: **PASS for configuration/mapping; experiments not launched.**

- Matrix size: 4 explicit method IDs × 5 paired seeds = 20 GPU tasks.
- Only the equal-positive-gain tie rule and zero/non-positive-gain fallback rule vary.
- Geometry, inverse-LID transformation, strict radius test, explicit self-coverage, candidate eligibility, and uncovered-set updates follow the frozen LIDCover reference.
- Random choices use a run-local generator keyed by experiment seed, selection event, query position, and decision stream. Global training/split RNG state is untouched.
- Per-round metrics and ordered per-selection JSONL diagnostics are written within each new run. Post-processing combines them into the requested CSV/Parquet outputs.
- All runs use CIFAR100/ResNet18, delta0=0.25, alpha=1, k_id=k_knn=50, batch=50, and an empty cold start.
- All 20 array mappings passed dry-run validation; no historical output is read as a substitute.
