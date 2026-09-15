# Matched batch-size pre-launch validation

Status: **PASS for configuration/mapping; experiments not launched.**

- Matrix size: 3 acquisition batch sizes × 2 new reference method IDs × 5 seeds = 30 GPU tasks.
- All runs start empty and permit their acquisition rule to select the first batch.
- Configured base radius is 0.25 for both methods; LIDCover uses alpha=1 and k_id=k_knn=50.
- Classifier configuration, dataset split convention, feature source convention, and seed are held fixed within each matched comparison.
- Exact evaluated sequences end at 5,000 labels: batch 50 uses 100 acquisition calls/episodes 0–99; batch 100 uses 50 calls/episodes 0–49; batch 500 uses 10 calls/episodes 0–9.
- Common comparisons use only exact saved checkpoints at 500, 1000, …, 5000 labels. Accuracy is never interpolated.
- AULC is normalised trapezoidal integration over those exact common checkpoints only (500–5000).
- All 30 array mappings passed dry-run validation; output IDs and directories are unique and outside historical output roots.
