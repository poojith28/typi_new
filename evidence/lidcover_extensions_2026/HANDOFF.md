# LIDCover extensions 2026 — handoff

## A. Files added

All additions are confined to:

- `experiments/lidcover_extensions_2026/` — automatic radius, new samplers, isolated runtime registration;
- `analysis/lidcover_extensions_2026/` — strict aggregation and read-only mechanism replay;
- `configs/lidcover_extensions_2026/` — protocol matrix;
- `scripts/lidcover_extensions_2026/` — manifests, preflight, shared runner, and seven sbatch files;
- `evidence/lidcover_extensions_2026/` — manifests, hashes, registry, and validation reports.

## B. Existing files changed

None. The exact diff attributable to this work against pre-existing files is empty. The repository already had unrelated/uncommitted modifications before this work; they were preserved.

## C. Historical source hashes before/after

The before ledger is `historical_source_hashes_before.json`. Final validation reproduced every digest exactly. In particular:

- ProbCover: `5e41d51543a3bb5c77633834d43f6f3361e83317130b24a4b466bda660a4191a`
- IDProbCover/LIDCover: `f700edd6b68881fee2c9992c75236a8c88610935ea10b461c4d3ca3866afe875`
- ActiveLearning routing: `c33610d8333a152095668b85110205701598746bb24d7cc32616b92b5aa7695e`
- train_al driver: `bca9dab55c236d544f8695e49b838caa0a274721138e002c06922aaaf7134ca8`
- representation loader: `000bcdfc8d4d5428dbe32432d3cee977dc43d2bcd18ccbac7929200a5d0b8d23`

## D. Experiment counts

- automatic radius: 30;
- matched batch size: 30;
- tie/fallback factorial: 20;
- optional SparseFixedCover: 5;
- total essential training runs: 80;
- total including optional control: 85.

## E. Expected GPU jobs

Four array submissions create 80 essential plus 5 optional array tasks. The read-only mechanism replay is a separate five-task GPU array. Post-processing and mechanism summary are CPU jobs.

## F. Estimated storage

Canonical historical runs sampled locally occupy roughly 120–185 MB for 101-episode runs. Accounting for shorter batch-size trajectories, 85 training runs, new kNN/LID caches, diagnostics, and figures, reserve approximately **15–20 GB**. This is an estimate, not a quota guarantee.

## G. Exact commands

```bash
cd /vast/s219110279

python scripts/lidcover_extensions_2026/validate_prelaunch.py --family auto_radius --check-squeue
sbatch scripts/lidcover_extensions_2026/launch_auto_radius.sbatch

python scripts/lidcover_extensions_2026/validate_prelaunch.py --family batch_size --check-squeue
sbatch scripts/lidcover_extensions_2026/launch_batch_size.sbatch

python scripts/lidcover_extensions_2026/validate_prelaunch.py --family tie_fallback --check-squeue
sbatch scripts/lidcover_extensions_2026/launch_tie_fallback.sbatch

# Optional
python scripts/lidcover_extensions_2026/validate_prelaunch.py --family sparse_fixed_optional --check-squeue
sbatch scripts/lidcover_extensions_2026/launch_sparse_fixed.sbatch

# Submit after all essential training arrays complete.
sbatch scripts/lidcover_extensions_2026/launch_postprocess.sbatch

# Historical read-only mechanism replay, then summary.
MECH_JOB=$(sbatch --parsable scripts/lidcover_extensions_2026/launch_mechanism_replay.sbatch)
sbatch --dependency="afterok:${MECH_JOB}" scripts/lidcover_extensions_2026/launch_mechanism_summary.sbatch
```

## H. Essential versus optional

The automatic-radius, matched batch-size, and tie/fallback arrays are essential. SparseFixedCover is optional. Mechanism replay is essential for the requested mechanism analysis but does not rerun training.

## I. Thesis Appendix C suitability

Suitable after strict completion/aggregation validation:

- automatic-radius table, learning curves, and selected-scale figure;
- exact-common-budget batch table/AULC/learning curves;
- mechanism per-selection provenance, seed-level summaries, tables, and five figures;
- tie/fallback performance and frequency tables/figures;
- method registry and experiment validation report.

The pre-launch `PENDING_RUN` provenance rows and any partial/quarantined run are not thesis-result evidence.

## J. Methodological problems / negative findings

1. The representation loader may fall back to `features_seed1.npy` for AL seeds 2–5. Provenance records the actual path/hash, but those seeds may not represent independently pretrained encoders.
2. Frozen LIDCover computes full-training-matrix LID/kNN before restricting acquisition vertices. Validation samples cannot be acquired but may influence geometry. This is preserved to avoid changing the historical algorithm and must be disclosed.
3. SparseFixedCover is kNN-truncated and is not historical dense ProbCover.
4. The current `typiclust` environment lacks both `pyarrow` and `fastparquet`. Training jobs are unaffected; install one before running the post-processing jobs that must write Parquet.
5. Scheduler queue validation timed out in the current sandbox. Run the provided `--check-squeue` preflight on the login node immediately before each `sbatch` command.
6. No predictive results exist yet; no final accuracy, AULC, or significance claim has been fabricated.
