# LIDCover ICONIP reviewer experiments

All arrays use the matched cold-start protocol (`initial_size=0`, batch 50,
five seeds, 100 acquisition rounds, final labeled budget 5050) and immutable
new output names. Every array is capped at fifteen simultaneous tasks.

| Array | New runs | Purpose |
|---|---:|---|
| `lidcover_review_fallback_30.sbatch` | 30 | Measure fallback fraction per round and isolate min-LID versus seeded-random zero-gain fallback on all three datasets. Positive-gain tie-breaking is unchanged. |
| `lidcover_review_probcover_radius_25.sbatch` | 25 | Add CIFAR-100 ProbCover radii 0.10, 0.20, 0.30, 0.40, and 0.50. Existing 0.25 and 0.60 rows complete the matched seven-radius sweep. |
| `lidcover_review_signal_tuning_50.sbatch` | 50 | Add exponents 0.5 and 2.0 for LID and four alternative signals. Existing exponent-1 rows complete an equal three-value tuning budget. |
| `lidcover_review_maxherding_15.sbatch` | 15 | Produce the missing MaxHerding numbers on CIFAR-10, CIFAR-100, and TinyImageNet. |
| `lidcover_review_k_disentangle_20.sbatch` | 20 | Vary `k_ID` and `k_NN` one at a time around the existing (50,50) control. |
| `lidcover_auto_delta_30.sbatch` | 30 | Matched label-free automatic-radius LIDCover/ProbCover experiment requested in the earlier thesis evidence plan. |

Total: **170 new runs** (140 reviewer-specific and 30 automatic-radius).

Submit the arrays independently:

```bash
bash /vast/s219110279/TypiClust/deep-al/slurm/submit_lidcover_review_experiments.sh
```

Preview without submitting:

```bash
DRY_RUN=1 bash /vast/s219110279/TypiClust/deep-al/slurm/submit_lidcover_review_experiments.sh
```

The wrapper does not add dependencies between arrays. A failure in one array
therefore does not block the other experiment groups. The `%15` throttle is
per array; cluster, user, and QoS limits determine total concurrent usage.
