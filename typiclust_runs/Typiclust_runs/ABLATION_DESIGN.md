# LIDCOVER Ablation Design

Goal: explain which part of LIDCOVER helps, not just whether it beats baselines.

Use the full baseline runs as the main comparison:

- random
- uncertainty
- entropy
- margin
- coreset
- dbal
- probcover
- adaptive cover variants
- LIDCOVER

Recommended ablations below should run on CIFAR-100 first. If results are clear, repeat the best 3-5 ablations on CIFAR-10 and TinyImageNet.

## Fixed Base Setup

Use this unless the ablation says otherwise:

```bash
--cfg=/scratch/s219110279/TypiClust/deep-al/configs/cifar100/al/RESNET18.yaml
--al=idprobcover
--budget=50
--initial_size=50
--initial_delta=0.25
--idpc_alpha=1.0
--idpc_mode=high_id_more_centers
--idpc_k_id=50
--idpc_k_knn=50
--idpc_cache_root=/scratch/s219110279/idpc_cache
--seed <1..5>
```

Primary metric: active-learning test accuracy curve and AUC over episodes.

Secondary diagnostics:

- selected ID mean/std
- selected radius mean/std
- coverage fraction after acquisition
- acquisition time

These are already written into `benchmark_summary.json` and `id_probcover_rounds.csv`.

## Ablation A: LID Strength

Question: does local intrinsic dimension scaling matter, and how much?

Run:

| Name | alpha |
| --- | --- |
| LIDCOVER_alpha0 | 0.0 |
| LIDCOVER_alpha05 | 0.5 |
| LIDCOVER_alpha1 | 1.0 |
| LIDCOVER_alpha2 | 2.0 |

Interpretation:

- `alpha=0` removes radius adaptation, but keeps LIDCOVER's tie-breaking behavior.
- If `alpha=1` or `alpha=2` beats `alpha=0`, the LID radius adaptation is doing real work.
- If `alpha=2` drops, the method is over-concentrating in high-ID regions.

Priority: high.

## Ablation B: Direction Of LID Bias

Question: should high-ID regions get more centers, or low-ID regions?

Run:

| Name | mode | alpha |
| --- | --- | --- |
| LIDCOVER_highID | high_id_more_centers | 1.0 |
| LIDCOVER_lowID | low_id_more_centers | 1.0 |

Interpretation:

- `high_id_more_centers` is the proposed method.
- `low_id_more_centers` is the opposite hypothesis.
- A clear gap supports the geometric argument.

Priority: high.

## Ablation C: Neighborhood Robustness

Question: is the method sensitive to the local-ID and coverage-neighborhood k values?

Run:

| Name | k_id | k_knn |
| --- | --- | --- |
| LIDCOVER_k20 | 20 | 20 |
| LIDCOVER_k50 | 50 | 50 |
| LIDCOVER_k75 | 75 | 75 |
| LIDCOVER_id20_knn50 | 20 | 50 |
| LIDCOVER_id75_knn50 | 75 | 50 |

Interpretation:

- First three test global sensitivity to neighborhood scale.
- Last two isolate ID-estimation sensitivity while holding coverage graph fixed.

Priority: high.

## Ablation D: Coverage Radius

Question: is the gain just from a lucky delta choice?

Run:

| Name | initial_delta |
| --- | --- |
| LIDCOVER_d010 | 0.10 |
| LIDCOVER_d020 | 0.20 |
| LIDCOVER_d025 | 0.25 |
| LIDCOVER_d030 | 0.30 |
| LIDCOVER_d040 | 0.40 |

Interpretation:

- Shows whether LIDCOVER remains stable under radius changes.
- Compare against ProbCover with the same delta values if runtime allows.

Priority: medium-high.

## Ablation E: Tie-Breaking

Question: are gains from LID radius scaling or only from selecting low-ID points during ties?

Run:

| Name | sampler |
| --- | --- |
| LIDCOVER_default | idprobcover |
| LIDCOVER_tie_random | idprobcover_tiebreak_random |
| LIDCOVER_tie_firstmax | idprobcover_tiebreak_first_max |
| LIDCOVER_tie_minid | idprobcover_tiebreak_min_id |

Interpretation:

- If default and `tie_minid` are similar, implementation is consistent.
- If random/firstmax are much worse, LID tie-breaking matters.
- If random/firstmax remain strong, radius scaling matters more than tie-breaking.

Priority: medium.

## Ablation F: Frontier Density Variant

Question: does choosing the densest candidate among near-best coverage candidates help?

Run:

| Name | sampler |
| --- | --- |
| LIDCOVER_default | idprobcover |
| LIDCOVER_frontier_density | idprobcover_frontier_density |

Interpretation:

- Tests whether the density refinement improves stability or just adds complexity.

Priority: medium.

## Minimal Run Plan

If runtime is limited, run only this on CIFAR-100 with seeds 1-5:

1. `LIDCOVER_alpha0`
2. `LIDCOVER_alpha05`
3. `LIDCOVER_alpha1`
4. `LIDCOVER_alpha2`
5. `LIDCOVER_lowID`
6. `LIDCOVER_k20`
7. `LIDCOVER_k75`
8. `LIDCOVER_d020`
9. `LIDCOVER_d030`
10. `LIDCOVER_tie_random`

That is 10 settings x 5 seeds = 50 runs.

## Budget Ablation

Question: does LIDCOVER still help when each acquisition round is larger?

Run only a small comparison set:

- random
- entropy
- coreset
- probcover
- LIDCOVER

Budgets:

- `100`, with `--initial_size=100`
- `500`, with `--initial_size=500`

Scripts:

```bash
/scratch/s219110279/Typiclust_runs/cifar100_budget_ablation
```

This is 5 methods x 2 budgets x 5 seeds = 50 runs.

Submit with:

```bash
cd /scratch/s219110279/Typiclust_runs/cifar100_budget_ablation
./submit_all_budget.sh
```

## Paper Table Structure

Recommended ablation table:

| Ablation | Setting | Final Acc | AUC | Coverage After | Selected ID Mean | Time/Round |
| --- | --- | --- | --- | --- | --- | --- |

Recommended figures:

- AL curve for alpha sweep.
- AL curve for high-ID vs low-ID direction.
- Bar plot of final accuracy for k sensitivity.
- Coverage fraction vs episode for delta sweep.

## Expected Claims

The strongest claims this design can support:

1. LID scaling helps beyond ProbCover-style fixed radius.
2. High-ID allocation is better than the opposite low-ID allocation.
3. Results are not highly sensitive to k.
4. Improvements are not from tie-breaking alone.
5. LIDCOVER has acceptable acquisition overhead.
