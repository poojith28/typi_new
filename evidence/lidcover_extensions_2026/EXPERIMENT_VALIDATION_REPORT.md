# LIDCover extensions 2026 — experiment validation report

## Current pre-launch verdict

**PASS with two operational checks remaining at submission time:** scheduler queue exclusion and availability of a Parquet engine (`pyarrow` or `fastparquet`) for post-processing. No job has been submitted and no predictive result is claimed.

## Historical repository map

1. Historical ProbCover: `TypiClust/deep-al/pycls/al/prob_cover.py`, class `ProbCover`.
2. Historical LIDCover/IDProbCover: `TypiClust/deep-al/pycls/al/IDProbCover.py`, class `IDProbCover`; `IDprocover.py` is its compatibility import.
3. Routing/driver: `pycls/al/ActiveLearning.py` selects samplers; `tools/train_al.py` performs cold-start acquisition, training/evaluation episodes, radius scheduling, and persistence.
4. Representation loading: `pycls/datasets/utils/features.py`; backbone is selected through `TYPI_FEATURE_BACKBONE`, root through `TYPI_FEATURES_ROOT`, requested AL-seed representation falls back to `features_seed1.npy` when absent.
5. Episode/budget convention: an empty-set acquisition occurs before episode 0. Episode `e` evaluates `(e+1)×batch` labels. A `MAX_ITER=m` run therefore makes `m+1` acquisitions and evaluates through `(m+1)×batch` labels.
6. Historical results live under `TypiClust/output/<DATASET>/<backbone>/<experiment>` with root partitions, `initial_pool`, `episode_N`, `episode_summary.json`, and `benchmark_summary.json` in newer canonical runs.
7. Relevant pre-existing variants include uncommitted auto-delta and IDProbCover tie/fallback work. Those identifiers and outputs are not reused by this extension.

## Required validation assertions

| Requirement | Verdict |
|---|---|
| Historical source unchanged by additions | PASS; eight-file SHA256 ledger matches |
| Old outputs untouched | PASS by output isolation and command audit; all new writes target extension roots |
| New sampler identifiers unique | PASS; absent from historical method source/routing |
| New output directories unique | PASS across 85 manifest rows |
| Dataset splits correct | PASS by dry-run reconstruction; runtime partitions are revalidated |
| Validation/test excluded from acquisition | PASS; candidate and held-out index sets are disjoint |
| Feature/candidate hashes recorded | PASS in pre-launch provenance; runtime provenance adds normalised-feature hashes |
| Seeds | PASS; exactly 1–5 per cell |
| Cold start | PASS; all training manifests specify empty labelled set |
| Final/common budgets | PASS; batch study ends exactly at 5000; other historical-trajectory studies at 5050 |
| Acquisition batch sizes | PASS |
| No performance-based hyperparameter selection | PASS by design/static dataflow |
| Auto radius uses no labels | PASS by design/static dataflow |
| AULC definition | PASS; normalised trapezoid, exact declared domains |
| Completed episode sequence | Enforced at run completion and again during aggregation |

## Methodological cautions / negative findings

- Representation loading can fall back to the same `features_seed1.npy` for AL seeds 2–5. This is the repository convention, and provenance exposes the actual path/hash. Thus AL seeds vary splits/training but may not represent independently pretrained encoders.
- Frozen LIDCover estimates LID/kNN on the full stored training representation before restricting graph vertices to labelled+unlabelled candidates. Held-out validation points cannot be acquired, but they may influence the historical local-ID/kNN geometry. Changing that would violate the frozen-reference requirement; it must be disclosed.
- Historical ProbCover is dense-radius coverage, whereas `sparsefixedcover_v1` is kNN-truncated. The optional control must not be interpreted or named as historical ProbCover.
- Individual acquisition positions/rounds are not independent statistical replicates. Inferential summaries use seed-level aggregation; no causal claim is supported.

## Dry-run record

All 85 configurations passed array-index, config-path, method-ID, seed, budget, final-budget, and output-uniqueness dry-run checks on 2026-09-03. The mechanism analysis is staged separately and will fail unless every selected ID exactly replays under recomputed geometry.
