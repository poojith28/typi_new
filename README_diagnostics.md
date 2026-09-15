# `posthoc_graph_diagnostics`

A post-hoc diagnostic pipeline for active-learning (AL) runs. It **never**
touches training code -- it only reads artifacts an AL run already produced
(fixed embeddings, optional labels, per-round selected indices, accuracy
logs) and computes intrinsic-dimensionality, H0 graph-filtration, kNN-graph
structural, community-coverage, and label-neighbourhood-impurity
diagnostics, plus their correlation with downstream accuracy.

```
posthoc_graph_diagnostics/     # the package (this is what you import / run -m)
configs/diagnostics_template.yaml   # copy this to configs/diagnostics.yaml and edit
results/posthoc_graph_diagnostics/  # default output root (created on first run)
```

Run everything from the repository root (`/vast/s219110279`) so that the
`posthoc_graph_diagnostics` package is importable:

```bash
pip install -r posthoc_graph_diagnostics/requirements.txt
# optional but recommended for large datasets / faster kNN:
pip install faiss-cpu python-louvain

cp configs/diagnostics_template.yaml configs/diagnostics.yaml
# ... edit configs/diagnostics.yaml to point at your runs ...

python -m posthoc_graph_diagnostics.run       --config configs/diagnostics.yaml
python -m posthoc_graph_diagnostics.aggregate --root results/posthoc_graph_diagnostics
python -m posthoc_graph_diagnostics.plot      --summary results/posthoc_graph_diagnostics/summary/all_round_diagnostics.csv
```

## 1. Expected inputs

| Input | Format | Notes |
|---|---|---|
| Embeddings | `.npy`, shape `[N, d]` | Fixed feature matrix for the full pool (e.g. SimCLR/SCAN pretext features). |
| Labels (optional) | `.npy`, shape `[N]` | Integer class ids, same length as embeddings. Only used for the post-hoc label-impurity diagnostics -- never for training or selection. |
| Per-round selected indices | see below | Indices into the embeddings array. |
| Accuracy per round (optional) | see below | Merged in for the correlation analysis. |

### 1.1 Selection loaders (`selection.type`)

* **`lset_episodes`** (default, matches TypiClust/deep-al output):
  `experiment_dir/episode_<r>/lSet.npy` holds the **cumulative** selected
  set `I_r` at round `r`. If `episode_<r>/episode_summary.json` has an
  `active_set_ids` field, it is used as the exact batch `ΔI_r` (more
  robust than diffing, since `lSet.npy` ordering is not guaranteed);
  otherwise `ΔI_r = I_r \ I_{r-1}` is computed automatically.
* **`picks_dir`**: a directory of per-round files matched by
  `picks_glob` (e.g. `picks/round_*.npy`); `picks_round_regex` extracts the
  round number from each filename. Files are treated as **batch** sets
  `ΔI_r` by default; set `cumulative: true` if they are already cumulative.
* **`csv`**: either long format (`csv_round_col`, `csv_index_col`: one row
  per selected index) or wide format (`csv_indices_col`: one row per round
  holding a JSON/Python-literal list of indices).
* **`json`**: `{round: [indices...]}` or `[{"round": r, "indices": [...]}, ...]`.

All loaders handle **missing round 0** gracefully: the first available
round's batch defaults to its own cumulative set (i.e. nothing is assumed
selected before it), unless `selection.initial_lset_path` points at an
explicit seed-pool file.

### 1.2 Accuracy loaders (`accuracy.type`)

* **`episode_json`** (default): reads `test_accuracy` / `best_val_accuracy`
  from each round's `episode_summary.json`.
* **`csv`**: a CSV with a round/episode column and an accuracy column
  (`csv_round_col`, `csv_accuracy_col`; auto-detects `test_accuracy` /
  `accuracy` / `best_val_accuracy` if not given).
* **`stdout_log`**: a free-text log parsed line-by-line with `log_regex`
  (named groups `round` and `accuracy`).

### 1.3 Multiple methods / seeds / datasets

Use `run_matrix` in the config to expand a templated cross-product of
dataset x backbone x method x seed into individual runs (see
`configs/diagnostics_template.yaml`). Missing files for any one combination
are skipped with a warning, not a crash. You can also list one-off `runs:`
entries explicitly (e.g. for non-conforming paths), which are appended
after the matrix expansion.

## 2. Output layout

```
results/posthoc_graph_diagnostics/
  <dataset>/<backbone>/<method>/<seed>/
    round_diagnostics.csv       # one row per AL round; all diagnostics + accuracy
    node_graph_metrics.csv      # long format over graph_k: per-node degree/PageRank/
                                 #   clustering/k-core/betweenness + per-node label
                                 #   impurity columns (suffixed _labelk<K>)
    node_lid_k10.csv            # reference-pool local ID per node (one file per k)
    node_lid_k20.csv
    node_lid_k50.csv
    node_lid_k100.csv
    h0_reference_events.csv     # MST edges: event_id, u, v, death_value
    community_assignments.csv   # long format over graph_k: node_id -> community_id
    figures/
      fig_batch_vs_cumulative_id_<dataset>_<backbone>.pdf
      fig_id_k_sensitivity_<dataset>_<backbone>.pdf
      fig_id_vs_accuracy_<dataset>_<backbone>.pdf
      fig_h0_gap_random_control_<dataset>_<backbone>.pdf
      fig_community_coverage_<dataset>_<backbone>.pdf
      fig_graph_metric_trajectories_<dataset>_<backbone>.pdf
      fig_label_impurity_<dataset>_<backbone>.pdf   # only if labels were provided
  summary/
    all_round_diagnostics.csv       # concatenation of every round_diagnostics.csv
                                     # + accuracy_gain, next_round_accuracy_gain, final_accuracy
    aulc_summary.csv                # AULC per dataset/backbone/method/seed
    correlation_summary.csv         # Spearman rho + bootstrap CI, per dataset/backbone,
                                     #   diagnostic x {accuracy, next_round_accuracy_gain}
    early_predictor_summary.csv     # early-round diagnostics -> final_accuracy correlation
    method_family_summary.csv       # AULC / final accuracy aggregated by method family
    figures/
      fig_diagnostic_accuracy_correlation_heatmap.pdf
      fig_diagnostic_nextgain_correlation_heatmap.pdf
      fig_topology_id_joint_scatter.pdf
      fig_batch_vs_cumulative_id_<dataset>_<backbone>.pdf   # cross-method comparison
      fig_h0_gap_random_control_<dataset>_<backbone>.pdf    #   (produced by `plot.py`,
      fig_community_coverage_<dataset>_<backbone>.pdf       #    overlaying all methods
      fig_graph_metric_trajectories_<dataset>_<backbone>.pdf#    and seeds per dataset/
      fig_label_impurity_<dataset>_<backbone>.pdf           #    backbone)
```

Per-run figures (inside each run's `figures/`) show that single
method/seed's trajectory. The same-named figures under `summary/figures/`
(produced by `python -m posthoc_graph_diagnostics.plot`) overlay every
method (mean +/- std across seeds) for a given dataset/backbone, which is
what you want for cross-method comparison plots in a paper/thesis.

## 3. Diagnostics computed per round

**Intrinsic dimensionality** (Levina-Bickel local ID; k in {10,20,50,100}):
`batch_mean_lid_k*`, `cumulative_mean_lid_k*` (+ q10/q25/q50/q75/q90 for
both), `global_mle_id_k*` (only when `|I_t| > k+2`), `adaptive_global_id_k*`
using `k_t = min(k, |I_t|-1)` with a `_valid` flag and `_k_used` column.

**H0 graph-filtration** (MST-edge death values as finite H0 events):
`raw_h0_ecdf_gap`, `random_mean_gap`, `random_std_gap`, `normalized_h0_gap`
(`= raw - random_mean`), `h0_ecdf_gap_zscore`, `beta0_gap` on a fixed
epsilon grid, and the matching `beta0_random_mean_gap` /
`beta0_random_std_gap` / `beta0_normalized_gap` / `beta0_gap_zscore`
same-cardinality random-subset control (100 samples/round by default).
Controlled by `h0_scope` (`cumulative` by default; `batch` or `both` also
supported).

**kNN-graph structural metrics** (for each `graph_k` in {10,20,50}):
mean/median/std/q10-q90 of `degree`, `weighted_degree`,
`clustering_coefficient`, `pagerank`, `kcore_number`, `betweenness_approx`
over the batch and cumulative selected sets, plus
`distance_to_nearest_selected_{mean,median}_k*` (graph-shortest-path
distance from every point to its nearest selected point, via a single
vectorized multi-source Dijkstra call).

**Community coverage** (Louvain, falling back to greedy-modularity then
label-propagation): `communities_touched_batch_k*`,
`communities_touched_cumulative_k*`, `community_coverage_fraction_k*`,
`selected_community_entropy_k*`, `selected_community_gini_k*`,
`largest_selected_community_share_k*`, and
`rounds_to_{50,75,90}_percent_community_coverage_k*`.

**Label-neighbourhood impurity** (only when labels are provided; k
defaults to `graph_k_values`, override with `label_k_values`):
mean/median/q10-q90 of `knn_label_entropy`, `same_label_neighbor_ratio`,
`local_class_count` over batch and cumulative selected sets.

## 4. Performance correlation (`aggregate` step)

`aggregate.py` merges every run's `round_diagnostics.csv`, computes
`accuracy_gain`, `next_round_accuracy_gain`, `final_accuracy`, and AULC
(trapezoidal, normalized by round-range), auto-discovers every numeric
diagnostic column (batch/cumulative ID, H0 gaps, graph metrics, community
metrics, label impurity), and computes Spearman correlations (with
percentile bootstrap confidence intervals) against `accuracy` and
`next_round_accuracy_gain` per dataset/backbone, plus a separate
early-round-diagnostics-vs-`final_accuracy` table
(`early_predictor_summary.csv`, cutoff controlled by `--early-round-cutoff`
/ `early_round_cutoff` in the config).

## 5. Performance & scalability notes

* Reference-pool LID, the H0 reference MST, kNN graphs, node graph
  metrics, communities, and label metrics are cached under `cache_dir`,
  keyed by a **cheap file fingerprint** (path + size + mtime) of the
  embeddings/labels file -- so N methods x M seeds sharing the same
  backbone/dataset embeddings only pay the O(N log N) cost once.
* Full pairwise distance matrices are only materialized when
  `N <= h0_max_exact_n` (H0) / via FAISS or scikit-learn `NearestNeighbors`
  otherwise (LID, kNN graphs). H0 on large point sets falls back to an
  approximate k-NN-graph MST (a minimum spanning **forest** if the graph is
  disconnected -- a warning is logged with the component count).
* The H0 same-cardinality random control exploits the fact that the
  reference structure is a tree/forest: any node-induced subgraph is
  automatically acyclic, so `beta0(eps)` for a subset `S` is
  `|S| - searchsorted(induced_sorted_weights, eps)` -- O(log|S|) per
  epsilon after one O(E) filter, making 100 random-control samples/round
  tractable even for ~10^5-point datasets.
* Betweenness centrality is approximated via sampling
  (`betweenness_sample_size` source nodes) whenever
  `N > betweenness_max_exact_n`.
* FAISS is used automatically when installed (`use_faiss: true`); the
  pipeline transparently falls back to `sklearn.neighbors.NearestNeighbors`
  otherwise. Louvain community detection falls back to
  `networkx`'s greedy-modularity or label-propagation if `python-louvain`
  (or `networkx>=3.2`'s built-in `louvain_communities`) is unavailable.
* **Exact brute-force kNN is `O(N^2 * d)`.** For high-dimensional features
  (e.g. 4096-d AlexNet fc7) at the ~10^4-10^5-point scale typical of AL
  embedding pools, this is impractical (a single reference kNN pass over
  50,000 x 4096-d points did not finish in several minutes on a loaded
  shared host during development). With `use_approx_knn: true` (default),
  any kNN search whose index set exceeds `approx_knn_min_n` points
  automatically switches to a FAISS IVF-Flat approximate index (`ann_nlist`
  Voronoi cells, probing `ann_nprobe` of them per query), which is an
  order of magnitude faster at a small, well-documented recall cost. This
  only applies when FAISS is installed; set `use_approx_knn: false` to force
  exact search (fine for smaller/lower-dimensional embeddings, e.g. ResNet
  features on CIFAR-sized pools).

## 6. Troubleshooting

* **"No runs configured"**: check that `run_matrix` and/or `runs` are
  present and produce at least one entry in the YAML config.
* **A run is silently skipped in `run.py`**: check the log for a `FAILED`
  line with a traceback; common causes are a wrong `experiment_dir`
  (missing `episode_*` folders) or an embeddings/labels length mismatch.
* **"No round_diagnostics.csv files found" in `aggregate.py`**: run
  `python -m posthoc_graph_diagnostics.run` first, and check `--root`
  points at the same `output_root` used there.
