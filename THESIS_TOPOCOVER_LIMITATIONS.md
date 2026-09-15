# TopoCover and MultiScaleTopoCover limitations

- The audit found 35 missing or partial topology runs; none were aggregated.
- Historical original TopoCover outputs that were removed were not reconstructed. Alpha endpoints are single-scale MSTC controls, not historical TopoCover results.
- Structural summaries use only quantities preserved in benchmark metadata. Exact $\beta_0$, uncovered-component counts, component-size distributions, batch component touch counts, and neighbourhood redundancy are left unavailable where saved graph objects were absent.
- The representation used for acquisition is a frozen self-supervised embedding. The supervised classifier is restarted on CIFAR-10 (`FINE_TUNE=False`) but warm-started between rounds on CIFAR-100 and TinyImageNet under the preserved configurations (`FINE_TUNE=True`).
- Historical validation/test pipelines used random crops (and, for TinyImageNet, a random flip). These curves are descriptive only; deterministic-transform reruns are required for corrected predictive comparisons.
- A stored GUDHI birth representative is not an invariant owner of an $H_0$ event. Birth-vertex touched coverage is therefore not used as central evidence; symmetric MST-edge endpoint definitions and matched null controls are required.
- Original TopoCover selected-index trajectories were not preserved. Newly instrumented reruns save ordered pool indices, every selected batch, configuration information, stopping metadata, and hashes; TopoCover cannot enter the corrected post-hoc diagnostic until those reruns finish.
- Small seed counts limit power; effect sizes, seed consistency, and bootstrap intervals should be considered alongside p-values.
- Exploratory variants have n=1 and cannot establish inferiority or superiority.
