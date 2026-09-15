# Trajectory-Adaptive LID Cover (TALC)

TALC is a coverage-first acquisition strategy motivated by the diagnostic
finding that successful coverage methods initially favour low-local-ID,
short-connectivity-scale regions and broaden later in acquisition.

For candidate `i`, the round-specific radius is

```text
r_i(t) = clip(delta * (LID_i / median(LID))^(-alpha_t))
alpha_t = alpha_max * (1 - min(base_coverage / coverage_target, 1))
```

At each greedy pick, TALC first retains candidates whose newly covered point
count is within `coverage_epsilon` of the current maximum. It only then applies
the curriculum score:

```text
(1-progress) * low_ID_preference
  + progress * (topology_weight * component_novelty
                + (1-topology_weight) * margin_uncertainty)
```

The topology term uses component membership at quantiles of the reference
kNN-MST merge scales. It does not use order-sensitive pointwise persistence
attribution.

## Recommended first experiment

```bash
python tools/train_al.py \
  --cfg configs/<dataset-config>.yaml \
  --exp-name TALC_1_50b \
  --al talc_auto_delta \
  --budget 50 \
  --initial_size 0 \
  --seed 1 \
  --talc_alpha_max 1.0 \
  --talc_coverage_target 0.9 \
  --talc_coverage_epsilon 0.05 \
  --talc_topology_weight 0.5
```

Use `--al talc` with `--initial_delta` for matched fixed-radius experiments.
The flags `--talc_use_topology false` and `--talc_use_uncertainty false`
provide the main ablations without changing sampler identifiers in the code.
