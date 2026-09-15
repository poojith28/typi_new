# Automatic sparse radius v3

This family adds two new runtime-only identifiers:

- `probcover_auto_sparse_radius_v3`
- `lidcover_auto_sparse_radius_v3`

No historical identifier, central sampler source, or prior output directory is
modified or reused.

The radius is not fixed to `0.25`. For each dataset and seed, the resolver uses
only the acquisition-eligible rows of the pretrained representation, applies
row-wise L2 normalisation, computes exact 50-nearest-neighbour distances, and
estimates local intrinsic dimension with the explicit Hill/MLE formula. It then
chooses the smallest base radius whose adaptive graph at the historical
post-cold-start scale (`0.85`) reaches mean non-self out-degree 2. Labels,
validation/test examples, and validation/test accuracy are excluded.

The resulting base-radius preview over five seeds is approximately `0.343` for
CIFAR-10, `0.269` for CIFAR-100, and `0.314` for Tiny ImageNet. The corresponding
post-cold-start effective radii are approximately `0.291`, `0.229`, and `0.267`.
Thus the operational radius is naturally around the requested `0.25` region,
while remaining dataset-derived and fully reproducible.

The paired ProbCover and LIDCover runs receive the same seed-specific scalar
base radius. Their selection implementations remain the historical samplers.
