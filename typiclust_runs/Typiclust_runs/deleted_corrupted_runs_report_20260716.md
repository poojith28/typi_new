# Deleted / Corrupted Runs Detected

Audit date: 2026-07-16

Completion requires a valid `benchmark_summary.json`. Active TopoCover jobs and partially launched MaxHerding experiments are excluded.

## Summary

| Dataset | Backbone | Strategy | Missing seeds | Count |
| --- | --- | --- | --- | ---: |
| CIFAR10 | AlexNet | `distance_variance_cover` | 4 | 1 |
| CIFAR10 | ResNet-50 | `distance_variance_cover` | 3 | 1 |
| CIFAR10 | ResNet-50 | `knn_distance_cover` | 2, 3, 5 | 3 |
| CIFAR10 | ResNet-50 | `density_cover` | 2, 3, 4, 5 | 4 |
| CIFAR10 | ResNet-50 | `dbal` | 4, 5 | 2 |
| TinyImageNet | ResNet-50 | `distance_variance_cover` | 5 | 1 |
| **Total** |  |  |  | **12** |

## Individual Missing Runs

| Dataset | Backbone | Strategy | Experiment name | Canonical script |
| --- | --- | --- | --- | --- |
| CIFAR10 | AlexNet | `distance_variance_cover` | `distance_variance_cover_4_50b` | `cifar10_full_50b_alexnet/distance_variance_cover.sh` |
| CIFAR10 | ResNet-50 | `distance_variance_cover` | `distance_variance_cover_3_50b` | `cifar10_full_50b/distance_variance_cover.sh` |
| CIFAR10 | ResNet-50 | `knn_distance_cover` | `knn_distance_cover_2_50b` | `cifar10_full_50b/knn_distance_cover.sh` |
| CIFAR10 | ResNet-50 | `knn_distance_cover` | `knn_distance_cover_3_50b` | `cifar10_full_50b/knn_distance_cover.sh` |
| CIFAR10 | ResNet-50 | `knn_distance_cover` | `knn_distance_cover_5_50b` | `cifar10_full_50b/knn_distance_cover.sh` |
| CIFAR10 | ResNet-50 | `density_cover` | `density_cover_2_50b` | `cifar10_full_50b/density_cover.sh` |
| CIFAR10 | ResNet-50 | `density_cover` | `density_cover_3_50b` | `cifar10_full_50b/density_cover.sh` |
| CIFAR10 | ResNet-50 | `density_cover` | `density_cover_4_50b` | `cifar10_full_50b/density_cover.sh` |
| CIFAR10 | ResNet-50 | `density_cover` | `density_cover_5_50b` | `cifar10_full_50b/density_cover.sh` |
| CIFAR10 | ResNet-50 | `dbal` | `dbal_4_50b` | `cifar10_full_50b/dbal.sh` |
| CIFAR10 | ResNet-50 | `dbal` | `dbal_5_50b` | `cifar10_full_50b/dbal.sh` |
| TinyImageNet | ResNet-50 | `distance_variance_cover` | `distance_variance_cover_5_50b` | `tinyimagenet_full_50b/distance_variance_cover.sh` |

## Audit Notes

- All 12 expected experiment directories are absent.
- None of the 12 runs currently appears in `squeue`.
- This report does not submit reruns.
- The older `unfinished_runs_report.md` remains the source for incomplete 500-budget runs.
