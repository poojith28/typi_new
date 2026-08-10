# MSTC single-scale via alpha — full 160, throttled to 20

Same `multiscale_topocover` code:
- `alpha=1.0` coarse-only
- `alpha=0.0` fine-only

## Full grid (160)

| kind | count |
|---|---|
| coarse_only (α=1, 3×3×5) | 45 |
| fine_only (α=0, 3×3×5) | 45 |
| scale_coarse (α=1, 3 ds × R18 × 2 δ) | 30 |
| scale_fine (α=0, same) | 30 |
| budget_coarse (α=1, B∈{100,500}) | 10 |

Submit throttled:
```bash
bash submit_throttled.sh
```
Uses `#SBATCH --array=0-159%20`.
