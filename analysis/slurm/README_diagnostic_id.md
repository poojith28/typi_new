# Submit intrinsic-dimensionality diagnostics

## Output layout (separate folders)

```
figures/diagnostic_id/          # all ID figures (PDF + PNG)
outputs/diagnostic_id/          # CSVs, TeX, markdown, latex snippets
outputs/diagnostic_id/id_cache/ # full-pool / full-L_t ID caches
```

Accuracy artefacts live separately under:
```
figures/diagnostic_accuracy/
outputs/diagnostic_accuracy/
```

## What the job does

Runs `analysis/generate_diagnostic_id_figures.py` with:

- **Full** fixed embedding pool for local ID (exact FAISS Flat L2 — no IVF / no sparsifying)
- **Full** cumulative labelled set `L_t` for global ID
- Same method colours as the accuracy figures

## Submit

```bash
cd /vast/s219110279/analysis/slurm
bash submit_diagnostic_id.sh resnet18          # main chapter
# bash submit_diagnostic_id.sh                 # all backbones
# MODE=gpu bash submit_diagnostic_id.sh resnet18
```

When done: `outputs/diagnostic_id/diagnostic_id_generation_report.md`
