# Running the LIDCover extension jobs

From `/vast/s219110279`, run the preflight immediately before each submission:

```bash
python scripts/lidcover_extensions_2026/validate_prelaunch.py --family auto_radius --check-squeue
sbatch scripts/lidcover_extensions_2026/launch_auto_radius.sbatch

python scripts/lidcover_extensions_2026/validate_prelaunch.py --family batch_size --check-squeue
sbatch scripts/lidcover_extensions_2026/launch_batch_size.sbatch

python scripts/lidcover_extensions_2026/validate_prelaunch.py --family tie_fallback --check-squeue
sbatch scripts/lidcover_extensions_2026/launch_tie_fallback.sbatch
```

The optional sparse fixed-radius control is separate:

```bash
python scripts/lidcover_extensions_2026/validate_prelaunch.py --family sparse_fixed_optional --check-squeue
sbatch scripts/lidcover_extensions_2026/launch_sparse_fixed.sbatch
```

After the three essential arrays are complete:

```bash
sbatch scripts/lidcover_extensions_2026/launch_postprocess.sbatch
```

The read-only historical mechanism replay and its dependent summary are:

```bash
MECH_JOB=$(sbatch --parsable scripts/lidcover_extensions_2026/launch_mechanism_replay.sbatch)
sbatch --dependency="afterok:${MECH_JOB}" scripts/lidcover_extensions_2026/launch_mechanism_summary.sbatch
```

No launch script resumes, repairs, renames, or overwrites an existing directory. Strictly complete outputs are skipped; any other existing directory is a hard error.
