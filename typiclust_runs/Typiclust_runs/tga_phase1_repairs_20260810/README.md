# TGA Phase-1 core-matrix repairs

This campaign contains only the four TinyImageNet endpoint-control runs found PARTIAL by `analysis/audit_tga_evidence.py` on 2026-08-10.

- Each array task is one unique dataset × method configuration × seed run.
- The Slurm array is `0-3%20`; collective workflow concurrency remains capped at 20.
- All targets use new `AUDIT_20260810_...` experiment identifiers.
- Existing partial directories listed in `jobs.tsv` are preserved and never reused.
- Fine-only rows report only the fine radius scientifically; coarse-only rows report only the coarse radius scientifically. The paired inactive radius is supplied only because the immutable MultiScaleTopoCover constructor builds both graphs.
- The 15 alpha-ablation jobs remain gated until all four runs complete and a fresh audit passes.
- Exit 75 denotes the known unsupported-GPU infrastructure condition. Retry at most once, with another unique experiment identifier and only after re-audit.

## Failure record

Original array `3473736` was accepted on 2026-08-10 and all four tasks failed before training because the wrapper enabled Bash nounset before Conda activation. Conda's MKL hook referenced unset `MKL_INTERFACE_LAYER`. No target run directory was created. The wrapper now enables nounset after Conda activation. `submit_phase1_retry1.sh` permits exactly one retry and preserves the original job/logs.

Submission is forbidden unless `squeue` first succeeds and confirms that none of these experiment IDs is COMPLETE, RUNNING, PENDING, or already submitted.
