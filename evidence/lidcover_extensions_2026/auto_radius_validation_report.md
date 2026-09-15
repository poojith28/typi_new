# Automatic-radius pre-launch validation

Status: **PASS for configuration/mapping; experiments not launched.**

- Matrix size: 3 datasets × 2 new method IDs × 5 AL seeds = 30 GPU tasks.
- Every array index 0–29 maps to exactly one unique experiment ID and output directory.
- Candidate indices reproduce the repository split convention: a seed-specific shuffle of training indices, with the final 10% held out and excluded from calibration/acquisition.
- The provenance manifest records the representation path/SHA256 and candidate/validation index SHA256 before launch.
- Runtime calibration loads candidate feature rows only, applies rowwise L2 normalisation, performs exact non-self kNN with k=50, and takes the median 50th-neighbour Euclidean distance.
- No label file, validation accuracy, or test accuracy is read by calibration.
- Each matched ProbCover/LIDCover pair independently verifies the same representation and candidate-index hashes; post-processing refuses a pair whose delta or hashes differ.
- The base radius is resolved before cold-start acquisition. Effective radii are 1.35× at the empty cold start and 0.85× thereafter.
- Existing output directories are never reused. A complete run is skipped only after strict validation; any incomplete directory causes a hard failure.
- Scheduler PENDING/RUNNING exclusion must be checked on the login node with `validate_prelaunch.py --family auto_radius --check-squeue` immediately before submission.

The `raw_delta_auto` fields in `auto_radius_provenance.csv` deliberately remain `PENDING_RUN`; no result is fabricated before geometry is computed by the job.
