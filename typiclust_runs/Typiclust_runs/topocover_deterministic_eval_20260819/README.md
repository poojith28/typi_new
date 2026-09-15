# TopoCover deterministic-evaluation rerun

This immutable campaign regenerates the historical main TopoCover matrix after
the validation/test transform repair. It preserves the cold-start pool, ordered
pre-acquisition pools, selected IDs, stopping reason, and index hashes.

- 3 datasets x 3 backbones x 5 seeds = 45 runs
- delta = 0.70, k = 50, acquisition batch = 50
- one experiment per Slurm array task, collective cap 20
- new experiment IDs; historical aggregate evidence is never overwritten
