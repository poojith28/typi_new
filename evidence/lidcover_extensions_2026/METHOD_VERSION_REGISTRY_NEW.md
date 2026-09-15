# New method-version registry

This registry is additive. None of these identifiers appears in the historical routing or historical output namespace.

| Identifier | Versioned behaviour | Frozen reference |
|---|---|---|
| `probcover_auto_radius_v2` | Historical dense ProbCover selection with a label-free, exact-kNN automatic base radius and the historical driver radius schedule | `pycls/al/prob_cover.py:ProbCover` |
| `lidcover_auto_radius_v2` | Historical LIDCover mechanics using the same paired automatic base radius, then the historical inverse-LID radius transformation | `pycls/al/IDProbCover.py:IDProbCover` |
| `probcover_batch_size_v2` | Historical ProbCover behaviour in the new matched batch-size family | `pycls/al/prob_cover.py:ProbCover` |
| `lidcover_batch_size_v2` | Historical LIDCover behaviour in the new matched batch-size family | `pycls/al/IDProbCover.py:IDProbCover` |
| `lidcover_tf_minlid_minlid` | Adaptive LIDCover geometry; minimum-LID positive-gain tie and fallback rules | frozen LIDCover specification |
| `lidcover_tf_minlid_random` | Adaptive LIDCover geometry; minimum-LID tie, run-local seeded-random fallback | frozen LIDCover specification |
| `lidcover_tf_random_minlid` | Adaptive LIDCover geometry; run-local seeded-random tie, minimum-LID fallback | frozen LIDCover specification |
| `lidcover_tf_random_random` | Adaptive LIDCover geometry; run-local seeded-random tie and fallback | frozen LIDCover specification |
| `sparsefixedcover_v1` | Fixed-radius kNN-truncated cover graph, strict radius, explicit self-coverage, deterministic first-index tie/fallback, no LID computation or use | new optional control; **not ProbCover** |

Random tie/fallback streams use `numpy.random.SeedSequence([experiment_seed, selection_event, query_position, decision_stream])`. They never mutate the global NumPy, Torch, model-training, or split RNG.

Registration occurs only at runtime in `experiments/lidcover_extensions_2026/extension_entrypoint.py`. The historical `ActiveLearning.py`, `train_al.py`, `prob_cover.py`, and `IDProbCover.py` files are not edited by this project area.
