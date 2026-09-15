# TopoCover and MultiScaleTopoCover findings

TopoCover and MultiScaleTopoCover test whether graph connectivity in a frozen representation can be used as an active-learning acquisition signal. The post-hoc analysis does not attribute an $H_0$ event intrinsically to one stored birth vertex; its primary definition uses either endpoint of the corresponding MST death edge.

The principal comparison uses the preselected MultiScaleTopoCover configuration $\alpha=0.7$, $\delta_c=0.70$, $\delta_f=0.50$, and acquisition batch size 50 with a ResNet-18 backbone. No dataset-specific configuration was silently substituted.

## Main evidence

- CIFAR10: among protocol-matched methods present, Margin had the highest mean final test accuracy (75.66%, SEM 0.39, n=5). This is a configuration-specific result, not evidence of universal superiority.
- CIFAR100: among protocol-matched methods present, MSTC had the highest mean final test accuracy (36.76%, SEM 0.22, n=5). This is a configuration-specific result, not evidence of universal superiority.
- TINYIMAGENET: among protocol-matched methods present, ProbCover had the highest mean final test accuracy (13.87%, SEM 0.12, n=5). This is a configuration-specific result, not evidence of universal superiority.

Multiscale effects are dataset- and scale-dependent. Single-seed topology variants remain exploratory and are not used for definitive ranking. All means exclude incomplete runs.

The original birth-vertex touched-event interpretation fails the invariance audit. At episode 100, birth touched count is effectively selected-set size (median ratio 1.000), and birth-versus-either-endpoint touched-fraction rankings are unstable (median Spearman $\rho=0.202$). Mean death and normalized $\beta_0$ AUC are much more stable under the symmetric endpoint definition (median $\rho=0.977$ and $0.976$, respectively), but must be presented with size- and MST-degree-matched null controls.

Historical accuracy comparisons use matched cumulative label budgets and retain seed-level dispersion, but their evaluation crops were stochastic. They remain descriptive until deterministic-transform reruns complete.
