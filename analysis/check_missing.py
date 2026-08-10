#!/usr/bin/env python3
import os
import re
from collections import defaultdict

ROOT = "/vast/s219110279/TypiClust/output"
DATASETS = ["CIFAR10", "CIFAR100", "TINYIMAGENET"]
BACKBONES = ["alexnet", "resnet18", "resnet50"]
SEEDS = [1, 2, 3, 4, 5]

# Canonical "main" strategies (base algorithms), longest names first so
# multi-word names match before short ones.
MAIN_STRATEGIES = [
    "distance_variance_cover",
    "distance_cv_cover",
    "knn_distance_cover",
    "density_cover",
    "coreset",
    "dbal",
    "entropy",
    "margin",
    "uncertainty",
    "random",
    "typiclust",
    "maxherding",
    "probcover",
    "LIDCOVER",
    "TOPOCOVER",
]

# dir name pattern: <strategy>_<seed>_<budget>b  (strategy may itself contain _)
# We only treat a dir as a "main" run if, after removing _<seed>_<budget>b,
# the remaining string EXACTLY equals a main strategy (no extra ablation token).
run_re = re.compile(r"^(?P<strat>.+)_(?P<seed>\d+)_(?P<budget>\d+)b$")

# present[(dataset, backbone, strategy, budget)] = set(seeds)
present = defaultdict(set)
variant_dirs = []

for ds in DATASETS:
    for bb in BACKBONES:
        d = os.path.join(ROOT, ds, bb)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            full = os.path.join(d, name)
            if not os.path.isdir(full):
                continue
            m = run_re.match(name)
            if not m:
                continue
            strat = m.group("strat")
            seed = int(m.group("seed"))
            budget = m.group("budget") + "b"
            if "partial" in name:
                continue
            if strat in MAIN_STRATEGIES:
                present[(ds, bb, strat, budget)].add(seed)
            else:
                variant_dirs.append((ds, bb, name))

# Report missing for main strategies at 50b grid
print("=" * 70)
print("MAIN GRID: 15 strategies x 3 datasets x 3 backbones x 5 seeds @ 50b")
print("=" * 70)
total_cells = 0
complete_cells = 0
missing = []
for ds in DATASETS:
    for bb in BACKBONES:
        for strat in MAIN_STRATEGIES:
            total_cells += 1
            seeds_have = present.get((ds, bb, strat, "50b"), set())
            need = set(SEEDS)
            miss = sorted(need - seeds_have)
            if not miss:
                complete_cells += 1
            else:
                missing.append((ds, bb, strat, miss, sorted(seeds_have)))

print(f"\nComplete cells (all 5 seeds): {complete_cells}/{total_cells}")
print(f"Incomplete/missing cells: {len(missing)}\n")
for ds, bb, strat, miss, have in missing:
    if not have:
        print(f"  [NONE]    {ds}/{bb}/{strat:<24} missing seeds {miss} (have none)")
    else:
        print(f"  [PARTIAL] {ds}/{bb}/{strat:<24} missing seeds {miss} (have {have})")

# Extra budgets summary (100b, 500b)
print("\n" + "=" * 70)
print("EXTRA BUDGET RUNS (100b / 500b) present:")
print("=" * 70)
for budget in ["100b", "500b"]:
    rows = sorted([(ds, bb, strat, sorted(s)) for (ds, bb, strat, b), s in present.items() if b == budget])
    print(f"\n-- {budget} --")
    for ds, bb, strat, s in rows:
        print(f"  {ds}/{bb}/{strat:<20} seeds {s}")
