#!/usr/bin/env python3
"""Emit the exact run directories flagged as corrupt (same logic as
flag_bad_accuracy.py) so they can be deleted deterministically."""
import glob
import json
import os

ROOT = "/vast/s219110279/TypiClust/output"
ZONE = {
    "CIFAR10":      (5.0, 96.0),
    "CIFAR100":     (0.5, 80.0),
    "TINYIMAGENET": (0.2, 70.0),
}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


dirs = []
for f in sorted(glob.glob(os.path.join(ROOT, "*/*/*/benchmark_summary.json"))):
    parts = f.split("/")
    dataset = parts[-4]
    run_dir = os.path.dirname(f)
    try:
        d = json.load(open(f))
    except Exception:
        dirs.append(run_dir)
        continue
    v = num(d.get("final_val_accuracy"))
    t = num(d.get("final_test_accuracy"))
    lo, hi = ZONE.get(dataset, (0.0, 100.0))
    bad = False
    for val in (v, t):
        if val is None:            # NaN / missing
            bad = True
        elif val > 100.0 + 1e-6:   # impossible
            bad = True
        elif val < 0:              # impossible
            bad = True
        elif val > hi or val < lo: # out of plausible zone
            bad = True
    if bad:
        dirs.append(run_dir)

for d in dirs:
    print(d)
