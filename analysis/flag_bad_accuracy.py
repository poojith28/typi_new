#!/usr/bin/env python3
"""Scan all benchmark_summary.json runs and flag accuracies that are
impossible or implausibly out of the expected zone."""
import glob
import json
import os
from collections import defaultdict

ROOT = "/vast/s219110279/TypiClust/output"

# Rough plausible upper bounds for FINAL accuracy (%) after a full 50-budget
# AL run. Used only to flag "out of zone" values, not to grade methods.
# Lower bound ~ random-chance floor; upper bound ~ generous ceiling.
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


rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "*/*/*/benchmark_summary.json"))):
    parts = f.split("/")
    dataset, backbone, run = parts[-4], parts[-3], parts[-2]
    try:
        d = json.load(open(f))
    except Exception as e:
        rows.append((dataset, backbone, run, None, None, f"UNREADABLE: {e}"))
        continue
    v = num(d.get("final_val_accuracy"))
    t = num(d.get("final_test_accuracy"))
    lo, hi = ZONE.get(dataset, (0.0, 100.0))

    flags = []
    for name, val in (("val", v), ("test", t)):
        if val is None:
            flags.append(f"{name}=NaN/missing")
        elif val > 100.0 + 1e-6:
            flags.append(f"{name}={val:.4g} >100 IMPOSSIBLE")
        elif val < 0:
            flags.append(f"{name}={val:.4g} <0 IMPOSSIBLE")
        elif val > hi:
            flags.append(f"{name}={val:.4g} > zone_hi({hi})")
        elif val < lo:
            flags.append(f"{name}={val:.4g} < zone_lo({lo})")
    # test vs val wildly inconsistent
    if v is not None and t is not None and abs(v - t) > 25 and t <= 100 and v <= 100:
        flags.append(f"|val-test|={abs(v-t):.1f} large gap")

    if flags:
        rows.append((dataset, backbone, run, v, t, "; ".join(flags)))

# categorize
impossible = [r for r in rows if "IMPOSSIBLE" in r[5] or "NaN" in r[5] or "UNREADABLE" in r[5]]
outzone = [r for r in rows if r not in impossible]

print("=" * 90)
print(f"IMPOSSIBLE / CORRUPT accuracies ({len(impossible)})")
print("=" * 90)
for ds, bb, run, v, t, fl in impossible:
    print(f"  {ds}/{bb}/{run:<40} {fl}")

print("\n" + "=" * 90)
print(f"OUT-OF-ZONE (suspicious but not strictly impossible) ({len(outzone)})")
print("=" * 90)
for ds, bb, run, v, t, fl in outzone:
    print(f"  {ds}/{bb}/{run:<40} val={v} test={t} | {fl}")

print(f"\nTotal flagged: {len(rows)}")
