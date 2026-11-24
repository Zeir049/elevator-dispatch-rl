"""
Aggregate per-checkpoint evaluate.py CSVs (runs/<variant>_s<seed>_eval.csv)
into one table: mean and std across *training* seeds of each metric, where
each training seed's value is itself the average over the 20 evaluation seeds.

The heuristic baselines do not depend on the training seed (random uses a
fixed rng), so they appear once with std = 0.

    python aggregate_seeds.py --runs runs --out compare_multiseed.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re

import numpy as np

METRICS = ["mean_wait", "p95_wait", "max_wait", "served", "unserved"]
LABELS = {"enh": "dqn_double_dueling_per", "van": "dqn_vanilla"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="compare_multiseed.csv")
    args = ap.parse_args()

    per_variant: dict[str, list[dict]] = {}
    baselines: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(args.runs, "*_s*_eval.csv"))):
        m = re.match(r"(\w+?)_s(\d+)_eval\.csv", os.path.basename(path))
        variant, seed = m.group(1), int(m.group(2))
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                vals = {k: float(row[k]) for k in METRICS}
                if row["policy"] == "dqn":
                    per_variant.setdefault(variant, []).append(dict(seed=seed, **vals))
                else:
                    baselines[row["policy"]] = vals  # identical in every file

    header = ["policy", "n_train_seeds", "train_seeds"]
    for k in METRICS:
        header += [f"{k}_mean", f"{k}_std"]
    rows = []
    for name, vals in baselines.items():
        r = [name, 0, ""]
        for k in METRICS:
            r += [vals[k], 0.0]
        rows.append(r)
    for variant, runs in per_variant.items():
        runs.sort(key=lambda d: d["seed"])
        r = [LABELS.get(variant, variant), len(runs),
             " ".join(str(d["seed"]) for d in runs)]
        for k in METRICS:
            x = np.array([d[k] for d in runs])
            r += [x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0]
        rows.append(r)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"{'policy':<24}{'n':>3}  {'mean wait':>14}  {'p95 wait':>14}  "
          f"{'served':>14}  {'unserved':>12}")
    for r in rows:
        d = dict(zip(header, r))
        print(f"{d['policy']:<24}{d['n_train_seeds']:>3}  "
              + "  ".join(f"{d[k+'_mean']:7.1f}+-{d[k+'_std']:5.1f}"
                          for k in ["mean_wait", "p95_wait", "served"])
              + f"  {d['unserved_mean']:6.2f}+-{d['unserved_std']:4.2f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
