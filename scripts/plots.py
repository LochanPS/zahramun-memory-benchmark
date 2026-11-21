"""
Zahramun — Figures from results/results.csv.

Reads the per-episode CSV produced by scripts/evaluate.py and renders:
  Figure 1  win-rate vs degradation level, one line per agent  (the headline plot)
  Figure 3  revisit-count vs degradation level, per agent       (forgetting signature)

Statistics: the seed is the unit of analysis. For each (agent, level, train_seed) we take
the mean over that run's episodes, then report the mean +/- 95% CI across seeds. With few
seeds this is a descriptive CI, not a strong significance claim (see EXPERIMENTS.md).

Usage:
  python scripts/plots.py                       # auto-detect static vs drain axis
  python scripts/plots.py --x drain
  python scripts/plots.py --csv results/results.csv --outdir results/figures
"""

from __future__ import annotations
import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Plot Zahramun results")
    p.add_argument("--csv", default=os.path.join(ROOT, "results", "results.csv"))
    p.add_argument("--outdir", default=os.path.join(ROOT, "results", "figures"))
    p.add_argument("--x", choices=["auto", "static_level", "drain"], default="auto")
    return p.parse_args()


def _f(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pick_x(rows, choice):
    if choice != "auto":
        return choice
    has_static = any(_f(r["static_level"]) is not None for r in rows)
    return "static_level" if has_static else "drain"


def aggregate(rows, xkey, metric):
    """
    Returns {agent: (xs, means, cis)} sorted by x.
    Per-seed mean first (seed = unit of analysis), then mean +/- 95% CI across seeds.
    """
    # (agent, x, seed) -> list of per-episode metric values
    per_run = defaultdict(list)
    for r in rows:
        x = _f(r[xkey])
        if x is None:
            continue
        val = _f(r[metric])
        if val is None:
            continue
        per_run[(r["agent"], x, r["train_seed"])].append(val)

    # (agent, x) -> list of per-seed means
    per_cell = defaultdict(list)
    for (agent, x, _seed), vals in per_run.items():
        per_cell[(agent, x)].append(float(np.mean(vals)))

    out = defaultdict(lambda: ([], [], []))
    agents = sorted({a for (a, _x) in per_cell})
    for agent in agents:
        xs = sorted({x for (a, x) in per_cell if a == agent})
        means, cis = [], []
        for x in xs:
            seed_means = np.asarray(per_cell[(agent, x)], dtype=float)
            m = float(seed_means.mean())
            n = len(seed_means)
            sem = float(seed_means.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            means.append(m)
            cis.append(1.96 * sem)
        out[agent] = (xs, means, cis)
    return out


def plot_metric(agg, xkey, ylabel, title, path, invert_x):
    plt.figure(figsize=(7, 5))
    for agent, (xs, means, cis) in sorted(agg.items()):
        xs = np.asarray(xs); means = np.asarray(means); cis = np.asarray(cis)
        plt.plot(xs, means, marker="o", label=agent)
        plt.fill_between(xs, means - cis, means + cis, alpha=0.2)
    # For static stability, HIGHER value = LESS degradation; flip so "worse" goes right.
    if invert_x:
        plt.gca().invert_xaxis()
        plt.xlabel("memory stability (higher = less degradation) → worse to the right")
    else:
        plt.xlabel("passive drain rate (higher = faster degradation)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  wrote {path}")


def main():
    args = parse_args()
    if not os.path.exists(args.csv):
        raise SystemExit(f"No results CSV at {args.csv} — run the sweep first.")
    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("results.csv is empty.")
    os.makedirs(args.outdir, exist_ok=True)

    xkey = pick_x(rows, args.x)
    invert = (xkey == "static_level")
    print(f"x-axis = {xkey} | {len(rows)} rows")

    win = aggregate(rows, xkey, "win")
    plot_metric(win, xkey, "win rate",
                "Zahramun: win rate vs degradation",
                os.path.join(args.outdir, "fig1_win_rate.png"), invert)

    rev = aggregate(rows, xkey, "revisit")
    plot_metric(rev, xkey, "revisit count (per episode)",
                "Zahramun: forgetting signature (cell revisits)",
                os.path.join(args.outdir, "fig3_revisit.png"), invert)


if __name__ == "__main__":
    main()
