"""
Zahramun — Experiment sweep orchestrator.

Runs the EXPERIMENTS.md matrix: for each (agent x degradation-level x seed) it trains a
model, then evaluates it with scripts/evaluate.py. With --jobs > 1 the cells run in
parallel (one subprocess chain each), which is the way to use a many-core machine. Each
cell writes its own partial CSV; they are merged into results/results.csv at the end so
parallel writers never corrupt a shared file.

Two experiments:
  A (static) : stability frozen per episode at fixed levels  -> clean degradation axis
  B (dynamic): native passive drain at several rates          -> native setting

Training and evaluation run in the SAME ablation/static/drain condition.

Examples:
  python scripts/run_sweep.py --experiment A --dry-run
  python scripts/run_sweep.py --experiment A --jobs 6 --seeds 0 1 2 --timesteps 300000
  python scripts/run_sweep.py --experiment B --jobs 4 --timesteps 500000
"""

from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

AGENT_LABEL = {"ppo": "mlp", "recurrent_ppo": "lstm", "frame_stack_ppo": "stack4"}

DEFAULT_STATIC_LEVELS = [90, 70, 50, 30, 15]     # NONE..CRITICAL
DEFAULT_DRAINS        = [0.05, 0.10, 0.15, 0.20]


def parse_args():
    p = argparse.ArgumentParser(description="Run the Zahramun experiment sweep")
    p.add_argument("--experiment", choices=["A", "B"], default="A",
                   help="A = static degradation levels, B = dynamic drain rates")
    p.add_argument("--agents", nargs="+",
                   default=["ppo", "recurrent_ppo", "frame_stack_ppo"],
                   choices=["ppo", "recurrent_ppo", "frame_stack_ppo"])
    p.add_argument("--levels", nargs="+", type=float, default=None,
                   help="Override static levels (A) or drain rates (B)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--ablation", default="perception_only",
                   choices=["full", "perception_only", "survival_only", "none"])
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--jobs", type=int, default=1,
                   help="Number of cells to train+evaluate in parallel")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "results.csv"))
    p.add_argument("--dry-run", action="store_true", help="Print the plan, run nothing")
    return p.parse_args()


def cell_tag(agent, static, drain, seed) -> str:
    cond = f"static{static:g}" if static is not None else f"drain{drain:g}"
    return f"{agent}_{cond}_s{seed}"


def train(agent, seed, ablation, static, drain, timesteps, envs, tag) -> str | None:
    """Train one model. Returns the saved .zip path (parsed from train.py stdout)."""
    cmd = [PY, os.path.join(ROOT, "training", "train.py"),
           "--algo", agent, "--seed", str(seed), "--ablation", ablation,
           "--timesteps", str(timesteps), "--envs", str(envs), "--no-eval",
           "--tag", f"sweep_{tag}"]
    if static is not None:
        cmd += ["--static", str(static)]
    if drain is not None:
        cmd += ["--drain", str(drain)]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-2000:])
        print(f"  !! train failed: {tag}")
        return None
    m = re.search(r"Model saved:\s*(.+\.zip)", res.stdout)
    if not m:
        print(f"  !! could not parse model path: {tag}")
        return None
    return m.group(1).strip()


def evaluate(model_path, agent, seed, ablation, static, drain, episodes, out) -> bool:
    cmd = [PY, os.path.join(ROOT, "scripts", "evaluate.py"),
           "--model", model_path, "--algo", agent,
           "--agent", AGENT_LABEL[agent], "--ablation", ablation,
           "--episodes", str(episodes), "--train-seed", str(seed), "--out", out]
    if static is not None:
        cmd += ["--static", str(static)]
    if drain is not None:
        cmd += ["--drain", str(drain)]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-2000:])
        print(f"  !! eval failed for {model_path}")
        return False
    return True


def run_cell(args, agent, static, drain, seed, partial_dir):
    tag = cell_tag(agent, static, drain, seed)
    out = os.path.join(partial_dir, tag + ".csv")
    if os.path.exists(out):
        os.remove(out)                       # fresh — avoid double-append on rerun
    model = train(agent, seed, args.ablation, static, drain,
                  args.timesteps, args.envs, tag)
    if model is None:
        return (tag, "train-failed")
    ok = evaluate(model, agent, seed, args.ablation, static, drain,
                  args.episodes, out)
    return (tag, "ok" if ok else "eval-failed")


def merge_partials(partial_dir, out):
    parts = sorted(glob.glob(os.path.join(partial_dir, "*.csv")))
    if not parts:
        print("  no partial CSVs to merge")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fout:
        header_written = False
        for pth in parts:
            with open(pth) as fin:
                lines = fin.readlines()
            if not lines:
                continue
            if not header_written:
                fout.write(lines[0]); header_written = True
            fout.writelines(lines[1:])
    print(f"  merged {len(parts)} partial files -> {out}")


def main():
    args = parse_args()
    if args.experiment == "A":
        levels = args.levels or DEFAULT_STATIC_LEVELS
        cond_cells = [(lvl, None) for lvl in levels]      # (static, drain)
    else:
        levels = args.levels or DEFAULT_DRAINS
        cond_cells = [(None, d) for d in levels]

    cells = [(agent, static, drain, seed)
             for agent in args.agents
             for (static, drain) in cond_cells
             for seed in args.seeds]

    print(f"Sweep {args.experiment}: {len(args.agents)} agents x {len(cond_cells)} levels "
          f"x {len(args.seeds)} seeds = {len(cells)} runs | timesteps={args.timesteps} "
          f"| jobs={args.jobs}")
    print(f"Results -> {args.out}\n")

    if args.dry_run:
        for i, (agent, static, drain, seed) in enumerate(cells, 1):
            print(f"  [{i}/{len(cells)}] {cell_tag(agent, static, drain, seed)}")
        print("\nDry run complete.")
        return

    partial_dir = os.path.join(os.path.dirname(args.out), "partial")
    os.makedirs(partial_dir, exist_ok=True)

    done = 0
    if args.jobs <= 1:
        for cell in cells:
            tag, status = run_cell(args, *cell, partial_dir)
            done += 1
            print(f"  [{done}/{len(cells)}] {tag}: {status}")
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_cell, args, *cell, partial_dir): cell for cell in cells}
            for fut in as_completed(futs):
                tag, status = fut.result()
                done += 1
                print(f"  [{done}/{len(cells)}] {tag}: {status}")

    merge_partials(partial_dir, args.out)
    print("\nSweep complete.")


if __name__ == "__main__":
    main()
