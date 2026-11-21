"""
Zahramun — Experiment sweep orchestrator.

Runs the EXPERIMENTS.md matrix: for each (agent x degradation-level x seed) it trains a
model, then evaluates it with scripts/evaluate.py, appending to results/results.csv.

Two experiments:
  A (static) : stability frozen per episode at fixed levels  -> clean degradation axis
  B (dynamic): native passive drain at several rates          -> native setting

This is a long job (hours per full sweep on CPU). Use --dry-run first to print the plan,
--timesteps to shorten training, and trim --agents / --levels / --seeds to scope it down.
Training and evaluation run in the SAME ablation/static/drain condition (train on the
degradation you test on).

Examples:
  python scripts/run_sweep.py --experiment A --dry-run
  python scripts/run_sweep.py --experiment A --agents ppo --seeds 0 1 2 --timesteps 500000
  python scripts/run_sweep.py --experiment B --timesteps 1000000
"""

from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys

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
    p.add_argument("--out", default=os.path.join(ROOT, "results", "results.csv"))
    p.add_argument("--dry-run", action="store_true", help="Print the plan, run nothing")
    return p.parse_args()


def train(agent, seed, ablation, static, drain, timesteps, envs, dry) -> str | None:
    """Train one model. Returns the saved .zip path (parsed from train.py stdout)."""
    cmd = [PY, os.path.join(ROOT, "training", "train.py"),
           "--algo", agent, "--seed", str(seed), "--ablation", ablation,
           "--timesteps", str(timesteps), "--envs", str(envs), "--no-eval",
           "--tag", f"sweep_s{seed}"]
    if static is not None:
        cmd += ["--static", str(static)]
    if drain is not None:
        cmd += ["--drain", str(drain)]

    print("TRAIN:", " ".join(cmd))
    if dry:
        return None
    res = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(res.stdout[-1500:])
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-2000:])
        print(f"  !! train failed for {agent} seed={seed}")
        return None
    m = re.search(r"Model saved:\s*(.+\.zip)", res.stdout)
    if not m:
        print("  !! could not parse model path from train output")
        return None
    return m.group(1).strip()


def evaluate(model_path, agent, seed, ablation, static, drain, episodes, out, dry):
    cmd = [PY, os.path.join(ROOT, "scripts", "evaluate.py"),
           "--model", model_path, "--algo", agent,
           "--agent", AGENT_LABEL[agent], "--ablation", ablation,
           "--episodes", str(episodes), "--train-seed", str(seed), "--out", out]
    if static is not None:
        cmd += ["--static", str(static)]
    if drain is not None:
        cmd += ["--drain", str(drain)]

    print("EVAL :", " ".join(cmd))
    if dry:
        return
    res = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(res.stdout[-800:])
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-2000:])
        print(f"  !! eval failed for {agent} seed={seed}")


def main():
    args = parse_args()
    if args.experiment == "A":
        levels = args.levels or DEFAULT_STATIC_LEVELS
        cells = [(lvl, None) for lvl in levels]          # (static, drain)
    else:
        levels = args.levels or DEFAULT_DRAINS
        cells = [(None, d) for d in levels]

    total = len(args.agents) * len(cells) * len(args.seeds)
    print(f"Sweep {args.experiment}: {len(args.agents)} agents x {len(cells)} levels "
          f"x {len(args.seeds)} seeds = {total} runs | timesteps={args.timesteps}")
    print(f"Results -> {args.out}\n")

    i = 0
    for agent in args.agents:
        for static, drain in cells:
            for seed in args.seeds:
                i += 1
                print(f"--- [{i}/{total}] {agent} static={static} drain={drain} seed={seed} ---")
                model = train(agent, seed, args.ablation, static, drain,
                              args.timesteps, args.envs, args.dry_run)
                if args.dry_run:
                    continue
                if model is None:
                    continue
                evaluate(model, agent, seed, args.ablation, static, drain,
                         args.episodes, args.out, args.dry_run)

    print("\nSweep complete." if not args.dry_run else "\nDry run complete.")


if __name__ == "__main__":
    main()
