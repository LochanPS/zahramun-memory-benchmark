# Zahramun

**A parametric POMDP benchmark for studying agent robustness under structured memory degradation.**

Zahramun is a Gymnasium reinforcement-learning environment in which an agent must
explore a procedurally-generated dungeon, collect an artifact, and defeat a guardian —
while its observations **progressively degrade**. Stale cells decay, remembered values
grow noisy, far-away things blur, and tracked entity positions drift. The agent carries a
depletable *Memory Stability* resource ("Memory Drought") that governs how badly its view
of the world is corrupted.

The research question: **how much does an internal memory mechanism (recurrence,
frame-stacking) recover the performance a memoryless agent loses as observations degrade —
and at what degradation level does memory stop helping?**

Unlike most POMDP testbeds, the degradation here is *parametric and decomposable* —
temporal decay, noise, masking and position-drift can each be dialed and ablated
independently. That makes Zahramun a controllable instrument for probing memory in RL,
not just another dungeon crawler.

> Lineage: the "Memory Drought" mechanic comes from an original fiction setting (Zahramun,
> a desert moon whose inhabitants' memories erode). This repository is the technical
> realization of that idea as an RL benchmark.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Quick start

```bash
# Sanity-check the environment
python -m zahramun.tests.test_env_sanity

# Train the memoryless baseline (MLP-PPO)
python training/train.py --algo ppo

# Train the recurrent baseline (LSTM)
python training/train.py --algo recurrent_ppo

# Increase the memory-drought drain rate
python training/train.py --algo ppo --drain 0.15

# Watch a trained agent play (pygame)
python scripts/watch.py --model saves/<run>/final_model.zip
```

## The Memory Drought

Observation degradation is applied in `zahramun/env/memory/degradation.py` as a pipeline:

1. **Temporal decay** — confidence in a cell drops with steps since it was last seen.
2. **Noise injection** — Gaussian noise, scaled by degradation level.
3. **Masking** — cells past a forget horizon become an `UNKNOWN` sentinel; moderately
   degraded far cells become `COARSE_FAR` ("something is there, unsure what").
4. **Position drift** — tracked entity/landmark positions acquire increasing error.

Degradation intensifies across five levels (NONE → CRITICAL) as Memory Stability drains.

## Baselines

| Policy | Memory | Config |
|---|---|---|
| MLP-PPO | none (feedforward) | `training/configs/ppo.yaml` |
| RecurrentPPO | LSTM | `training/configs/recurrent_ppo.yaml` |
| Frame-stack PPO | short window | `training/configs/frame_stack_ppo.yaml` |

## Experiments

The full research protocol — hypotheses, the confound fix, the run matrix, metrics and
figures — is in [`EXPERIMENTS.md`](EXPERIMENTS.md). Results and figures will be added here
as the sweep completes.

## Evaluation & sweep

```bash
# Batch-evaluate a trained model -> results/results.csv
python scripts/evaluate.py --model saves/<run>/final_model.zip --algo ppo \
    --agent mlp --ablation perception_only --static 50 --episodes 100

# Run the full matrix (train + evaluate every cell). --dry-run prints the plan.
python scripts/run_sweep.py --experiment A --dry-run
python scripts/run_sweep.py --experiment A --seeds 0 1 2 --timesteps 500000

# Render figures from results/results.csv -> results/figures/
python scripts/plots.py
```

### Ablation modes

`memory_stability` is decoupled into independent jobs so results are interpretable
(`--ablation`): `full` (native), `perception_only` (degrade obs only — the headline
condition), `survival_only` (full obs, collapse still kills), `none` (control). `--static N`
freezes stability at `N` for a clean per-episode degradation level.

## Status

Environment, baselines, training loop, ablation modes, batch evaluation, sweep
orchestration and plotting are implemented and smoke-tested. Next: run the sweep and add
results/figures (see [`EXPERIMENTS.md`](EXPERIMENTS.md)).

## Limitations

Degradation is a hand-designed heuristic (a controllable benchmark knob, not an empirical
sensor-failure model). Single environment family. Standard baselines — no novel memory
architecture is claimed.

## License

MIT (see `LICENSE`).
