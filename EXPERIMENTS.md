# Zahramun: Experiment Plan

**A parametric POMDP benchmark for agent robustness under structured memory degradation.**

This document is the research protocol that turns the Zahramun engine into reportable
evidence. It is written to be reproducible by a third party (and to be pushed to GitHub
as part of the scholarship portfolio).

---

## 1. Research question

> When an agent's observations degrade in a *structured, controllable* way — stale cells
> decay, values grow noisy, far things blur, positions drift — how much does an internal
> memory mechanism (recurrence, frame-stacking) recover the performance that a memoryless
> agent loses? And at what degradation level does memory stop helping?

This is a **benchmark / empirical-study contribution**, not a new-algorithm claim. The
novelty is the environment: degradation is *parametric and decomposable* (decay vs noise
vs masking vs position drift), so it can be dialed and ablated — which most POMDP
testbeds do not expose.

### Hypotheses (pre-registered)

- **H1.** Memoryless MLP-PPO win-rate falls monotonically as degradation rises; recurrent
  PPO (LSTM) falls more slowly. (Stated already in `configs/ppo.yaml` and
  `recurrent_ppo.yaml`.)
- **H2.** There is a crossover / collapse region: below some degradation level memory
  gives little benefit (perception is good enough); above some higher level even LSTM
  collapses (signal too sparse to integrate). The *interesting* result is locating that
  band.
- **H3.** The degradation *mechanisms are not equivalent*: masking (information removed)
  hurts more than noise (information corrupted) at matched intensity, and position drift
  hurts navigation-dependent sub-tasks (reaching artifact/guardian) more than tile
  masking.
- **H4.** Frame-stacking (k frames) is a weak, cheap substitute for recurrence: it helps
  at low degradation but plateaus because the forget horizon (`full_forget_age=100`)
  exceeds any practical stack depth.

---

## 2. The confound, and how we remove it

**Problem.** Right now `memory_stability` does three jobs at once:

1. **Perception** — low stability degrades the observation (`degradation.py`).
2. **Survival** — `is_collapsed` ends the episode with `memory_collapse_penalty`.
3. **Reward shaping** — `memory_stability_bonus` paid every step.

So a naive "memory agent wins more" result is uninterpretable: did memory help *perceive*,
or just help *not die* / farm the stability bonus? A reviewer will flag this immediately.

**Fix — add an `ablation_mode` to the env** (small change, see §6). Four modes:

| Mode | Obs degraded? | Collapse kills? | Stability reward? | Isolates |
|---|---|---|---|---|
| `full` (native game) | yes | yes | yes | end-to-end task |
| `perception_only` | yes | **no** | **no** | *does memory fix bad perception?* ← main claim |
| `survival_only` | **no** (full obs) | yes | yes | pure resource management |
| `none` (control) | no | no | no | upper bound / sanity |

The headline experiment runs in **`perception_only`**. That cleanly answers H1–H3 without
the survival/reward confound. `full` is reported as the "native task" for context.

**Second cleanliness lever — static vs dynamic degradation.**

- **Static (Experiment A, cleanest):** fix the degradation level for the whole episode
  (freeze `memory_stability` at a set value, disable drain). Degradation level is now a
  clean independent variable. This is the plot reviewers trust.
- **Dynamic (Experiment B, native):** the drought drains over time (`--drain`). More
  realistic, messier. Reported as the applied / native setting.

---

## 3. Experimental matrix

### Agents (independent variable 1)

| ID | Policy | Memory | Config |
|---|---|---|---|
| `mlp` | MlpPolicy PPO | none | `ppo.yaml` |
| `lstm` | MlpLstmPolicy RecurrentPPO | recurrent | `recurrent_ppo.yaml` |
| `stack4` | PPO + VecFrameStack(4) | short window | `frame_stack_ppo.yaml` |

Optional 4th: `stack8` (k=8) to test H4 plateau.

### Degradation level (independent variable 2)

- **Experiment A (static):** frozen stability ∈ {90, 70, 50, 30, 15}
  → maps to degradation levels NONE / MINOR / MODERATE / HEAVY / CRITICAL.
- **Experiment B (dynamic):** `--drain` ∈ {0.05, 0.10, 0.15, 0.20}.

### Mechanism ablation (independent variable 3, for H3) — run in `perception_only`, static MODERATE

Toggle one mechanism at a time (set the others off in `memory_drought`):

- decay only (`noise_std_*`=0, masking off)
- noise only (`decay_rate`=0, masking off)
- masking only (`full_forget_age` short, noise=0, decay=0)
- position-drift only
- all (native)

### Seeds

**5 seeds** per cell (minimum 3 if compute-bound). Seeds fixed and listed: `{0,1,2,3,4}`.
Report mean ± 95% CI (or ±1 std).

### Total run count

- Exp A: 3 agents × 5 levels × 5 seeds = **75 runs**
- Exp B: 3 agents × 4 drains × 5 seeds = **60 runs**
- Exp C (mechanism): 3 agents × 5 mechanisms × 5 seeds = **75 runs**

That is a lot on CPU. **Scope control:** start with Exp A at 3 seeds (45 runs) — that
alone proves H1/H2 and is enough for the portfolio. Add Exp C for H3 if time allows.
Exp B is "nice to have."

---

## 4. Metrics

Primary (already logged by `WinRateCallback`):

- **win_rate** (episode reaches VICTORY) — headline.
- **artifact_rate**, **guardian_rate** — sub-goal completion.
- **memory_collapse_rate** — only meaningful in modes where collapse is on.

Secondary (add to eval harness):

- **episode_return** (mean ± CI).
- **steps_to_victory** (efficiency; conditional on win).
- **revisit_count** — times agent re-enters an already-seen cell. *This is the direct
  behavioral signature of forgetting* and ties straight back to the project's origin
  ("agent forgets where it has been"). Worth adding; strong figure.
- **coverage** — fraction of reachable map explored.

Evaluation protocol: **100 episodes per trained model**, on a *held-out seed range*
(train seeds `0–999`, eval seeds `5000–5099`), deterministic=False, no gradient updates.

---

## 5. Analysis & figures (the deliverables)

1. **Figure 1 — Degradation curve (money plot).** x = degradation level (static),
   y = win_rate, one line per agent, shaded CI. Shows H1 (lines diverge) and H2 (where
   they cross / collapse). *This single plot is the paper.*
2. **Figure 2 — Mechanism bars.** Grouped bar: win_rate drop per mechanism (decay / noise
   / mask / drift) per agent. Tests H3.
3. **Figure 3 — Forgetting signature.** revisit_count vs degradation, per agent. Links
   quantitative result to the intuitive story.
4. **Figure 4 — Learning curves.** TensorBoard win_rate vs steps, to show training was
   stable / converged (defends "did you train long enough?").
5. **Table 1 — Full results** (win/artifact/guardian/return ± CI) for every cell.

Stats: report CIs; for the headline MLP-vs-LSTM gap at each level, a simple
bootstrap or Mann–Whitney over per-seed means is enough (don't overclaim significance
with 3–5 seeds).

---

## 6. Code changes required (minimal)

Engine is ~done. Needed additions:

1. **`ablation_mode` in `ZahramunEnv`** (§2). ~30 lines: gate the collapse termination,
   the stability reward, and the obs-degradation call on the mode. Add a `static_level`
   option that freezes stability and disables drain.
2. **Mechanism toggles** already mostly expressible via `memory_drought` config
   (set `noise_std_*`, `decay_rate`, `full_forget_age`); add a `position_drift: on/off`
   flag for clean isolation.
3. **`revisit_count` / `coverage`** counters in `GameState` → surfaced in `info`.
4. **Batch eval harness `scripts/evaluate.py`** — does NOT exist yet. Loads a model,
   runs N eval episodes, writes one CSV row per (agent, level, seed) with all metrics.
   `watch.py` is visual only; `WinRateCallback` only logs to TensorBoard.
5. **`scripts/run_sweep.py`** — loops the matrix, calls `train.py` then `evaluate.py`,
   writes `results/results.csv`.
6. **`scripts/plots.py`** — reads `results/results.csv`, emits Figures 1–3 as PNG/PDF.
7. **Pin versions** — freeze `requirements.txt` (sb3, sb3-contrib, gymnasium, torch,
   numpy) so results reproduce. Record git commit hash in each run's `meta.json`.

No new ML algorithm. No new architecture. All contribution is in the environment +
rigorous evaluation.

---

## 7. Compute & time budget

- MLP-PPO 1M steps, 4 envs, CPU: order ~1–2 h/run (per train.py comment, MLP is CPU-fine).
- LSTM slower: ~2–4 h/run.
- Frame-stack ≈ MLP.

Exp A at 3 seeds (45 runs) ≈ a few days of wall-clock if run serially; much less if the
machine runs them back-to-back overnight or several in parallel. **Reduce `total_timesteps`
to 500k for the sweep if curves have already plateaued by then** (check Figure 4 first on
one run). GPU gives little for these small nets — don't block on it.

---

## 8. Timeline (to Jan deadlines: DEAI ~11 Jan, EDISS ~12 Jan)

| Dates | Task |
|---|---|
| Nov 22–26 | Add `ablation_mode`, counters, `evaluate.py`, `run_sweep.py`. Smoke-test 1 run end-to-end. Freeze `requirements.txt`. |
| Nov 27 – Dec 5 | Run Exp A (static, 3 agents × 5 levels × 3–5 seeds). Monitor TensorBoard. |
| Dec 6–10 | `plots.py`, Figures 1 & 4, Table 1. Confirm H1/H2 hold. |
| Dec 11–18 | Run Exp C (mechanisms) if H1/H2 clean. Figures 2–3. |
| Dec 19–28 | Write technical report (4–6 pp: question, env, method, results, limitations). Clean README + reproduce instructions. Push public GitHub repo. |
| Dec 29 – Jan 8 | Fold into SOPs: one paragraph per program (ECE → signals/bounded-memory → POMDP memory benchmark). Request recommendation letters referencing the project. |
| Jan 11–12 | Submit DEAI, then EDISS. |

GATE prep continues in parallel; this is evenings/weekends, not a full-time pivot.

---

## 9. Honest limitations (put these IN the report)

- Degradation is a **hand-designed heuristic**, not learned or empirically derived from a
  real sensor failure model. Framed as a controllable benchmark knob — that is the point,
  but state it.
- Single environment family; results may not transfer to other POMDPs.
- Small seed count → CIs, not strong significance claims.
- LSTM/frame-stack are standard baselines; no claim of a novel memory architecture.
- Reward shaping (`memory_stability_bonus`, step penalty) is tuned, not ablated for
  sensitivity (optional future work).

Stating limits honestly is a *strength* in a research application. Do not hide them.

---

## 10. Program / researcher fit (why this lands)

- **EDISS** — intelligent systems, edge/efficient AI. Bounded-memory framing (agent must
  act under a memory *budget*) is a direct hook. Mention Åbo Akademi embedded/CV, SCOPIA
  vision.
- **Tampere (AISS 3rd sem / general)** — Iosifidis (COIN, RL + vision), Peltonen
  (probabilistic / high-dim ML), Gabbouj (SAMI, signal + pattern). The decay/noise
  degradation is literally a signal-corruption model → ECE-native angle for SAMI.
- **DEAI** — ML + data systems; the clean benchmark + reproducible evaluation pipeline
  demonstrates the engineering rigor DEAI scores for. Pair with transcript CS-credit map.

SOP throughline: *ECE (signals, probability, bounded hardware) → curiosity about agents
that fail when they forget → built a parametric POMDP benchmark that measures exactly that
→ want to extend it with [group]'s work on [memory / efficient / probabilistic ML].*
