"""
Zahramun — Watch a trained RL agent play.

Loads a saved model and renders it playing in the pygame window.
Press SPACE to pause/unpause, R to restart episode, ESC to quit.

Usage:
  python scripts/watch.py --model saves/<run>/final_model.zip
  python scripts/watch.py --model saves/<run>/final_model.zip --algo recurrent_ppo
  python scripts/watch.py --model saves/<run>/final_model.zip --speed 2
"""

from __future__ import annotations
import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame
import numpy as np

from zahramun.env.zahramun_env import ZahramunEnv
from zahramun.env.world.game_state import GamePhase
from zahramun.env.rendering.pygame_renderer import PygameRenderer


def parse_args():
    p = argparse.ArgumentParser(description="Watch trained Zahramun agent")
    p.add_argument("--model",  required=True, help="Path to .zip model file")
    p.add_argument("--algo",   default="ppo",
                   choices=["ppo", "recurrent_ppo", "frame_stack_ppo"],
                   help="Algorithm used to train the model")
    p.add_argument("--speed",  type=float, default=1.0,
                   help="Playback speed multiplier (1=normal, 2=2x fast, 0.5=slow)")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--windowed", action="store_true", help="Run windowed (not fullscreen)")
    return p.parse_args()


def load_model(model_path: str, algo: str, env):
    if algo == "recurrent_ppo":
        try:
            from sb3_contrib import RecurrentPPO
            return RecurrentPPO.load(model_path, env=env)
        except ImportError:
            print("ERROR: sb3-contrib not installed. Run: pip install sb3-contrib")
            sys.exit(1)
    else:
        from stable_baselines3 import PPO
        return PPO.load(model_path, env=env)


def main():
    args  = parse_args()
    seed  = args.seed
    speed = max(0.1, args.speed)

    # Create env
    env = ZahramunEnv(seed=seed)
    obs, _ = env.reset(seed=seed)

    # Load model
    print(f"Loading model: {args.model}")
    model = load_model(args.model, args.algo, env)
    print(f"Model loaded. Algorithm: {args.algo}")

    # Renderer
    fullscreen = not args.windowed
    env._renderer = PygameRenderer(env.map_w, env.map_h, env.cfg, fullscreen=fullscreen)
    print("Controls: SPACE=pause  R=restart  T=truth  F11=fullscreen  ESC=quit")

    # LSTM state (RecurrentPPO only)
    lstm_state   = None
    episode_done = False
    paused       = False
    episode_num  = 1
    ep_reward    = 0.0
    step_delay   = (1.0 / 30.0) / speed    # seconds per step

    # Stats
    ep_rewards: list[float] = []
    ep_wins:    int = 0

    running = True
    while running:
        # ── Render ────────────────────────────────────────────────────────────
        if env.state and env._renderer:
            env._renderer.render(env.state, env.mem)

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                    print("PAUSED" if paused else "RESUMED")
                elif k == pygame.K_t:
                    if env._renderer:
                        env._renderer._show_truth = not env._renderer._show_truth
                elif k == pygame.K_F11:
                    if env._renderer:
                        env._renderer.toggle_fullscreen()
                elif k == pygame.K_r:
                    seed += 1
                    episode_num += 1
                    obs, _ = env.reset(seed=seed)
                    lstm_state   = None
                    episode_done = False
                    ep_reward    = 0.0

        if paused or episode_done:
            time.sleep(0.05)
            continue

        # ── Agent step ────────────────────────────────────────────────────────
        if args.algo == "recurrent_ppo":
            action, lstm_state = model.predict(
                obs, state=lstm_state, deterministic=False)
        else:
            action, _ = model.predict(obs, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(int(action))
        ep_reward += reward

        if terminated or truncated:
            episode_done = True
            ep_rewards.append(ep_reward)
            phase = env.state.phase
            won   = phase == GamePhase.VICTORY
            if won:
                ep_wins += 1
            print(
                f"  Ep {episode_num:3d} | "
                f"{'WIN' if won else 'FAIL':4s} | "
                f"reward={ep_reward:7.1f} | "
                f"step={info['step']:4d} | "
                f"hp={info['hp']:3d} | "
                f"mem={info['memory_stability']:5.1f} | "
                f"art={'Y' if info['artifact_collected'] else 'N'} "
                f"grd={'Y' if info['guardian_defeated'] else 'N'}"
            )
            if len(ep_rewards) > 0:
                print(
                    f"         Win rate: {ep_wins}/{len(ep_rewards)} "
                    f"({100*ep_wins/len(ep_rewards):.1f}%) | "
                    f"mean reward: {np.mean(ep_rewards[-20:]):.1f}  "
                    f"[Press R to restart]"
                )

        time.sleep(step_delay)

    env.close()
    if ep_rewards:
        print(f"\nSession over: {len(ep_rewards)} episodes | "
              f"win rate {ep_wins}/{len(ep_rewards)} | "
              f"mean reward {np.mean(ep_rewards):.1f}")


if __name__ == "__main__":
    main()
