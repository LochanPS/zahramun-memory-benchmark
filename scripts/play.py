"""
Zahramun — Human Play Mode (pygame)

Controls:
  Arrow keys / WASD  : move
  F / Space          : attack (explore) / attack (combat)
  E                  : defend (combat only)
  C                  : Memory Pulse skill (combat only)
  H                  : use HP potion
  N                  : use Memory potion
  X                  : flee (combat only)
  T                  : toggle truth mode (see full map)
  F11                : toggle fullscreen / windowed
  R                  : restart episode
  ESC / Q            : quit
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame
from zahramun.env.zahramun_env import ZahramunEnv
from zahramun.env.world.game_state import GamePhase

# Action indices — match zahramun_env.py
MOVE_N  = 0
MOVE_S  = 1
MOVE_E  = 2
MOVE_W  = 3
ATTACK  = 4
DEFEND  = 5
SKILL   = 6
USE_HP  = 7
USE_MEM = 8
FLEE    = 9

# Clean key → action table, no conflicts
EXPLORE_KEYS = {
    pygame.K_UP:    MOVE_N,
    pygame.K_w:     MOVE_N,
    pygame.K_DOWN:  MOVE_S,
    pygame.K_s:     MOVE_S,
    pygame.K_RIGHT: MOVE_E,
    pygame.K_d:     MOVE_E,
    pygame.K_LEFT:  MOVE_W,
    pygame.K_a:     MOVE_W,
    pygame.K_f:     ATTACK,
    pygame.K_SPACE: ATTACK,
    pygame.K_h:     USE_HP,
    pygame.K_n:     USE_MEM,
}

COMBAT_KEYS = {
    pygame.K_f:     ATTACK,
    pygame.K_SPACE: ATTACK,
    pygame.K_e:     DEFEND,
    pygame.K_c:     SKILL,
    pygame.K_h:     USE_HP,
    pygame.K_n:     USE_MEM,
    pygame.K_x:     FLEE,
}


def run():
    seed = 42
    env  = ZahramunEnv(render_mode="human", seed=seed)
    obs, _ = env.reset(seed=seed)

    # Explicitly create renderer — owns the window (fullscreen by default, F11 to toggle)
    from zahramun.env.rendering.pygame_renderer import PygameRenderer
    env._renderer = PygameRenderer(env.map_w, env.map_h, env.cfg, fullscreen=True)

    show_truth      = False
    episode_num     = 1
    running         = True
    waiting_restart = False

    while running:
        # ── Render — renderer owns clock.tick internally ───────────────────────
        if env.state and env._renderer:
            env._renderer.render(env.state, env.mem)

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type != pygame.KEYDOWN:
                continue

            k = event.key

            # ── Global keys (always active) ───────────────────────────────────
            if k in (pygame.K_ESCAPE, pygame.K_q):
                running = False
                break

            if k == pygame.K_F11:
                if env._renderer:
                    env._renderer.toggle_fullscreen()
                continue

            if k == pygame.K_t:
                show_truth = not show_truth
                if env._renderer:
                    env._renderer._show_truth = show_truth
                continue

            if k == pygame.K_r:
                seed += 1
                episode_num += 1
                # Ramp difficulty: +8% enemy count per episode, cap at 1.8x
                env._difficulty_scale = min(1.0 + (episode_num - 1) * 0.08, 1.8)
                obs, _ = env.reset(seed=seed)
                waiting_restart = False
                continue

            # ── Dead / victory — only R works ─────────────────────────────────
            if waiting_restart:
                continue

            phase = env.state.phase
            if phase in (GamePhase.GAME_OVER, GamePhase.VICTORY):
                waiting_restart = True
                continue

            # ── Route action by phase ─────────────────────────────────────────
            if phase == GamePhase.COMBAT:
                action = COMBAT_KEYS.get(k)
            else:
                action = EXPLORE_KEYS.get(k)

            if action is None:
                continue

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                waiting_restart = True

    env.close()
    print(f"Quit after {episode_num} episode(s).")


if __name__ == "__main__":
    run()
