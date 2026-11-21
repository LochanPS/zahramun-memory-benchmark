"""
Sanity checks — run without pytest: python -m zahramun.tests.test_env_sanity
"""

import numpy as np
from zahramun.env.zahramun_env import ZahramunEnv


def test_reset_obs_shape():
    env = ZahramunEnv(seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape, \
        f"Obs shape {obs.shape} != space shape {env.observation_space.shape}"
    print(f"  obs shape: {obs.shape}  OK")


def test_step_cycle():
    env = ZahramunEnv(seed=42)
    obs, _ = env.reset(seed=42)
    total_reward = 0.0
    done = False
    steps = 0
    while not done and steps < 200:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
    print(f"  {steps} steps, total_reward={total_reward:.2f}, done={done}  OK")
    env.close()


def test_ansi_render():
    env = ZahramunEnv(seed=42, render_mode="ansi")
    env.reset(seed=42)
    rendered = env.render()
    assert isinstance(rendered, str)
    assert "@" in rendered, "Agent char '@' missing from render"
    print(f"  ansi render OK, {len(rendered)} chars  OK")
    env.close()


def test_obs_values_in_range():
    env = ZahramunEnv(seed=7)
    obs, _ = env.reset(seed=7)
    # UNKNOWN sentinel = -1.0 is allowed; all others should be in [-1, 2]
    assert obs.min() >= -1.0, f"Obs min {obs.min()} below -1"
    assert obs.max() <= 2.0, f"Obs max {obs.max()} above 2"
    print(f"  obs range [{obs.min():.3f}, {obs.max():.3f}]  OK")
    env.close()


def test_memory_collapse():
    """Force memory collapse by draining stability."""
    env = ZahramunEnv(seed=1)
    obs, _ = env.reset(seed=1)
    env.mem.stability = 0.1  # near zero
    # One step should collapse
    _, _, terminated, _, info = env.step(5)  # WAIT
    assert terminated, "Episode should terminate on memory collapse"
    print(f"  memory collapse terminates episode  OK")
    env.close()


if __name__ == "__main__":
    tests = [
        test_reset_obs_shape,
        test_step_cycle,
        test_ansi_render,
        test_obs_values_in_range,
        test_memory_collapse,
    ]
    print("Running Zahramun sanity tests...")
    passed = 0
    for t in tests:
        try:
            print(f"\n{t.__name__}")
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
    print(f"\n{passed}/{len(tests)} passed")
