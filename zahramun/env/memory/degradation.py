"""
Memory Drought — Information Degradation Filter.

Takes ground truth game state + memory stability level.
Produces degraded observation values.

Three degradation modes applied in sequence:
  1. Temporal decay  — confidence drops per step since last seen
  2. Noise injection — Gaussian noise on partially-known values
  3. Masking         — fully unknown cells become UNKNOWN sentinel

Degradation intensifies as memory_stability drops.
"""

from __future__ import annotations
import numpy as np
from enum import IntEnum

# Sentinel values in observation space
UNKNOWN = -1
COARSE_FAR = -2   # "something is there but I don't know what"


class DegradationLevel(IntEnum):
    NONE     = 0   # stability > 80
    MINOR    = 1   # stability 60-80
    MODERATE = 2   # stability 40-60
    HEAVY    = 3   # stability 20-40
    CRITICAL = 4   # stability 0-20


def get_degradation_level(memory_stability: float, cfg: dict) -> DegradationLevel:
    ms = memory_stability
    if ms > cfg["layer_minor"]:
        return DegradationLevel.NONE
    elif ms > cfg["layer_moderate"]:
        return DegradationLevel.MINOR
    elif ms > cfg["layer_heavy"]:
        return DegradationLevel.MODERATE
    elif ms > cfg["layer_critical"]:
        return DegradationLevel.HEAVY
    else:
        return DegradationLevel.CRITICAL


class DegradationFilter:
    """
    Applies memory drought to raw observations.

    Usage:
        filter = DegradationFilter(config["memory_drought"], rng)
        obs = filter.apply(raw_obs, cell_ages, memory_stability)
    """

    def __init__(self, drought_cfg: dict, rng: np.random.Generator):
        self.cfg = drought_cfg
        self.rng = rng

    def apply_to_cell(
        self,
        raw_value: float,
        cell_age: int,        # steps since last seen, -1 = never seen
        memory_stability: float,
        level: DegradationLevel,
    ) -> float:
        """
        Apply degradation to a single cell value.

        Pipeline:
          never seen        → UNKNOWN
          very old          → UNKNOWN (temporal decay)
          old + low stab    → COARSE_FAR (knows something, not what)
          recent + minor    → noisy value
          recent + none     → raw value
        """
        cfg = self.cfg

        # Never seen
        if cell_age < 0:
            return UNKNOWN

        # Temporal decay — full forget threshold
        full_forget_age = cfg["full_forget_age"]
        if cell_age > full_forget_age:
            return UNKNOWN

        # Decay-based confidence (1.0 = perfectly remembered, 0.0 = forgotten)
        decay_confidence = max(0.0, 1.0 - cell_age * cfg["decay_rate"])

        # At CRITICAL level: heavy age penalty
        if level == DegradationLevel.CRITICAL:
            decay_confidence *= 0.4

        # Below threshold → coarse knowledge
        if decay_confidence < 0.2:
            if level >= DegradationLevel.MODERATE:
                return COARSE_FAR
            return UNKNOWN

        # Noise injection based on level
        noise_std = self._get_noise_std(level, decay_confidence)
        if noise_std > 0:
            noise = self.rng.normal(0, noise_std)
            noisy = raw_value + noise * raw_value  # proportional noise
            return float(np.clip(noisy, -1, raw_value * 2))  # don't go negative for most values

        return raw_value

    def apply_to_position(
        self,
        true_x: int,
        true_y: int,
        steps_since_update: int,
        memory_stability: float,
        level: DegradationLevel,
    ) -> tuple[float, float]:
        """
        Degrade entity position. Position drift increases with level.
        Returns (estimated_x, estimated_y) — may be floats representing uncertainty.
        """
        if steps_since_update < 0:
            return (UNKNOWN, UNKNOWN)

        # Position confidence decays faster than tile knowledge
        pos_decay = max(0.0, 1.0 - steps_since_update * self.cfg["decay_rate"] * 2)

        if pos_decay <= 0 or (level >= DegradationLevel.HEAVY and pos_decay < 0.3):
            return (UNKNOWN, UNKNOWN)

        # Add position drift noise
        drift_scale = self._get_noise_std(level, pos_decay) * 3.0
        if drift_scale > 0:
            dx = self.rng.normal(0, drift_scale)
            dy = self.rng.normal(0, drift_scale)
            return (true_x + dx, true_y + dy)

        return (float(true_x), float(true_y))

    def _get_noise_std(self, level: DegradationLevel, confidence: float) -> float:
        cfg = self.cfg
        base = {
            DegradationLevel.NONE:     0.0,
            DegradationLevel.MINOR:    cfg["noise_std_minor"],
            DegradationLevel.MODERATE: cfg["noise_std_moderate"],
            DegradationLevel.HEAVY:    cfg["noise_std_heavy"],
            DegradationLevel.CRITICAL: cfg["noise_std_heavy"] * 1.5,
        }[level]
        # More noise when confidence is lower
        return base * (1.0 - confidence * 0.5)

    def compute_passive_drain(self, level: DegradationLevel) -> float:
        """Passive memory stability drain per step. Worsens as stability drops."""
        base = self.cfg["passive_drain_per_step"]
        multiplier = {
            DegradationLevel.NONE:     1.0,
            DegradationLevel.MINOR:    1.2,
            DegradationLevel.MODERATE: 1.5,
            DegradationLevel.HEAVY:    2.0,
            DegradationLevel.CRITICAL: 3.0,
        }[level]
        return base * multiplier
