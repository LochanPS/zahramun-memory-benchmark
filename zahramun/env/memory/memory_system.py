"""
Memory System — tracks Memory Stability resource and coordinates degradation.
"""

from __future__ import annotations
import numpy as np
from .degradation import DegradationFilter, DegradationLevel, get_degradation_level, UNKNOWN


class MemorySystem:
    """
    Owns memory_stability resource.
    Applies drought each step.
    Exposes current degradation level.
    """

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.drought_cfg = cfg["memory_drought"]
        self.stability_max = cfg["resources"]["memory_stability_max"]
        self.stability: float = float(self.stability_max)
        self.filter = DegradationFilter(self.drought_cfg, rng)

    @property
    def level(self) -> DegradationLevel:
        return get_degradation_level(self.stability, self.drought_cfg)

    @property
    def is_collapsed(self) -> bool:
        return self.stability <= 0

    def step_drain(self) -> float:
        """Apply passive drain. Returns amount drained."""
        drain = self.filter.compute_passive_drain(self.level)
        self.stability = max(0.0, self.stability - drain)
        return drain

    def damage(self, amount: float) -> float:
        """Direct stability damage (memory eater, hazard). Returns actual damage dealt."""
        actual = min(self.stability, amount)
        self.stability = max(0.0, self.stability - amount)
        return actual

    def restore(self, amount: float) -> float:
        """Restore stability (memory item). Returns actual restored."""
        before = self.stability
        self.stability = min(self.stability_max, self.stability + amount)
        return self.stability - before

    def reset(self) -> None:
        self.stability = float(self.stability_max)

    def normalized(self) -> float:
        """0.0 – 1.0 for observation space."""
        return self.stability / self.stability_max
