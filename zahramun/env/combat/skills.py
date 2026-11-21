"""
Agent skills — usable in JRPG combat.
Each skill has research value: forces agent to manage limited resources under degradation.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum


class SkillID(IntEnum):
    MEMORY_PULSE = 0   # restore stability, deal psychic damage


@dataclass
class SkillResult:
    stability_restored: float = 0.0
    damage_dealt: int = 0
    message: str = ""


def use_memory_pulse(agent_stability: float, stability_max: float,
                     enemy_hp: int, rng) -> SkillResult:
    """
    Memory Pulse — agent focuses remaining memory, clears some drought,
    and unleashes a psychic burst at enemy.

    Stability restored: 15% of max
    Damage dealt: scales with current stability (healthier memory = harder hit)
    """
    restore = stability_max * 0.15
    stability_ratio = agent_stability / stability_max
    damage = int(rng.integers(8, 16) * (0.5 + stability_ratio * 0.5))

    return SkillResult(
        stability_restored=restore,
        damage_dealt=damage,
        message=f"Memory Pulse! Restored {restore:.0f} stability, dealt {damage} psychic dmg.",
    )
