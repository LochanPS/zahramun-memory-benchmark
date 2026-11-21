"""
Exploration-phase enemy AI — movement + out-of-combat attacks.
JRPG combat (turn-based) handled by jrpg_combat.py.
"""

from __future__ import annotations
import numpy as np
from ..world.game_state import (
    GameState, AgentState, Entity, EntityType,
    StatusEffect, Tile
)
from ..memory.memory_system import MemorySystem

WALKABLE = {Tile.FLOOR, Tile.SAND, Tile.RUIN}

# Aggro ranges per enemy type
AGGRO_RANGE = {
    EntityType.SCORPION:     5,
    EntityType.MEMORY_EATER: 8,   # hunts from farther away
    EntityType.WANDERER:     3,
    EntityType.GUARDIAN:     6,
}


class CombatSystem:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg        = cfg
        self.rng        = rng
        self.reward_cfg = cfg["rewards"]

    def enemy_turns(self, state: GameState, mem: MemorySystem) -> float:
        """
        All living enemies act during exploration phase.
        Returns reward delta (penalties for hits taken).
        """
        reward = 0.0
        agent  = state.agent
        ax, ay = agent.x, agent.y

        # Tick status effects on agent
        reward += self._tick_status(agent, mem)

        for enemy in state.living_enemies():
            dist = abs(enemy.x - ax) + abs(enemy.y - ay)
            aggro = AGGRO_RANGE.get(enemy.entity_type, 5)

            if dist <= 1:
                # Adjacent — quick strike (will trigger JRPG combat next step)
                reward += self._quick_strike(enemy, agent, mem)
            elif dist <= aggro:
                self._move_enemy(enemy, ax, ay, state)

        return reward

    # ─── Movement AI ──────────────────────────────────────────────────────────

    def _move_enemy(self, enemy: Entity, tx: int, ty: int, state: GameState) -> None:
        if enemy.entity_type == EntityType.WANDERER:
            # 60% wander randomly, 40% drift toward player
            if self.rng.random() < 0.6:
                self._wander(enemy, state)
            else:
                self._chase(enemy, tx, ty, state)
        elif enemy.entity_type == EntityType.MEMORY_EATER:
            # Always aggressively chase
            self._chase(enemy, tx, ty, state)
        else:
            self._chase(enemy, tx, ty, state)

    def _chase(self, enemy: Entity, tx: int, ty: int, state: GameState) -> None:
        dx, dy = tx - enemy.x, ty - enemy.y
        if abs(dx) >= abs(dy):
            moves = [(int(np.sign(dx)), 0), (0, int(np.sign(dy)))]
        else:
            moves = [(0, int(np.sign(dy))), (int(np.sign(dx)), 0)]
        for mdx, mdy in moves:
            if mdx == 0 and mdy == 0:
                continue
            nx, ny = enemy.x + mdx, enemy.y + mdy
            if self._can_move(nx, ny, state):
                enemy.x, enemy.y = nx, ny
                break

    def _wander(self, enemy: Entity, state: GameState) -> None:
        dirs = [(0,-1),(0,1),(1,0),(-1,0)]
        self.rng.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = enemy.x + dx, enemy.y + dy
            if self._can_move(nx, ny, state):
                enemy.x, enemy.y = nx, ny
                break

    def _can_move(self, nx: int, ny: int, state: GameState) -> bool:
        return (0 <= nx < state.map_width
                and 0 <= ny < state.map_height
                and Tile(state.tiles[ny, nx]) in WALKABLE
                and not state.get_entities_at(nx, ny))

    # ─── Quick Strike (exploration phase) ─────────────────────────────────────

    def _quick_strike(self, enemy: Entity, agent: AgentState, mem: MemorySystem) -> float:
        """Light hit before JRPG combat initiates next step."""
        reward = 0.0
        if enemy.entity_type == EntityType.MEMORY_EATER:
            drain = enemy.stability_drain * 0.3
            mem.damage(drain)
            reward -= drain * 0.2
        else:
            dmg = max(1, int(self.rng.normal(enemy.damage * 0.4, 1)))
            agent.hp -= dmg
            # Apply status on scorpion hit
            if (enemy.entity_type == EntityType.SCORPION
                    and agent.status == StatusEffect.NONE
                    and self.rng.random() < 0.25):
                agent.status = StatusEffect.POISONED
                agent.status_duration = 4
            reward -= dmg * 0.05
        return reward

    # ─── Status Effect Tick ───────────────────────────────────────────────────

    def _tick_status(self, agent: AgentState, mem: MemorySystem) -> float:
        reward = 0.0
        if agent.status == StatusEffect.NONE or agent.status_duration <= 0:
            agent.status = StatusEffect.NONE
            agent.status_duration = 0
            return reward

        if agent.status == StatusEffect.POISONED:
            dmg = 3
            agent.hp -= dmg
            reward -= dmg * 0.05

        agent.status_duration -= 1
        if agent.status_duration <= 0:
            agent.status = StatusEffect.NONE

        return reward
