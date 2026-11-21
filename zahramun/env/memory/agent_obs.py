"""
Agent Observation Builder — reads ground truth → applies degradation → flat numpy array.

Observation layout:
  [0]   phase (0=explore, 1=combat, normalized)
  [1]   agent_x (norm)
  [2]   agent_y (norm)
  [3]   agent_hp (norm)
  [4]   memory_stability (norm)
  [5]   degradation_level (0-4, norm)
  [6]   artifact_collected
  [7]   guardian_defeated
  [8]   hp_potions (norm, max 5)
  [9]   memory_potions (norm, max 5)
  [10]  memory_pulse_charges (norm, max 3)

  -- Map channels (H*W each) --
  tile_type channel       : degraded tile values
  tile_confidence channel : 0=unknown, 1=fresh
  entity_channel          : entity type at cell (degraded)

  -- Enemy tracking (MAX_TRACKED slots * 4) --
  per enemy: [deg_x, deg_y, deg_type, deg_hp]

  -- Artifact pos (2) --
  -- Guardian pos  (2) --

  -- Combat state (8, zero when exploring) --
  [0] in_combat
  [1] enemy_type (norm)
  [2] enemy_hp (norm)
  [3] enemy_hp_max (norm)
  [4] agent_turn
  [5] defending
  [6] turn_count (norm, /20)
  [7] can_flee

  -- Action mask (10) --
"""

from __future__ import annotations
import numpy as np
from ..world.game_state import GameState, GamePhase, EntityType, Tile
from .degradation import DegradationFilter, DegradationLevel, UNKNOWN
from .memory_system import MemorySystem

MAX_TRACKED_ENEMIES = 8
FOV_RADIUS = 7


class ObservationBuilder:
    def __init__(self, map_w: int, map_h: int, memory_system: MemorySystem,
                 rng: np.random.Generator, degrade_enabled: bool = True):
        self.map_w = map_w
        self.map_h = map_h
        self.mem = memory_system
        self.filter = memory_system.filter
        self.rng = rng
        # When False, remembered cells are returned perfectly (no decay/noise/mask);
        # never-seen cells are still UNKNOWN. Isolates perception corruption from the
        # underlying exploration/mapping task (ablation modes survival_only / none).
        self.degrade_enabled = degrade_enabled

        map_cells = map_w * map_h
        self.obs_size = (
            11                        # global
            + map_cells * 3           # map channels
            + MAX_TRACKED_ENEMIES * 4 # enemy slots
            + 2                       # artifact
            + 2                       # guardian
            + 8                       # combat state
            + 10                      # action mask
        )

    def _cell(self, raw: float, age: int, level) -> float:
        if not self.degrade_enabled:
            return raw if age >= 0 else UNKNOWN
        return self.filter.apply_to_cell(raw, age, self.mem.stability, level)

    def _pos(self, x: int, y: int, age: int, level) -> tuple[float, float]:
        if not self.degrade_enabled:
            if age < 0:
                return (UNKNOWN, UNKNOWN)
            return (float(x), float(y))
        return self.filter.apply_to_position(x, y, age, self.mem.stability, level)

    def build(self, state: GameState) -> np.ndarray:
        obs = np.zeros(self.obs_size, dtype=np.float32)
        level = self.mem.level
        agent = state.agent
        idx = 0

        # ── Global ───────────────────────────────────────────────────────────
        obs[0]  = float(state.phase) / 3.0
        obs[1]  = agent.x / self.map_w
        obs[2]  = agent.y / self.map_h
        obs[3]  = agent.hp / agent.hp_max
        obs[4]  = self.mem.normalized()
        obs[5]  = float(level) / 4.0
        obs[6]  = float(agent.artifact_collected)
        obs[7]  = float(agent.guardian_defeated)
        obs[8]  = agent.inventory.hp_potions / 5.0
        obs[9]  = agent.inventory.memory_potions / 5.0
        obs[10] = agent.memory_pulse_charges / 3.0
        idx = 11

        # ── Map channels ─────────────────────────────────────────────────────
        map_cells = self.map_w * self.map_h
        tile_ch   = np.full(map_cells, UNKNOWN, dtype=np.float32)
        conf_ch   = np.zeros(map_cells, dtype=np.float32)
        entity_ch = np.full(map_cells, UNKNOWN, dtype=np.float32)
        tile_max  = float(max(Tile))
        etype_max = float(max(EntityType))

        for r in range(self.map_h):
            for c in range(self.map_w):
                ci = r * self.map_w + c
                age = state.cell_age(c, r)
                raw_tile = float(state.tiles[r, c]) / tile_max
                deg = self._cell(raw_tile, age, level)
                tile_ch[ci] = deg

                if age < 0:
                    conf_ch[ci] = 0.0
                elif not self.degrade_enabled:
                    conf_ch[ci] = 1.0
                else:
                    conf_ch[ci] = max(0.0, 1.0 - age * self.filter.cfg["decay_rate"])

                if deg != UNKNOWN:
                    ents = state.get_entities_at(c, r)
                    if ents:
                        raw_et = float(ents[0].entity_type) / etype_max
                        entity_ch[ci] = self._cell(raw_et, age, level)

        obs[idx:idx + map_cells] = tile_ch;   idx += map_cells
        obs[idx:idx + map_cells] = conf_ch;   idx += map_cells
        obs[idx:idx + map_cells] = entity_ch; idx += map_cells

        # ── Enemy slots ──────────────────────────────────────────────────────
        ax, ay = agent.x, agent.y
        enemies = sorted(
            state.living_enemies(),
            key=lambda e: abs(e.x - ax) + abs(e.y - ay)
        )[:MAX_TRACKED_ENEMIES]

        for i in range(MAX_TRACKED_ENEMIES):
            if i < len(enemies):
                e = enemies[i]
                age = state.cell_age(e.x, e.y)
                ex, ey = self._pos(e.x, e.y, age, level)
                obs[idx]   = ex / self.map_w if ex != UNKNOWN else UNKNOWN
                obs[idx+1] = ey / self.map_h if ey != UNKNOWN else UNKNOWN
                obs[idx+2] = self._cell(float(e.entity_type) / etype_max, age, level)
                obs[idx+3] = self._cell(e.hp / e.hp_max, age, level)
            else:
                obs[idx:idx+4] = UNKNOWN
            idx += 4

        # ── Artifact ─────────────────────────────────────────────────────────
        if agent.artifact_collected:
            obs[idx] = obs[idx+1] = 1.0
        elif state.artifact_pos:
            ax_p, ay_p = state.artifact_pos
            age = state.cell_age(ax_p, ay_p)
            ex, ey = self._pos(ax_p, ay_p, age, level)
            obs[idx]   = ex / self.map_w if ex != UNKNOWN else UNKNOWN
            obs[idx+1] = ey / self.map_h if ey != UNKNOWN else UNKNOWN
        else:
            obs[idx:idx+2] = UNKNOWN
        idx += 2

        # ── Guardian ─────────────────────────────────────────────────────────
        if agent.guardian_defeated:
            obs[idx] = obs[idx+1] = 1.0
        elif state.guardian_id and state.guardian_id in state.entities:
            g = state.entities[state.guardian_id]
            age = state.cell_age(g.x, g.y)
            gx, gy = self._pos(g.x, g.y, age, level)
            obs[idx]   = gx / self.map_w if gx != UNKNOWN else UNKNOWN
            obs[idx+1] = gy / self.map_h if gy != UNKNOWN else UNKNOWN
        else:
            obs[idx:idx+2] = UNKNOWN
        idx += 2

        # ── Combat state ─────────────────────────────────────────────────────
        if state.phase == GamePhase.COMBAT and state.combat:
            c = state.combat
            enemy = state.entities.get(c.enemy_id)
            obs[idx]   = 1.0
            obs[idx+1] = float(enemy.entity_type) / etype_max if enemy else 0.0
            obs[idx+2] = enemy.hp / enemy.hp_max if enemy else 0.0
            obs[idx+3] = float(enemy.hp_max) / 100.0 if enemy else 0.0
            obs[idx+4] = float(c.agent_turn)
            obs[idx+5] = float(c.defending)
            obs[idx+6] = min(c.turn_count / 20.0, 1.0)
            obs[idx+7] = 1.0
        idx += 8

        # ── Action mask ──────────────────────────────────────────────────────
        obs[idx:idx+10] = state.action_mask.astype(np.float32)
        idx += 10

        assert idx == self.obs_size, f"Obs size mismatch: {idx} != {self.obs_size}"
        return obs
