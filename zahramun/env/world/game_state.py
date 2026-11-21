"""
Ground truth game state. The env knows everything here.
Agent NEVER sees this directly — always goes through degradation filter.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum, auto


# ─── Game Phase ──────────────────────────────────────────────────────────────

class GamePhase(IntEnum):
    EXPLORING = 0
    COMBAT    = 1
    GAME_OVER = 2
    VICTORY   = 3


# ─── Tile Types ──────────────────────────────────────────────────────────────

class Tile(IntEnum):
    VOID   = 0
    FLOOR  = 1
    WALL   = 2
    SAND   = 3
    RUIN   = 4
    DOOR   = 5
    HAZARD = 6   # drains memory stability on step


# ─── Entity Types ────────────────────────────────────────────────────────────

class EntityType(IntEnum):
    AGENT        = 0
    SCORPION     = 1
    MEMORY_EATER = 2
    WANDERER     = 3
    GUARDIAN     = 4
    ARTIFACT     = 5
    ITEM_HP      = 6
    ITEM_MEMORY  = 7
    HAZARD_ENT   = 8
    SHRINE       = 9   # Memory Shrine — restores stability, one-use


@dataclass
class Entity:
    entity_id: int
    entity_type: EntityType
    x: int
    y: int
    hp: int
    hp_max: int
    alive: bool = True

    # Combat stats
    damage: int = 0
    defense: int = 0
    stability_drain: int = 0   # memory eaters / guardian

    # Item values (for ITEM_HP, ITEM_MEMORY)
    item_value: int = 0

    def is_enemy(self) -> bool:
        return self.entity_type in (
            EntityType.SCORPION,
            EntityType.MEMORY_EATER,
            EntityType.WANDERER,
            EntityType.GUARDIAN,
        )

    def is_item(self) -> bool:
        return self.entity_type in (EntityType.ITEM_HP, EntityType.ITEM_MEMORY)

    def is_shrine(self) -> bool:
        return self.entity_type == EntityType.SHRINE


# ─── Combat State ─────────────────────────────────────────────────────────────

@dataclass
class CombatState:
    """Active when phase == COMBAT."""
    enemy_id: int                       # which entity we're fighting
    agent_turn: bool = True             # True = agent picks action
    defending: bool = False             # agent chose Defend last turn
    turn_count: int = 0
    last_action_result: str = ""        # human-readable feedback
    combat_log: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.combat_log.append(msg)
        if len(self.combat_log) > 8:
            self.combat_log.pop(0)


# ─── Agent Inventory ──────────────────────────────────────────────────────────

@dataclass
class Inventory:
    hp_potions: int = 0
    memory_potions: int = 0

    HP_RESTORE    = 30
    MEM_RESTORE   = 25


# ─── Agent State ──────────────────────────────────────────────────────────────

class StatusEffect(IntEnum):
    NONE      = 0
    POISONED  = 1   # scorpion: lose HP each step
    CONFUSED  = 2   # memory eater: movement direction randomly flipped
    WEAKENED  = 3   # wanderer: deal less damage


@dataclass
class AgentState:
    x: int
    y: int
    hp: int
    hp_max: int
    memory_stability: float
    memory_stability_max: float
    steps_alive: int = 0
    artifact_collected: bool = False
    guardian_defeated: bool = False
    inventory: Inventory = field(default_factory=Inventory)

    # Skills
    memory_pulse_charges: int = 3   # combat skill: restore stability

    # Status effects
    status: StatusEffect = StatusEffect.NONE
    status_duration: int = 0        # steps remaining


# ─── Full Game State ──────────────────────────────────────────────────────────

@dataclass
class GameState:
    """
    Complete ground truth. env.step() mutates this.
    Degradation filter reads this to produce agent observation.
    """
    map_width: int
    map_height: int

    phase: GamePhase = GamePhase.EXPLORING

    # Tile grid — shape (H, W), dtype int8
    tiles: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.int8))

    # Entity registry
    entities: dict[int, Entity] = field(default_factory=dict)
    next_entity_id: int = 0

    # Agent
    agent: Optional[AgentState] = None

    # Combat (only valid when phase == COMBAT)
    combat: Optional[CombatState] = None

    # Objective positions
    artifact_pos: Optional[tuple[int, int]] = None
    guardian_id: Optional[int] = None

    # Step counter
    step: int = 0
    done: bool = False
    truncated: bool = False

    # Cell visibility — last step each cell was in agent FOV, -1 = never seen
    last_seen: np.ndarray = field(default_factory=lambda: np.full((1, 1), -1, dtype=np.int32))

    # Action validity mask (for RL)
    action_mask: np.ndarray = field(default_factory=lambda: np.ones(10, dtype=bool))

    # Hazard tile positions
    hazard_positions: set = field(default_factory=set)

    # Exploration tracking (behavioral metrics — forgetting signature)
    visited: set = field(default_factory=set)   # unique cells the agent has stood on
    revisit_count: int = 0                       # times agent re-entered a known cell

    def add_entity(self, entity_type: EntityType, x: int, y: int,
                   hp: int, damage: int = 0, defense: int = 0,
                   stability_drain: int = 0, item_value: int = 0) -> Entity:
        eid = self.next_entity_id
        self.next_entity_id += 1
        e = Entity(
            entity_id=eid,
            entity_type=entity_type,
            x=x, y=y,
            hp=hp, hp_max=hp,
            damage=damage,
            defense=defense,
            stability_drain=stability_drain,
            item_value=item_value,
        )
        self.entities[eid] = e
        return e

    def get_entities_at(self, x: int, y: int) -> list[Entity]:
        return [e for e in self.entities.values() if e.alive and e.x == x and e.y == y]

    def living_enemies(self) -> list[Entity]:
        return [e for e in self.entities.values() if e.alive and e.is_enemy()]

    def update_visibility(self, fov_cells: set[tuple[int, int]]) -> None:
        for (x, y) in fov_cells:
            if 0 <= y < self.map_height and 0 <= x < self.map_width:
                self.last_seen[y, x] = self.step

    def cell_age(self, x: int, y: int) -> int:
        last = self.last_seen[y, x]
        if last < 0:
            return -1
        return self.step - last

    def record_visit(self, x: int, y: int) -> None:
        """Track unique vs repeated cell visits — behavioral forgetting signal."""
        if (x, y) in self.visited:
            self.revisit_count += 1
        else:
            self.visited.add((x, y))

    def coverage(self) -> float:
        """Fraction of the map the agent has ever had in view (0..1)."""
        total = self.map_width * self.map_height
        if total == 0:
            return 0.0
        return float((self.last_seen >= 0).sum()) / total

    def update_action_mask(self) -> None:
        """Recompute valid actions for current phase."""
        mask = np.zeros(10, dtype=bool)
        inv = self.agent.inventory
        if self.phase == GamePhase.EXPLORING:
            mask[0:4] = True   # move
            mask[4] = True     # attack (if adjacent enemy exists)
            mask[5] = True     # wait
            mask[7] = inv.hp_potions > 0
            mask[8] = inv.memory_potions > 0
        elif self.phase == GamePhase.COMBAT:
            mask[4] = True     # attack
            mask[5] = True     # defend
            mask[6] = self.agent.memory_pulse_charges > 0  # skill
            mask[7] = inv.hp_potions > 0
            mask[8] = inv.memory_potions > 0
            mask[9] = True     # flee
        self.action_mask = mask
