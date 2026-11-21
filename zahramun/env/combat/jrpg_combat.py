"""
JRPG Combat System — Pokemon-style turn-based.

Flow:
  Agent enters cell with enemy → CombatState created → phase = COMBAT
  Agent picks action (attack/defend/skill/item/flee)
  Enemy responds
  Repeat until: agent dead, enemy dead, or agent flees
  Phase returns to EXPLORING
"""

from __future__ import annotations
import numpy as np
from ..world.game_state import (
    GameState, GamePhase, CombatState, Entity, EntityType, Inventory, AgentState
)
from ..memory.memory_system import MemorySystem
from .skills import use_memory_pulse

# Combat actions (match action space indices)
ACT_ATTACK  = 4
ACT_DEFEND  = 5
ACT_SKILL   = 6
ACT_USE_HP  = 7
ACT_USE_MEM = 8
ACT_FLEE    = 9

# Defend damage reduction
DEFEND_REDUCTION = 0.5

# Flee base success chance
FLEE_BASE_CHANCE = 0.55


class JRPGCombat:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.reward_cfg = cfg["rewards"]
        self.combat_cfg = cfg["combat"]

    # ─── Initiate / End ──────────────────────────────────────────────────────

    def start_combat(self, state: GameState, enemy: Entity) -> None:
        """Transition to COMBAT phase."""
        state.phase = GamePhase.COMBAT
        state.combat = CombatState(enemy_id=enemy.entity_id)
        state.combat.log(f"Encountered {_enemy_name(enemy.entity_type)}!")
        state.update_action_mask()

    def end_combat(self, state: GameState, victory: bool) -> None:
        """Return to EXPLORING phase."""
        state.phase = GamePhase.EXPLORING
        state.combat = None
        state.update_action_mask()

    # ─── Main Step ───────────────────────────────────────────────────────────

    def step(
        self,
        action: int,
        state: GameState,
        mem: MemorySystem,
    ) -> float:
        """
        Process one combat action.
        Returns reward delta.
        Called by ZahramunEnv when phase == COMBAT.
        """
        combat = state.combat
        agent = state.agent
        enemy = state.entities.get(combat.enemy_id)

        if enemy is None or not enemy.alive:
            self.end_combat(state, victory=True)
            return 0.0

        reward = 0.0
        combat.defending = False

        # ── Agent turn ───────────────────────────────────────────────────────
        if action == ACT_ATTACK:
            dmg = self._agent_attack_damage()
            enemy.hp -= dmg
            combat.log(f"You attack for {dmg} damage.")
            reward += dmg * 0.1

        elif action == ACT_DEFEND:
            combat.defending = True
            combat.log("You brace for impact. Defense up!")
            reward += 0.5  # small reward for smart play

        elif action == ACT_SKILL:
            if agent.memory_pulse_charges > 0:
                result = use_memory_pulse(
                    mem.stability, mem.stability_max, enemy.hp, self.rng)
                agent.memory_pulse_charges -= 1
                mem.restore(result.stability_restored)
                enemy.hp -= result.damage_dealt
                combat.log(result.message)
                reward += result.damage_dealt * 0.15 + result.stability_restored * 0.05
            else:
                combat.log("No Memory Pulse charges left!")

        elif action == ACT_USE_HP:
            inv = agent.inventory
            if inv.hp_potions > 0:
                inv.hp_potions -= 1
                restored = min(Inventory.HP_RESTORE, agent.hp_max - agent.hp)
                agent.hp += restored
                combat.log(f"Used HP Potion. Restored {restored} HP.")
                reward += restored * 0.05
            else:
                combat.log("No HP Potions!")

        elif action == ACT_USE_MEM:
            inv = agent.inventory
            if inv.memory_potions > 0:
                inv.memory_potions -= 1
                restored = mem.restore(Inventory.MEM_RESTORE)
                combat.log(f"Used Memory Potion. Restored {restored:.0f} stability.")
                reward += restored * 0.05
            else:
                combat.log("No Memory Potions!")

        elif action == ACT_FLEE:
            flee_chance = FLEE_BASE_CHANCE
            # Easier to flee weak enemies
            hp_ratio = enemy.hp / enemy.hp_max
            flee_chance += (1.0 - hp_ratio) * 0.2
            if self.rng.random() < flee_chance:
                combat.log("Fled successfully!")
                self.end_combat(state, victory=False)
                reward -= 2.0  # small penalty for fleeing
                return reward
            else:
                combat.log("Couldn't escape!")

        # ── Check enemy death ────────────────────────────────────────────────
        if enemy.hp <= 0:
            enemy.hp = 0
            enemy.alive = False
            combat.log(f"{_enemy_name(enemy.entity_type)} defeated!")
            reward += self.reward_cfg["enemy_defeat"]
            if enemy.entity_id == state.guardian_id:
                agent.guardian_defeated = True
                reward += self.reward_cfg["guardian_defeat"]
                combat.log("GUARDIAN FALLS! The desert trembles.")
                # Guardian always drops a memory potion
                state.add_entity(EntityType.ITEM_MEMORY, enemy.x, enemy.y,
                                  hp=1, item_value=Inventory.MEM_RESTORE)
            else:
                self._maybe_drop_item(state, enemy)
            self.end_combat(state, victory=True)
            return reward

        # ── Enemy turn ───────────────────────────────────────────────────────
        reward += self._enemy_turn(state, enemy, mem, combat)

        combat.turn_count += 1
        state.update_action_mask()
        return reward

    # ─── Enemy Turn ──────────────────────────────────────────────────────────

    def _enemy_turn(
        self,
        state: GameState,
        enemy: Entity,
        mem: MemorySystem,
        combat: CombatState,
    ) -> float:
        reward = 0.0
        agent = state.agent

        if enemy.entity_type == EntityType.MEMORY_EATER:
            # Memory Eater drains stability — no HP damage
            drain = enemy.stability_drain
            actual = mem.damage(drain)
            combat.log(f"Memory Eater corrupts your mind! -{actual:.0f} stability.")
            reward -= actual * 0.2

            # Occasionally also does small HP hit
            if self.rng.random() < 0.3:
                hp_dmg = int(self.rng.integers(3, 8))
                if combat.defending:
                    hp_dmg = int(hp_dmg * DEFEND_REDUCTION)
                agent.hp -= hp_dmg
                combat.log(f"Also bites for {hp_dmg} HP!")
                reward -= hp_dmg * 0.05

        elif enemy.entity_type == EntityType.GUARDIAN:
            # Guardian alternates: big hit and stability drain
            if combat.turn_count % 2 == 0:
                dmg = self._enemy_damage(enemy)
                if combat.defending:
                    dmg = int(dmg * DEFEND_REDUCTION)
                agent.hp -= dmg
                drain = enemy.stability_drain
                mem.damage(drain)
                combat.log(
                    f"Guardian strikes for {dmg} HP and -{drain} stability!")
                reward -= dmg * 0.1 + drain * 0.15
            else:
                dmg = self._enemy_damage(enemy)
                if combat.defending:
                    dmg = int(dmg * DEFEND_REDUCTION)
                agent.hp -= dmg
                combat.log(f"Guardian crushes you for {dmg} HP!")
                reward -= dmg * 0.1

        else:
            # Normal physical attack
            dmg = self._enemy_damage(enemy)
            if combat.defending:
                dmg = int(dmg * DEFEND_REDUCTION)
                combat.log(f"Defended! Took only {dmg} HP damage.")
            else:
                combat.log(f"{_enemy_name(enemy.entity_type)} attacks for {dmg} HP!")
            agent.hp -= dmg
            reward -= dmg * 0.1

        return reward

    # ─── Damage Helpers ──────────────────────────────────────────────────────

    def _agent_attack_damage(self) -> int:
        return max(1, int(self.rng.normal(15, 3)))

    def _enemy_damage(self, enemy: Entity) -> int:
        return max(1, int(self.rng.normal(enemy.damage, enemy.damage * 0.2)))

    # ─── Exploration Auto-Combat ──────────────────────────────────────────────

    # ─── Item Drops ──────────────────────────────────────────────────────────

    def _maybe_drop_item(self, state: GameState, enemy: Entity) -> None:
        """
        Random item drop on enemy death.
        35% HP potion, 20% memory potion, 45% nothing.
        """
        roll = self.rng.random()
        if roll < 0.35:
            state.add_entity(EntityType.ITEM_HP, enemy.x, enemy.y,
                              hp=1, item_value=Inventory.HP_RESTORE)
        elif roll < 0.55:
            state.add_entity(EntityType.ITEM_MEMORY, enemy.x, enemy.y,
                              hp=1, item_value=Inventory.MEM_RESTORE)

    # ─── Exploration Auto-Combat ──────────────────────────────────────────────

    def handle_exploration_attack(
        self,
        state: GameState,
        target: Entity,
        mem: MemorySystem,
    ) -> float:
        """
        Called when agent attacks during exploration (adjacent enemy).
        Does NOT enter JRPG mode — quick strike, then initiates combat if enemy survives.
        """
        dmg = self._agent_attack_damage()
        target.hp -= dmg
        reward = dmg * 0.1

        if target.hp <= 0:
            target.hp = 0
            target.alive = False
            if target.entity_id == state.guardian_id:
                state.agent.guardian_defeated = True
                reward += self.reward_cfg["guardian_defeat"]
                state.add_entity(EntityType.ITEM_MEMORY, target.x, target.y,
                                  hp=1, item_value=Inventory.MEM_RESTORE)
            else:
                self._maybe_drop_item(state, target)
            reward += self.reward_cfg["enemy_defeat"]
        else:
            # Enemy survived — enter full JRPG combat
            self.start_combat(state, target)

        return reward


def _enemy_name(etype: EntityType) -> str:
    return {
        EntityType.SCORPION:     "Scorpion",
        EntityType.MEMORY_EATER: "Memory Eater",
        EntityType.WANDERER:     "Forgotten Wanderer",
        EntityType.GUARDIAN:     "The Guardian",
    }.get(etype, "Enemy")
