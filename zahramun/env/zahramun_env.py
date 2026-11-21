"""
ZahramunEnv — Gymnasium environment, Zahramun RL Benchmark.

Action space: Discrete(10)
  0-3 : Move N/S/E/W        (explore only)
  4   : Attack               (explore: quick strike; combat: attack)
  5   : Defend/Wait          (explore: wait; combat: defend)
  6   : Skill (Mem Pulse)    (combat only)
  7   : Use HP Potion        (both)
  8   : Use Memory Potion    (both)
  9   : Flee                 (combat only)
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import yaml, os

from .world.game_state import (
    GameState, GamePhase, AgentState, EntityType, Tile, Inventory
)
from .world.map_generator import MapGenerator
from .world.fov import compute_fov
from .memory.memory_system import MemorySystem
from .memory.agent_obs import ObservationBuilder, FOV_RADIUS
from .combat.jrpg_combat import JRPGCombat

N_ACTIONS = 10

MOVE_DELTAS = {0: (0,-1), 1: (0,1), 2: (1,0), 3: (-1,0)}
WALKABLE    = {Tile.FLOOR, Tile.SAND, Tile.RUIN, Tile.HAZARD}


def load_config(path: Optional[str] = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


class ZahramunEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(
        self,
        config_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = load_config(config_path)
        self.render_mode = render_mode
        self._seed = seed

        map_cfg = self.cfg["env"]
        self.map_w = map_cfg["map_width"]
        self.map_h = map_cfg["map_height"]
        self.max_steps = map_cfg["max_steps"]

        # Compute obs size via dummy builder
        _rng = np.random.default_rng(0)
        _mem = MemorySystem(self.cfg, _rng)
        _obs = ObservationBuilder(self.map_w, self.map_h, _mem, _rng)
        obs_size = _obs.obs_size

        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.state: Optional[GameState]        = None
        self.mem:   Optional[MemorySystem]     = None
        self.obs_builder: Optional[ObservationBuilder] = None
        self.combat_sys:  Optional[JRPGCombat] = None
        self._episode_rng: Optional[np.random.Generator] = None
        self._renderer = None
        self._episode_count: int = 0         # for difficulty scaling in play mode
        self._difficulty_scale: float = 1.0  # set externally by play.py

    # ─── Gym Interface ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ep_seed = seed if seed is not None else self._seed
        self._episode_rng = np.random.default_rng(ep_seed)
        self._build_episode(ep_seed)
        obs = self.obs_builder.build(self.state)
        return obs, self._get_info()

    def step(self, action: int):
        assert self.state is not None
        state  = self.state
        agent  = state.agent
        reward = 0.0

        # Mask invalid actions silently (treat as wait)
        if not state.action_mask[action]:
            action = 5  # wait/defend

        # ── Route by phase ────────────────────────────────────────────────────
        if state.phase == GamePhase.COMBAT:
            reward += self.combat_sys.step(action, state, self.mem)
        else:
            reward += self._explore_action(action)

        # ── FOV update ───────────────────────────────────────────────────────
        if state.phase == GamePhase.EXPLORING:
            fov = compute_fov(agent.x, agent.y, FOV_RADIUS,
                              state.tiles, self.map_h, self.map_w)
            state.update_visibility(fov)

            # Hazard tile check
            if (agent.x, agent.y) in state.hazard_positions:
                drain = self.cfg["memory_drought"]["hazard_drain"]
                self.mem.damage(drain)
                reward -= drain * 0.1

            # Enemy aggro — enemies walk into agent → start combat
            adjacent = [
                e for e in state.living_enemies()
                if abs(e.x - agent.x) + abs(e.y - agent.y) <= 1
                and state.phase == GamePhase.EXPLORING
            ]
            if adjacent and state.phase == GamePhase.EXPLORING:
                self.combat_sys.start_combat(state, adjacent[0])

        # ── Passive memory drain ──────────────────────────────────────────────
        self.mem.step_drain()
        reward += self.cfg["rewards"]["step_penalty"]
        reward += self.mem.normalized() * self.cfg["rewards"]["memory_stability_bonus"]

        state.step += 1
        agent.steps_alive += 1

        # ── Terminal conditions ───────────────────────────────────────────────
        terminated = truncated = False

        if agent.hp <= 0:
            reward += self.cfg["rewards"]["death_penalty"]
            terminated = True
            state.phase = GamePhase.GAME_OVER

        elif self.mem.is_collapsed:
            reward += self.cfg["rewards"]["memory_collapse_penalty"]
            terminated = True
            state.phase = GamePhase.GAME_OVER

        elif agent.artifact_collected and agent.guardian_defeated:
            reward += self.cfg["rewards"]["episode_complete"]
            terminated = True
            state.phase = GamePhase.VICTORY

        elif state.step >= self.max_steps:
            truncated = True

        state.done = terminated
        state.truncated = truncated

        obs = self.obs_builder.build(state)
        # Note: human render driven externally by play.py, not here
        return obs, reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "ansi":
            return self._ansi_render()

    def close(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    # ─── Exploration Actions ──────────────────────────────────────────────────

    def _explore_action(self, action: int) -> float:
        state = self.state
        agent = state.agent
        reward = 0.0

        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            nx, ny = agent.x + dx, agent.y + dy

            if 0 <= nx < self.map_w and 0 <= ny < self.map_h:
                tile_at = Tile(state.tiles[ny, nx])

                # Door: open it and step through in one action
                if tile_at == Tile.DOOR:
                    state.tiles[ny, nx] = int(Tile.FLOOR)
                    tile_at = Tile.FLOOR

                if tile_at in WALKABLE:
                    enemies_at = [e for e in state.get_entities_at(nx, ny) if e.is_enemy()]
                    items_at   = [e for e in state.get_entities_at(nx, ny) if e.is_item()]
                    shrines_at = [e for e in state.get_entities_at(nx, ny)
                                  if e.is_shrine() and e.alive]

                    if enemies_at:
                        # Bump-to-attack → may trigger JRPG combat
                        reward += self.combat_sys.handle_exploration_attack(
                            state, enemies_at[0], self.mem)
                    else:
                        agent.x, agent.y = nx, ny

                        # Artifact pickup
                        if ((nx, ny) == state.artifact_pos
                                and not agent.artifact_collected):
                            agent.artifact_collected = True
                            reward += self.cfg["rewards"]["artifact_pickup"]

                        # Item pickup
                        for item in items_at:
                            reward += self._pickup_item(item)

                        # Shrine activation
                        for shrine in shrines_at:
                            reward += self._activate_shrine(shrine)

        elif action == 4:  # Attack — strike adjacent enemy
            for dx, dy in [(0,-1),(0,1),(1,0),(-1,0)]:
                nx, ny = agent.x + dx, agent.y + dy
                adj = [e for e in state.get_entities_at(nx, ny) if e.is_enemy()]
                if adj:
                    reward += self.combat_sys.handle_exploration_attack(
                        state, adj[0], self.mem)
                    break

        elif action == 7:  # Use HP potion
            inv = agent.inventory
            if inv.hp_potions > 0:
                inv.hp_potions -= 1
                healed = min(Inventory.HP_RESTORE, agent.hp_max - agent.hp)
                agent.hp += healed
                reward += healed * 0.03

        elif action == 8:  # Use memory potion
            inv = agent.inventory
            if inv.memory_potions > 0:
                inv.memory_potions -= 1
                restored = self.mem.restore(Inventory.MEM_RESTORE)
                reward += restored * 0.03

        # action 5 = wait, 6/9 invalid in explore → already masked to 5
        return reward

    def _activate_shrine(self, shrine) -> float:
        """One-use Memory Shrine — restores fixed stability, then deactivates."""
        shrine.alive = False
        restored = self.mem.restore(float(shrine.item_value))   # item_value = stability pts
        return restored * 0.08   # reward signal for finding + using shrine

    def _pickup_item(self, item) -> float:
        state = self.state
        agent = state.agent
        item.alive = False  # consumed

        if item.entity_type == EntityType.ITEM_HP:
            agent.inventory.hp_potions = min(5, agent.inventory.hp_potions + 1)
            return 2.0
        elif item.entity_type == EntityType.ITEM_MEMORY:
            agent.inventory.memory_potions = min(5, agent.inventory.memory_potions + 1)
            return 2.0
        return 0.0

    # ─── Episode Builder ──────────────────────────────────────────────────────

    def _build_episode(self, seed) -> None:
        self._episode_count += 1
        cfg  = self.cfg
        rcfg = cfg["resources"]
        ecfg = cfg["enemies"]
        ccfg = cfg["combat"]

        # Difficulty scale: each episode slightly harder (play mode only)
        sc = self._difficulty_scale

        import random
        rng_r = random.Random(seed)

        n_scorpions    = rng_r.randint(ecfg["scorpion_count_min"],
                                       max(ecfg["scorpion_count_min"],
                                           int(ecfg["scorpion_count_max"] * sc)))
        n_mem_eaters   = rng_r.randint(ecfg["memory_eater_count_min"],
                                       max(ecfg["memory_eater_count_min"],
                                           int(ecfg["memory_eater_count_max"] * sc)))
        n_wanderers    = rng_r.randint(ecfg["wanderer_count_min"],
                                       max(ecfg["wanderer_count_min"],
                                           int(ecfg["wanderer_count_max"] * sc)))
        n_enemies      = n_scorpions + n_mem_eaters + n_wanderers + 1
        n_hp_items     = rng_r.randint(2, 4)
        n_mem_items    = rng_r.randint(2, 4)

        map_gen = MapGenerator(self.map_w, self.map_h, seed)
        tiles, rooms, corridors, hazards = map_gen.generate()
        spawns = map_gen.pick_spawn_positions(
            tiles, rooms, n_enemies, n_hp_items, n_mem_items)

        state = GameState(
            map_width=self.map_w, map_height=self.map_h,
            tiles=tiles,
            last_seen=np.full((self.map_h, self.map_w), -1, dtype=np.int32),
            hazard_positions=hazards,
        )

        ax, ay = spawns["agent"]
        state.agent = AgentState(
            x=ax, y=ay,
            hp=rcfg["hp_max"], hp_max=rcfg["hp_max"],
            memory_stability=rcfg["memory_stability_max"],
            memory_stability_max=rcfg["memory_stability_max"],
        )
        state.artifact_pos = spawns["artifact"]

        eq = list(spawns["enemies"])

        for _ in range(n_scorpions):
            if eq:
                p = eq.pop(0)
                state.add_entity(EntityType.SCORPION, p[0], p[1],
                    hp=ccfg["scorpion_hp"], damage=ccfg["scorpion_damage"], defense=2)

        for _ in range(n_mem_eaters):
            if eq:
                p = eq.pop(0)
                state.add_entity(EntityType.MEMORY_EATER, p[0], p[1],
                    hp=ccfg["memory_eater_hp"], damage=ccfg["memory_eater_damage"],
                    stability_drain=ccfg["memory_eater_stability_drain"])

        for _ in range(n_wanderers):
            if eq:
                p = eq.pop(0)
                state.add_entity(EntityType.WANDERER, p[0], p[1],
                    hp=ccfg["wanderer_hp"], damage=ccfg["wanderer_damage"], defense=1)

        if eq:
            p = eq.pop(0)
        else:
            p = spawns["guardian"]
        guardian = state.add_entity(EntityType.GUARDIAN, p[0], p[1],
            hp=ccfg["guardian_hp"], damage=ccfg["guardian_damage"],
            stability_drain=ccfg["guardian_stability_drain"], defense=5)
        state.guardian_id = guardian.entity_id

        # Items
        for pos in spawns.get("hp_items", []):
            state.add_entity(EntityType.ITEM_HP, pos[0], pos[1], hp=1, item_value=Inventory.HP_RESTORE)
        for pos in spawns.get("mem_items", []):
            state.add_entity(EntityType.ITEM_MEMORY, pos[0], pos[1], hp=1, item_value=Inventory.MEM_RESTORE)

        # Memory Shrines — restore 30 stability points, one-use
        SHRINE_RESTORE = 30
        for pos in spawns.get("shrines", []):
            state.add_entity(EntityType.SHRINE, pos[0], pos[1],
                             hp=1, item_value=SHRINE_RESTORE)

        self.state = state
        self.mem   = MemorySystem(cfg, self._episode_rng)
        self.obs_builder = ObservationBuilder(self.map_w, self.map_h, self.mem, self._episode_rng)
        self.combat_sys  = JRPGCombat(cfg, self._episode_rng)

        state.update_action_mask()

        # Initial FOV
        fov = compute_fov(ax, ay, FOV_RADIUS, tiles, self.map_h, self.map_w)
        state.update_visibility(fov)

    # ─── Info / Render ────────────────────────────────────────────────────────

    def _get_info(self) -> dict:
        a = self.state.agent
        info = {
            "step": self.state.step,
            "phase": self.state.phase.name,
            "hp": a.hp,
            "memory_stability": self.mem.stability,
            "degradation_level": int(self.mem.level),
            "artifact_collected": a.artifact_collected,
            "guardian_defeated": a.guardian_defeated,
            "hp_potions": a.inventory.hp_potions,
            "memory_potions": a.inventory.memory_potions,
            "is_victory": self.state.phase.name == "VICTORY",
            "is_death": a.hp <= 0,
            "is_memory_collapse": self.mem.is_collapsed,
        }
        if self.state.phase == GamePhase.COMBAT and self.state.combat:
            info["combat_log"] = self.state.combat.combat_log[-3:]
        return info

    def _render_frame(self) -> None:
        try:
            from .rendering.pygame_renderer import PygameRenderer
            if self._renderer is None:
                self._renderer = PygameRenderer(self.map_w, self.map_h, self.cfg)
            self._renderer.render(self.state, self.mem)
        except Exception as e:
            pass

    def _ansi_render(self) -> str:
        state  = self.state
        agent  = state.agent
        lines  = []
        tchars = {
            Tile.VOID:"  ", Tile.FLOOR:".", Tile.WALL:"#",
            Tile.SAND:"~", Tile.RUIN:"%", Tile.DOOR:"+", Tile.HAZARD:"^",
        }
        echars = {
            EntityType.SCORPION:"s", EntityType.MEMORY_EATER:"M",
            EntityType.WANDERER:"w", EntityType.GUARDIAN:"G",
            EntityType.ITEM_HP:"!", EntityType.ITEM_MEMORY:"?",
            EntityType.SHRINE:"$",
        }
        for r in range(self.map_h):
            row = []
            for c in range(self.map_w):
                if state.cell_age(c, r) < 0:
                    row.append(" "); continue
                if (c,r) == (agent.x, agent.y):
                    row.append("@"); continue
                if (c,r) == state.artifact_pos and not agent.artifact_collected:
                    row.append("*"); continue
                ents = state.get_entities_at(c,r)
                e = next((x for x in ents if x.is_enemy()), None) or \
                    next((x for x in ents if x.is_item()), None)
                if e:
                    row.append(echars.get(e.entity_type, "?"))
                else:
                    row.append(tchars.get(Tile(state.tiles[r,c]), "?"))
            lines.append("".join(row))

        lines.append(
            f"HP:{agent.hp:3d}  MEM:{self.mem.stability:5.1f}"
            f"  [{self.mem.level.name}]  Step:{state.step}"
            f"  Phase:{state.phase.name}"
        )
        if state.phase == GamePhase.COMBAT and state.combat:
            lines += state.combat.combat_log[-3:]
        return "\n".join(lines)
