"""
Pygame Renderer — 1280×720 fixed, sprite-first with colored-rect fallback.

Layout:
  LEFT  (992px)  : map viewport, camera-centered on agent
  RIGHT (288px)  : HUD — HP/MEM bars, objectives, inventory, minimap, log

Visual effects:
  - Sprites from world.png / battle.png (auto-detected)
  - Starfield texture for never-seen void cells
  - Blue-shift + alpha fade on memory-decayed cells
  - Red vignette pulsing at HEAVY/CRITICAL degradation
  - JRPG combat overlay with large battle sprite
  - Degradable minimap in HUD
"""

from __future__ import annotations
import pygame
import numpy as np
import math
import os
from typing import Optional

from ..world.game_state import (
    GameState, GamePhase, EntityType, Tile, CombatState, StatusEffect
)
from ..memory.memory_system import MemorySystem
from ..memory.degradation import DegradationLevel
from .sprite_loader import SpriteLoader

# ─── Layout constants ─────────────────────────────────────────────────────────

WIN_W, WIN_H = 1280, 720
HUD_W        = 288
MAP_W        = WIN_W - HUD_W   # 992
TILE_SZ      = 24
BATTLE_SZ    = 128

# ─── Palette ──────────────────────────────────────────────────────────────────

DARK_BG    = (8,   6,  14)
PANEL_BG   = (16,  12,  26)
PANEL_LINE = (50,  40,  70)
WHITE      = (255, 255, 255)
DIM        = (120, 110, 140)

TILE_FALLBACK = {
    Tile.FLOOR:  (55,  50,  40),
    Tile.WALL:   (75,  70,  85),
    Tile.SAND:   (140, 120, 70),
    Tile.RUIN:   (65,  55,  50),
    Tile.HAZARD: (110, 25,  15),
    Tile.DOOR:   (90,  130, 110),
    Tile.VOID:   (4,   4,   8),
}

ENTITY_FALLBACK = {
    EntityType.AGENT:        (220, 220, 255),
    EntityType.SCORPION:     (200,  55,  35),
    EntityType.MEMORY_EATER: (170,  35, 195),
    EntityType.WANDERER:     (195, 155,  35),
    EntityType.GUARDIAN:     (255,  75,  25),
    EntityType.ARTIFACT:     (255, 215,  50),
    EntityType.ITEM_HP:      (55,  195,  75),
    EntityType.ITEM_MEMORY:  (70,  150, 215),
    EntityType.SHRINE:       (80,  210, 255),   # cyan-blue pillar
}

TILE_SPRITE_KEY = {
    Tile.FLOOR:  "floor",
    Tile.WALL:   "wall",
    Tile.SAND:   "sand",
    Tile.RUIN:   "ruin",
    Tile.HAZARD: "hazard",
    Tile.DOOR:   "door",
}

ENTITY_SPRITE_KEY = {
    EntityType.AGENT:        "agent",
    EntityType.SCORPION:     "scorpion",
    EntityType.MEMORY_EATER: "memory_eater",
    EntityType.WANDERER:     "wanderer",
    EntityType.GUARDIAN:     "guardian",
    EntityType.ARTIFACT:     "artifact",
    EntityType.ITEM_HP:      "item_hp",
    EntityType.ITEM_MEMORY:  "item_mem",
    EntityType.SHRINE:       "shrine",   # no sprite yet — draws custom fallback
}

BATTLE_SPRITE_KEY = {
    EntityType.SCORPION:     "scorpion",
    EntityType.MEMORY_EATER: "memory_eater",
    EntityType.WANDERER:     "wanderer",
    EntityType.GUARDIAN:     "guardian",
}

LEVEL_COLOR = {
    DegradationLevel.NONE:     (55,  195,  95),
    DegradationLevel.MINOR:    (195, 195,  55),
    DegradationLevel.MODERATE: (215, 155,  35),
    DegradationLevel.HEAVY:    (215,  75,  35),
    DegradationLevel.CRITICAL: (255,  25,  25),
}

LEVEL_LABEL = {
    DegradationLevel.NONE:     "STABLE",
    DegradationLevel.MINOR:    "MINOR DROUGHT",
    DegradationLevel.MODERATE: "MOD. DROUGHT",
    DegradationLevel.HEAVY:    "HEAVY DROUGHT",
    DegradationLevel.CRITICAL: "!! CRITICAL !!",
}

ENEMY_NAME = {
    EntityType.SCORPION:     "Scorpion",
    EntityType.MEMORY_EATER: "Memory Eater",
    EntityType.WANDERER:     "Forgotten Wanderer",
    EntityType.GUARDIAN:     "The Guardian",
}

STATUS_COLOR = {
    StatusEffect.NONE:     None,
    StatusEffect.POISONED: (55, 200, 55),
    StatusEffect.CONFUSED: (180, 55, 200),
    StatusEffect.WEAKENED: (200, 155, 55),
}


class PygameRenderer:
    def __init__(self, map_w: int, map_h: int, cfg: dict, fullscreen: bool = True):
        self.map_w = map_w
        self.map_h = map_h
        self.cfg   = cfg
        self._fullscreen = fullscreen

        pygame.init()
        flags = (pygame.FULLSCREEN | pygame.SCALED) if fullscreen else 0
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), flags)
        pygame.display.set_caption("Zahramun  |  Memory-Augmented RL Benchmark")
        pygame.display.set_icon(self._make_icon())

        self.sprites = SpriteLoader()

        self.font_title = self._font(20, bold=True)
        self.font_lg    = self._font(17)
        self.font_md    = self._font(14)
        self.font_sm    = self._font(12)
        self.font_mono  = self._font(13, mono=True)

        self.clock  = pygame.time.Clock()
        self._frame = 0
        self._show_truth = False

        # Pre-bake surfaces
        self._starfield   = self._bake_starfield()
        self._vignette    = self._bake_vignette()
        self._decay_cache: dict[tuple, pygame.Surface] = {}

    # ─── Public API ───────────────────────────────────────────────────────────

    def render(self, state: GameState, mem: MemorySystem) -> None:
        self._frame += 1
        self.screen.fill(DARK_BG)

        if state.phase == GamePhase.COMBAT and state.combat:
            self._draw_map(state, mem)          # dim background
            self._draw_combat_overlay(state, mem)
        else:
            self._draw_map(state, mem)

        self._draw_hud(state, mem)
        self._draw_vignette(mem)

        pygame.display.flip()
        self.clock.tick(30)

        for ev in pygame.event.get(pygame.QUIT):
            self.close()

    def toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        flags = (pygame.FULLSCREEN | pygame.SCALED) if self._fullscreen else 0
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), flags)

    def close(self) -> None:
        pygame.quit()

    # ─── Map Drawing ──────────────────────────────────────────────────────────

    def _draw_map(self, state: GameState, mem: MemorySystem) -> None:
        agent  = state.agent
        # Camera: center on agent
        cam_x = agent.x * TILE_SZ - MAP_W // 2
        cam_y = agent.y * TILE_SZ - WIN_H // 2
        cam_x = max(0, min(cam_x, self.map_w * TILE_SZ - MAP_W))
        cam_y = max(0, min(cam_y, self.map_h * TILE_SZ - WIN_H))

        # Clip to map area
        map_clip = pygame.Rect(0, 0, MAP_W, WIN_H)
        self.screen.set_clip(map_clip)

        for r in range(self.map_h):
            for c in range(self.map_w):
                px = c * TILE_SZ - cam_x
                py = r * TILE_SZ - cam_y
                if px > MAP_W or py > WIN_H or px < -TILE_SZ or py < -TILE_SZ:
                    continue

                age  = state.cell_age(c, r)
                rect = pygame.Rect(px, py, TILE_SZ, TILE_SZ)

                if age < 0 and not self._show_truth:
                    # Never seen — starfield void
                    sx = (c * TILE_SZ) % self._starfield.get_width()
                    sy = (r * TILE_SZ) % self._starfield.get_height()
                    self.screen.blit(self._starfield, (px, py),
                                     (sx, sy, TILE_SZ, TILE_SZ))
                    continue

                tile = Tile(state.tiles[r, c])
                self._draw_tile(px, py, tile, age, mem)

        # Entities
        for ent in state.entities.values():
            if not ent.alive:
                continue
            px = ent.x * TILE_SZ - cam_x
            py = ent.y * TILE_SZ - cam_y
            if px > MAP_W or py > WIN_H or px < -TILE_SZ or py < -TILE_SZ:
                continue
            age = state.cell_age(ent.x, ent.y)
            if age < 0 and not self._show_truth:
                continue
            if (ent.x, ent.y) == (agent.x, agent.y):
                continue  # agent drawn separately
            self._draw_entity(px, py, ent, age)

        # Agent (always on top)
        apx = agent.x * TILE_SZ - cam_x
        apy = agent.y * TILE_SZ - cam_y
        self._draw_agent(apx, apy, agent)

        self.screen.set_clip(None)

    def _draw_tile(self, px: int, py: int, tile: Tile, age: int, mem: MemorySystem) -> None:
        rect   = pygame.Rect(px, py, TILE_SZ, TILE_SZ)
        skey   = TILE_SPRITE_KEY.get(tile)
        sprite = self.sprites.get(skey, TILE_SZ) if skey else None
        decay  = min(age / 80.0, 1.0) if age >= 0 else 0.0

        if sprite:
            if decay > 0.05:
                alpha = int(255 * (1.0 - decay * 0.75))
                s = sprite.copy()
                s.set_alpha(max(40, alpha))
                # Blue-tint overlay for decay
                if decay > 0.2:
                    tint = pygame.Surface((TILE_SZ, TILE_SZ), pygame.SRCALPHA)
                    tint.fill((10, 15, 60, int(decay * 160)))
                    self.screen.blit(sprite, (px, py))
                    self.screen.blit(tint, (px, py))
                else:
                    self.screen.blit(sprite, (px, py))
            else:
                self.screen.blit(sprite, (px, py))
        else:
            color = TILE_FALLBACK.get(tile, (30, 30, 30))
            color = _decay_color(color, decay)
            pygame.draw.rect(self.screen, color, rect)

        # Hazard pulse overlay
        if tile == Tile.HAZARD:
            pulse = 0.4 + 0.4 * math.sin(self._frame * 0.14)
            overlay = pygame.Surface((TILE_SZ, TILE_SZ), pygame.SRCALPHA)
            overlay.fill((200, 30, 10, int(pulse * 120)))
            self.screen.blit(overlay, (px, py))

    def _draw_entity(self, px: int, py: int, ent, age: int) -> None:
        # ── Shrine — custom pulsing diamond ──────────────────────────────────
        if ent.entity_type == EntityType.SHRINE:
            pulse  = 0.5 + 0.5 * math.sin(self._frame * 0.10)
            scol   = (80, int(180 + 75 * pulse), 255)
            # Outer glow
            glow   = pygame.Surface((TILE_SZ * 2, TILE_SZ * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*scol, int(45 * pulse)),
                               (TILE_SZ, TILE_SZ), TILE_SZ - 2)
            self.screen.blit(glow, (px - TILE_SZ // 2, py - TILE_SZ // 2))
            # Diamond shape
            cx, cy = px + TILE_SZ // 2, py + TILE_SZ // 2
            r      = TILE_SZ // 2 - 2
            pts    = [(cx, py + 2), (px + TILE_SZ - 2, cy),
                      (cx, py + TILE_SZ - 2), (px + 2, cy)]
            pygame.draw.polygon(self.screen, _darken(scol, 0.5), pts)
            pygame.draw.polygon(self.screen, scol, pts, 2)
            return

        skey   = ENTITY_SPRITE_KEY.get(ent.entity_type)
        sprite = self.sprites.get(skey, TILE_SZ) if skey else None
        decay  = min(age / 80.0, 1.0) if age >= 0 else 0.0

        if sprite:
            if decay > 0.1:
                s = sprite.copy()
                s.set_alpha(max(60, int(255 * (1.0 - decay * 0.6))))
                self.screen.blit(s, (px, py))
            else:
                self.screen.blit(sprite, (px, py))
        else:
            color = ENTITY_FALLBACK.get(ent.entity_type, WHITE)
            color = _decay_color(color, decay * 0.5)
            cx, cy = px + TILE_SZ // 2, py + TILE_SZ // 2
            r = TILE_SZ // 2 - 2
            pygame.draw.circle(self.screen, _darken(color, 0.4), (cx, cy), r)
            pygame.draw.circle(self.screen, color, (cx, cy), r, 2)
            sym = ENTITY_SPRITE_KEY.get(ent.entity_type, "?")[0].upper()
            s = self.font_mono.render(sym, True, color)
            self.screen.blit(s, (px + (TILE_SZ - s.get_width()) // 2,
                                  py + (TILE_SZ - s.get_height()) // 2))

        # Artifact pulsing glow — makes it easy to spot
        if ent.entity_type == EntityType.ARTIFACT:
            pulse = 0.4 + 0.6 * abs(math.sin(self._frame * 0.07))
            glow  = pygame.Surface((TILE_SZ * 3, TILE_SZ * 3), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 215, 50, int(60 * pulse)),
                               (TILE_SZ * 3 // 2, TILE_SZ * 3 // 2), TILE_SZ)
            self.screen.blit(glow, (px - TILE_SZ, py - TILE_SZ))

        # Enemy HP micro-bar
        if ent.is_enemy():
            pct = ent.hp / ent.hp_max
            pygame.draw.rect(self.screen, (50, 15, 15), (px, py + TILE_SZ - 3, TILE_SZ, 3))
            pygame.draw.rect(self.screen, (200, 50, 35),
                             (px, py + TILE_SZ - 3, int(TILE_SZ * pct), 3))

    def _draw_agent(self, px: int, py: int, agent) -> None:
        sprite = self.sprites.get("agent", TILE_SZ)
        # Glow
        glow = pygame.Surface((TILE_SZ * 3, TILE_SZ * 3), pygame.SRCALPHA)
        g_alpha = 25 + int(18 * math.sin(self._frame * 0.09))
        col = ENTITY_FALLBACK[EntityType.AGENT]
        pygame.draw.circle(glow, (*col, g_alpha),
                           (TILE_SZ * 3 // 2, TILE_SZ * 3 // 2), TILE_SZ)
        self.screen.blit(glow, (px - TILE_SZ, py - TILE_SZ))

        if sprite:
            self.screen.blit(sprite, (px, py))
        else:
            cx, cy = px + TILE_SZ // 2, py + TILE_SZ // 2
            pygame.draw.circle(self.screen, _darken(col, 0.3), (cx, cy), TILE_SZ // 2 - 1)
            pygame.draw.circle(self.screen, col, (cx, cy), TILE_SZ // 2 - 1, 2)
            s = self.font_mono.render("@", True, col)
            self.screen.blit(s, (px + (TILE_SZ - s.get_width()) // 2,
                                  py + (TILE_SZ - s.get_height()) // 2))

        # Status indicator
        if agent.status != StatusEffect.NONE:
            scol = STATUS_COLOR.get(agent.status, (200, 200, 200))
            if scol:
                pulse = 0.5 + 0.5 * math.sin(self._frame * 0.2)
                ind = pygame.Surface((6, 6), pygame.SRCALPHA)
                ind.fill((*scol, int(180 * pulse)))
                self.screen.blit(ind, (px + TILE_SZ - 7, py))

    # ─── Combat Overlay ───────────────────────────────────────────────────────

    def _draw_combat_overlay(self, state: GameState, mem: MemorySystem) -> None:
        combat = state.combat
        enemy  = state.entities.get(combat.enemy_id)
        if not enemy:
            return

        agent  = state.agent
        ecolor = ENTITY_FALLBACK.get(enemy.entity_type, WHITE)
        ename  = ENEMY_NAME.get(enemy.entity_type, "Enemy")

        # ── Full-area dim ─────────────────────────────────────────────────────
        dim = pygame.Surface((MAP_W, WIN_H), pygame.SRCALPHA)
        dim.fill((4, 0, 16, 215))
        self.screen.blit(dim, (0, 0))

        # ── Panel ─────────────────────────────────────────────────────────────
        pw, ph   = 740, 520
        px_off   = (MAP_W - pw) // 2
        py_off   = (WIN_H - ph) // 2
        inner_w  = pw - 40          # 700

        # Enemy-coloured glow behind panel
        glow = pygame.Surface((pw + 80, ph + 80), pygame.SRCALPHA)
        glow_a = int(14 + 8 * math.sin(self._frame * 0.05))
        pygame.draw.rect(glow, (*ecolor, glow_a), (0, 0, pw + 80, ph + 80),
                         border_radius=24)
        self.screen.blit(glow, (px_off - 40, py_off - 40))

        # Panel body
        pygame.draw.rect(self.screen, (13, 10, 27),
                         (px_off, py_off, pw, ph), border_radius=16)
        pygame.draw.rect(self.screen, PANEL_LINE,
                         (px_off, py_off, pw, ph), 2, border_radius=16)

        ix = px_off + 20
        iy = py_off + 16

        # ── Section 1: Enemy name + HP bar ───────────────────────────────────
        self._txt(ename, ix, iy, self.font_title, ecolor)
        iy += 28

        ehp_pct = enemy.hp / enemy.hp_max
        self._bar(ix, iy, inner_w, 20, ehp_pct,
                  (195, 55, 35), (55, 18, 18),
                  label=f"HP  {enemy.hp} / {enemy.hp_max}")
        iy += 28
        self._divider(ix, iy, inner_w)
        iy += 10

        # ── Section 2: Sprite arena ───────────────────────────────────────────
        SPRITE_SZ = 190
        AGENT_SZ  = 95
        col_left  = 120     # agent column
        col_right = SPRITE_SZ + 20
        col_mid   = inner_w - col_left - col_right   # log column

        # --- Enemy sprite (right, bobbing) ---
        bob    = int(5 * math.sin(self._frame * 0.08))
        esp_x  = ix + col_left + col_mid + 10
        esp_y  = iy + bob
        bkey   = BATTLE_SPRITE_KEY.get(enemy.entity_type)
        bsprite = self.sprites.get_battle(bkey, SPRITE_SZ) if bkey else None

        # Soft glow ellipse behind enemy sprite
        eg = pygame.Surface((SPRITE_SZ + 40, SPRITE_SZ + 10), pygame.SRCALPHA)
        eg_a = int(30 + 18 * math.sin(self._frame * 0.06))
        pygame.draw.ellipse(eg, (*ecolor, eg_a), eg.get_rect())
        self.screen.blit(eg, (esp_x - 20, esp_y + SPRITE_SZ // 2 - 5))

        if bsprite:
            self.screen.blit(bsprite, (esp_x, esp_y))
        else:
            bg = pygame.Surface((SPRITE_SZ, SPRITE_SZ), pygame.SRCALPHA)
            bg.fill((*_darken(ecolor, 0.7), 200))
            self.screen.blit(bg, (esp_x, esp_y))
            pygame.draw.rect(self.screen, ecolor,
                             pygame.Rect(esp_x, esp_y, SPRITE_SZ, SPRITE_SZ),
                             2, border_radius=8)
            sym       = ename[0]
            big_font  = self._font(80, bold=True)
            s         = big_font.render(sym, True, ecolor)
            self.screen.blit(s, (esp_x + (SPRITE_SZ - s.get_width()) // 2,
                                  esp_y + (SPRITE_SZ - s.get_height()) // 2))

        # --- Agent side (left) ---
        self._txt("YOU", ix, iy, self.font_sm, (175, 175, 220))
        asp_y   = iy + SPRITE_SZ - AGENT_SZ
        asprite = self.sprites.get("agent", AGENT_SZ)
        if asprite:
            self.screen.blit(asprite, (ix, asp_y))
        else:
            acol = ENTITY_FALLBACK[EntityType.AGENT]
            pygame.draw.circle(self.screen, acol,
                               (ix + AGENT_SZ // 2, asp_y + AGENT_SZ // 2),
                               AGENT_SZ // 2 - 2, 2)

        # Agent HP + MEM bars below sprite
        bar_y  = iy + SPRITE_SZ + 6
        hp_pct = agent.hp / agent.hp_max
        self._bar(ix, bar_y, col_left - 4, 13, hp_pct,
                  (85, 185, 85), (22, 50, 22),
                  label=f"{agent.hp}/{agent.hp_max}")
        bar_y += 17
        lc = LEVEL_COLOR[mem.level]
        self._bar(ix, bar_y, col_left - 4, 13, mem.normalized(),
                  lc, (22, 16, 38),
                  label=f"{mem.stability:.0f}")

        # --- Combat log (center column) ---
        lx      = ix + col_left + 8
        log_w   = col_mid - 16
        self._txt("- Combat Log -", lx, iy, self.font_sm, DIM)
        ly      = iy + 20
        log     = combat.combat_log[-7:]
        for i, msg in enumerate(log):
            frac  = (i + 1) / max(len(log), 1)
            c     = int(70 + 185 * frac)
            shown = msg[:33] + ("..." if len(msg) > 33 else "")
            self._txt(shown, lx, ly + i * 21, self.font_sm, (c, c, min(255, c + 25)))

        # ── Section 3: Action area ────────────────────────────────────────────
        act_y = iy + SPRITE_SZ + 44

        self._divider(ix, act_y, inner_w)
        act_y += 10

        # Turn indicator (pulsing)
        turn_col = (155, 215, 155)
        pulse    = 0.8 + 0.2 * math.sin(self._frame * 0.14)
        tc       = tuple(int(c * pulse) for c in turn_col)
        ts       = self.font_md.render("- YOUR TURN -", True, tc)
        self.screen.blit(ts, (px_off + pw // 2 - ts.get_width() // 2, act_y))
        act_y += 28

        # Action buttons — 3x2 grid with styled boxes
        inv     = agent.inventory
        mpc     = agent.memory_pulse_charges
        actions = [
            ("F",  "Attack",            (210,  75,  55), True),
            ("E",  "Defend  (1/2 dmg)", ( 75, 135, 200), True),
            ("C",  f"Pulse  {mpc}/3",   (155,  70, 215), mpc > 0),
            ("H",  f"HP Pot  x{inv.hp_potions}",    ( 55, 190,  90), inv.hp_potions > 0),
            ("N",  f"Mem Pot x{inv.memory_potions}", ( 70, 145, 215), inv.memory_potions > 0),
            ("X",  "Flee",              (130, 125, 145), True),
        ]
        n_cols  = 3
        gap     = 8
        btn_w   = (inner_w - gap * (n_cols - 1)) // n_cols
        btn_h   = 44
        for i, (key, desc, col, avail) in enumerate(actions):
            col_i  = i % n_cols
            row_i  = i // n_cols
            bx     = ix + col_i * (btn_w + gap)
            by     = act_y + row_i * (btn_h + gap)

            btn_bg  = _darken(col, 0.82) if avail else (16, 14, 24)
            btn_bdr = col if avail else (45, 40, 58)
            pygame.draw.rect(self.screen, btn_bg,
                             (bx, by, btn_w, btn_h), border_radius=7)
            pygame.draw.rect(self.screen, btn_bdr,
                             (bx, by, btn_w, btn_h), 1, border_radius=7)

            tcol = col if avail else (55, 50, 68)
            self._txt(f"[{key}]", bx + 10, by + 7,  self.font_md, tcol)
            self._txt(desc,       bx + 10, by + 27, self.font_sm,
                      _darken(tcol, 0.1) if avail else (45, 42, 58))

    # ─── HUD ──────────────────────────────────────────────────────────────────

    def _draw_hud(self, state: GameState, mem: MemorySystem) -> None:
        hx = MAP_W
        pygame.draw.rect(self.screen, PANEL_BG, (hx, 0, HUD_W, WIN_H))
        pygame.draw.line(self.screen, PANEL_LINE, (hx, 0), (hx, WIN_H), 2)

        x = hx + 14
        y = 14
        agent = state.agent
        level = mem.level
        lc    = LEVEL_COLOR[level]

        # Title
        self._txt("ZAHRAMUN", x, y, self.font_title, (175, 155, 220))
        y += 30

        # Phase + step
        phase_c = {
            GamePhase.EXPLORING: (95,  195, 115),
            GamePhase.COMBAT:    (215, 115,  55),
            GamePhase.VICTORY:   (215, 195,  55),
            GamePhase.GAME_OVER: (195,  55,  55),
        }.get(state.phase, WHITE)
        self._txt(f"{state.phase.name}   step {state.step}",
                  x, y, self.font_sm, phase_c)
        y += 20

        # Status effect
        if agent.status != StatusEffect.NONE:
            scol = STATUS_COLOR.get(agent.status, (200, 200, 200)) or (200, 200, 200)
            self._txt(f"STATUS: {agent.status.name}  ({agent.status_duration})",
                      x, y, self.font_sm, scol)
            y += 18

        y += 4
        # HP
        self._txt("HP", x, y, self.font_sm, (195, 75, 75))
        y += 16
        self._bar(x, y, HUD_W - 28, 18,
                  agent.hp / agent.hp_max,
                  (195, 65, 55), (55, 18, 18),
                  label=f"{agent.hp}/{agent.hp_max}")
        y += 28

        # Memory
        self._txt("MEMORY", x, y, self.font_sm, lc)
        y += 16
        self._bar(x, y, HUD_W - 28, 18,
                  mem.normalized(), lc, (28, 18, 40),
                  label=f"{mem.stability:.0f}/{mem.stability_max:.0f}")
        y += 26
        self._txt(LEVEL_LABEL[level], x, y, self.font_sm, lc)
        y += 22

        self._divider(x, y, HUD_W - 28)
        y += 10

        # Objectives
        self._txt("OBJECTIVES", x, y, self.font_sm, DIM)
        y += 18
        ac = (255, 215, 50) if agent.artifact_collected else (80, 75, 90)
        gc = (255, 115, 50) if agent.guardian_defeated  else (80, 75, 90)
        self._txt(("[X]" if agent.artifact_collected else "[ ]") + "  Artifact", x, y, self.font_sm, ac)
        y += 18
        self._txt(("[X]" if agent.guardian_defeated  else "[ ]") + "  Guardian", x, y, self.font_sm, gc)
        y += 18

        # Artifact compass — direction + distance when not collected
        if not agent.artifact_collected and state.artifact_pos:
            ax_p, ay_p = state.artifact_pos
            ddx = ax_p - agent.x
            ddy = ay_p - agent.y
            dist = int(abs(ddx) + abs(ddy))
            arrow = _compass_arrow(ddx, ddy)
            pulse = 0.7 + 0.3 * math.sin(self._frame * 0.12)
            ac2   = tuple(int(c * pulse) for c in (255, 215, 50))
            self._txt(f"  {arrow}  {dist} tiles", x, y, self.font_sm, ac2)
            y += 16
        else:
            y += 2

        self._divider(x, y, HUD_W - 28)
        y += 10

        # Inventory
        self._txt("INVENTORY", x, y, self.font_sm, DIM)
        y += 18
        self._txt(f"HP Potions    x{agent.inventory.hp_potions}", x, y, self.font_sm, (55, 195, 80))
        y += 18
        self._txt(f"Mem Potions   x{agent.inventory.memory_potions}", x, y, self.font_sm, (70, 148, 215))
        y += 18
        self._txt(f"Memory Pulse  {agent.memory_pulse_charges}/3", x, y, self.font_sm, (155, 70, 215))
        y += 22

        self._divider(x, y, HUD_W - 28)
        y += 10

        # Minimap
        y = self._draw_minimap(state, mem, x, y)
        y += 8

        self._divider(x, y, HUD_W - 28)
        y += 10

        # Controls (compact)
        self._txt("CONTROLS", x, y, self.font_sm, DIM)
        y += 16
        ctrl = [
            "WASD/Arrows  move",
            "F/Space      attack",
            "E            defend",
            "C            Mem Pulse",
            "H/N          potions",
            "X            flee",
            "T            truth mode",
            "R            restart",
        ]
        for line in ctrl:
            self._txt(line, x, y, self.font_sm, (90, 85, 108))
            y += 15

        # Position
        self._txt(f"({agent.x},{agent.y})", x, WIN_H - 18, self.font_sm, (65, 60, 82))

    # ─── Minimap ──────────────────────────────────────────────────────────────

    def _draw_minimap(self, state: GameState, mem: MemorySystem, x: int, y: int) -> int:
        """
        Degradable minimap — memory drought corrupts it.
        Returns new y after drawing.
        """
        self._txt("MINIMAP", x, y, self.font_sm, DIM)
        y += 16

        mw = HUD_W - 28
        mh = 90
        cell_w = mw / self.map_w
        cell_h = mh / self.map_h

        bg = pygame.Surface((mw, mh))
        bg.fill((8, 6, 14))

        level  = mem.level
        agent  = state.agent

        for r in range(self.map_h):
            for c in range(self.map_w):
                age = state.cell_age(c, r)
                if age < 0:
                    continue  # never seen — black

                tile  = Tile(state.tiles[r, c])
                decay = min(age / 80.0, 1.0)

                # Minimap degrades: at HEAVY+ level, old cells go blank
                if level >= DegradationLevel.HEAVY and decay > 0.4:
                    continue
                if level >= DegradationLevel.CRITICAL and decay > 0.2:
                    continue

                base_color = TILE_FALLBACK.get(tile, (30, 30, 30))
                color = _decay_color(base_color, decay * 0.6)

                cx = int(c * cell_w)
                cy = int(r * cell_h)
                cw = max(1, int(cell_w))
                ch = max(1, int(cell_h))
                pygame.draw.rect(bg, color, (cx, cy, cw, ch))

        # Enemies on minimap (degrade with memory)
        for ent in state.living_enemies():
            age = state.cell_age(ent.x, ent.y)
            if age < 0:
                continue
            decay = min(age / 60.0, 1.0)
            if level >= DegradationLevel.MODERATE and decay > 0.5:
                continue
            ec = ENTITY_FALLBACK.get(ent.entity_type, (200, 50, 50))
            ex = int(ent.x * cell_w)
            ey = int(ent.y * cell_h)
            pygame.draw.rect(bg, ec, (ex, ey, max(2, int(cell_w)), max(2, int(cell_h))))

        # Artifact
        if state.artifact_pos and not agent.artifact_collected:
            ax_p, ay_p = state.artifact_pos
            age = state.cell_age(ax_p, ay_p)
            if age >= 0:
                ex = int(ax_p * cell_w)
                ey = int(ay_p * cell_h)
                pygame.draw.rect(bg, (255, 215, 50),
                                 (ex, ey, max(2, int(cell_w)), max(2, int(cell_h))))

        # Agent dot (always shown)
        adx = int(agent.x * cell_w)
        ady = int(agent.y * cell_h)
        pygame.draw.rect(bg, (220, 220, 255), (adx, ady, max(2, int(cell_w)), max(2, int(cell_h))))

        # Border
        pygame.draw.rect(bg, PANEL_LINE, (0, 0, mw, mh), 1)

        self.screen.blit(bg, (x, y))
        return y + mh

    # ─── Vignette ─────────────────────────────────────────────────────────────

    def _draw_vignette(self, mem: MemorySystem) -> None:
        level = mem.level
        if level < DegradationLevel.HEAVY:
            return
        base_alpha = {
            DegradationLevel.HEAVY:    70,
            DegradationLevel.CRITICAL: 140,
        }.get(level, 0)
        if level == DegradationLevel.CRITICAL:
            pulse = 0.5 + 0.5 * math.sin(self._frame * 0.07)
            base_alpha = int(base_alpha * (0.65 + 0.35 * pulse))
        v = self._vignette.copy()
        v.set_alpha(base_alpha)
        self.screen.blit(v, (0, 0))

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _txt(self, text, x, y, font, color):
        self.screen.blit(font.render(str(text), True, color), (x, y))

    def _bar(self, x, y, w, h, pct, fill, bg, label=""):
        pygame.draw.rect(self.screen, bg,   (x, y, w, h), border_radius=4)
        fw = max(0, int(w * max(0.0, min(1.0, pct))))
        if fw:
            pygame.draw.rect(self.screen, fill, (x, y, fw, h), border_radius=4)
        pygame.draw.rect(self.screen, PANEL_LINE, (x, y, w, h), 1, border_radius=4)
        if label:
            ls = self.font_sm.render(label, True, WHITE)
            self.screen.blit(ls, (x + w // 2 - ls.get_width() // 2,
                                  y + h // 2 - ls.get_height() // 2))

    def _divider(self, x, y, w):
        pygame.draw.line(self.screen, PANEL_LINE, (x, y), (x + w, y), 1)

    # ─── Surface Factories ────────────────────────────────────────────────────

    def _bake_starfield(self) -> pygame.Surface:
        """96×96 tiling starfield texture — dark void with tiny stars."""
        sz  = 96
        surf = pygame.Surface((sz, sz))
        surf.fill((3, 2, 8))
        rng = np.random.default_rng(999)
        n_stars = 40
        xs = rng.integers(0, sz, n_stars)
        ys = rng.integers(0, sz, n_stars)
        for sx, sy in zip(xs, ys):
            b = rng.integers(80, 200)
            col = (int(b * 0.6), int(b * 0.7), b)
            pygame.draw.circle(surf, col, (int(sx), int(sy)), 1)
        return surf

    def _bake_vignette(self) -> pygame.Surface:
        surf = pygame.Surface((MAP_W, WIN_H), pygame.SRCALPHA)
        surf.fill((90, 0, 0, 0))

        cx, cy = MAP_W // 2, WIN_H // 2
        max_r  = math.sqrt(cx**2 + cy**2)
        xs = np.arange(MAP_W, dtype=np.float32)
        ys = np.arange(WIN_H, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)          # shape (WIN_H, MAP_W)
        dist   = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        # pygame surfarray uses (W, H) order — transpose
        alpha  = np.clip((dist / max_r) ** 1.8 * 255, 0, 255).astype(np.uint8).T

        # Set RGB via pixels3d (W, H, 3)
        rgb = pygame.surfarray.pixels3d(surf)
        rgb[:, :, 0] = 90
        rgb[:, :, 1] = 0
        rgb[:, :, 2] = 0
        del rgb

        # Set alpha via pixels_alpha (W, H)
        alp = pygame.surfarray.pixels_alpha(surf)
        alp[:] = alpha
        del alp

        return surf

    def _make_icon(self) -> pygame.Surface:
        icon = pygame.Surface((32, 32))
        icon.fill((8, 6, 14))
        pygame.draw.circle(icon, (220, 220, 255), (16, 16), 12, 2)
        return icon

    @staticmethod
    def _font(size: int, bold: bool = False, mono: bool = False) -> pygame.font.Font:
        names = (["Consolas", "Courier New", "Lucida Console"] if mono
                 else ["Segoe UI", "Arial", "Helvetica"])
        for name in names:
            try:
                f = pygame.font.SysFont(name, size, bold=bold)
                if f:
                    return f
            except Exception:
                pass
        return pygame.font.Font(None, size)


# ─── Compass helper ──────────────────────────────────────────────────────────

def _compass_arrow(ddx: int, ddy: int) -> str:
    """8-direction arrow pointing from agent toward artifact."""
    if ddx == 0 and ddy == 0:
        return "HERE"
    angle = math.atan2(ddy, ddx)   # radians, +y = down (screen coords)
    # Map to 8 sectors
    idx = int((angle + math.pi + math.pi / 8) / (math.pi / 4)) % 8
    arrows = [">", "v>", "v", "v<", "<", "^<", "^", "^>"]
    labels = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    return labels[idx]


# ─── Color utils ─────────────────────────────────────────────────────────────

def _decay_color(color: tuple, decay: float) -> tuple:
    target = (5, 8, 22)
    return _lerp(color, target, min(decay, 1.0) * 0.8)

def _lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

def _darken(color, f):
    return tuple(int(c * (1 - f)) for c in color)
