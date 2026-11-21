"""
Sprite Loader — extracts individual surfaces from spritesheets.
Falls back to colored rectangles if files not found.
Renderer works with or without sprites.
"""

from __future__ import annotations
import os
import pygame
from typing import Optional

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "tiles")

WORLD_PATH  = os.path.join(ASSETS_DIR, "world.png")
BATTLE_PATH = os.path.join(ASSETS_DIR, "battle.png")

# ── Spritesheet config ─────────────────────────────────────────────────────────
# world.png:  5 cols x 3 rows, black background
# battle.png: 5 cols x 1 row,  black background
WORLD_COLS = 5
WORLD_ROWS = 3
# ──────────────────────────────────────────────────────────────────────────────

# world.png layout — (row, col) → name
# Row 0: floor | wall | sand | ruin | hazard
# Row 1: agent | scorpion | memory_eater | wanderer | guardian
# Row 2: artifact | (spare) | (spare) | item_hp | item_mem
WORLD_MAP = {
    "floor":        (0, 0),
    "wall":         (0, 1),
    "sand":         (0, 2),
    "ruin":         (0, 3),
    "hazard":       (0, 4),
    "agent":        (1, 0),
    "scorpion":     (1, 1),
    "memory_eater": (1, 2),
    "wanderer":     (1, 3),
    "guardian":     (1, 4),
    "artifact":     (2, 0),
    "item_hp":      (2, 3),
    "item_mem":     (2, 4),
}

# battle.png layout — col → entity key (1 row, N cols)
BATTLE_MAP = {
    "scorpion":     0,
    "memory_eater": 1,
    "wanderer":     2,
    "guardian":     3,
    "artifact":     4,
}


class SpriteLoader:
    """
    Loads spritesheets and exposes sprites by name.
    Returns None if sheet not found — renderer falls back to colored rects.
    """

    def __init__(self):
        self.world_sheet:  Optional[pygame.Surface] = None
        self.battle_sheet: Optional[pygame.Surface] = None
        self.cache: dict[str, pygame.Surface] = {}
        self.has_world  = False
        self.has_battle = False

        # World cell dimensions (col_w != row_h when image isn't square-grid)
        self._col_w = 24   # image_width  / WORLD_COLS
        self._row_h = 24   # image_height / WORLD_ROWS

        # Battle cell dimensions
        self._bat_w = 256  # image_width  / len(BATTLE_MAP)
        self._bat_h = 256  # image_height (single row)

        self._load()

    # ─── Load ─────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(WORLD_PATH):
            try:
                raw = pygame.image.load(WORLD_PATH).convert()
                raw.set_colorkey((0, 0, 0))        # black → transparent
                self.world_sheet = raw.convert_alpha()
                self.has_world = True
                w, h = self.world_sheet.get_size()
                self._col_w = w // WORLD_COLS      # e.g. 1536/5 = 307
                self._row_h = h // WORLD_ROWS      # e.g. 1024/3 = 341
            except Exception:
                pass
            if self.has_world:
                w, h = self.world_sheet.get_size()
                print(f"[sprites] world.png {w}x{h} "
                      f"-> {WORLD_COLS}x{WORLD_ROWS} grid "
                      f"-> cell {self._col_w}x{self._row_h}px")
            else:
                print("[sprites] world.png load failed - using colored rects")
        else:
            print("[sprites] world.png not found - using colored rects")

        if os.path.exists(BATTLE_PATH):
            try:
                raw = pygame.image.load(BATTLE_PATH).convert()
                raw.set_colorkey((0, 0, 0))        # black → transparent
                self.battle_sheet = raw.convert_alpha()
                self.has_battle = True
                w, h = self.battle_sheet.get_size()
                self._bat_w = w // len(BATTLE_MAP)  # e.g. 1774/5 = 354
                self._bat_h = h                      # e.g. 887
            except Exception as e:
                print(f"[sprites] battle.png load failed: {e}")
            if self.has_battle:
                w, h = self.battle_sheet.get_size()
                print(f"[sprites] battle.png {w}x{h} "
                      f"-> {len(BATTLE_MAP)} sprites "
                      f"-> cell {self._bat_w}x{self._bat_h}px")
        else:
            print("[sprites] battle.png not found - using colored rects in combat")

    # ─── Get World Sprite ─────────────────────────────────────────────────────

    def get(self, name: str, target_size: int = 24) -> Optional[pygame.Surface]:
        """
        Get world sprite scaled to target_size x target_size.
        Uses correct col_w / row_h offsets — fixes the dimension bug.
        """
        cache_key = f"{name}_{target_size}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        if not self.has_world or name not in WORLD_MAP:
            return None

        row, col = WORLD_MAP[name]
        # KEY FIX: x uses col_w, y uses row_h (they differ when grid isn't square)
        px = col * self._col_w
        py = row * self._row_h
        raw = self._cut(self.world_sheet, px, py, self._col_w, self._row_h)

        # Scale to render tile size
        if (self._col_w, self._row_h) != (target_size, target_size):
            raw = pygame.transform.smoothscale(raw, (target_size, target_size))

        self.cache[cache_key] = raw
        return raw

    # ─── Get Battle Sprite ────────────────────────────────────────────────────

    def get_battle(self, entity_key: str, target_size: int = 200) -> Optional[pygame.Surface]:
        """
        Get battle sprite (full cell height) scaled to target_size x target_size.
        """
        cache_key = f"battle_{entity_key}_{target_size}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        if not self.has_battle or entity_key not in BATTLE_MAP:
            return None

        col = BATTLE_MAP[entity_key]
        px  = col * self._bat_w
        raw = self._cut(self.battle_sheet, px, 0, self._bat_w, self._bat_h)

        if (self._bat_w, self._bat_h) != (target_size, target_size):
            raw = pygame.transform.smoothscale(raw, (target_size, target_size))

        self.cache[cache_key] = raw
        return raw

    # ─── Util ─────────────────────────────────────────────────────────────────

    def get_darkened(self, name: str, alpha: int) -> Optional[pygame.Surface]:
        cache_key = f"{name}_a{alpha}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        base = self.get(name)
        if base is None:
            return None
        s = base.copy()
        s.set_alpha(alpha)
        self.cache[cache_key] = s
        return s

    @staticmethod
    def _cut(sheet: pygame.Surface, x: int, y: int, w: int, h: int) -> pygame.Surface:
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.blit(sheet, (0, 0), (x, y, w, h))
        return surf
