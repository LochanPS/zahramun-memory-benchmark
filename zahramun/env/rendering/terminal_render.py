"""
Terminal Renderer — blessed-based, human viewing only.
Training loops NEVER call this. render_mode="human" only.

Layout:
  ┌─ MAP ─────────────────────────┐ ┌─ STATUS ──────────────┐
  │ procedural world              │ │ HP / MEM / LEVEL       │
  │ fog of war shown              │ │ degradation warning     │
  └───────────────────────────────┘ └────────────────────────┘
"""

from __future__ import annotations
from typing import Optional

try:
    import blessed
    BLESSED_AVAILABLE = True
except ImportError:
    BLESSED_AVAILABLE = False

from ..world.game_state import GameState, EntityType, Tile
from ..memory.memory_system import MemorySystem
from ..memory.degradation import DegradationLevel


# ─── Visual Config ────────────────────────────────────────────────────────────

TILE_CHARS = {
    Tile.VOID:  (" ",  "on_black"),
    Tile.FLOOR: (".", "dim"),
    Tile.WALL:  ("#", "bold"),
    Tile.SAND:  ("~", "yellow"),
    Tile.RUIN:  ("%", "dim"),
    Tile.DOOR:  ("+", "cyan"),
}

ENTITY_CHARS = {
    EntityType.AGENT:        ("@", "bold_white"),
    EntityType.SCORPION:     ("s", "red"),
    EntityType.MEMORY_EATER: ("M", "magenta"),
    EntityType.WANDERER:     ("w", "yellow"),
    EntityType.GUARDIAN:     ("G", "bold_red"),
    EntityType.ARTIFACT:     ("*", "bold_yellow"),
    EntityType.ITEM_HP:      ("!", "green"),
    EntityType.ITEM_MEMORY:  ("?", "cyan"),
    EntityType.HAZARD:       ("^", "bold_red"),
}

LEVEL_COLORS = {
    DegradationLevel.NONE:     "green",
    DegradationLevel.MINOR:    "yellow",
    DegradationLevel.MODERATE: "yellow",
    DegradationLevel.HEAVY:    "red",
    DegradationLevel.CRITICAL: "bold_red",
}

LEVEL_LABELS = {
    DegradationLevel.NONE:     "STABLE",
    DegradationLevel.MINOR:    "MINOR DROUGHT",
    DegradationLevel.MODERATE: "MODERATE DROUGHT",
    DegradationLevel.HEAVY:    "HEAVY DROUGHT",
    DegradationLevel.CRITICAL: "CRITICAL DROUGHT",
}


class TerminalRenderer:
    def __init__(self, map_w: int, map_h: int):
        if not BLESSED_AVAILABLE:
            raise ImportError("blessed not installed. pip install blessed")
        self.map_w = map_w
        self.map_h = map_h
        self.term = blessed.Terminal()
        self._setup()

    def _setup(self) -> None:
        print(self.term.clear + self.term.hide_cursor, end="", flush=True)

    def render(self, state: GameState, mem: MemorySystem) -> None:
        t = self.term
        out = []

        # ── Map ───────────────────────────────────────────────────────────────
        for r in range(min(self.map_h, t.height - 4)):
            out.append(t.move(r, 0))
            for c in range(min(self.map_w, t.width - 22)):
                age = state.cell_age(c, r)
                out.append(self._render_cell(state, mem, c, r, age, t))

        # ── Status panel ──────────────────────────────────────────────────────
        status_col = min(self.map_w + 2, t.width - 20)
        level = mem.level
        level_color = LEVEL_LABELS[level]

        status_lines = [
            f"ZAHRAMUN",
            f"",
            f"Step: {state.step}",
            f"",
            f"HP:  {state.agent.hp:3d} / {state.agent.hp_max}",
            f"MEM: {mem.stability:5.1f} / {mem.stability_max:.0f}",
            f"",
            LEVEL_LABELS[level],
            f"",
            f"Artifact: {'YES' if state.agent.artifact_collected else 'no'}",
            f"Guardian: {'DEAD' if state.agent.guardian_defeated else 'alive'}",
        ]

        for i, line in enumerate(status_lines):
            if i < t.height - 2:
                out.append(t.move(i, status_col))
                if "DROUGHT" in line:
                    out.append(getattr(t, LEVEL_COLORS.get(level, "white"), t.white))
                out.append(line[:20])
                out.append(t.normal)

        # ── Memory bar ────────────────────────────────────────────────────────
        bar_row = self.map_h + 1
        if bar_row < t.height:
            mem_pct = mem.normalized()
            bar_len = min(40, t.width - 20)
            filled = int(mem_pct * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            out.append(t.move(bar_row, 0))
            color = LEVEL_COLORS.get(level, "white")
            out.append(getattr(t, color, t.white))
            out.append(f"MEM [{bar}] {mem.stability:.0f}")
            out.append(t.normal)

        print("".join(out), end="", flush=True)

    def _render_cell(
        self, state: GameState, mem: MemorySystem,
        c: int, r: int, age: int, t
    ) -> str:
        """Render single cell with correct char and color."""
        if age < 0:
            # Never seen — true dark
            return t.on_black + " " + t.normal

        agent = state.agent

        if c == agent.x and r == agent.y:
            return t.bold + t.white + "@" + t.normal

        if (c, r) == state.artifact_pos and not state.agent.artifact_collected:
            return t.bold + t.yellow + "*" + t.normal

        # Check entities
        entities = [e for e in state.get_entities_at(c, r) if e.is_enemy()]
        if entities:
            e = entities[0]
            char, style = ENTITY_CHARS.get(e.entity_type, ("?", "white"))
            color = getattr(t, style.split("_")[0] if "_" in style else style, t.white)
            return color + char + t.normal

        # Tile
        tile = Tile(state.tiles[r, c])
        char, style = TILE_CHARS.get(tile, ("?", "white"))

        # Dim older cells to show memory decay visually
        if age > 20:
            return t.dim + char + t.normal
        return char

    def close(self) -> None:
        t = self.term
        print(t.move(self.map_h + 3, 0) + t.normal_cursor, end="", flush=True)
