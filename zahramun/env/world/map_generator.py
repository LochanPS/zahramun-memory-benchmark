"""
Procedural map generator — BSP room-based dungeon with desert aesthetic.
Produces tile grid + spawn candidates for entities, items, hazards.
"""

from __future__ import annotations
import numpy as np
import random
from dataclasses import dataclass
from typing import Optional
from .game_state import Tile


@dataclass
class Room:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def inner_cells(self) -> list[tuple[int, int]]:
        cells = []
        for row in range(self.y + 1, self.y + self.h - 1):
            for col in range(self.x + 1, self.x + self.w - 1):
                cells.append((col, row))
        return cells

    def overlaps(self, other: Room, padding: int = 1) -> bool:
        return (
            self.x - padding < other.x + other.w and
            self.x + self.w + padding > other.x and
            self.y - padding < other.y + other.h and
            self.y + self.h + padding > other.y
        )


class MapGenerator:
    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate(self) -> tuple[np.ndarray, list[Room], list[tuple[int, int]], set[tuple[int, int]]]:
        """
        Returns:
            tiles:    np.ndarray shape (H, W)
            rooms:    list of Room
            corridors: list of (x,y) floor tuples in corridors
            hazards:  set of (x,y) hazard positions
        """
        tiles = np.full((self.height, self.width), Tile.WALL, dtype=np.int8)

        rooms = self._place_rooms(tiles)
        corridors = self._connect_rooms(tiles, rooms)
        self._add_sand_patches(tiles)
        self._add_ruins(tiles, rooms)
        hazards = self._add_hazards(tiles, rooms)
        self._add_doors(tiles, rooms, corridors)

        return tiles, rooms, corridors, hazards

    # ─── Room Placement ───────────────────────────────────────────────────────

    def _place_rooms(self, tiles: np.ndarray) -> list[Room]:
        rooms = []
        attempts = 0
        max_attempts = 200
        target_rooms = self.rng.randint(8, 14)

        while len(rooms) < target_rooms and attempts < max_attempts:
            attempts += 1
            w = self.rng.randint(4, 10)
            h = self.rng.randint(4, 8)
            x = self.rng.randint(1, self.width - w - 1)
            y = self.rng.randint(1, self.height - h - 1)
            room = Room(x, y, w, h)

            if any(room.overlaps(r) for r in rooms):
                continue

            rooms.append(room)
            self._carve_room(tiles, room)

        return rooms

    def _carve_room(self, tiles: np.ndarray, room: Room) -> None:
        for row in range(room.y + 1, room.y + room.h - 1):
            for col in range(room.x + 1, room.x + room.w - 1):
                tiles[row, col] = Tile.FLOOR

    # ─── Corridor Connection ──────────────────────────────────────────────────

    def _connect_rooms(self, tiles: np.ndarray, rooms: list[Room]) -> list[tuple[int, int]]:
        corridors = []
        for i in range(len(rooms) - 1):
            cx1, cy1 = rooms[i].center
            cx2, cy2 = rooms[i + 1].center
            if self.rng.random() < 0.5:
                corridors.extend(self._h_corridor(tiles, cx1, cx2, cy1))
                corridors.extend(self._v_corridor(tiles, cy1, cy2, cx2))
            else:
                corridors.extend(self._v_corridor(tiles, cy1, cy2, cx1))
                corridors.extend(self._h_corridor(tiles, cx1, cx2, cy2))
        return corridors

    def _h_corridor(self, tiles: np.ndarray, x1: int, x2: int, y: int) -> list[tuple[int, int]]:
        cells = []
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if 0 < y < self.height and 0 < x < self.width:
                tiles[y, x] = Tile.FLOOR
                cells.append((x, y))
        return cells

    def _v_corridor(self, tiles: np.ndarray, y1: int, y2: int, x: int) -> list[tuple[int, int]]:
        cells = []
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if 0 < y < self.height and 0 < x < self.width:
                tiles[y, x] = Tile.FLOOR
                cells.append((x, y))
        return cells

    # ─── Desert Flavour ───────────────────────────────────────────────────────

    def _add_sand_patches(self, tiles: np.ndarray) -> None:
        floor_positions = list(zip(*np.where(tiles == Tile.FLOOR)))
        n_patches = len(floor_positions) // 8
        for _ in range(n_patches):
            if not floor_positions:
                break
            idx = self.rng.randint(0, len(floor_positions) - 1)
            r, c = floor_positions[idx]
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.height and 0 <= nc < self.width
                            and tiles[nr, nc] == Tile.FLOOR
                            and self.rng.random() < 0.5):
                        tiles[nr, nc] = Tile.SAND

    def _add_ruins(self, tiles: np.ndarray, rooms: list[Room]) -> None:
        n_ruins = max(1, len(rooms) // 3)
        ruin_rooms = self.rng.sample(rooms, min(n_ruins, len(rooms)))
        for room in ruin_rooms:
            for (x, y) in room.inner_cells():
                if tiles[y, x] in (Tile.FLOOR, Tile.SAND) and self.rng.random() < 0.3:
                    tiles[y, x] = Tile.RUIN

    def _add_hazards(self, tiles: np.ndarray, rooms: list[Room]) -> set[tuple[int, int]]:
        """Place hazard tiles in corridors and some rooms. Returns set of (x,y)."""
        hazard_positions: set[tuple[int, int]] = set()
        floor_positions = [
            (c, r) for r in range(self.height)
            for c in range(self.width)
            if tiles[r, c] in (Tile.FLOOR, Tile.SAND, Tile.RUIN)
        ]
        n_hazards = max(3, len(floor_positions) // 15)
        candidates = self.rng.sample(floor_positions, min(n_hazards, len(floor_positions)))
        for (x, y) in candidates:
            tiles[y, x] = Tile.HAZARD
            hazard_positions.add((x, y))
        return hazard_positions

    def _add_doors(self, tiles: np.ndarray, rooms: list[Room],
                   corridors: list[tuple[int, int]]) -> None:
        """
        Place door tiles at corridor-to-room junctions.
        65% of rooms get a door at one entrance.
        """
        corridor_set = set(corridors)
        # Collect per-room inner cell sets once
        room_inners = [set(room.inner_cells()) for room in rooms]

        for room_idx, room in enumerate(rooms):
            if self.rng.random() < 0.35:   # 35% rooms stay open
                continue
            inner = room_inners[room_idx]

            # Find corridor tiles immediately adjacent to this room's inner cells
            door_spots: list[tuple[int, int]] = []
            for (cx, cy) in inner:
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if ((nx, ny) in corridor_set
                            and 0 <= ny < self.height
                            and 0 <= nx < self.width
                            and tiles[ny, nx] == Tile.FLOOR):
                        door_spots.append((nx, ny))

            if not door_spots:
                continue

            # Deduplicate, shuffle, place one door
            door_spots = list(set(door_spots))
            self.rng.shuffle(door_spots)
            dx, dy = door_spots[0]
            tiles[dy, dx] = Tile.DOOR

    # ─── Spawn Candidates ─────────────────────────────────────────────────────

    def get_walkable_cells(self, tiles: np.ndarray) -> list[tuple[int, int]]:
        walkable = {Tile.FLOOR, Tile.SAND, Tile.RUIN}
        return [
            (c, r)
            for r in range(self.height)
            for c in range(self.width)
            if tiles[r, c] in walkable
        ]

    def pick_spawn_positions(
        self,
        tiles: np.ndarray,
        rooms: list[Room],
        n_enemies: int,
        n_hp_items: int,
        n_mem_items: int,
        n_shrines: int = 2,
    ) -> dict:
        walkable = self.get_walkable_cells(tiles)
        used: set[tuple[int, int]] = set()
        shuffled_rooms = self.rng.sample(rooms, len(rooms))

        def pick_from_room(room: Room) -> tuple[int, int]:
            inner = [c for c in room.inner_cells()
                     if c not in used and tiles[c[1], c[0]] in {Tile.FLOOR, Tile.SAND, Tile.RUIN}]
            if not inner:
                inner = [c for c in walkable if c not in used]
            pos = self.rng.choice(inner)
            used.add(pos)
            return pos

        agent_pos    = pick_from_room(shuffled_rooms[0])
        artifact_pos = pick_from_room(shuffled_rooms[-1])
        guardian_pos = pick_from_room(shuffled_rooms[-2])

        enemies  = [pick_from_room(shuffled_rooms[i % len(shuffled_rooms)])
                    for i in range(1, n_enemies + 1)]
        hp_items = [pick_from_room(shuffled_rooms[i % len(shuffled_rooms)])
                    for i in range(n_enemies + 1, n_enemies + n_hp_items + 1)]
        mem_items = [pick_from_room(shuffled_rooms[i % len(shuffled_rooms)])
                     for i in range(n_enemies + n_hp_items + 1,
                                    n_enemies + n_hp_items + n_mem_items + 1)]

        # Shrines — placed in mid-dungeon rooms (not agent/artifact/guardian rooms)
        total_so_far = n_enemies + n_hp_items + n_mem_items + 1
        shrines = [pick_from_room(shuffled_rooms[i % len(shuffled_rooms)])
                   for i in range(total_so_far + 1, total_so_far + n_shrines + 1)]

        return {
            "agent":     agent_pos,
            "artifact":  artifact_pos,
            "guardian":  guardian_pos,
            "enemies":   enemies,
            "hp_items":  hp_items,
            "mem_items": mem_items,
            "shrines":   shrines,
        }
