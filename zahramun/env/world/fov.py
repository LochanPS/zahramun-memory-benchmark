"""
Field of View — recursive shadowcasting.
Much more realistic than square radius: walls actually block vision.
"""

from __future__ import annotations
import numpy as np
from ..world.game_state import Tile

OPAQUE_TILES = {Tile.WALL, Tile.VOID}


def compute_fov(
    origin_x: int,
    origin_y: int,
    radius: int,
    tiles: np.ndarray,
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    """
    Recursive shadowcasting FOV. Returns set of (x, y) visible from origin.
    Always includes origin tile.
    """
    visible: set[tuple[int, int]] = {(origin_x, origin_y)}

    # 8 octants
    for octant in range(8):
        _cast_light(
            visible, tiles, height, width,
            origin_x, origin_y, radius,
            row=1, start=1.0, end=0.0,
            octant=octant,
        )

    return visible


# Octant transform tables
_MULT = [
    ( 1,  0,  0,  1),
    ( 0,  1,  1,  0),
    ( 0, -1,  1,  0),
    (-1,  0,  0,  1),
    (-1,  0,  0, -1),
    ( 0, -1, -1,  0),
    ( 0,  1, -1,  0),
    ( 1,  0,  0, -1),
]


def _cast_light(
    visible: set,
    tiles: np.ndarray,
    height: int,
    width: int,
    cx: int,
    cy: int,
    radius: int,
    row: int,
    start: float,
    end: float,
    octant: int,
) -> None:
    if start < end:
        return

    xx, xy, yx, yy = _MULT[octant]
    new_start = 0.0

    for j in range(row, radius + 1):
        blocked = False
        dx = -j - 1
        dy = -j

        while dx <= 0:
            dx += 1
            # Map to actual coordinates
            mx = cx + dx * xx + dy * xy
            my = cy + dx * yx + dy * yy

            l_slope = (dx - 0.5) / (dy + 0.5)
            r_slope = (dx + 0.5) / (dy - 0.5)

            if start < r_slope:
                continue
            if end > l_slope:
                break

            if 0 <= mx < width and 0 <= my < height:
                if dx * dx + dy * dy <= radius * radius:
                    visible.add((mx, my))

                if blocked:
                    if Tile(tiles[my, mx]) in OPAQUE_TILES:
                        new_start = r_slope
                    else:
                        blocked = False
                        start = new_start
                else:
                    if Tile(tiles[my, mx]) in OPAQUE_TILES and j < radius:
                        blocked = True
                        _cast_light(
                            visible, tiles, height, width,
                            cx, cy, radius,
                            j + 1, start, l_slope, octant,
                        )
                        new_start = r_slope

        if blocked:
            break
