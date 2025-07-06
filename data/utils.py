from typing import Tuple
from enum import Enum

import numpy as np

class MVTecCategory(Enum):
    BOTTLE = "bottle"
    CABLE = "cable"
    CAPSULE = "capsule"
    CARPET = "carpet"
    GRID = "grid"
    HAZELNUT = "hazelnut"
    LEATHER = "leather"
    METAL_NUT = "metal_nut"
    PILL = "pill"
    SCREW = "screw"
    TILE = "tile"
    TOOTHBRUSH = "toothbrush"
    TRANSISTOR = "transistor"
    WOOD = "wood"
    ZIPPER = "zipper"
    
POS_DIFF_ARRAY = np.array([
    [-1, -1], [-1, 0], [-1, 1],
    [ 0, -1],          [ 0, 1],
    [ 1, -1], [ 1, 0], [ 1, 1]
], dtype=np.int32)

def is_texture_category(category: MVTecCategory) -> bool:
    if isinstance(category, str):
        category = MVTecCategory(category)
    
    texture_categories = {
        MVTecCategory.CARPET,
        MVTecCategory.GRID,
        MVTecCategory.LEATHER,
        MVTecCategory.TILE,
        MVTecCategory.WOOD,
        MVTecCategory.CABLE
    }
    
    return category in texture_categories

def is_object_category(category: MVTecCategory) -> bool:
    return is_texture_category(category)

def generate_coords_position(H: int, W: int, K: int) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    p1 = np.random.randint([0, 0], [H - K + 1, W - K + 1])
    pos = np.random.randint(8)

    direction = POS_DIFF_ARRAY[pos]
    jitter = np.random.randint(K // 4, size=2)
    offset = direction * (jitter + (3 * K // 4))

    p2 = np.clip(p1 + offset, 0, [H - K, W - K])

    return tuple(p1), tuple(p2), pos

def generate_coords_svdd(H: int, W: int, K: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    p1 = np.random.randint([0, 0], [H - K + 1, W - K + 1])

    jitter_range = K // 32
    jitter = np.random.randint(-jitter_range, jitter_range + 1, size=2)
    while np.all(jitter == 0):
        jitter = np.random.randint(-jitter_range, jitter_range + 1, size=2)

    p2 = np.clip(p1 + jitter, 0, [H - K, W - K])

    return tuple(p1), tuple(p2)