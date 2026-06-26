from __future__ import annotations
import numpy as np
from typing import Dict, List, Set


class HNSWNode:
    def __init__(self, node_id: int, vector: np.ndarray, level: int):
        self.id = node_id
        self.vector = vector
        self.level = level
        # neighbours[layer] -> set of neighbour node ids
        self.neighbours: Dict[int, List[int]] = {i: [] for i in range(level + 1)}

    def add_neighbour(self, layer: int, neighbour_id: int):
        if layer not in self.neighbours:
            self.neighbours[layer] = []
        if neighbour_id not in self.neighbours[layer]:
            self.neighbours[layer].append(neighbour_id)

    def get_neighbours(self, layer: int) -> List[int]:
        return self.neighbours.get(layer, [])
