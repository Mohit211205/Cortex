"""
HNSW (Hierarchical Navigable Small World) index — implemented from scratch.

Key parameters:
  M              — max connections per node per layer (default 16)
  ef_construction— beam width during index build (default 200)
  ef_search      — beam width during search (default 50)

Complexity:
  Insert : O(log N) average
  Search : O(log N) average
"""

from __future__ import annotations

import math
import random
import heapq
import threading
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .node import HNSWNode
from .distance import cosine_distance


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2          # max connections at layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1 / math.log(M)    # level multiplier

        self.nodes: Dict[int, HNSWNode] = {}
        self.entry_point: Optional[int] = None
        self.max_level: int = -1
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, node_id: int, vector: np.ndarray):
        vector = self._normalise(vector)
        level = self._random_level()

        with self._lock:
            node = HNSWNode(node_id, vector, level)
            self.nodes[node_id] = node

            if self.entry_point is None:
                self.entry_point = node_id
                self.max_level = level
                return

            ep = self.entry_point
            current_max = self.max_level

            # Phase 1 — greedily descend from top to level+1
            for lc in range(current_max, level, -1):
                ep = self._greedy_search(vector, ep, lc)

            # Phase 2 — insert with beam search from level down to 0
            for lc in range(min(level, current_max), -1, -1):
                candidates = self._beam_search(vector, ep, self.ef_construction, lc)
                neighbours = self._select_neighbours(node_id, candidates, self._m_at(lc), lc)

                for nb_id in neighbours:
                    node.add_neighbour(lc, nb_id)
                    self.nodes[nb_id].add_neighbour(lc, node_id)
                    # prune if over limit
                    self._prune(nb_id, lc)

                if candidates:
                    ep = candidates[0][1]   # closest found so far

            if level > current_max:
                self.entry_point = node_id
                self.max_level = level

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        if self.entry_point is None:
            return []

        query = self._normalise(query)
        ep = self.entry_point

        # Descend to layer 1
        for lc in range(self.max_level, 0, -1):
            ep = self._greedy_search(query, ep, lc)

        # Beam search at layer 0
        candidates = self._beam_search(query, ep, max(self.ef_search, k), 0)

        results = sorted(candidates, key=lambda x: x[0])[:k]
        return [(node_id, dist) for dist, node_id in results]

    def __len__(self) -> int:
        return len(self.nodes)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "HNSWIndex":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        return int(-math.log(random.random()) * self.ml)

    def _m_at(self, layer: int) -> int:
        return self.M_max0 if layer == 0 else self.M

    def _normalise(self, v: np.ndarray) -> np.ndarray:
        v = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return cosine_distance(a, b)

    def _greedy_search(self, query: np.ndarray, ep_id: int, layer: int) -> int:
        """Single greedy step — returns id of closest node at this layer."""
        best = ep_id
        best_dist = self._dist(query, self.nodes[ep_id].vector)
        improved = True
        while improved:
            improved = False
            for nb_id in self.nodes[best].get_neighbours(layer):
                if nb_id not in self.nodes:
                    continue
                d = self._dist(query, self.nodes[nb_id].vector)
                if d < best_dist:
                    best_dist = d
                    best = nb_id
                    improved = True
        return best

    def _beam_search(
        self, query: np.ndarray, ep_id: int, ef: int, layer: int
    ) -> List[Tuple[float, int]]:
        """
        Beam search: maintains a candidate min-heap and a dynamic result set.
        Returns list of (distance, node_id) sorted by distance ascending.
        """
        ep_dist = self._dist(query, self.nodes[ep_id].vector)
        candidates = [(ep_dist, ep_id)]   # min-heap by dist
        results = [(ep_dist, ep_id)]       # best ef found so far (min-heap)
        visited = {ep_id}

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            # furthest result so far
            worst_dist = max(r[0] for r in results)
            if c_dist > worst_dist and len(results) >= ef:
                break

            for nb_id in self.nodes[c_id].get_neighbours(layer):
                if nb_id not in self.nodes or nb_id in visited:
                    continue
                visited.add(nb_id)
                nb_dist = self._dist(query, self.nodes[nb_id].vector)
                worst_dist = max(r[0] for r in results)

                if nb_dist < worst_dist or len(results) < ef:
                    heapq.heappush(candidates, (nb_dist, nb_id))
                    results.append((nb_dist, nb_id))
                    if len(results) > ef:
                        results.sort()
                        results = results[:ef]

        return results

    def _select_neighbours(
        self,
        node_id: int,
        candidates: List[Tuple[float, int]],
        m: int,
        layer: int,
    ) -> List[int]:
        """Simple heuristic: take closest m candidates, excluding self."""
        sorted_c = sorted(candidates, key=lambda x: x[0])
        return [nid for _, nid in sorted_c if nid != node_id][:m]

    def _prune(self, node_id: int, layer: int):
        """Trim neighbour list to M_max."""
        limit = self._m_at(layer)
        node = self.nodes[node_id]
        nbs = node.get_neighbours(layer)
        if len(nbs) <= limit:
            return
        # keep closest
        scored = sorted(
            nbs,
            key=lambda nb_id: self._dist(node.vector, self.nodes[nb_id].vector)
            if nb_id in self.nodes else float("inf"),
        )
        node.neighbours[layer] = scored[:limit]
