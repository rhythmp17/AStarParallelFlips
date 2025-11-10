"""
A* Search for Flip Distance with Parallel Flips (Pure Python Version)

This implementation finds the flip distance between two triangulations of the same
set of points. A flip replaces one diagonal of a convex quadrilateral by the other.
Multiple non-conflicting flips (no shared triangle) may occur in parallel (up to k).

Author: Rhythm Patni
Date: 2025-11-10
"""

from __future__ import annotations
from dataclasses import dataclass, field
import heapq
import itertools
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from inputparser import CGSHOP2026InstanceParser

Edge = Tuple[int, int]
TriSet = FrozenSet[Edge]


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    count: int
    node: Any = field(compare=False)


class AStarParallel:
    """A* over the flip graph allowing up to k parallel flips per step."""

    def __init__(self, instance, k: int = 1):
        self.instance = instance
        self.k = max(1, k)

        self.points = list(zip(instance.points_x, instance.points_y))
        self.triangulations_raw = instance.triangulations
        self.tri_edge_sets: List[TriSet] = [self._to_edge_set(t) for t in self.triangulations_raw]

        # compute boundary edges once (edges shared by only 1 triangle)
        self._boundary_edges = self._compute_boundary_edges()

    # -------------------- Helper Functions --------------------
    def _canonical_edge(self, e: Iterable[int]) -> Edge:
        a, b = tuple(e)
        return (a, b) if a < b else (b, a)

    def _to_edge_set(self, interior_edges: Iterable[Iterable[int]]) -> TriSet:
        return frozenset(self._canonical_edge(e) for e in interior_edges)

    def _compute_boundary_edges(self) -> Set[Edge]:
        """Boundary edges are those belonging to only one triangle in the convex hull triangulation."""
        all_edges = set()
        for t in self.tri_edge_sets:
            all_edges |= set(t)
        # We cannot know hull precisely without coordinates; assume convex hull edges are those
        # that appear in every triangulation (common edges).
        common_edges = set(self.tri_edge_sets[0])
        for tri in self.tri_edge_sets[1:]:
            common_edges &= tri
        return common_edges

    # -------------------- Core Geometry Logic --------------------
    def _build_triangle_map(self, edges: TriSet) -> List[Tuple[int, int, int]]:
        """
        Reconstruct triangles from edges by finding all triplets (a,b,c)
        that are pairwise connected.
        """
        triangles = []
        vertices = set(v for e in edges for v in e)
        for a, b, c in itertools.combinations(vertices, 3):
            if (
                (a, b) in edges or (b, a) in edges
            ) and (
                (b, c) in edges or (c, b) in edges
            ) and (
                (a, c) in edges or (c, a) in edges
            ):
                triangles.append(tuple(sorted((a, b, c))))
        return triangles

    def _list_flippable(self, cur_edges: TriSet) -> List[Edge]:
        """Return all edges that are diagonals of a quadrilateral."""
        triangles = self._build_triangle_map(cur_edges)
        edge_to_tris: Dict[Edge, List[Tuple[int, int, int]]] = {}

        for tri in triangles:
            for e in itertools.combinations(tri, 2):
                e = self._canonical_edge(e)
                edge_to_tris.setdefault(e, []).append(tri)

        flippable = []
        for e, tris in edge_to_tris.items():
            if e in self._boundary_edges:
                continue
            if len(tris) == 2:
                t1, t2 = tris
                c1 = [v for v in t1 if v not in e]
                c2 = [v for v in t2 if v not in e]
                if c1 and c2:
                    c, d = c1[0], c2[0]
                    new_diag = self._canonical_edge((c, d))
                    if new_diag not in cur_edges:
                        flippable.append(e)
        return flippable

    def _apply_parallel_flip(self, cur_edges: TriSet, edges_to_flip: Iterable[Edge]) -> Optional[TriSet]:
        """
        Perform valid geometric (quadrilateral-based) flips combinatorially.
        """
        triangles = self._build_triangle_map(cur_edges)
        edge_to_tris: Dict[Edge, List[Tuple[int, int, int]]] = {}

        for tri in triangles:
            for e in itertools.combinations(tri, 2):
                e = self._canonical_edge(e)
                edge_to_tris.setdefault(e, []).append(tri)

        new_edges = set(cur_edges)

        for e in edges_to_flip:
            if e in self._boundary_edges:
                continue
            tris = edge_to_tris.get(e)
            if not tris or len(tris) != 2:
                continue
            t1, t2 = tris
            c1 = [v for v in t1 if v not in e]
            c2 = [v for v in t2 if v not in e]
            if not c1 or not c2:
                continue
            c, d = c1[0], c2[0]
            new_diag = self._canonical_edge((c, d))
            if new_diag in new_edges:
                continue
            # valid flip
            new_edges.remove(e)
            new_edges.add(new_diag)
        return frozenset(new_edges)

    def _valid_parallel_sets(self, flippable: List[Edge]) -> Iterable[Tuple[Edge, ...]]:
        """Yield all non-conflicting edge sets (no shared vertex) up to size k."""
        for r in range(1, min(self.k, len(flippable)) + 1):
            for comb in itertools.combinations(flippable, r):
                vertices = set()
                ok = True
                for e in comb:
                    if set(e) & vertices:
                        ok = False
                        break
                    vertices.update(e)
                if ok:
                    yield comb

    def simple_heuristic(self, cur_edges: TriSet, target_edges: TriSet) -> int:
        """Simple heuristic: number of edges not in target triangulation."""
        return len([e for e in cur_edges if e not in target_edges])

    # -------------------- A* Search --------------------
    def run(self, start_idx: int = 0, target_idx: int = 1, max_expansions: Optional[int] = None):
        start = self.tri_edge_sets[start_idx]
        target = self.tri_edge_sets[target_idx]

        if start == target:
            return 0, [start]

        open_heap: List[PrioritizedItem] = []
        entry_finder: Dict[TriSet, Tuple[int, int]] = {}
        parent: Dict[TriSet, Optional[TriSet]] = {}
        counter = 0

        g_start = 0
        h_start = self.simple_heuristic(start, target)
        start_item = PrioritizedItem(priority=g_start + h_start, count=counter, node=start)
        heapq.heappush(open_heap, start_item)
        entry_finder[start] = (g_start, counter)
        parent[start] = None

        closed: Set[TriSet] = set()
        expansions = 0

        while open_heap:
            current_item = heapq.heappop(open_heap)
            cur = current_item.node

            if cur in closed:
                continue

            if cur == target:
                # reconstruct path
                path = []
                node = cur
                while node is not None:
                    path.append(node)
                    node = parent.get(node)
                path.reverse()
                return entry_finder[cur][0], path

            closed.add(cur)
            expansions += 1
            if max_expansions is not None and expansions > max_expansions:
                break

            flippable = self._list_flippable(cur)
            # print(f"Expanding {cur}, flippable = {flippable}")

            for flip_set in self._valid_parallel_sets(flippable):
                new_edges = self._apply_parallel_flip(cur, flip_set)
                if new_edges is None or new_edges in closed:
                    continue

                tentative_g = entry_finder[cur][0] + 1
                known = entry_finder.get(new_edges)
                if known is None or tentative_g < known[0]:
                    counter += 1
                    h = self.simple_heuristic(new_edges, target)
                    item = PrioritizedItem(priority=tentative_g + h, count=counter, node=new_edges)
                    heapq.heappush(open_heap, item)
                    entry_finder[new_edges] = (tentative_g, counter)
                    parent[new_edges] = cur

        return None, None


# -------------------- CLI Entry --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("instance", help="Path to instance JSON file")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("-k", type=int, default=1, help="Max edges to flip in parallel")
    args = parser.parse_args()

    p = CGSHOP2026InstanceParser(args.instance)
    p.load()
    inst = p.instance

    astar = AStarParallel(inst, k=args.k)
    dist, path = astar.run(args.start, args.target)

    if dist is None:
        print("❌ Target not reached.")
    else:
        print(f"✅ Flip distance (k={args.k}): {dist}")
        print(f"Path length: {len(path)}")
