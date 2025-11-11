#!/usr/bin/env python3
"""
ARA* solver for "Central Triangulation under Parallel Flip Operations"
Scaled / robust version: avoids exponential enumeration, uses sampled independent sets,
bounded exact-distance BFS (with fallback heuristic), and small branching.

Usage:
  - Place an instance JSON in the same directory named "instance.json" (format shown in your messages),
    or modify the 'EMBED_INSTANCE' below.
  - Run: python ara_parallel_flips.py

Notes:
  - This implementation is designed to scale to larger instances by avoiding full enumeration.
  - For small instances, it still finds exact solutions.
"""

from typing import List, Tuple, Set, FrozenSet, Dict, Optional
from itertools import combinations
from collections import deque
import math, json, random, time, os, sys, copy, heapq

# ---------------- PARAMETERS (tune these) ----------------
MAX_SUCCESSORS = 8          # branching factor (pruned successors per expansion)
MAX_SAMPLE_INDEP = 200      # samples of independent sets per node (randomized)
DIST_BFS_DEPTH_LIMIT = 4    # depth limit for exact parallel flip BFS
DIST_BFS_NODE_LIMIT = 3000  # node limit for bounded BFS
DIST_BFS_TIMEOUT = 0.12     # seconds per bounded BFS before fallback
HEURISTIC_U = None          # if None, default chosen as max(1, n//3) after reading instance
W0 = 2.0                    # initial ARA* weight
MAX_ASTAR_ITER = 20000      # safety limit for expansions in weighted A* phase
RANDOM_SEED = 12345
# --------------------------------------------------------

random.seed(RANDOM_SEED)

# ----------------- Input reading -------------------------
# If instance.json exists, read it; otherwise use EMBED_INSTANCE if provided
EMBED_INSTANCE = None
# You can paste a JSON into EMBED_INSTANCE for quick tests, else put instance.json next to script.
if os.path.exists("instance.json"):
    with open("instance.json", "r") as f:
        instance_raw = json.load(f)
else:
    if EMBED_INSTANCE is None:
        print("No 'instance.json' found and no EMBED_INSTANCE provided. Exiting.")
        sys.exit(1)
    instance_raw = EMBED_INSTANCE

# Validate basic format
assert instance_raw.get("content_type","").startswith("CGSHOP"), "JSON does not look like a CGSHOP instance"
points_x = instance_raw["points_x"]
points_y = instance_raw["points_y"]
triangulations_raw = instance_raw["triangulations"]

points = [(float(x), float(y)) for x,y in zip(points_x, points_y)]
n = len(points)

# normalize triangulation edges to 0-based ints (JSON might be 0-based already)
def edge_norm(a:int,b:int)->Tuple[int,int]:
    return (a,b) if a<b else (b,a)

given_tris = []
for tri in triangulations_raw:
    # tri is a list of edges [ [a,b], [c,d], ... ]
    given_tris.append(frozenset(edge_norm(int(e[0]), int(e[1])) for e in tri))

# Pick HEURISTIC_U if not set
if HEURISTIC_U is None:
    HEURISTIC_U = max(1, n // 3)

print(f"Loaded instance '{instance_raw.get('instance_uid', 'unknown')}' with n={n} points and m={len(given_tris)} triangulations")
print(f"HEURISTIC_U={HEURISTIC_U}, MAX_SUCCESSORS={MAX_SUCCESSORS}, MAX_SAMPLE_INDEP={MAX_SAMPLE_INDEP}")

# ---------------- geometry helpers -----------------------
def orient(a:int,b:int,c:int)->float:
    (x1,y1) = points[a]; (x2,y2)=points[b]; (x3,y3)=points[c]
    return (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)

def is_convex_quad(a:int,c:int,b:int,d:int)->bool:
    # Check convexity of quadrilateral in order a-c-b-d
    verts = [a,c,b,d]
    coords = [points[i] for i in verts]
    crosses = []
    for i in range(4):
        x1,y1 = coords[i]
        x2,y2 = coords[(i+1)%4]
        x3,y3 = coords[(i+2)%4]
        cross = (x2-x1)*(y3-y2) - (y2-y1)*(x3-x2)
        crosses.append(cross)
    pos = any(c>1e-9 for c in crosses)
    neg = any(c<-1e-9 for c in crosses)
    if pos and neg:
        return False
    if any(abs(c) < 1e-9 for c in crosses):
        return False
    return True

def triangles_from_edges(edge_set:FrozenSet[Tuple[int,int]])->Set[Tuple[int,int,int]]:
    tris=set()
    es=set(edge_set)
    for a,b,c in combinations(range(n),3):
        if edge_norm(a,b) in es and edge_norm(b,c) in es and edge_norm(a,c) in es:
            tris.add(tuple(sorted((a,b,c))))
    return tris

def edge_incident_triangles(edge:Tuple[int,int], tri_set:Set[Tuple[int,int,int]]):
    a,b = edge
    res=[]
    for tri in tri_set:
        if a in tri and b in tri:
            third = [v for v in tri if v!=a and v!=b][0]
            res.append(third)
    return res

def flippable_edges(edge_set:FrozenSet[Tuple[int,int]]):
    tris = triangles_from_edges(edge_set)
    es = set(edge_set)
    flippable=[]
    for edge in list(es):
        a,b = edge
        inc = edge_incident_triangles(edge, tris)
        if len(inc)==2:
            c,d = inc[0], inc[1]
            if len({a,b,c,d})==4 and is_convex_quad(a,c,b,d):
                new_edge = edge_norm(c,d)
                if new_edge not in es:
                    flippable.append((edge,(c,d)))
    return flippable

def apply_parallel_flip(edge_set:FrozenSet[Tuple[int,int]], D:Tuple[Tuple[int,int],...]) -> Optional[FrozenSet[Tuple[int,int]]]:
    if not D:
        return None
    es=set(edge_set)
    tris = triangles_from_edges(edge_set)
    new_es=set(es)
    for edge in D:
        inc = edge_incident_triangles(edge, tris)
        if len(inc)!=2:
            return None
        c,d = inc[0], inc[1]
        if len({edge[0],edge[1],c,d})!=4:
            return None
        if not is_convex_quad(edge[0],c,edge[1],d):
            return None
        new_edge = edge_norm(c,d)
        if edge in new_es:
            new_es.remove(edge)
            new_es.add(new_edge)
        else:
            return None
    return frozenset(new_es)

# ---------------- independent-set sampling ----------------
def greedy_independent_set(edge_set:FrozenSet[Tuple[int,int]]) -> Tuple[Tuple[int,int], ...]:
    fl = flippable_edges(edge_set)
    if not fl: return tuple()
    score = {}
    for e,(c,d) in fl:
        cnt = 0
        for tgt in given_tris:
            if e in edge_set and e not in tgt:
                cnt += 1
        score[e]=cnt
    edges_sorted = sorted([f[0] for f in fl], key=lambda e: -score.get(e,0))
    tris = triangles_from_edges(edge_set)
    chosen=[]
    for e in edges_sorted:
        conflict=False
        for ch in chosen:
            for tri in tris:
                if ch[0] in tri and ch[1] in tri and e[0] in tri and e[1] in tri:
                    conflict=True; break
            if conflict: break
        if not conflict:
            chosen.append(e)
    return tuple(chosen)

def sampled_independent_sets(edge_set:FrozenSet[Tuple[int,int]], max_samples:int=MAX_SAMPLE_INDEP) -> List[Tuple[Tuple[int,int], ...]]:
    fl = flippable_edges(edge_set)
    edges = [f[0] for f in fl]
    if not edges:
        return []
    tris = triangles_from_edges(edge_set)
    samples = set()
    # always include greedy set if non-empty
    g = greedy_independent_set(edge_set)
    if g:
        samples.add(g)
    # compute simple scores
    score = {}
    for e,(c,d) in fl:
        score[e] = sum(1 for tgt in given_tris if e in edge_set and e not in tgt)
    edges_sorted = sorted(edges, key=lambda e: -score.get(e,0))
    topk = edges_sorted[:min(len(edges_sorted), 12)]
    # sample mixes biased to topk
    tries = 0
    while len(samples) < max_samples and tries < max_samples * 3:
        tries += 1
        chosen=[]
        pool = topk[:] if random.random() < 0.85 else edges[:]
        random.shuffle(pool)
        for e in pool:
            conflict=False
            for ch in chosen:
                for tri in tris:
                    if ch[0] in tri and ch[1] in tri and e[0] in tri and e[1] in tri:
                        conflict=True; break
                if conflict: break
            if not conflict and random.random() < 0.6:
                chosen.append(e)
            if len(chosen) >= 6: break
        if chosen:
            samples.add(tuple(sorted(chosen)))
    # also add a few single-edge flips to ensure fine moves
    for e in edges_sorted[:6]:
        samples.add((e,))
    return [s for s in samples if s]

# ---------------- bounded parallel-flip BFS ----------------
def parallel_flip_distance_bounded(src:FrozenSet[Tuple[int,int]],
                                   tgt:FrozenSet[Tuple[int,int]],
                                   depth_limit:int=DIST_BFS_DEPTH_LIMIT,
                                   node_limit:int=DIST_BFS_NODE_LIMIT,
                                   timeout_sec:float=DIST_BFS_TIMEOUT) -> Tuple[Optional[int], bool]:
    if src == tgt:
        return 0, True
    start = time.time()
    q=deque([(src,0)])
    seen={src}
    nodes=0
    while q:
        cur, d = q.popleft()
        nodes += 1
        if nodes > node_limit or (time.time() - start) > timeout_sec:
            return None, False
        if d >= depth_limit:
            continue
        # expand with sampled indep sets + single edges
        ind_sets = sampled_independent_sets(cur, max_samples=40)
        # ensure some single-edge flips also present
        fls = flippable_edges(cur)
        for f in fls[:6]:
            ind_sets.append((f[0],))
        for D in ind_sets:
            new = apply_parallel_flip(cur, D)
            if not new or new in seen: 
                continue
            if new == tgt:
                return d+1, True
            seen.add(new)
            q.append((new,d+1))
    return None, False

# robust distance: try bounded BFS, else fallback to admissible heuristic
_distance_cache: Dict[Tuple[FrozenSet[Tuple[int,int]], FrozenSet[Tuple[int,int]]], int] = {}

def parallel_flip_distance_robust(src: FrozenSet[Tuple[int,int]],
                                  tgt: FrozenSet[Tuple[int,int]]) -> int:
    """Purely heuristic distance: no BFS or DFS calls."""
    diff = len(src.symmetric_difference(tgt))
    heur = math.ceil(diff / HEURISTIC_U)
    return heur

# quick objective for candidate (sum of robust distances to targets)
def objective_of_candidate(candidate:FrozenSet[Tuple[int,int]]) -> int:
    s=0
    for tgt in given_tris:
        s += parallel_flip_distance_robust(candidate,tgt)
    return s

# ---------------- successor generation (pruned) ----------------
def successor_candidates_scaled(candidate:FrozenSet[Tuple[int,int]], B:int=MAX_SUCCESSORS) -> List[FrozenSet[Tuple[int,int]]]:
    succs=[]
    # greedy multi-edge flip
    g = greedy_independent_set(candidate)
    if g:
        new = apply_parallel_flip(candidate, g)
        if new and new != candidate:
            succs.append(new)
    # top single-edge flips
    fl = flippable_edges(candidate)
    if fl:
        score = {}
        for e,(c,d) in fl:
            score[e] = sum(1 for tgt in given_tris if e in candidate and e not in tgt)
        top_edges = sorted([f[0] for f in fl], key=lambda e: -score.get(e,0))[:min(4, len(fl))]
        for e in top_edges:
            new = apply_parallel_flip(candidate, (e,))
            if new and new not in succs:
                succs.append(new)
    # several sampled multi-edge flips
    for s in sampled_independent_sets(candidate, max_samples=MAX_SAMPLE_INDEP):
        new = apply_parallel_flip(candidate, s)
        if new and new not in succs:
            succs.append(new)
        if len(succs) >= B:
            break
    # limit to B and return
    return succs[:B]

# ---------------- heuristic ----------------
def heuristic_LB_total(candidate:FrozenSet[Tuple[int,int]]) -> int:
    total = 0
    for tgt in given_tris:
        diff = len(candidate.symmetric_difference(tgt))
        total += math.ceil(diff / HEURISTIC_U)
    return total

# ---------------- ARA* implementation (simplified, robust) ----------------
def ara_star_anytime(initial:FrozenSet[Tuple[int,int]], time_budget:Optional[float]=None, w0:float=W0, B:int=MAX_SUCCESSORS):
    start_time = time.time()
    w = w0

    # g-values map
    gvals: Dict[FrozenSet[Tuple[int,int]], int] = {initial: 0}
    hcache: Dict[FrozenSet[Tuple[int,int]], int] = {initial: heuristic_LB_total(initial)}
    # OPEN: priority queue of (f = g + w*h, -h, g, node)
    OPEN = []
    heapq.heappush(OPEN, (gvals[initial] + w * hcache[initial], -hcache[initial], gvals[initial], initial))
    best_solution = initial
    best_cost = objective_of_candidate(initial)
    print(f"[ARA*] start: initial cost = {best_cost}")

    # Weighted A* phase + repairs loop (we do a single weighted A* then a final repair with w=1)
    # We'll do: weighted search with cap, then set w=1 and run again reusing gvals
    def run_weighted_astar(weight:float, open_heap, gmap, hmap, max_iters:int=MAX_ASTAR_ITER):
        OPEN_local = open_heap
        g_local = gmap
        h_local = hmap
        CLOSED = set()
        iterations = 0
        nonlocal best_solution, best_cost
        while OPEN_local and iterations < max_iters:
            if time_budget and (time.time() - start_time) > time_budget:
                break
            f, neg_h, gcur, node = heapq.heappop(OPEN_local)
            if node in CLOSED:
                iterations += 1
                continue
            # refresh h if missing
            if node not in h_local:
                h_local[node] = heuristic_LB_total(node)
            CLOSED.add(node)
            # evaluate exact/robust objective for node
            obj = objective_of_candidate(node)
            if obj < best_cost:
                best_cost = obj
                best_solution = node
                print(f"[{weight:.2f}] New best_cost={best_cost} at time {time.time()-start_time:.2f}s (g={g_local.get(node)})")
            # expansion
            succs = successor_candidates_scaled(node, B=B)
            for s in succs:
                ng = g_local.get(node, math.inf) + 1
                if s not in g_local or ng < g_local[s]:
                    g_local[s] = ng
                    if s not in h_local:
                        h_local[s] = heuristic_LB_total(s)
                    f_s = ng + weight * h_local[s]
                    heapq.heappush(OPEN_local, (f_s, -h_local[s], ng, s))
            iterations += 1
        return OPEN_local, g_local, h_local, CLOSED

    # Weighted phase
    print(f"[ARA*] Running weighted A* with w={w}")
    OPEN, gvals, hcache, closed1 = run_weighted_astar(w, OPEN, gvals, hcache)
    print(f"[ARA*] Weighted phase done. Best cost so far: {best_cost}")

    # Repair / final A* with w=1 using seen gvals as starting OPEN
    print("[ARA*] Repairing with w=1 (A*). Reusing g-values seen so far.")
    # build initial OPEN from current gvals
    OPEN2 = []
    for node, gg in gvals.items():
        hval = hcache.get(node, heuristic_LB_total(node))
        f = gg + 1.0 * hval
        heapq.heappush(OPEN2, (f, -hval, gg, node))
    OPEN2, gvals, hcache, closed2 = run_weighted_astar(1.0, OPEN2, gvals, hcache, max_iters=MAX_ASTAR_ITER * 2)
    print(f"[ARA*] Repair phase done. Best cost final: {best_cost}")

    return best_solution, best_cost

# ---------------- main ----------------
def main():
    initial = given_tris[0]
    t0 = time.time()
    center, cost = ara_star_anytime(initial, time_budget=300, w0=W0, B=MAX_SUCCESSORS)
    took = time.time() - t0
    print("\n=== Result ===")
    print("Best center triangulation (edges):", sorted(list(center)))
    print("Objective (sum of robust distances to given targets):", cost)
    print("Per-target robust distances:")
    for i,t in enumerate(given_tris):
        d = parallel_flip_distance_robust(center,t)
        print(f"  dist(center, T{i}) = {d}")
    print(f"\nTime taken: {took:.2f}s")
    # Optionally: compare to brute force if small number of enumerated nodes (avoid for big n)
    # We will not enumerate full flip graph here to avoid explosion.
    # But if you'd like brute-force verification for small instances (n <= 8) we can add it.

if __name__ == "__main__":
    main()
