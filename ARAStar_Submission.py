#!/usr/bin/env python3
"""
ARA* solver for "Central Triangulation under Parallel Flip Operations"

- Produces CGSHOP2026-style solution JSON with a top-level "flips" array:
    "flips": [
      [ [[a,b], [c,d]], [[e,f]] ],   # rounds for triangulation 0
      [ [[g,h]] ]                    # rounds for triangulation 1
    ]

- Most algorithmic choices and helper functions are adapted from your original script,
  but the output schema and per-instance data handling have been corrected and cleaned.
"""

import os
import sys
import json
import time
import math
import random
import heapq
import concurrent.futures
from itertools import combinations
from functools import lru_cache
from typing import List, Tuple, Set, FrozenSet, Dict, Optional

# ---------------------- Tunable parameters -----------------------
MAX_SUCCESSORS = 8
MAX_SAMPLE_INDEP = 200
W0 = 2.0
MAX_ASTAR_ITER = 20000
RANDOM_SEED = 12345
DEFAULT_INST_DIR = "benchmark_instances_rev1"
DEFAULT_SOL_DIR = "solutions"
MAX_WORKERS = 6
TIME_BUDGET_PER_INSTANCE = 120.0
# ----------------------------------------------------------------

random.seed(RANDOM_SEED)

# ------------------ geometry helpers (instance-local globals) ------------------
# These globals are set per-instance inside process_one_instance.
points: List[Tuple[float, float]] = []
n: int = 0
given_tris: List[FrozenSet[Tuple[int, int]]] = []
HEURISTIC_U = None

def edge_norm(a:int, b:int) -> Tuple[int,int]:
    return (a,b) if a < b else (b,a)

def orient(a:int,b:int,c:int) -> float:
    (x1,y1), (x2,y2), (x3,y3) = points[a], points[b], points[c]
    return (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)

def is_convex_quad(a:int,c:int,b:int,d:int)->bool:
    # Check simple polygon convexity for quad (a,c,b,d) using cross products.
    verts = [a,c,b,d]
    coords = [points[i] for i in verts]
    crosses = []
    for i in range(4):
        x1,y1 = coords[i]
        x2,y2 = coords[(i+1)%4]
        x3,y3 = coords[(i+2)%4]
        cross = (x2-x1)*(y3-y2) - (y2-y1)*(x3-x2)
        crosses.append(cross)
    # if signs mixed -> non-convex; if any near-zero -> degenerate
    pos = any(c > 1e-9 for c in crosses)
    neg = any(c < -1e-9 for c in crosses)
    if pos and neg:
        return False
    if any(abs(c) < 1e-9 for c in crosses):
        return False
    return True

# Cached helpers that depend on global points and n.
# We will clear caches between instances by calling .cache_clear().
@lru_cache(maxsize=None)
def triangles_from_edges_cached(edge_tuple:Tuple[Tuple[int,int], ...]) -> Tuple[Tuple[int,int,int], ...]:
    es = set(edge_tuple)
    tris = []
    for a,b,c in combinations(range(n), 3):
        if edge_norm(a,b) in es and edge_norm(b,c) in es and edge_norm(a,c) in es:
            tris.append(tuple(sorted((a,b,c))))
    return tuple(tris)

def triangles_from_edges(edge_set:FrozenSet[Tuple[int,int]]) -> Set[Tuple[int,int,int]]:
    tup = tuple(sorted(edge_set))
    return set(triangles_from_edges_cached(tup))

@lru_cache(maxsize=None)
def flippable_edges_cached(edge_tuple:Tuple[Tuple[int,int], ...]) -> Tuple[Tuple[Tuple[int,int], Tuple[int,int]], ...]:
    es = set(edge_tuple)
    flippable = []
    tris = set(triangles_from_edges_cached(edge_tuple))
    for edge in list(es):
        a,b = edge
        inc = []
        for tri in tris:
            if a in tri and b in tri:
                third = [v for v in tri if v!=a and v!=b][0]
                inc.append(third)
        if len(inc) == 2:
            c,d = inc[0], inc[1]
            if len({a,b,c,d}) == 4 and is_convex_quad(a,c,b,d):
                new_edge = edge_norm(c,d)
                if new_edge not in es:
                    flippable.append((edge,(c,d)))
    return tuple(flippable)

def flippable_edges(edge_set:FrozenSet[Tuple[int,int]]):
    return list(flippable_edges_cached(tuple(sorted(edge_set))))

def edge_incident_triangles(edge:Tuple[int,int], tri_set:Set[Tuple[int,int,int]]):
    a,b = edge
    res = []
    for tri in tri_set:
        if a in tri and b in tri:
            third = [v for v in tri if v!=a and v!=b][0]
            res.append(third)
    return res

def apply_parallel_flip(edge_set:FrozenSet[Tuple[int,int]], D:Tuple[Tuple[int,int], ...]) -> Optional[FrozenSet[Tuple[int,int]]]:
    if not D:
        return None
    es = set(edge_set)
    tris = triangles_from_edges(edge_set)
    new_es = set(es)
    for edge in D:
        inc = edge_incident_triangles(edge, tris)
        if len(inc) != 2:
            return None
        c,d = inc[0], inc[1]
        if len({edge[0], edge[1], c, d}) != 4:
            return None
        if not is_convex_quad(edge[0], c, edge[1], d):
            return None
        new_edge = edge_norm(c, d)
        if edge in new_es:
            new_es.remove(edge)
            new_es.add(new_edge)
        else:
            return None
    return frozenset(new_es)

# ------------------ independent-set sampling & greedy helpers ------------------
def greedy_independent_set(edge_set:FrozenSet[Tuple[int,int]]) -> Tuple[Tuple[int,int], ...]:
    fl = flippable_edges(edge_set)
    if not fl:
        return tuple()
    score = {}
    for e,(c,d) in fl:
        cnt = 0
        for tgt in given_tris:
            if e in edge_set and e not in tgt:
                cnt += 1
        score[e] = cnt
    edges_sorted = sorted([f[0] for f in fl], key=lambda e: -score.get(e,0))
    tris = triangles_from_edges(edge_set)
    chosen = []
    for e in edges_sorted:
        conflict = False
        for ch in chosen:
            # conflict if share a triangle
            for tri in tris:
                if ch[0] in tri and ch[1] in tri and e[0] in tri and e[1] in tri:
                    conflict = True
                    break
            if conflict:
                break
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
    g = greedy_independent_set(edge_set)
    if g:
        samples.add(g)
    score = {}
    for e,(c,d) in fl:
        score[e] = sum(1 for tgt in given_tris if e in edge_set and e not in tgt)
    edges_sorted = sorted(edges, key=lambda e: -score.get(e,0))
    topk = edges_sorted[:min(len(edges_sorted), 12)]
    tries = 0
    while len(samples) < max_samples and tries < max_samples * 3:
        tries += 1
        chosen = []
        pool = topk[:] if random.random() < 0.85 else edges[:]
        random.shuffle(pool)
        for e in pool:
            conflict = False
            for ch in chosen:
                for tri in tris:
                    if ch[0] in tri and ch[1] in tri and e[0] in tri and e[1] in tri:
                        conflict = True
                        break
                if conflict: break
            if not conflict and random.random() < 0.6:
                chosen.append(e)
            if len(chosen) >= 6: break
        if chosen:
            samples.add(tuple(sorted(chosen)))
    for e in edges_sorted[:6]:
        samples.add((e,))
    return [s for s in samples if s]

# ------------------ heuristic & objective ------------------
def parallel_flip_distance_heuristic(src:FrozenSet[Tuple[int,int]], tgt:FrozenSet[Tuple[int,int]]) -> int:
    diff = len(src.symmetric_difference(tgt))
    return math.ceil(diff / HEURISTIC_U)

def objective_of_candidate(candidate:FrozenSet[Tuple[int,int]]) -> int:
    s = 0
    for tgt in given_tris:
        s += parallel_flip_distance_heuristic(candidate, tgt)
    return s

# ------------------ successors ------------------
def successor_candidates_scaled(candidate:FrozenSet[Tuple[int,int]], B:int=MAX_SUCCESSORS):
    succs = []
    g = greedy_independent_set(candidate)
    if g:
        new = apply_parallel_flip(candidate, g)
        if new and new != candidate:
            succs.append((new, tuple(g)))
    fl = flippable_edges(candidate)
    if fl:
        score = {}
        for e,(c,d) in fl:
            score[e] = sum(1 for tgt in given_tris if e in candidate and e not in tgt)
        top_edges = sorted([f[0] for f in fl], key=lambda e: -score.get(e,0))[:min(4, len(fl))]
        for e in top_edges:
            new = apply_parallel_flip(candidate, (e,))
            if new:
                succs.append((new, (e,)))
    for s in sampled_independent_sets(candidate, max_samples=MAX_SAMPLE_INDEP):
        new = apply_parallel_flip(candidate, s)
        if new:
            succs.append((new, tuple(s)))
        if len(succs) >= B:
            break
    seen = {}
    dedup = []
    for new, D in succs:
        if new not in seen:
            seen[new] = D
            dedup.append((new, D))
    return dedup[:B]

def heuristic_LB_total(candidate:FrozenSet[Tuple[int,int]]):
    total = 0
    for tgt in given_tris:
        diff = len(candidate.symmetric_difference(tgt))
        total += math.ceil(diff / HEURISTIC_U)
    return total

# ------------------ ARA* (heuristic-only) ------------------
def ara_star_anytime(initial:FrozenSet[Tuple[int,int]], time_budget:Optional[float]=None, w0:float=W0, B:int=MAX_SUCCESSORS):
    start_time = time.time()
    w = w0
    gvals: Dict[FrozenSet[Tuple[int,int]], int] = {initial: 0}
    hcache: Dict[FrozenSet[Tuple[int,int]], int] = {initial: heuristic_LB_total(initial)}
    OPEN = []
    heapq.heappush(OPEN, (gvals[initial] + w * hcache[initial], -hcache[initial], gvals[initial], initial))
    best_solution = initial
    best_cost = objective_of_candidate(initial)
    print(f"[ARA*] start: initial objective = {best_cost}")

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
            if node not in h_local:
                h_local[node] = heuristic_LB_total(node)
            CLOSED.add(node)
            # evaluate objective
            obj = objective_of_candidate(node)
            if obj < best_cost:
                best_cost = obj
                best_solution = node
                print(f"[w={weight:.2f}] New best objective={best_cost} (g={g_local.get(node)}) at t={time.time()-start_time:.2f}s")
            succs = successor_candidates_scaled(node, B=B)
            for s_node, s_D in succs:
                ng = g_local.get(node, math.inf) + 1
                if s_node not in g_local or ng < g_local[s_node]:
                    g_local[s_node] = ng
                    if s_node not in h_local:
                        h_local[s_node] = heuristic_LB_total(s_node)
                    f_s = ng + weight * h_local[s_node]
                    heapq.heappush(OPEN_local, (f_s, -h_local[s_node], ng, s_node))
            iterations += 1
        return OPEN_local, g_local, h_local, CLOSED

    # weighted A*
    print(f"[ARA*] Running weighted A* w={w}")
    OPEN, gvals, hcache, closed1 = run_weighted_astar(w, OPEN, gvals, hcache)
    print(f"[ARA*] Weighted phase done. Best objective so far: {best_cost}")

    # repair with w=1
    OPEN2 = []
    for node, gg in gvals.items():
        hval = hcache.get(node, heuristic_LB_total(node))
        f = gg + 1.0 * hval
        heapq.heappush(OPEN2, (f, -hval, gg, node))
    print("[ARA*] Repair with w=1 (A*)")
    OPEN2, gvals, hcache, closed2 = run_weighted_astar(1.0, OPEN2, gvals, hcache, max_iters=MAX_ASTAR_ITER * 2)
    print(f"[ARA*] Repair phase done. Final best objective: {best_cost}")

    return best_solution, best_cost

# ------------------ constructive schedule (greedy parallel rounds) ------------------
def construct_flip_schedule_to_target(start:FrozenSet[Tuple[int,int]], target:FrozenSet[Tuple[int,int]], max_rounds:int=1000) -> List[List[List[int]]]:
    cur = start
    rounds: List[List[List[int]]] = []
    safety = 0
    while cur != target and safety < max_rounds:
        safety += 1
        fl = flippable_edges(cur)
        if not fl:
            break
        preferred = []
        other = []
        for e,(c,d) in fl:
            new_edge = edge_norm(c,d)
            if new_edge in target and e not in target:
                preferred.append(e)
            else:
                new_es = set(cur)
                if e in new_es:
                    new_es.remove(e)
                    new_es.add(new_edge)
                    old_diff = len(cur.symmetric_difference(target))
                    new_diff = len(frozenset(new_es).symmetric_difference(target))
                    if new_diff < old_diff:
                        other.append(e)
        chosen = []
        tris = triangles_from_edges(cur)
        for e in preferred:
            conflict=False
            for ch in chosen:
                for tri in tris:
                    if ch[0] in tri and ch[1] in tri and e[0] in tri and e[1] in tri:
                        conflict=True; break
                if conflict: break
            if not conflict:
                chosen.append(e)
        if not chosen:
            for e in other:
                conflict=False
                for ch in chosen:
                    for tri in tris:
                        if ch[0] in tri and ch[1] in tri and e[0] in tri and e[1] in tri:
                            conflict=True; break
                    if conflict: break
                if not conflict:
                    chosen.append(e)
                    if len(chosen) >= 6:
                        break
        if not chosen:
            best_e = None; best_gain = 0
            old_diff = len(cur.symmetric_difference(target))
            for e,(c,d) in fl:
                new_edge = edge_norm(c,d)
                new_es = set(cur)
                if e in new_es:
                    new_es.remove(e); new_es.add(new_edge)
                    new_diff = len(frozenset(new_es).symmetric_difference(target))
                    gain = old_diff - new_diff
                    if gain > best_gain:
                        best_gain = gain; best_e = e
            if best_e is not None:
                chosen.append(best_e)
            else:
                break
        D = tuple(chosen)
        new = apply_parallel_flip(cur, D)
        if not new:
            applied_any = False
            for e in D:
                new_single = apply_parallel_flip(cur, (e,))
                if new_single:
                    rounds.append([[e[0], e[1]]])
                    cur = new_single
                    applied_any = True
                    break
            if not applied_any:
                break
        else:
            rounds.append([[e[0], e[1]] for e in D])
            cur = new
    # return partial rounds if cannot reach target fully
    return rounds

def apply_rounds_return_final(start:FrozenSet[Tuple[int,int]], rounds:List[List[List[int]]]) -> FrozenSet[Tuple[int,int]]:
    cur = start
    for rnd in rounds:
        D = []
        for e in rnd:
            D.append(edge_norm(int(e[0]), int(e[1])))
        new = apply_parallel_flip(cur, tuple(D))
        if new is None:
            # try single flips
            for e in D:
                cur_single = apply_parallel_flip(cur, (e,))
                if cur_single:
                    cur = cur_single
            continue
        cur = new
    return cur

# ------------------ per-instance processing & main ------------------
def process_one_instance(instance_raw:dict, out_path:str, time_budget:Optional[float]=TIME_BUDGET_PER_INSTANCE):
    global points, n, given_tris, HEURISTIC_U
    # basic validation & load
    if not instance_raw.get("content_type", "").startswith("CGSHOP"):
        raise ValueError("JSON does not look like a CGSHOP instance (content_type mismatch)")
    points_x = instance_raw.get("points_x")
    points_y = instance_raw.get("points_y")
    triangulations_raw = instance_raw.get("triangulations")
    if points_x is None or points_y is None or triangulations_raw is None:
        raise ValueError("Missing required fields: points_x, points_y, triangulations")

    points = [(float(x), float(y)) for x,y in zip(points_x, points_y)]
    n = len(points)

    given_tris = []
    for tri in triangulations_raw:
        # each tri is expected to be list of edges, each edge is [u,v]
        given_tris.append(frozenset(edge_norm(int(e[0]), int(e[1])) for e in tri))

    if HEURISTIC_U is None:
        HEURISTIC_U = max(1, n // 3)

    # clear caches
    triangles_from_edges_cached.cache_clear()
    flippable_edges_cached.cache_clear()

    print(f"\nInstance: {instance_raw.get('instance_uid', 'unknown')} (n={n}, m={len(given_tris)})")
    print(f"HEURISTIC_U={HEURISTIC_U}, MAX_SUCCESSORS={MAX_SUCCESSORS}, MAX_SAMPLE_INDEP={MAX_SAMPLE_INDEP}")

    # run ARA*
    initial = given_tris[0]
    t0 = time.time()
    center, cost = ara_star_anytime(initial, time_budget=time_budget, w0=W0, B=MAX_SUCCESSORS)
    took = time.time() - t0
    print(f"ARA* done in {took:.2f}s: best objective={cost}")

    # build flips: for each triangulation, get rounds -> center (list of rounds)
    flips_for_output: List[List[List[List[int]]]] = []
    for idx, T in enumerate(given_tris):
        print(f"[Schedule] triangulation #{idx} -> center ...")
        rounds = construct_flip_schedule_to_target(T, center, max_rounds=500)
        flips_for_output.append(rounds)
        final_tf = apply_rounds_return_final(T, rounds) if rounds else T
        symdiff = len(center.symmetric_difference(final_tf))
        print(f"  rounds={len(rounds)}, final symmetric-diff={symdiff}")

    # prepare output following the CGSHOP2026 template exactly
    sol = {
        "content_type": "CGSHOP2026_Solution",
        "instance_uid": instance_raw.get("instance_uid", os.path.splitext(os.path.basename(out_path))[0]),
        "flips": flips_for_output,
        "meta": {
            "algorithm": "ARA*-heuristic-center + greedy-parallel-schedule",
            "seed": RANDOM_SEED,
            "notes": "Heuristic-only distances (no BFS). Flip schedules built greedily; indices are 0-based."
        }
    }

    # write output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sol, f, indent=2)
    print(f"Saved solution to {out_path}")

def main(argv):
    inst_dir = DEFAULT_INST_DIR
    sol_dir = DEFAULT_SOL_DIR
    max_workers = MAX_WORKERS
    if len(argv) >= 1 and argv[0]:
        inst_dir = argv[0]
    if len(argv) >= 2 and argv[1]:
        sol_dir = argv[1]
    if len(argv) >= 3 and argv[2]:
        try:
            max_workers = int(argv[2])
        except:
            pass

    if not os.path.isdir(inst_dir):
        print(f"Instance directory '{inst_dir}' not found. Exiting.")
        sys.exit(1)
    os.makedirs(sol_dir, exist_ok=True)

    # gather files
    files = []
    for root, dirs, filenames in os.walk(inst_dir):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                files.append(os.path.join(root, fn))
    files = sorted(files)
    if not files:
        print(f"No JSON files found in '{inst_dir}'. Exiting.")
        sys.exit(1)

    print(f"Processing {len(files)} instances from '{inst_dir}' -> '{sol_dir}' with up to {max_workers} workers")

    def run_file(in_path):
        base = os.path.basename(in_path)
        out_filename = base.replace(".json", ".solution.json")
        out_path = os.path.join(sol_dir, out_filename)
        print(f"\n=== {in_path} -> {out_path} ===")
        try:
            with open(in_path, "r") as f:
                inst_raw = json.load(f)
        except Exception as e:
            print(f"Failed to load {in_path}: {e}")
            return
        try:
            process_one_instance(inst_raw, out_path, time_budget=TIME_BUDGET_PER_INSTANCE)
        except Exception as e:
            print(f"Error processing {in_path}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_file, p) for p in files]
        for fut in concurrent.futures.as_completed(futures):
            # exceptions are already printed inside run_file
            pass

if __name__ == "__main__":
    # usage: python ara_center_solver.py [instance_dir] [solution_dir] [max_workers]
    main(sys.argv[1:])
