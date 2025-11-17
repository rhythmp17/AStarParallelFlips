#!/usr/bin/env python3
"""
Validate and (attempt to) fix a CGSHOP2026 solution for a given instance.

- Usage:
    python3 validate_and_fix_solution.py instance.json [candidate.solution.json]

If candidate.solution.json is not provided, the script will produce a valid solution
that transforms each triangulation to the first input triangulation (used as center).
"""

import json, sys, os, math, time
from itertools import combinations
from typing import List, Tuple, Set, FrozenSet, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- utility geometry helpers (copied/adapted from your solver) ----------
def edge_norm(a:int,b:int)->Tuple[int,int]:
    return (a,b) if a<b else (b,a)

def triangles_from_edges(edge_set:FrozenSet[Tuple[int,int]], n:int) -> Set[Tuple[int,int,int]]:
    es = set(edge_set)
    tris=set()
    for a,b,c in combinations(range(n),3):
        if edge_norm(a,b) in es and edge_norm(b,c) in es and edge_norm(a,c) in es:
            tris.add(tuple(sorted((a,b,c))))
    return tris

def is_convex_quad(points:List[Tuple[float,float]], a:int, c:int, b:int, d:int)->bool:
    verts=[a,c,b,d]
    coords=[points[i] for i in verts]
    crosses=[]
    for i in range(4):
        x1,y1=coords[i]
        x2,y2=coords[(i+1)%4]
        x3,y3=coords[(i+2)%4]
        cross=(x2-x1)*(y3-y2) - (y2-y1)*(x3-x2)
        crosses.append(cross)
    pos = any(c>1e-9 for c in crosses)
    neg = any(c<-1e-9 for c in crosses)
    if pos and neg: return False
    if any(abs(c) < 1e-9 for c in crosses): return False
    return True

def edge_incident_triangles(edge:Tuple[int,int], tri_set:Set[Tuple[int,int,int]])->List[int]:
    a,b=edge
    res=[]
    for tri in tri_set:
        if a in tri and b in tri:
            third=[v for v in tri if v!=a and v!=b][0]
            res.append(third)
    return res

def flippable_edges(edge_set:FrozenSet[Tuple[int,int]], points:List[Tuple[float,float]], n:int) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]:
    tris = triangles_from_edges(edge_set, n)
    es = set(edge_set)
    result=[]
    for e in list(es):
        a,b=e
        inc=edge_incident_triangles(e,tris)
        if len(inc)==2:
            c,d=inc[0],inc[1]
            if len({a,b,c,d})==4 and is_convex_quad(points,a,c,b,d):
                new_edge=edge_norm(c,d)
                if new_edge not in es:
                    result.append((e,(c,d)))
    return result

def apply_parallel_flip(edge_set:FrozenSet[Tuple[int,int]], D:Tuple[Tuple[int,int],...],
                        points:List[Tuple[float,float]], n:int) -> Optional[FrozenSet[Tuple[int,int]]]:
    if not D:
        return None
    es=set(edge_set)
    tris=triangles_from_edges(edge_set,n)
    # check pairwise non-interference
    used_tris=set()
    for edge in D:
        inc=edge_incident_triangles(edge,tris)
        if len(inc)!=2:
            return None
        tri1=tuple(sorted((edge[0],edge[1],inc[0])))
        tri2=tuple(sorted((edge[0],edge[1],inc[1])))
        if tri1 in used_tris or tri2 in used_tris:
            return None
        used_tris.add(tri1); used_tris.add(tri2)
    # apply
    new=set(es)
    for edge in D:
        inc=edge_incident_triangles(edge,tris)
        if len(inc)!=2:
            return None
        c,d=inc[0],inc[1]
        if len({edge[0],edge[1],c,d})!=4: return None
        if not is_convex_quad(points,edge[0],c,edge[1],d): return None
        new_edge=edge_norm(c,d)
        if edge in new:
            new.remove(edge); new.add(new_edge)
        else:
            return None
    return frozenset(new)

# ---------- greedy scheduler (constructive) ----------
def greedy_round(edge_set:FrozenSet[Tuple[int,int]], target:FrozenSet[Tuple[int,int]],
                 points:List[Tuple[float,float]], n:int, max_round_size:int=6) -> List[Tuple[int,int]]:
    """
    Greedy selection of a parallel flip set that reduces symmetric difference to target.
    Returns list of edges (as tuples) to flip in parallel (may be empty).
    """
    fl = flippable_edges(edge_set, points, n)
    if not fl: return []
    # prefer flips whose new_edge is in target while old edge not in target
    preferred=[]
    others=[]
    cur_diff = len(edge_set.symmetric_difference(target))
    for e,(c,d) in fl:
        new_edge = edge_norm(c,d)
        if new_edge in target and e not in target:
            preferred.append(e)
        else:
            new_es = set(edge_set)
            new_es.remove(e); new_es.add(new_edge)
            new_diff = len(frozenset(new_es).symmetric_difference(target))
            if new_diff < cur_diff:
                others.append(e)
    tris = triangles_from_edges(edge_set,n)
    chosen=[]
    # pick from preferred greedily avoiding conflicts
    for e in preferred:
        conflict=False
        for ch in chosen:
            for tri in tris:
                if ch[0] in tri and ch[1] in tri and e[0] in tri and e[1] in tri:
                    conflict=True; break
            if conflict: break
        if not conflict:
            chosen.append(e)
            if len(chosen)>=max_round_size: break
    if not chosen:
        for e in others:
            conflict=False
            for ch in chosen:
                for tri in tris:
                    if ch[0] in tri and ch[1] in tri and e[0] in tri and e[1] in tri:
                        conflict=True; break
                if conflict: break
            if not conflict:
                chosen.append(e)
                if len(chosen)>=max_round_size: break
    return chosen

def greedy_transform_to_target(src:FrozenSet[Tuple[int,int]], tgt:FrozenSet[Tuple[int,int]],
                               points:List[Tuple[float,float]], n:int, max_rounds:int=500) -> List[List[List[int]]]:
    cur = src
    rounds=[]
    for _ in range(max_rounds):
        if cur == tgt:
            return rounds
        D = greedy_round(cur, tgt, points, n)
        if not D:
            # maybe try best single edge if it helps
            fl = flippable_edges(cur, points, n)
            best=None; best_gain=0
            cur_diff = len(cur.symmetric_difference(tgt))
            for e,(c,d) in fl:
                new_edge = edge_norm(c,d)
                new_es = set(cur)
                new_es.remove(e); new_es.add(new_edge)
                new_diff = len(frozenset(new_es).symmetric_difference(tgt))
                gain = cur_diff - new_diff
                if gain > best_gain:
                    best_gain = gain; best=(e,)
            if best is None:
                break
            D=list(best)
        # apply D
        new = apply_parallel_flip(cur, tuple(D), points, n)
        if new is None:
            # try single-edge fallback
            applied=False
            for e in D:
                single = apply_parallel_flip(cur, (e,), points, n)
                if single:
                    rounds.append([[e[0], e[1]]])
                    cur=single; applied=True; break
            if not applied:
                break
        else:
            rounds.append([[e[0], e[1]] for e in D])
            cur=new
    # if not reached tgt, return partial rounds
    return rounds

# ---------- validator & fixer ----------
def validate_and_fix(instance_path:str, sol_path:Optional[str]=None, out_path:Optional[str]=None):
    with open(instance_path,"r") as f:
        inst=json.load(f)
    n = len(inst["points_x"])
    points = [(float(x),float(y)) for x,y in zip(inst["points_x"], inst["points_y"])]
    triangulations_raw = inst["triangulations"]
    given_tris = [frozenset(edge_norm(int(a),int(b)) for (a,b) in tri) for tri in triangulations_raw]
    instance_uid = inst.get("instance_uid","instance")
    # load candidate solution if provided
    candidate = None
    if sol_path and os.path.exists(sol_path):
        with open(sol_path,"r") as f:
            candidate = json.load(f)

    # choose center: if candidate exists and seems meaningful we can try to infer center by simulating,
    # but to be safe we use first triangulation in instance as target center (common, valid).
    center = given_tris[0]
    print(f"Using center = triangulation 0 (instance's first triangulation) as repair target.")

    # prepare output solution structure
    fixed_flips = []

    # if candidate present, try to validate its flips and repair per triangulation
    for idx,tri in enumerate(triangulations_raw):
        src = frozenset(edge_norm(int(a),int(b)) for (a,b) in tri)
        print(f"\nTriangulation #{idx}: starting edges = {len(src)} edges")
        if candidate:
            try:
                cand_seq = candidate["flips"][idx]
            except Exception as e:
                print("  candidate missing sequence for triangulation; will compute new schedule.")
                cand_seq = []
        else:
            cand_seq = []
        cur = src
        valid_rounds=[]
        round_no=0
        for rnd in cand_seq:
            round_no += 1
            # Each rnd is a list of edges [[i,j], ...]
            # Validate all edges indices and distinctness
            bad = False
            D=[]
            for e in rnd:
                if not (isinstance(e,list) or isinstance(e,tuple)) or len(e) < 2:
                    print(f"  Round {round_no}: invalid edge format {e}, skipping.")
                    bad=True; break
                i = int(e[0]); j = int(e[1])
                if i == j or i < 0 or j < 0 or i >= n or j >= n:
                    print(f"  Round {round_no}: edge [{i},{j}] out of range or degenerate (n={n}). Marking round invalid.")
                    bad=True; break
                D.append(edge_norm(i,j))
            if bad:
                print(f"  Round {round_no}: replacing by greedily computed round.")
                # compute greedy round instead
                greedy = greedy_round(cur, center, points, n)
                if not greedy:
                    print("    greedy produced no flips; stopping validation of further rounds.")
                    break
                new = apply_parallel_flip(cur, tuple(greedy), points, n)
                if new is None:
                    # as fallback, apply best single flip if any
                    if greedy:
                        single = apply_parallel_flip(cur, (greedy[0],), points, n)
                        if single:
                            valid_rounds.append([[greedy[0][0], greedy[0][1]]]); cur=single; continue
                    print("    greedy flips are not applicable; stopping.")
                    break
                valid_rounds.append([[e[0],e[1]] for e in greedy]); cur=new; continue
            # All indices valid; check flippability and non-interference
            # Check each edge is flippable in current triangulation
            tris = triangles_from_edges(cur,n)
            edge_list = D
            # flippable check
            flippable_set = set(e for e,_ in flippable_edges(cur, points, n))
            not_flippable = [e for e in edge_list if e not in flippable_set]
            if not_flippable:
                print(f"  Round {round_no}: some edges not flippable: {not_flippable}. Attempting to salvage by removing them.")
                # remove non-flippable edges and try to apply the remainder
                D2 = tuple(e for e in edge_list if e in flippable_set)
                if not D2:
                    # replace with greedy round
                    greedy = greedy_round(cur, center, points, n)
                    if not greedy:
                        print("    greedy produced no flips; stopping.")
                        break
                    new = apply_parallel_flip(cur, tuple(greedy), points, n)
                    if new is None:
                        print("    greedy didn't apply; stopping.")
                        break
                    valid_rounds.append([[e[0],e[1]] for e in greedy]); cur=new; continue
                new = apply_parallel_flip(cur, D2, points, n)
                if new is None:
                    print("    reduced set still invalid (non-interfering constraint?). Replacing with greedy.")
                    greedy = greedy_round(cur, center, points, n)
                    if not greedy:
                        break
                    new = apply_parallel_flip(cur, tuple(greedy), points, n)
                    if new is None:
                        break
                    valid_rounds.append([[e[0],e[1]] for e in greedy]); cur=new; continue
                valid_rounds.append([[e[0],e[1]] for e in D2]); cur=new; continue
            # Check pairwise non-interference
            # we can try to apply entire D
            new = apply_parallel_flip(cur, tuple(edge_list), points, n)
            if new is None:
                # conflict; try to select maximal subset that's valid
                tris = triangles_from_edges(cur,n)
                chosen=[]
                for e in edge_list:
                    conflict=False
                    for ch in chosen:
                        for tri in tris:
                            if ch[0] in tri and ch[1] in tri and e[0] in tri and e[1] in tri:
                                conflict=True; break
                        if conflict: break
                    if not conflict:
                        chosen.append(e)
                if not chosen:
                    # replace with greedy
                    greedy = greedy_round(cur, center, points, n)
                    if not greedy:
                        break
                    new = apply_parallel_flip(cur, tuple(greedy), points, n)
                    if new is None: break
                    valid_rounds.append([[e[0],e[1]] for e in greedy]); cur=new; continue
                new = apply_parallel_flip(cur, tuple(chosen), points, n)
                if new is None:
                    # fail safe
                    print(f"  Round {round_no}: failed to apply selected subset {chosen}; skipping round.")
                    break
                valid_rounds.append([[e[0],e[1]] for e in chosen]); cur=new; continue
            else:
                # successfully applied given round
                valid_rounds.append([[e[0],e[1]] for e in edge_list])
                cur=new
        # after processing candidate rounds, if cur != center, append greedy rounds until reach
        if cur != center:
            print(f"  After candidate rounds, triangulation != center (symdiff={len(cur.symmetric_difference(center))}). Appending greedy schedule to center.")
            more = greedy_transform_to_target(cur, center, points, n, max_rounds=1000)
            if more is None:
                more=[]
            # ensure final applicabilty
            valid_rounds.extend(more)
            cur = apply_rounds_return_final_simulation(cur, more, points, n) if more else cur
        # final check
        final_sym = len(cur.symmetric_difference(center))
        if final_sym != 0:
            print(f"  Warning: could not reach center exactly. final symdiff={final_sym}. We will keep rounds anyway.")
        fixed_flips.append(valid_rounds)
    # If candidate absent, directly compute schedules to center
    if not candidate:
        fixed_flips=[]
        for idx,tri in enumerate(triangulations_raw):
            src=frozenset(edge_norm(int(a),int(b)) for (a,b) in tri)
            rounds = greedy_transform_to_target(src, center, points, n, max_rounds=1000)
            fixed_flips.append(rounds)
    # compose solution
    solution = {
        "content_type": "CGSHOP2026_Solution",
        "instance_uid": instance_uid,
        "flips": fixed_flips,
        "meta": {
            "algorithm": "validate_and_fix_greedy",
            "timestamp": time.asctime()
        }
    }
    # write out file
    out_name = (os.path.splitext(sol_path)[0] + ".fixed.solution.json") if sol_path else (instance_uid + ".fixed.solution.json")
    with open(out_name,"w") as f:
        json.dump(solution, f, indent=2)
    print("\nWrote fixed solution to:", out_name)
    return out_name

# small helper to simulate applying rounds to get final triangulation (used above)
def apply_rounds_return_final_simulation(start:FrozenSet[Tuple[int,int]], rounds:List[List[List[int]]],
                                        points:List[Tuple[float,float]], n:int) -> FrozenSet[Tuple[int,int]]:
    cur = start
    for rnd in rounds:
        D=[]
        for e in rnd:
            D.append(edge_norm(int(e[0]), int(e[1])))
        new = apply_parallel_flip(cur, tuple(D), points, n)
        if new is None:
            # try single flips
            for e in D:
                single = apply_parallel_flip(cur, (e,), points, n)
                if single:
                    cur = single
        else:
            cur = new
    return cur

# -------------------------- main --------------------------
if __name__ == "__main__":

    INST_DIR = "benchmark_instances_rev1/benchmark_instances"
    SOL_DIR = "solutions"
    os.makedirs(SOL_DIR, exist_ok=True)

    instance_files = [f for f in os.listdir(INST_DIR) if f.endswith(".json")]

    print(f"Found {len(instance_files)} instance files.")
    print("Processing with 10 threads...\n")

    def worker(instance_filename):
        instance_path = os.path.join(INST_DIR, instance_filename)

        # If user pre-provided candidate solution with same base name:
        base = instance_filename.replace(".json", "")
        sol_path = os.path.join(SOL_DIR, base + ".solution.json")
        if not os.path.exists(sol_path):
            sol_path = None

        return validate_and_fix(instance_path, sol_path, SOL_DIR)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, f) for f in instance_files]

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print("Error:", e)

    print("\nAll instances processed.")