from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Dict, Iterable, List, Optional, Set, Tuple



def compute_separation_metrics(
    queries: List[dict],
    all_row_ids: Optional[Iterable[int]] = None,
) -> dict:
    """
    Computes:
      - average_separation: average over unordered row pairs (i,j) of
            sep(i,j) = #{q : exactly one of i,j is in q['ids']}
      - bottleneck_separation: min_{i<j} sep(i,j)
      - min_row_degree: min_i deg(i), where deg(i) = #{q : i in q['ids']}
      - num_twins: number of row pairs (i,j) where i and j always appear together
            across all queries (i in q iff j in q)

    Input:
      queries: list of dicts, each must have key 'ids' containing a list of row integers.
              Each element of `queries` is treated as one "sample-bearing query" (i.e., one
              incidence set). If you want each (q,k) sample to count separately, include
              duplicates in this list accordingly.
      all_row_ids: optional iterable of all rows in the universe. If provided, rows that
              never appear in any query will have degree 0 and will affect min_row_degree
              and bottleneck_separation.

    Returns:
      dict

    Notes:
      - Exact bottleneck computation avoids O(R^2) enumeration by using co-occurrence counts
        and a "find smallest non-neighbor" scan. Worst-case can still be heavy for huge R.
    """
    if all_row_ids is None:
        universe: Set[int] = set()
        for q in queries:
            universe.update(q.get("ids", []))
    else:
        universe = set(all_row_ids)

    row_ids = sorted(universe)
    R = len(row_ids)
    N = len(queries)

    if R == 0:
        return {
            "average": 0.0,
            "bottleneck": 0,
            "min_row_degree": 0,
            "num_twins": 0,
        }
    if R == 1:
        # No pairs exist; define bottleneck as 0 by convention.
        # min_row_degree is just the single row's degree.
        only = row_ids[0]
        deg = 0
        for q in queries:
            if only in set(q.get("ids", [])):
                deg += 1
        return {
            "average": 0.0,
            "bottleneck": 0,
            "min_row_degree": deg,
            "num_twins": 0,
        }
    id_to_idx: Dict[int, int] = {rid: i for i, rid in enumerate(row_ids)}

    degrees = [0] * R

    # cooccur[(i,j)] = number of queries where i and j both appear
    cooccur: Dict[Tuple[int, int], int] = {}
    # neighbors[i] = set of j where (i,j) ever co-occurred at least once
    neighbors: List[Set[int]] = [set() for _ in range(R)]

    total_separated_pairs_across_queries = 0  # sum_q |S_q| * (R - |S_q|)

    for q in queries:
        ids = q.get("ids", [])
        if not ids:
            continue

        # Dedup within a query
        S = sorted({id_to_idx[i] for i in ids if i in id_to_idx})
        s = len(S)
        if s == 0:
            continue

        # degrees
        for i in S:
            degrees[i] += 1

        # average separation aggregate
        total_separated_pairs_across_queries += s * (R - s)

        # pair co-occurrences
        for i, j in combinations(S, 2):
            key = (i, j) if i < j else (j, i)
            cooccur[key] = cooccur.get(key, 0) + 1
            neighbors[i].add(j)
            neighbors[j].add(i)

    # ----- average separation -----
    denom_pairs = comb(R, 2)
    average_separation = total_separated_pairs_across_queries / denom_pairs

    # ----- min row degree -----
    min_row_degree = min(degrees)

    # ----- bottleneck separation -----
    # sep(i,j) = deg(i) + deg(j) - 2*cooccur(i,j)
    bottleneck = float("inf")

    # First consider pairs that co-occur at least once (edges)
    for (i, j), c in cooccur.items():
        sep = degrees[i] + degrees[j] - 2 * c
        if sep < bottleneck:
            bottleneck = sep

    # Now consider pairs that never co-occur (non-edges): sep = deg(i) + deg(j)
    # We find the smallest deg(i)+deg(j) among non-neighbor pairs without enumerating all pairs.
    order = sorted(range(R), key=lambda i: degrees[i])
    deg_sorted = [degrees[i] for i in order]

    # If the co-occurrence graph is complete, there are no non-edges.
    # We'll detect that on the fly; if complete, we keep bottleneck from edges.
    for pos_i, i in enumerate(order):
        # Lower bound using the smallest possible partner degree
        lb = degrees[i] + deg_sorted[0]
        if bottleneck != float("inf") and lb >= bottleneck:
            # since degrees only increase as we move through order, we can stop early
            break

        # Scan for the smallest-degree j that is not i and not a neighbor of i
        for j in order:
            if j == i:
                continue
            if j not in neighbors[i]:
                sep = degrees[i] + degrees[j]  # cooccur=0
                if sep < bottleneck:
                    bottleneck = sep
                break  # best possible j for this i (degrees sorted)

    if bottleneck == float("inf"):
        # This happens only if R<2 (handled) or if something went odd; safe fallback.
        bottleneck = 0

    # ----- num twins -----
    zero_degree = sum(1 for d in degrees if d == 0)
    num_twins = comb(zero_degree, 2)
    for (i, j), c in cooccur.items():
        if degrees[i] == c and degrees[j] == c:
            num_twins += 1

    return {
        "average": float(average_separation),
        "bottleneck": int(bottleneck),
        "min_row_degree": int(min_row_degree),
        "num_twins": int(num_twins),
    }

if __name__ == '__main__':
    queries = [
    {"ids": [0, 1, 2]},
    {"ids": [2, 3]},
    {"ids": [1, 3]},
    ]
    m = compute_separation_metrics(queries)
    print(m)
