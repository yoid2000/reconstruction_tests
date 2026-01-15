from __future__ import annotations

import itertools
import math
import random
from typing import Optional

import pandas as pd


def simulated_mixing(R: int, C: int, D: int, T: int, seed: Optional[int] = None) -> Optional[float]:
    """
    Simulate mixing for a random R x C dataset with D distinct values per column.

    Returns None if distinct rows are impossible (R > D**C). Otherwise, samples
    rows without replacement to ensure uniqueness, evaluates all counting queries
    for every subset of columns and all value assignments, suppressing those with
    count < T, and returns the average pair co-appearance count.
    """
    if any(x < 0 for x in (R, C, D, T)):
        raise ValueError("R, C, D, T must be nonnegative (and D>0).")
    if D <= 0:
        raise ValueError("D must be > 0.")
    if R < 2:
        return 0.0

    total_space = D ** C
    if total_space < R:
        return None

    rng = random.Random(seed)
    col_names = [f"c{i}" for i in range(C)]
    indices = rng.sample(range(total_space), R)
    rows = []
    for idx in indices:
        row = [0] * C
        tmp = idx
        for pos in range(C - 1, -1, -1):
            row[pos] = tmp % D
            tmp //= D
        rows.append(row)
    df = pd.DataFrame(rows, columns=col_names)

    total_pairs = math.comb(R, 2)
    if total_pairs == 0:
        return 0.0

    total_pair_hits = 0
    for k in range(0, C + 1):
        if k == 0:
            count = R
            if count >= T:
                total_pair_hits += math.comb(count, 2)
            continue

        for cols in itertools.combinations(col_names, k):
            counts = df.groupby(list(cols)).size()
            for count in counts.values:
                if count >= T:
                    total_pair_hits += math.comb(int(count), 2)

    return total_pair_hits / total_pairs


if __name__ == "__main__":
    print(simulated_mixing(10, 3, 3, 2, seed=0))
