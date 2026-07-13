import itertools
import random
from typing import List

import pandas as pd


def contingency_table_columns(df: pd.DataFrame, num_contingency_tables: int) -> List[List[str]]:
    """Return QI column groups ordered by distinct value-vector count."""
    if num_contingency_tables < 1:
        return []

    qi_cols = sorted([col for col in df.columns if col.startswith('qi')])
    candidates = []
    for subset_size in range(1, len(qi_cols) + 1):
        for cols_tuple in itertools.combinations(qi_cols, subset_size):
            cols = list(cols_tuple)
            num_distinct_vectors = int(df[cols].drop_duplicates().shape[0])
            candidates.append((num_distinct_vectors, subset_size, cols))

    candidates_by_key: dict[tuple[int, int], list[list[str]]] = {}
    for num_distinct_vectors, subset_size, cols in candidates:
        candidates_by_key.setdefault((num_distinct_vectors, subset_size), []).append(cols)

    ordered_candidates: list[list[str]] = []
    for key in sorted(candidates_by_key):
        tied_candidates = candidates_by_key[key]
        random.shuffle(tied_candidates)
        ordered_candidates.extend(tied_candidates)

    return ordered_candidates[:num_contingency_tables]
