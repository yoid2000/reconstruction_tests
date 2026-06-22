from __future__ import annotations

import argparse
import json
import math
from typing import Dict


DEFAULT_MAX_NUM_CONTINGENCY_TABLES = 100000


def selected_table_size_counts(
    nqi: int,
    max_num_contingency_tables: int,
) -> Dict[int, int]:
    """Return how many selected contingency tables have each QI-column count."""
    total_available = sum(math.comb(nqi, subset_size) for subset_size in range(1, nqi + 1))
    remaining = min(max(0, max_num_contingency_tables), total_available)
    counts: Dict[int, int] = {}

    for subset_size in range(1, nqi + 1):
        if remaining == 0:
            break

        available_at_size = math.comb(nqi, subset_size)
        selected_at_size = min(remaining, available_at_size)
        if selected_at_size > 0:
            counts[subset_size] = selected_at_size
        remaining -= selected_at_size

    return counts


def estimate_nrows(
    *,
    max_num_contingency_tables: int,
    nqi: int,
    nunique: int,
    vals_per_qi: int,
    min_num_rows: int,
) -> dict:
    """Estimate rows needed for no QI contingency-table cell suppression."""
    if max_num_contingency_tables < 0:
        raise ValueError("max_num_contingency_tables must be non-negative.")
    if nqi < 0:
        raise ValueError("nqi must be non-negative.")
    if nunique < 1:
        raise ValueError("nunique must be at least 1.")
    if vals_per_qi < 1:
        raise ValueError("vals_per_qi must be at least 1.")
    if min_num_rows < 1:
        raise ValueError("min_num_rows must be at least 1.")

    table_size_counts = selected_table_size_counts(nqi, max_num_contingency_tables)
    max_selected_qi_cols = max(table_size_counts.keys(), default=0)
    max_cells_per_table = vals_per_qi ** max_selected_qi_cols if max_selected_qi_cols else 0
    estimated_nrows = min_num_rows * max_cells_per_table
    full_factorial_nrows = vals_per_qi ** nqi if nqi else 0
    selected_contingency_tables = sum(table_size_counts.values())
    total_possible_contingency_tables = sum(
        math.comb(nqi, subset_size) for subset_size in range(1, nqi + 1)
    )
    selected_cells_across_tables = sum(
        count * (vals_per_qi ** subset_size)
        for subset_size, count in table_size_counts.items()
    )
    if max_selected_qi_cols == 0:
        estimate_changes_at_table = None
        estimate_constant_through_table = 0
    elif max_selected_qi_cols == nqi:
        estimate_changes_at_table = None
        estimate_constant_through_table = total_possible_contingency_tables
    else:
        estimate_constant_through_table = sum(
            math.comb(nqi, subset_size)
            for subset_size in range(1, max_selected_qi_cols + 1)
        )
        estimate_changes_at_table = estimate_constant_through_table + 1

    return {
        "estimated_nrows": estimated_nrows,
        "max_selected_qi_cols": max_selected_qi_cols,
        "max_cells_per_table": max_cells_per_table,
        "selected_table_size_counts": table_size_counts,
        "selected_contingency_tables": selected_contingency_tables,
        "total_possible_contingency_tables": total_possible_contingency_tables,
        "selected_cells_across_tables": selected_cells_across_tables,
        "estimate_constant_through_table": estimate_constant_through_table,
        "estimate_changes_at_table": estimate_changes_at_table,
        "full_factorial_nrows": full_factorial_nrows,
        "nunique_used_for_suppression": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate nrows needed so selected QI contingency-table cells have "
            "at least min_num_rows rows under an ideal balanced distribution."
        )
    )
    parser.add_argument(
        "--max_num_contingency_tables",
        type=int,
        default=DEFAULT_MAX_NUM_CONTINGENCY_TABLES,
    )
    parser.add_argument("--nqi", type=int, default=3)
    parser.add_argument("--nunique", type=int, default=2)
    parser.add_argument("--vals_per_qi", type=int, default=2)
    parser.add_argument("--min_num_rows", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = estimate_nrows(
        max_num_contingency_tables=args.max_num_contingency_tables,
        nqi=args.nqi,
        nunique=args.nunique,
        vals_per_qi=args.vals_per_qi,
        min_num_rows=args.min_num_rows,
    )

    if args.json:
        print(json.dumps(result, indent=4))
        return

    print(f"estimated_nrows: {result['estimated_nrows']}")
    print(f"max_selected_qi_cols: {result['max_selected_qi_cols']}")
    print(f"max_cells_per_table: {result['max_cells_per_table']}")
    print(f"selected_table_size_counts: {result['selected_table_size_counts']}")
    print(f"selected_contingency_tables: {result['selected_contingency_tables']}")
    print(f"total_possible_contingency_tables: {result['total_possible_contingency_tables']}")
    print(f"selected_cells_across_tables: {result['selected_cells_across_tables']}")
    print(f"estimate_constant_through_table: {result['estimate_constant_through_table']}")
    print(f"estimate_changes_at_table: {result['estimate_changes_at_table']}")
    print(f"full_factorial_nrows: {result['full_factorial_nrows']}")
    print("nunique_used_for_suppression: False")
    print(
        "Assumption: rows are ideally balanced across the largest selected "
        "QI contingency table; current suppression does not depend on nunique."
    )


if __name__ == "__main__":
    main()
