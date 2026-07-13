from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ctgan_common import get_seeded_output_path, resolve_local_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete seeded CTGAN outputs for seeds > 4 and rewrite results.parquet."
    )
    parser.add_argument(
        "--info",
        required=True,
        help="Path to the results.parquet file to clean in place.",
    )
    return parser.parse_args()


def resolve_seeded_output_path(output_path: str, seed: int) -> Path:
    seeded_output_path = get_seeded_output_path_str(output_path, seed)
    return resolve_local_path(seeded_output_path)


def get_seeded_output_path_str(output_path: str, seed: int) -> str:
    output_file = Path(output_path)
    expected_suffix = f"_{seed}"
    if output_file.stem.endswith(expected_suffix):
        return output_path
    return get_seeded_output_path(output_path, seed)


def get_contingency_table_width(raw_value: str) -> int:
    parsed = json.loads(raw_value)
    if not isinstance(parsed, list):
        raise ValueError(f"Invalid p__contingency_table value: {raw_value!r}")
    return len(parsed)


def main() -> None:
    args = parse_args()
    info_path = resolve_local_path(args.info)
    if not info_path.exists():
        raise FileNotFoundError(f"Results parquet not found: {info_path}")

    df_results = pd.read_parquet(info_path)
    required_columns = {"seed", "p__output_path", "p__contingency_table"}
    missing_columns = required_columns - set(df_results.columns)
    if missing_columns:
        raise ValueError(
            "results.parquet must contain columns: "
            f"{sorted(required_columns)}; missing {sorted(missing_columns)}"
        )

    df_results = df_results.copy()
    df_results["contingency_table_width"] = df_results["p__contingency_table"].map(
        lambda raw_value: get_contingency_table_width(str(raw_value))
    )

    seed_mask = df_results["seed"] > 4
    width_mask = df_results["contingency_table_width"] >= 6
    rows_to_remove = df_results[seed_mask | width_mask].copy()
    print(
        f"Found {len(rows_to_remove)} rows to remove "
        f"(seed_gt_4={int(seed_mask.sum())}, width_ge_6={int(width_mask.sum())})"
    )

    deleted_count = 0
    missing_count = 0
    deleted_paths: set[Path] = set()
    for _, row in rows_to_remove.iterrows():
        seed = int(row["seed"])
        output_path = str(row["p__output_path"])
        seeded_output_path = resolve_seeded_output_path(output_path, seed)
        if seeded_output_path in deleted_paths:
            continue
        if seeded_output_path.exists():
            seeded_output_path.unlink()
            deleted_paths.add(seeded_output_path)
            deleted_count += 1
            print(f"Deleted {seeded_output_path}")
        else:
            missing_count += 1
            print(f"Missing file, skipped delete: {seeded_output_path}")

    cleaned_df = df_results[~(seed_mask | width_mask)].copy()
    cleaned_df["p__output_path"] = cleaned_df.apply(
        lambda row: get_seeded_output_path_str(str(row["p__output_path"]), int(row["seed"])),
        axis=1,
    )
    cleaned_df = cleaned_df.drop(columns=["contingency_table_width"])
    cleaned_df.to_parquet(info_path, index=False)
    print(
        f"Wrote cleaned results to {info_path} "
        f"(deleted={deleted_count}, missing={missing_count}, remaining_rows={len(cleaned_df)})"
    )


if __name__ == "__main__":
    main()
