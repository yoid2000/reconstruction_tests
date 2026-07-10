from __future__ import annotations

import argparse
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


def main() -> None:
    args = parse_args()
    info_path = resolve_local_path(args.info)
    if not info_path.exists():
        raise FileNotFoundError(f"Results parquet not found: {info_path}")

    df_results = pd.read_parquet(info_path)
    if "seed" not in df_results.columns or "p__output_path" not in df_results.columns:
        raise ValueError("results.parquet must contain 'seed' and 'p__output_path' columns.")

    rows_to_remove = df_results[df_results["seed"] > 4].copy()
    print(f"Found {len(rows_to_remove)} rows with seed > 4")

    deleted_count = 0
    missing_count = 0
    for _, row in rows_to_remove.iterrows():
        seed = int(row["seed"])
        output_path = str(row["p__output_path"])
        seeded_output_path = resolve_seeded_output_path(output_path, seed)
        if seeded_output_path.exists():
            seeded_output_path.unlink()
            deleted_count += 1
            print(f"Deleted {seeded_output_path}")
        else:
            missing_count += 1
            print(f"Missing file, skipped delete: {seeded_output_path}")

    cleaned_df = df_results[df_results["seed"] <= 4].copy()
    cleaned_df["p__output_path"] = cleaned_df.apply(
        lambda row: get_seeded_output_path_str(str(row["p__output_path"]), int(row["seed"])),
        axis=1,
    )
    cleaned_df.to_parquet(info_path, index=False)
    print(
        f"Wrote cleaned results to {info_path} "
        f"(deleted={deleted_count}, missing={missing_count}, remaining_rows={len(cleaned_df)})"
    )


if __name__ == "__main__":
    main()
