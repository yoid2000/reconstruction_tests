from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from ctgan_common import resolve_local_path

MAX_SAMPLES_PER_MEASURE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure synthetic count noise against a source parquet file."
    )
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--synth_dir", type=str, required=True)
    return parser.parse_args()


def list_synth_paths(synth_dir: str) -> list[Path]:
    resolved_dir = resolve_local_path(synth_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Synthetic parquet directory not found: {resolved_dir}")

    synth_paths = sorted(path for path in resolved_dir.rglob("*.parquet") if path.is_file())
    if not synth_paths:
        raise FileNotFoundError(f"No synthetic parquet files found under: {resolved_dir}")

    return synth_paths


def measure_group_noise(
    df_source: pd.DataFrame,
    df_synth: pd.DataFrame,
    group_columns: Iterable[str],
) -> pd.Series:
    group_columns = list(group_columns)
    if not set(group_columns).issubset(df_source.columns):
        missing = [column for column in group_columns if column not in df_source.columns]
        raise ValueError(f"Source file is missing columns required by synthetic file: {missing}")

    source_counts = (
        df_source[group_columns]
        .groupby(group_columns, dropna=False)
        .size()
        .rename("source_count")
        .reset_index()
    )
    synth_counts = (
        df_synth.groupby(group_columns, dropna=False)
        .size()
        .rename("synth_count")
        .reset_index()
    )

    merged = source_counts.merge(synth_counts, on=group_columns, how="outer")
    merged["source_count"] = merged["source_count"].fillna(0).astype(int)
    merged["synth_count"] = merged["synth_count"].fillna(0).astype(int)
    return merged["synth_count"] - merged["source_count"]


def summarize_noise_by_num_columns(rows: list[dict[str, object]]) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    if not rows:
        return pd.DataFrame(summary_rows)

    noise_df = pd.DataFrame(rows)
    for (table_width, num_columns), group in noise_df.groupby(["table_width", "num_columns"]):
        available_group_count = int(len(group))
        sampled_group = (
            group.sample(n=MAX_SAMPLES_PER_MEASURE, random_state=0)
            if len(group) > MAX_SAMPLES_PER_MEASURE
            else group
        )
        noise = sampled_group["noise"].astype(float)
        summary_rows.append(
            {
                "table_width": int(table_width),
                "num_columns": int(num_columns),
                "available_group_count": available_group_count,
                "sampled_group_count": int(len(sampled_group)),
                "synthetic_files": int(group["synth_file"].nunique()),
                "mean_noise": float(noise.mean()),
                "std_noise": float(noise.std(ddof=1)) if len(sampled_group) > 1 else 0.0,
                "mean_abs_noise": float(noise.abs().mean()),
                "rmse_noise": math.sqrt(float((noise**2).mean())),
                "min_noise": float(noise.min()),
                "max_noise": float(noise.max()),
            }
        )

    return pd.DataFrame(summary_rows).sort_values(["table_width", "num_columns"])


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    if value == 0:
        return "0.0000"

    abs_value = abs(value)
    fixed_precision = f"{abs_value:.16f}".split(".", maxsplit=1)[1]
    first_nonzero_index = next(
        (index for index, digit in enumerate(fixed_precision, start=1) if digit != "0"),
        None,
    )
    precision = 4 if first_nonzero_index is None else max(4, first_nonzero_index + 4)
    return f"{value:.{precision}f}"


def print_summary_text(summary_df: pd.DataFrame) -> None:
    for table_width, width_group in summary_df.groupby("table_width", sort=True):
        print(f"Table width {int(table_width)}")
        for _, row in width_group.sort_values("num_columns").iterrows():
            print(
                f"  columns={int(row['num_columns'])}: "
                f"available_groups={int(row['available_group_count'])}, "
                f"sampled_groups={int(row['sampled_group_count'])}, "
                f"synthetic_files={int(row['synthetic_files'])}, "
                f"mean_noise={format_float(float(row['mean_noise']))}, "
                f"std_noise={format_float(float(row['std_noise']))}, "
                f"mean_abs_noise={format_float(float(row['mean_abs_noise']))}, "
                f"rmse_noise={format_float(float(row['rmse_noise']))}, "
                f"min_noise={format_float(float(row['min_noise']))}, "
                f"max_noise={format_float(float(row['max_noise']))}"
            )
        print("")


def print_table_width_counts(synth_paths: list[Path]) -> None:
    width_counts: dict[int, int] = {}
    for synth_path in synth_paths:
        df_synth = pd.read_parquet(synth_path, columns=None)
        table_width = len(df_synth.columns)
        width_counts[table_width] = width_counts.get(table_width, 0) + 1

    print("Synthetic table counts by width")
    for table_width in sorted(width_counts):
        print(f"  width={table_width}: tables={width_counts[table_width]}")
    print("")


def main() -> None:
    args = parse_args()
    source_path = resolve_local_path(args.source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source parquet file not found: {source_path}")

    df_source = pd.read_parquet(source_path)
    synth_paths = list_synth_paths(args.synth_dir)
    print_table_width_counts(synth_paths)

    rows: list[dict[str, object]] = []
    for synth_path in synth_paths:
        df_synth = pd.read_parquet(synth_path)
        if df_synth.empty:
            continue

        synth_columns = list(df_synth.columns)
        table_width = len(synth_columns)
        for num_columns in range(1, table_width + 1):
            group_columns = synth_columns[:num_columns]
            noise = measure_group_noise(df_source, df_synth, group_columns)
            for value in noise.tolist():
                rows.append(
                    {
                        "synth_file": str(synth_path),
                        "table_width": table_width,
                        "num_columns": num_columns,
                        "noise": int(value),
                    }
                )

    summary_df = summarize_noise_by_num_columns(rows)
    if summary_df.empty:
        print("No noise measurements were produced.")
        return

    print(f"Source file: {source_path}")
    print(f"Synthetic files processed: {len(synth_paths)}")
    print(f"Max samples per measure: {MAX_SAMPLES_PER_MEASURE}")
    print("")
    print("Noise summary by table width and group-by column count")
    print("")
    print_summary_text(summary_df)


if __name__ == "__main__":
    main()
