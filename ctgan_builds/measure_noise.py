from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from ctgan_common import resolve_local_path


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


def measure_group_noise(df_source: pd.DataFrame, df_synth: pd.DataFrame) -> pd.Series:
    group_columns = list(df_synth.columns)

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
    for num_columns, group in noise_df.groupby("num_columns"):
        noise = group["noise"].astype(float)
        summary_rows.append(
            {
                "num_columns": int(num_columns),
                "group_count": int(len(group)),
                "synthetic_files": int(group["synth_file"].nunique()),
                "mean_noise": float(noise.mean()),
                "std_noise": float(noise.std(ddof=1)) if len(group) > 1 else 0.0,
                "mean_abs_noise": float(noise.abs().mean()),
                "rmse_noise": math.sqrt(float((noise**2).mean())),
                "min_noise": float(noise.min()),
                "max_noise": float(noise.max()),
            }
        )

    return pd.DataFrame(summary_rows).sort_values("num_columns")


def main() -> None:
    args = parse_args()
    source_path = resolve_local_path(args.source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source parquet file not found: {source_path}")

    df_source = pd.read_parquet(source_path)
    synth_paths = list_synth_paths(args.synth_dir)

    rows: list[dict[str, object]] = []
    for synth_path in synth_paths:
        df_synth = pd.read_parquet(synth_path)
        if df_synth.empty:
            continue

        noise = measure_group_noise(df_source, df_synth)
        num_columns = len(df_synth.columns)
        for value in noise.tolist():
            rows.append(
                {
                    "synth_file": str(synth_path),
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
    print("")
    print("Noise summary by group-by column count")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))


if __name__ == "__main__":
    main()
