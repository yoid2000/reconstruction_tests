from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)

from plotters.solver_metric_scatterplots import plot_solver_metric_scatterplots


DEFAULT_RESULTS_PATH = Path(__file__).parent / "results" / "results.parquet"
DEFAULT_PLOTS_DIR = Path(__file__).parent / "plots"


def values_all_same(series: pd.Series) -> bool:
    return series.nunique(dropna=False) <= 1


def max_value(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return None
    try:
        return non_null.max()
    except TypeError:
        return max(str(value) for value in non_null)


def average_datetime(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NaT
    return pd.to_datetime(int(non_null.astype("int64").mean()))


def average_timedelta(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NaT
    return pd.to_timedelta(int(non_null.astype("int64").mean()))


def grouped_value(series: pd.Series) -> Any:
    if values_all_same(series):
        return series.iloc[0]
    if is_bool_dtype(series):
        return max_value(series)
    if is_numeric_dtype(series):
        return series.mean()
    if is_datetime64_any_dtype(series):
        return average_datetime(series)
    if is_timedelta64_dtype(series):
        return average_timedelta(series)
    return max_value(series)


def build_grouped_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "experiment_id" not in df.columns:
        raise ValueError("results dataframe must contain an experiment_id column.")

    rows = []
    for _, group in df.groupby("experiment_id", sort=False):
        rows.append({column: grouped_value(group[column]) for column in df.columns})
    return pd.DataFrame(rows, columns=df.columns)


def print_alc_tables(df_group: pd.DataFrame) -> None:
    required_columns = {
        "p__max_num_contingency_tables",
        "p__noise",
        "p__min_num_rows",
        "alc_alc",
    }
    missing_columns = sorted(required_columns - set(df_group.columns))
    if missing_columns:
        raise ValueError(f"df_group is missing required columns: {missing_columns}")

    for max_tables in sorted(df_group["p__max_num_contingency_tables"].dropna().unique()):
        table_df = df_group[df_group["p__max_num_contingency_tables"] == max_tables]
        table = table_df.pivot_table(
            index="p__noise",
            columns="p__min_num_rows",
            values="alc_alc",
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)

        print(f"\np__max_num_contingency_tables = {max_tables}")
        print(table.to_string(float_format=lambda value: f"{value:.3g}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to results.parquet.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Directory where PNG plots are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.results)
    df_group = build_grouped_dataframe(df)
    print_alc_tables(df_group)
    plot_path = plot_solver_metric_scatterplots(df_group, args.plots_dir)
    print(f"\nWrote plot: {plot_path}")


if __name__ == "__main__":
    main()
