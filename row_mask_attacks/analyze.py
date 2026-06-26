from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)

from experiments import read_experiments
from plotters.solver_metric_scatterplots import plot_solver_metric_scatterplots


DEFAULT_RESULTS_PATH = Path(__file__).parent / "results" / "results.parquet"
DEFAULT_PLOTS_DIR = Path(__file__).parent / "plots"
NOT_PARAM_KEYS = {"not_params"}
RESULT_FIELD = "alc_alc"
SUPPRESSION_FIELD = "num_suppressed"
CONTINGENCY_TABLES_USED_FIELD = "num_contingency_tables_used"
TABLE_VALUE_FIELDS = [
    RESULT_FIELD,
    SUPPRESSION_FIELD,
    CONTINGENCY_TABLES_USED_FIELD,
]
IGNORED_TABLE_COLUMNS = {"p__supp_thresh"}


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


def as_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def experiment_group_value(experiment: dict[str, Any]) -> Any:
    return experiment.get(
        "experiment_group",
        experiment.get("not_params", {}).get("experiment_group"),
    )


def distinct_experiment_groups(experiments: list[dict[str, Any]]) -> list[Any]:
    groups = []
    seen = set()
    for experiment in experiments:
        group = experiment_group_value(experiment)
        if group is None or group in seen:
            continue
        seen.add(group)
        groups.append(group)
    return groups


def matching_rows_for_experiment(
    df_group: pd.DataFrame,
    experiment: dict[str, Any],
) -> pd.Series:
    matches = pd.Series(True, index=df_group.index)
    for key, value in experiment.items():
        if key in NOT_PARAM_KEYS:
            continue

        column = f"p__{key}"
        if column not in df_group.columns:
            continue

        matches &= df_group[column].isin(as_values(value))
    return matches


def rows_for_experiment_group(
    df_group: pd.DataFrame,
    experiments: list[dict[str, Any]],
    experiment_group: Any,
) -> pd.DataFrame:
    matches = pd.Series(False, index=df_group.index)
    for experiment in experiments:
        if experiment_group_value(experiment) != experiment_group:
            continue
        matches |= matching_rows_for_experiment(df_group, experiment)
    return df_group[matches].copy()


def varying_parameter_columns(df_experiment_group: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df_experiment_group.columns
        if (
            column.startswith("p__")
            and column not in IGNORED_TABLE_COLUMNS
            and df_experiment_group[column].nunique(dropna=False) > 1
        )
    ]


def sorted_unique_values(series: pd.Series) -> list[Any]:
    values = series.drop_duplicates().tolist()
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: str(value))


def matching_value_rows(df: pd.DataFrame, column: str, value: Any) -> pd.Series:
    if pd.isna(value):
        return df[column].isna()
    return df[column] == value


def filter_by_values(df: pd.DataFrame, values_by_column: dict[str, Any]) -> pd.DataFrame:
    matches = pd.Series(True, index=df.index)
    for column, value in values_by_column.items():
        matches &= matching_value_rows(df, column, value)
    return df[matches]


def format_filter_values(values_by_column: dict[str, Any]) -> str:
    if not values_by_column:
        return "all rows"
    return ", ".join(f"{column}={value}" for column, value in values_by_column.items())


def remaining_value_combinations(
    df_experiment_group: pd.DataFrame,
    remaining_columns: list[str],
) -> list[dict[str, Any]]:
    if not remaining_columns:
        return [{}]

    return [
        row.to_dict()
        for _, row in df_experiment_group[remaining_columns].drop_duplicates().iterrows()
    ]


def print_value_pivot(
    df_table: pd.DataFrame,
    row_column: str,
    column_column: str,
    value_column: str,
) -> None:
    table = df_table.pivot_table(
        index=row_column,
        columns=column_column,
        values=value_column,
        aggfunc="mean",
    )
    table = table.reindex(index=sorted_unique_values(df_table[row_column]))
    table = table.reindex(columns=sorted_unique_values(df_table[column_column]))
    print(table.to_string(float_format=lambda value: f"{value:10.3f}"))


def print_experiment_group_tables(
    experiment_group: Any,
    df_experiment_group: pd.DataFrame,
) -> None:
    print(f"\nexperiment_group: {experiment_group}")
    if df_experiment_group.empty:
        print("No matching rows.")
        return
    missing_value_fields = [
        column for column in TABLE_VALUE_FIELDS
        if column not in df_experiment_group.columns
    ]
    if missing_value_fields:
        raise ValueError(
            f"df_experiment_group is missing required columns: {missing_value_fields}"
        )

    varying_columns = varying_parameter_columns(df_experiment_group)
    print(f"varying variables: {varying_columns}")
    if len(varying_columns) < 2:
        print("Fewer than two varying variables; no 2D tables generated.")
        return

    for row_column, column_column in combinations(varying_columns, 2):
        remaining_columns = [
            column for column in varying_columns
            if column not in {row_column, column_column}
        ]
        for remaining_values in remaining_value_combinations(
            df_experiment_group,
            remaining_columns,
        ):
            df_table = filter_by_values(df_experiment_group, remaining_values)
            print(f"\nrows={row_column}, columns={column_column}")
            print(f"fixed: {format_filter_values(remaining_values)}")
            for value_column in TABLE_VALUE_FIELDS:
                print(f"value={value_column}")
                print_value_pivot(df_table, row_column, column_column, value_column)


def handle_experiment_group(experiment_group: Any, df_experiment_group: pd.DataFrame) -> None:
    print_experiment_group_tables(experiment_group, df_experiment_group)
    # Placeholder for future logic keyed by experiment_group.
    if experiment_group == "agg_dinur_noise_no_supp":
        pass
    elif experiment_group == "agg_dinur_supp_no_noise":
        pass


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
    experiment_definitions = read_experiments()
    for experiment_group in distinct_experiment_groups(experiment_definitions):
        df_experiment_group = rows_for_experiment_group(
            df_group,
            experiment_definitions,
            experiment_group,
        )
        handle_experiment_group(experiment_group, df_experiment_group)
    plot_path = plot_solver_metric_scatterplots(df_group, args.plots_dir)
    print(f"\nWrote plot: {plot_path}")


if __name__ == "__main__":
    main()
