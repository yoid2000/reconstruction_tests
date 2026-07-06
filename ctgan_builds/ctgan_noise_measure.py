from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from common.contingency_tables import contingency_table_columns

try:
    from .ctgan_common import (
        generate_synthetic_dataframe,
        load_ctgan_config,
        resolve_local_path,
        set_random_seed,
    )
except ImportError:
    from ctgan_common import (
        generate_synthetic_dataframe,
        load_ctgan_config,
        resolve_local_path,
        set_random_seed,
    )


@dataclass
class NoiseAccumulator:
    count: int = 0
    sum_noise: float = 0.0
    sum_noise_sq: float = 0.0
    sum_abs_noise: float = 0.0
    min_noise: float = math.inf
    max_noise: float = -math.inf
    zero_count: int = 0
    source_only_count: int = 0
    synth_only_count: int = 0
    both_count: int = 0
    run_count: int = 0
    table_count: int = 0
    input_file_count: int = 0

    def update(
        self,
        noise_df: pd.DataFrame,
        *,
        source_only_count: int,
        synth_only_count: int,
        both_count: int,
    ) -> None:
        if noise_df.empty:
            return

        noise_series = noise_df["noise"].astype(float)
        self.count += int(noise_series.shape[0])
        self.sum_noise += float(noise_series.sum())
        self.sum_noise_sq += float((noise_series ** 2).sum())
        self.sum_abs_noise += float(noise_series.abs().sum())
        self.min_noise = min(self.min_noise, float(noise_series.min()))
        self.max_noise = max(self.max_noise, float(noise_series.max()))
        self.zero_count += int((noise_series == 0).sum())
        self.source_only_count += source_only_count
        self.synth_only_count += synth_only_count
        self.both_count += both_count

    def mean_noise(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum_noise / self.count

    def std_noise(self) -> float:
        if self.count <= 1:
            return 0.0
        variance = (self.sum_noise_sq - ((self.sum_noise ** 2) / self.count)) / (self.count - 1)
        return math.sqrt(max(variance, 0.0))

    def mean_abs_noise(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum_abs_noise / self.count

    def rmse_noise(self) -> float:
        if self.count == 0:
            return 0.0
        return math.sqrt(self.sum_noise_sq / self.count)

    def exact_match_fraction(self) -> float:
        if self.count == 0:
            return 0.0
        return self.zero_count / self.count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure group-count noise introduced by CTGAN synthetic datasets."
    )
    parser.add_argument("--input_dir", type=str, default="input_files")
    parser.add_argument("--ctgan_configs_dir", type=str, default="ctgan_configs")
    parser.add_argument("--contingency_tables", type=int, default=200)
    parser.add_argument("--synthetic_datasets", type=int, default=20)
    parser.add_argument("--seed_base", type=int, default=1000)
    return parser.parse_args()


def list_input_files(input_dir: str) -> list[Path]:
    resolved_dir = resolve_local_path(input_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {resolved_dir}")

    input_paths = sorted(resolved_dir.glob("*.parquet"))
    if not input_paths:
        raise FileNotFoundError(f"No parquet files found under: {resolved_dir}")

    return input_paths


def list_config_files(ctgan_configs_dir: str) -> list[Path]:
    resolved_dir = resolve_local_path(ctgan_configs_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(f"CTGAN configs directory not found: {resolved_dir}")

    config_paths = sorted(resolved_dir.glob("*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No CTGAN yaml files found under: {resolved_dir}")

    return config_paths


def build_contingency_tables_with_val(
    df: pd.DataFrame,
    *,
    num_contingency_tables: int,
) -> list[list[str]]:
    if "val" not in df.columns:
        raise ValueError("Expected input dataframe to contain a 'val' column.")

    qi_tables = contingency_table_columns(df, num_contingency_tables)
    tables_with_val: list[list[str]] = []
    for columns in qi_tables:
        updated_columns = list(columns)
        if "val" not in updated_columns:
            updated_columns.append("val")
        tables_with_val.append(updated_columns)
    return tables_with_val


def measure_group_count_noise(df_source: pd.DataFrame, df_synth: pd.DataFrame) -> pd.DataFrame:
    group_columns = list(df_source.columns)

    source_counts = (
        df_source.groupby(group_columns, dropna=False)
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

    noise_df = source_counts.merge(synth_counts, on=group_columns, how="outer")
    noise_df["source_count"] = noise_df["source_count"].fillna(0).astype(int)
    noise_df["synth_count"] = noise_df["synth_count"].fillna(0).astype(int)
    noise_df["noise"] = noise_df["synth_count"] - noise_df["source_count"]
    noise_df["abs_noise"] = noise_df["noise"].abs()
    return noise_df


def summarize_noise_run(noise_df: pd.DataFrame) -> dict[str, float | int]:
    source_only_mask = (noise_df["source_count"] > 0) & (noise_df["synth_count"] == 0)
    synth_only_mask = (noise_df["source_count"] == 0) & (noise_df["synth_count"] > 0)
    both_mask = (noise_df["source_count"] > 0) & (noise_df["synth_count"] > 0)

    noise_series = noise_df["noise"].astype(float)
    abs_noise_series = noise_df["abs_noise"].astype(float)

    return {
        "num_groups": int(len(noise_df)),
        "num_source_only_groups": int(source_only_mask.sum()),
        "num_synth_only_groups": int(synth_only_mask.sum()),
        "num_shared_groups": int(both_mask.sum()),
        "mean_noise": float(noise_series.mean()),
        "std_noise": float(noise_series.std(ddof=1)) if len(noise_df) > 1 else 0.0,
        "mean_abs_noise": float(abs_noise_series.mean()),
        "median_abs_noise": float(abs_noise_series.median()),
        "p95_abs_noise": float(abs_noise_series.quantile(0.95)),
        "max_abs_noise": float(abs_noise_series.max()),
        "rmse_noise": math.sqrt(float((noise_series ** 2).mean())),
        "exact_match_fraction": float((noise_series == 0).mean()),
        "l1_count_shift": float(abs_noise_series.sum() / 2.0),
    }


def seed_for_run(
    *,
    seed_base: int,
    input_index: int,
    config_index: int,
    table_index: int,
    sample_index: int,
) -> int:
    return (
        seed_base
        + (input_index * 1_000_000)
        + (config_index * 100_000)
        + (table_index * 100)
        + sample_index
    )


def print_header(args: argparse.Namespace, input_paths: list[Path], config_paths: list[Path]) -> None:
    print("CTGAN noise measurement")
    print(f"Input files: {len(input_paths)}")
    print(f"CTGAN configs: {len(config_paths)}")
    print(f"Contingency tables per input file: {args.contingency_tables}")
    print(f"Synthetic datasets per config/table pair: {args.synthetic_datasets}")
    print("")


def print_group_level_summary(group_summary_df: pd.DataFrame) -> None:
    if group_summary_df.empty:
        print("No group-level noise summary was produced.")
        return

    print("Group-level noise summary by config and contingency-table column count")
    print(group_summary_df.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))
    print("")


def print_run_level_summary(run_summary_df: pd.DataFrame) -> None:
    if run_summary_df.empty:
        print("No run-level summary was produced.")
        return

    print("Per-synthetic-dataset summary aggregated by config and contingency-table column count")
    print(run_summary_df.to_string(index=False, float_format=lambda value: f"{value:0.4f}"))
    print("")


def build_summary_frames(
    combo_stats: dict[tuple[str, int], NoiseAccumulator],
    run_summaries: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_summary_rows: list[dict[str, object]] = []
    for (config_name, num_columns), stats in sorted(combo_stats.items()):
        group_summary_rows.append(
            {
                "config_name": config_name,
                "num_columns": num_columns,
                "group_count": stats.count,
                "synthetic_datasets": stats.run_count,
                "tables_processed": stats.table_count,
                "input_files": stats.input_file_count,
                "mean_noise": stats.mean_noise(),
                "std_noise": stats.std_noise(),
                "mean_abs_noise": stats.mean_abs_noise(),
                "rmse_noise": stats.rmse_noise(),
                "min_noise": stats.min_noise if stats.count else 0.0,
                "max_noise": stats.max_noise if stats.count else 0.0,
                "exact_match_fraction": stats.exact_match_fraction(),
                "source_only_groups": stats.source_only_count,
                "synth_only_groups": stats.synth_only_count,
                "shared_groups": stats.both_count,
            }
        )

    group_summary_df = pd.DataFrame(group_summary_rows)
    if not group_summary_df.empty:
        group_summary_df = group_summary_df.sort_values(["config_name", "num_columns"])

    run_summary_df = pd.DataFrame(run_summaries)
    if not run_summary_df.empty:
        run_summary_df = (
            run_summary_df.groupby(["config_name", "num_columns"], as_index=False)
            .agg(
                synthetic_datasets=("seed", "count"),
                input_files=("input_file", "nunique"),
                tables_processed=("table_signature", "nunique"),
                mean_groups=("num_groups", "mean"),
                mean_noise=("mean_noise", "mean"),
                mean_std_noise=("std_noise", "mean"),
                mean_abs_noise=("mean_abs_noise", "mean"),
                median_abs_noise=("median_abs_noise", "mean"),
                p95_abs_noise=("p95_abs_noise", "mean"),
                max_abs_noise=("max_abs_noise", "mean"),
                mean_exact_match_fraction=("exact_match_fraction", "mean"),
                mean_source_only_groups=("num_source_only_groups", "mean"),
                mean_synth_only_groups=("num_synth_only_groups", "mean"),
                mean_l1_count_shift=("l1_count_shift", "mean"),
                mean_elapsed_seconds=("elapsed_seconds", "mean"),
            )
            .sort_values(["config_name", "num_columns"])
        )

    return group_summary_df, run_summary_df


def print_progress_summary(
    *,
    sample_index: int,
    total_samples: int,
    combo_stats: dict[tuple[str, int], NoiseAccumulator],
    run_summaries: list[dict[str, object]],
) -> None:
    group_summary_df, run_summary_df = build_summary_frames(combo_stats, run_summaries)
    print("")
    print(
        f"Completed repeat {sample_index + 1} of {total_samples}. "
        "Statistics so far:",
        flush=True,
    )
    print_group_level_summary(group_summary_df)
    print_run_level_summary(run_summary_df)
    sys.stdout.flush()


def main() -> None:
    args = parse_args()
    input_paths = list_input_files(args.input_dir)
    config_paths = list_config_files(args.ctgan_configs_dir)
    loaded_configs = [(config_path.stem, load_ctgan_config(str(config_path))) for config_path in config_paths]

    print_header(args, input_paths, config_paths)

    combo_stats: dict[tuple[str, int], NoiseAccumulator] = {}
    run_summaries: list[dict[str, object]] = []
    input_specs: list[tuple[int, Path, list[list[str]], pd.DataFrame]] = []

    for input_index, input_path in enumerate(input_paths):
        df_input = pd.read_parquet(input_path)
        contingency_tables = build_contingency_tables_with_val(
            df_input,
            num_contingency_tables=args.contingency_tables,
        )
        print(
            f"Processing {input_path.name}: {len(contingency_tables)} contingency tables, "
            f"{len(df_input)} rows"
        )
        input_specs.append((input_index, input_path, contingency_tables, df_input))

    for input_index, _input_path, contingency_tables, _df_input in input_specs:
        combos_seen_for_input: set[tuple[str, int]] = set()
        for contingency_table in contingency_tables:
            num_columns = len(contingency_table)
            for config_name, _ctgan_config in loaded_configs:
                combo_key = (config_name, num_columns)
                stats = combo_stats.setdefault(combo_key, NoiseAccumulator())
                if combo_key not in combos_seen_for_input:
                    stats.input_file_count += 1
                    combos_seen_for_input.add(combo_key)
                stats.table_count += 1

    for sample_index in range(args.synthetic_datasets):
        for input_index, input_path, contingency_tables, df_input in input_specs:
            for table_index, contingency_table in enumerate(contingency_tables):
                num_columns = len(contingency_table)
                df_source = df_input[contingency_table].copy()

                for config_index, (config_name, ctgan_config) in enumerate(loaded_configs):
                    seed = seed_for_run(
                        seed_base=args.seed_base,
                        input_index=input_index,
                        config_index=config_index,
                        table_index=table_index,
                        sample_index=sample_index,
                    )
                    set_random_seed(seed)

                    started_at = time.perf_counter()
                    df_synth, _ = generate_synthetic_dataframe(df_source, ctgan_config)
                    elapsed_seconds = time.perf_counter() - started_at

                    noise_df = measure_group_count_noise(df_source, df_synth)
                    run_summary = summarize_noise_run(noise_df)
                    run_summary.update(
                        {
                            "input_file": input_path.name,
                            "config_name": config_name,
                            "num_columns": num_columns,
                            "table_index": table_index,
                            "contingency_table": ",".join(contingency_table),
                            "table_signature": f"{input_path.name}:{','.join(contingency_table)}",
                            "sample_index": sample_index,
                            "seed": seed,
                            "elapsed_seconds": elapsed_seconds,
                        }
                    )
                    run_summaries.append(run_summary)

                    stats.run_count += 1
                    stats.update(
                        noise_df,
                        source_only_count=run_summary["num_source_only_groups"],
                        synth_only_count=run_summary["num_synth_only_groups"],
                        both_count=run_summary["num_shared_groups"],
                    )
        print_progress_summary(
            sample_index=sample_index,
            total_samples=args.synthetic_datasets,
            combo_stats=combo_stats,
            run_summaries=run_summaries,
        )

    group_summary_df, run_summary_df = build_summary_frames(combo_stats, run_summaries)
    print("", flush=True)
    print_group_level_summary(group_summary_df)
    print_run_level_summary(run_summary_df)


if __name__ == "__main__":
    main()
