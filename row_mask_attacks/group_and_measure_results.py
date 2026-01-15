from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reconstruction_measure import run_heteroD_analysis
from expected_samples import estimate_samples_needed


GROUP_COLS = [
    "solve_type",
    "nrows",
    "mask_size",
    "nunique",
    "noise",
    "nqi",
    "vals_per_qi",
    "max_samples",
    "target_accuracy",
    "min_num_rows",
    "known_qi_fraction",
]


def build_grouped(df_result: pd.DataFrame) -> pd.DataFrame:
    df = df_result.copy()

    missing_group_cols = [col for col in GROUP_COLS if col not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Missing grouping columns: {missing_group_cols}")

    if "final_attack" in df.columns:
        df = df[df["final_attack"] == True]
    else:
        print("Warning: final_attack column missing; skipping final_attack filter.")

    if "solve_type" in df.columns:
        df = df[df["solve_type"] == "agg_row"]
    else:
        print("Warning: solve_type column missing; skipping solve_type filter.")

    if df.empty:
        return df

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg_map: dict[str, str] = {}
    for col in df.columns:
        if col in GROUP_COLS:
            continue
        if col in numeric_cols:
            agg_map[col] = "mean"
        else:
            agg_map[col] = "first"

    df_grouped = df.groupby(GROUP_COLS, dropna=False).agg(agg_map).reset_index()

    if "measure" in df.columns:
        measure_stats = (
            df.groupby(GROUP_COLS, dropna=False)["measure"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
        )
        se = measure_stats["std"].fillna(0.0) / np.sqrt(
            measure_stats["count"].replace(0, np.nan)
        )
        se = se.fillna(0.0)
        z95 = 1.96
        z99 = 2.576
        measure_stats["measure_ci_95_lower"] = measure_stats["mean"] - z95 * se
        measure_stats["measure_ci_95_upper"] = measure_stats["mean"] + z95 * se
        measure_stats["measure_ci_99_lower"] = measure_stats["mean"] - z99 * se
        measure_stats["measure_ci_99_upper"] = measure_stats["mean"] + z99 * se
        df_grouped = df_grouped.merge(
            measure_stats[
                GROUP_COLS
                + [
                    "measure_ci_95_lower",
                    "measure_ci_95_upper",
                    "measure_ci_99_lower",
                    "measure_ci_99_upper",
                ]
            ],
            on=GROUP_COLS,
            how="left",
        )
    else:
        print("Warning: measure column missing; skipping confidence intervals.")

    return df_grouped


def run_analysis_for_grouped(df_grouped: pd.DataFrame) -> pd.DataFrame:
    if df_grouped.empty:
        return df_grouped

    required_cols = ["nrows", "nunique", "noise", "min_num_rows", "nqi", "actual_vals_per_qi"]
    missing = [col for col in required_cols if col not in df_grouped.columns]
    if missing:
        raise ValueError(f"Missing required columns for analysis: {missing}")

    heteroD_analysis_cols = [
        "method",
        "pmf_mass",
        "d_eff",
        "d_min",
        "d_true_eff",
        "d_true_min",
        "I_eff",
        "I_min",
        "I_true_eff",
        "I_true_min",
        "var_eff",
        "var_true",
        "measure_elapsed_time",
    ]

    estimated_samples_cols = [
        "m_bucket",
        "p_suppress_per_class",
        "expected_samples_per_query",
        "L_appearances_per_row",
        "Q_queries",
        "N_samples",
    ]

    analysis_rows = []
    for idx, row in df_grouped.iterrows():
        D_val = int(row["actual_vals_per_qi"])
        D_cols = [D_val] * int(row["nqi"])
        analysis = run_heteroD_analysis(
            R=int(row["nrows"]),
            D_cols=D_cols,
            K=int(row["nunique"]),
            T=int(row["min_num_rows"]),
            e=int(row["noise"]),
            class_probs=None,
            method="auto",
            subset_samples=30000,
            seed=0,
        )
        row_out = {col: analysis.get(col) for col in heteroD_analysis_cols}
        est = estimate_samples_needed(
            R=int(row["nrows"]),
            K=int(row["nunique"]),
            T=int(row["min_num_rows"]),
            e=int(row["noise"]),
        )
        if isinstance(est, dict):
            row_out.update({col: est.get(col) for col in estimated_samples_cols})
        else:
            row_out.update({col: getattr(est, col, None) for col in estimated_samples_cols})
        analysis_rows.append(row_out)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} / {len(df_grouped)} grouped rows")

    analysis_df = pd.DataFrame(analysis_rows)
    return pd.concat([df_grouped.reset_index(drop=True), analysis_df], axis=1)


def main() -> None:
    results_path = Path("results/result.parquet")
    output_path = Path("results/grouped_result.parquet")

    if not results_path.exists():
        print(f"Missing results file: {results_path}")
        return

    df_result = pd.read_parquet(results_path)
    df_grouped = build_grouped(df_result)
    df_grouped = run_analysis_for_grouped(df_grouped)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_grouped.to_parquet(output_path, index=False)
    print(f"Saved grouped results to {output_path}")


if __name__ == "__main__":
    main()
