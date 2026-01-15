from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plotters.plot_measure_and_bounds import plot_measure_and_bounds
from plotters.plot_sim_theory_accuracy import plot_sim_theory_accuracy
from reconstruction_measure import HeteroDAnalysis


# Approach:
# - Use expected_accuracy_and_ci() to compute a theoretical expected accuracy and CI
#   under a Binomial(R, p) proxy model.
# - Compare theory vs simulation in three ways:
#     1) expected theoretical accuracy is inside the simulated CI (95% and 99%).
#     2) theoretical CI overlaps simulated CI (95% and 99%).
#     3) expected theoretical accuracy is within 5% (absolute) of simulated accuracy.
# - Store per-row diagnostics and emit summary counts.


def build_analysis_dict(row: pd.Series) -> dict:
    d_val = int(row["actual_vals_per_qi"])
    c_val = int(row["nqi"])
    return {
        "R": int(row["nrows"]),
        "C": c_val,
        "D_cols": [d_val] * c_val,
        "K": int(row["nunique"]),
        "T": int(row["min_num_rows"]),
        "e": int(row["noise"]),
        "class_probs": None,
        "method": row.get("method"),
        "pmf_mass": row.get("pmf_mass"),
        "d_eff": row.get("d_eff"),
        "d_min": row.get("d_min"),
        "d_true_eff": row.get("d_true_eff"),
        "d_true_min": row.get("d_true_min"),
        "I_eff": row.get("I_eff"),
        "I_min": row.get("I_min"),
        "I_true_eff": row.get("I_true_eff"),
        "I_true_min": row.get("I_true_min"),
        "var_eff": row.get("var_eff"),
        "var_true": row.get("var_true"),
    }


def evaluate_match(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "measure",
        "measure_ci_95_lower",
        "measure_ci_95_upper",
        "measure_ci_99_lower",
        "measure_ci_99_upper",
        "nrows",
        "nunique",
        "noise",
        "min_num_rows",
        "nqi",
        "actual_vals_per_qi",
        "I_true_min",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    for _, row in df.iterrows():
        analysis_dict = build_analysis_dict(row)
        analysis = HeteroDAnalysis(**analysis_dict)

        sim_measure = float(row["measure"])
        sim_ci_95 = (float(row["measure_ci_95_lower"]), float(row["measure_ci_95_upper"]))
        sim_ci_99 = (float(row["measure_ci_99_lower"]), float(row["measure_ci_99_upper"]))

        theory_95 = analysis.expected_accuracy_and_ci(ci_level=0.95, use="eff")
        theory_99 = analysis.expected_accuracy_and_ci(ci_level=0.99, use="eff")
        theory_expected = float(theory_95["expected_accuracy"])
        theory_ci_95 = (float(theory_95["accuracy_ci"][0]), float(theory_95["accuracy_ci"][1]))
        theory_ci_99 = (float(theory_99["accuracy_ci"][0]), float(theory_99["accuracy_ci"][1]))

        theory_expected_in_sim_ci_95 = sim_ci_95[0] <= theory_expected <= sim_ci_95[1]
        theory_expected_in_sim_ci_99 = sim_ci_99[0] <= theory_expected <= sim_ci_99[1]

        theory_sim_overlap_95 = max(sim_ci_95[0], theory_ci_95[0]) <= min(sim_ci_95[1], theory_ci_95[1])
        theory_sim_overlap_99 = max(sim_ci_99[0], theory_ci_99[0]) <= min(sim_ci_99[1], theory_ci_99[1])

        theory_expected_within_5pct = abs(theory_expected - sim_measure) <= 0.05

        rows.append(
            {
                "theory_expected_accuracy": theory_expected,
                "theory_ci_95_lower": theory_ci_95[0],
                "theory_ci_95_upper": theory_ci_95[1],
                "theory_ci_99_lower": theory_ci_99[0],
                "theory_ci_99_upper": theory_ci_99[1],
                "theory_expected_in_sim_ci_95": theory_expected_in_sim_ci_95,
                "theory_expected_in_sim_ci_99": theory_expected_in_sim_ci_99,
                "theory_sim_ci_overlap_95": theory_sim_overlap_95,
                "theory_sim_ci_overlap_99": theory_sim_overlap_99,
                "theory_expected_within_5pct": theory_expected_within_5pct,
            }
        )

    df_out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    df_out["theory_measure"] = df_out["theory_expected_accuracy"]
    return df_out


def summarize(df: pd.DataFrame) -> None:
    total = len(df)
    if total == 0:
        print("No rows to summarize.")
        return

    in_ci_95 = int(df["theory_expected_in_sim_ci_95"].sum())
    in_ci_99 = int(df["theory_expected_in_sim_ci_99"].sum())
    overlap_95 = int(df["theory_sim_ci_overlap_95"].sum())
    overlap_99 = int(df["theory_sim_ci_overlap_99"].sum())
    within_5 = int(df["theory_expected_within_5pct"].sum())

    print(f"Total rows: {total}")
    print(f"Theory expected accuracy within simulated 95% CI: {in_ci_95} / {total}")
    print(f"Theory expected accuracy within simulated 99% CI: {in_ci_99} / {total}")
    print(f"Theory CI overlaps simulated 95% CI: {overlap_95} / {total}")
    print(f"Theory CI overlaps simulated 99% CI: {overlap_99} / {total}")
    print(f"Theory expected accuracy within 5% of simulated: {within_5} / {total}")

    def print_measure_stats(flag_col: str, label: str) -> None:
        for flag_value, flag_name in [(True, "within"), (False, "outside")]:
            subset = df[df[flag_col] == flag_value]
            if subset.empty:
                mean_val = float("nan")
                std_val = float("nan")
            else:
                mean_val = float(subset["measure"].mean())
                std_val = float(subset["measure"].std())
            print(
                f"{label} ({flag_name}): mean={mean_val:.4f}, std={std_val:.4f}, n={len(subset)}"
            )

    print_measure_stats("theory_expected_in_sim_ci_95", "Theory expected in simulated 95% CI")
    print_measure_stats("theory_expected_in_sim_ci_99", "Theory expected in simulated 99% CI")
    print_measure_stats("theory_sim_ci_overlap_95", "Theory CI overlaps simulated 95% CI")
    print_measure_stats("theory_sim_ci_overlap_99", "Theory CI overlaps simulated 99% CI")
    print_measure_stats("theory_expected_within_5pct", "Theory expected within 5% of simulated")

    def report_threshold(threshold: float) -> None:
        subset = df[df["measure"] >= threshold]
        if subset.empty:
            print(f"Simulated >= {threshold:.2f}: no rows")
            return
        theory_ok = subset[subset["theory_expected_accuracy"] >= threshold]
        min_theory = float(subset["theory_expected_accuracy"].min())
        print(
            f"Simulated >= {threshold:.2f}: "
            f"theory >= {threshold:.2f} in {len(theory_ok)} / {len(subset)}, "
            f"min theory expected={min_theory:.4f}"
        )
        theory_subset = df[df["theory_expected_accuracy"] >= threshold]
        if theory_subset.empty:
            print(f"Theory >= {threshold:.2f}: no rows")
            return
        sim_ok = theory_subset[theory_subset["measure"] >= threshold]
        min_sim = float(theory_subset["measure"].min())
        print(
            f"Theory >= {threshold:.2f}: "
            f"simulated >= {threshold:.2f} in {len(sim_ok)} / {len(theory_subset)}, "
            f"min simulated={min_sim:.4f}"
        )

    report_threshold(0.99)
    report_threshold(0.90)


def main() -> None:
    input_path = Path("results/grouped_result.parquet")
    output_path = Path("results/theory_vs_simulated.parquet")
    if not input_path.exists():
        print(f"Missing grouped results: {input_path}")
        return

    df = pd.read_parquet(input_path)
    for col in ["measure_ci_95_upper", "measure_ci_99_upper"]:
        if col in df.columns:
            df[col] = df[col].clip(upper=1.0)
    for col in ["measure_ci_95_lower", "measure_ci_99_lower"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)
    plot_measure_and_bounds(df)
    df_out = evaluate_match(df)
    plot_sim_theory_accuracy(df_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    print(f"Saved annotated results to {output_path}")
    summarize(df_out)


if __name__ == "__main__":
    main()
