from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_measure_and_bounds(df: pd.DataFrame) -> None:
    required = [
        "measure",
        "measure_ci_95_lower",
        "measure_ci_95_upper",
        "measure_ci_99_lower",
        "measure_ci_99_upper",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"plot_measure_and_bounds: missing columns {missing}; skipping plot")
        return

    df_sorted = df.sort_values("measure").reset_index(drop=True)
    if df_sorted.empty:
        print("plot_measure_and_bounds: no rows to plot")
        return

    x = range(len(df_sorted))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax_left.plot(x, df_sorted["measure"], label="measure", color="tab:blue")
    ax_left.plot(x, df_sorted["measure_ci_95_lower"], label="ci_95_lower", color="tab:orange")
    ax_left.plot(x, df_sorted["measure_ci_95_upper"], label="ci_95_upper", color="tab:green")
    ax_left.set_title("Measure with 95% CI")
    ax_left.set_xlabel("Sorted rows")
    ax_left.set_ylabel("Measure")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(fontsize=8)

    ax_right.plot(x, df_sorted["measure"], label="measure", color="tab:blue")
    ax_right.plot(x, df_sorted["measure_ci_99_lower"], label="ci_99_lower", color="tab:orange")
    ax_right.plot(x, df_sorted["measure_ci_99_upper"], label="ci_99_upper", color="tab:green")
    ax_right.set_title("Measure with 99% CI")
    ax_right.set_xlabel("Sorted rows")
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(fontsize=8)

    fig.tight_layout()

    output_dir = Path("results/plots_theory")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "measure_and_bounds.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")
