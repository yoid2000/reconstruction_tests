from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


X_COLUMNS = [
    "solver_metrics_simplex_iterations",
    "solver_metrics_num_vars",
    "solver_metrics_num_constrs",
    "solver_metrics_runtime",
]
Y_COLUMN = "alc_alc"
OUTPUT_FILENAME = "alc_by_solver_metrics.png"


def plot_solver_metric_scatterplots(df: pd.DataFrame, output_dir: Path) -> Path:
    required_columns = [Y_COLUMN, *X_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"dataframe is missing required plot columns: {missing_columns}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes_flat = axes.ravel()

    y_values = pd.to_numeric(df[Y_COLUMN], errors="coerce")
    for axis, x_column in zip(axes_flat, X_COLUMNS):
        x_values = pd.to_numeric(df[x_column], errors="coerce")
        plot_df = pd.DataFrame({"x": x_values, "y": y_values}).dropna()

        axis.scatter(plot_df["x"], plot_df["y"], alpha=0.75)
        axis.set_xlabel(x_column)
        axis.set_ylabel(Y_COLUMN)
        axis.set_title(f"{Y_COLUMN} vs {x_column}")
        axis.grid(True, alpha=0.25)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
