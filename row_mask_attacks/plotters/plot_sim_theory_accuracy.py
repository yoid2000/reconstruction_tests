from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_sim_theory_accuracy(df: pd.DataFrame) -> None:
    required = ["measure", "theory_measure"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"plot_sim_theory_accuracy: missing columns {missing}; skipping plot")
        return

    df_sorted = df.sort_values("measure").reset_index(drop=True)
    if df_sorted.empty:
        print("plot_sim_theory_accuracy: no rows to plot")
        return

    x = range(len(df_sorted))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, df_sorted["measure"], label="Simulated accuracy", color="tab:blue")
    ax.plot(x, df_sorted["theory_measure"], label="Theoretical accuracy", color="tab:orange")
    ax.set_xlabel("Sorted rows")
    ax.set_ylabel("Accuracy")
    ax.set_title("Simulated vs Theoretical Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_dir = Path("results/plots_theory")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sim_theory_accuracy.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")
