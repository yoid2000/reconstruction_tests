from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize


def plot_scatter_acc_cov(df: pd.DataFrame):
    """Create three ALC scatter plots for real datasets."""
    required_cols = {
        'path_to_dataset',
        'alc_attack_precision',
        'alc_baseline_precision',
        'alc_attack_recall',
        'alc_baseline_recall',
        'alc_alc',
    }
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"plot_scatter_acc_cov: missing columns {missing}; skipping plot")
        return

    df_plot = df[df['path_to_dataset'] != ''].copy()
    if df_plot.empty:
        print("plot_scatter_acc_cov: no rows with non-empty path_to_dataset; skipping plot")
        return

    df_plot['x_delta_recall'] = df_plot['alc_attack_recall'] - df_plot['alc_baseline_recall']
    df_plot['y_delta_precision'] = df_plot['alc_attack_precision'] - df_plot['alc_baseline_precision']
    df_plot = df_plot.dropna(
        subset=[
            'alc_attack_precision',
            'alc_baseline_precision',
            'alc_attack_recall',
            'alc_baseline_recall',
            'x_delta_recall',
            'y_delta_precision',
            'alc_alc',
        ]
    )

    if df_plot.empty:
        print("plot_scatter_acc_cov: no rows left after dropping NaNs; skipping plot")
        return

    output_dir = Path('./results/plots_real')
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'scatter_acc_cov.png'

    alc_min = float(df_plot['alc_alc'].min())
    alc_max = float(df_plot['alc_alc'].max())
    if alc_min == alc_max:
        alc_max = alc_min + 1e-9
    norm = Normalize(vmin=alc_min, vmax=alc_max)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    scatter_specs = [
        (
            'alc_baseline_recall',
            'alc_attack_precision',
            'Baseline Recall',
            'Attack Precision',
            'Attack Precision vs Baseline Recall',
        ),
        (
            'alc_attack_recall',
            'alc_baseline_precision',
            'Attack Recall',
            'Baseline Precision',
            'Baseline Precision vs Attack Recall',
        ),
        (
            'x_delta_recall',
            'y_delta_precision',
            'Attack Recall - Baseline Recall',
            'Attack Precision - Baseline Precision',
            'Precision/Recall Delta',
        ),
    ]

    for idx, (x_col, y_col, x_label, y_label, title) in enumerate(scatter_specs):
        ax = axes[idx]
        sc = ax.scatter(
            df_plot[x_col],
            df_plot[y_col],
            c=df_plot['alc_alc'],
            cmap='viridis',
            norm=norm,
            s=20,
            alpha=0.85,
        )
        if idx == 2:
            ax.axhline(0.0, color='gray', linewidth=1, alpha=0.5)
            ax.axvline(0.0, color='gray', linewidth=1, alpha=0.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('alc_alc')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")
