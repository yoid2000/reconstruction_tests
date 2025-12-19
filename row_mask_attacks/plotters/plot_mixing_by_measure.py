import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_mixing_by_measure(df: pd.DataFrame, output_dir: Path):
    """Scatterplot of mixing_avg vs measure for agg_row runs, colored by exit_reason."""
    required_cols = {'solve_type', 'mixing_avg', 'measure', 'exit_reason'}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"plot_mixing_by_measure: missing columns {missing}; skipping plot")
        return

    df_plot = df[df['solve_type'] == 'agg_row'].copy()
    if df_plot.empty:
        print("plot_mixing_by_measure: no rows with solve_type=='agg_row'; skipping plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    df_plot['exit_reason_plot'] = df_plot['exit_reason'].fillna('unknown').astype(str)
    reasons = sorted(df_plot['exit_reason_plot'].unique())

    plt.figure(figsize=(6, 4))
    cmap = plt.get_cmap('tab10')

    for idx, reason in enumerate(reasons):
        subset = df_plot[df_plot['exit_reason_plot'] == reason]
        plt.scatter(
            subset['measure'],
            subset['mixing_avg'],
            color=cmap(idx % cmap.N),
            alpha=0.3,
            s=2,
            label=reason,
        )

    plt.xlabel('Measure')
    plt.ylabel('Mixing Average')
    plt.title('Mixing Average vs Measure (solve_type = agg_row)')
    plt.grid(True, alpha=0.3)
    plt.legend(title='exit_reason', fontsize=8)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        out_path = output_dir / f'mixing_avg_vs_measure_agg_row.{ext}'
        plt.savefig(out_path, dpi=300 if ext == 'png' else None)
    plt.close()
    print(f"Saved: {output_dir / 'mixing_avg_vs_measure_agg_row.png'}")
