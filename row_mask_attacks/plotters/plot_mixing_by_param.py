import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_mixing_by_param(df: pd.DataFrame, param_col: str, tag: str = ""):
    """Line plots of mixing_avg vs noise, split by nunique and nqi.

    - Drops rows where target_accuracy is not reached
    - Left subplot: one line per nunique
    - Right subplot: one line per nqi
    """
    required_cols = {'mixing_avg', 'noise', 'nqi', 'nunique', 'exit_reason', 'final_attack'}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"plot_mixing_by_param: missing columns {missing}; skipping plot")
        return None, None

    df_plot = df.copy()
    # keep rows where final_attack is True and exit_reason is 'target_accuracy'
    df_plot = df_plot[(df_plot['final_attack'] == True) & (df_plot['exit_reason'] == 'target_accuracy')]
    df_plot = df_plot.dropna(subset=['mixing_avg', 'noise', 'nqi', 'nunique'])

    if df_plot.empty:
        print("plot_mixing_by_param: no rows left after filtering; skipping plot")
        return None, None

    grouped_nunique = df_plot.groupby(['nunique', 'noise'], as_index=False)['mixing_avg'].mean()
    grouped_nqi = df_plot.groupby(['nqi', 'noise'], as_index=False)['mixing_avg'].mean()

    if grouped_nunique.empty or grouped_nqi.empty:
        print("plot_mixing_by_param: no grouped rows to plot; skipping plot")
        return None, None

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    palette = plt.get_cmap('tab10')

    nunique_values = sorted(pd.unique(grouped_nunique['nunique']))
    for idx, val in enumerate(nunique_values):
        subset = grouped_nunique[grouped_nunique['nunique'] == val].sort_values('noise')
        ax_left.plot(
            subset['noise'],
            subset['mixing_avg'],
            marker='o',
            linewidth=1.5,
            color=palette(idx % palette.N),
            label=f"nunique={val}",
        )

    nqi_values = sorted(pd.unique(grouped_nqi['nqi']))
    for idx, val in enumerate(nqi_values):
        subset = grouped_nqi[grouped_nqi['nqi'] == val].sort_values('noise')
        ax_right.plot(
            subset['noise'],
            subset['mixing_avg'],
            marker='o',
            linewidth=1.5,
            color=palette(idx % palette.N),
            label=f"nqi={val}",
        )

    for ax in (ax_left, ax_right):
        ax.set_yscale('log')
        ax.set_xlabel('Noise')
        ax.grid(True, alpha=0.3)

    ax_left.set_ylabel('Mixing Average')
    ax_left.set_title('Mixing Average vs Noise by nunique')
    ax_right.set_title('Mixing Average vs Noise by nqi')
    ax_left.legend(title='nunique', fontsize=9, framealpha=0.9, loc='best')
    ax_right.legend(title='nqi', fontsize=9, framealpha=0.9, loc='best')

    title_parts = [part for part in [param_col, tag] if part]
    if title_parts:
        fig.suptitle(f"Mixing Average vs Noise ({', '.join(title_parts)})")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    output_dir = Path('./results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    param_part = f"_{param_col}" if param_col else ""
    tag_part = f"_{tag}" if tag else ""
    for plottype in ['png']:
        filename = f'mixing_by_noise_nunique_nqi{param_part}{tag_part}.{plottype}'
        filepath = output_dir / filename
        plt.savefig(filepath)
        print(f"Saved: {filepath}")
    plt.close()
