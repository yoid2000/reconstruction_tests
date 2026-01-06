import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def plot_mixing_by_param(df: pd.DataFrame, param_col: str, tag: str = ""):
    """Scatterplot of mixing_avg vs noise with categorical styling.

    - Drops rows where target_accuracy is not reached
    - Point shape is determined by nqi.
    - Point color is determined by param_col (categorical, using tab10 palette).
    """
    required_cols = {'mixing_avg', 'noise', 'nqi', 'exit_reason', 'final_attack', param_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"plot_mixing_by_param: missing columns {missing}; skipping plot")
        return None, None

    df_plot = df.copy()
    # keep rows where final_attack is True and exit_reason is 'target_accuracy'
    df_plot = df_plot[(df_plot['final_attack'] == True) & (df_plot['exit_reason'] == 'target_accuracy')]
    df_plot = df_plot.dropna(subset=['mixing_avg', 'noise', 'nqi', param_col])

    if df_plot.empty:
        print("plot_mixing_by_param: no rows left after filtering; skipping plot")
        return None, None

    # Marker shapes keyed by nqi
    marker_cycle = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    nqi_values = list(pd.unique(df_plot['nqi']))
    marker_map = {val: marker_cycle[idx % len(marker_cycle)] for idx, val in enumerate(nqi_values)}

    # Colors keyed by param_col, treated categorically
    palette = plt.get_cmap('tab10')
    param_values = list(pd.unique(df_plot[param_col]))
    color_map = {val: palette(idx % palette.N) for idx, val in enumerate(param_values)}

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False

    for param_value in param_values:
        subset_param = df_plot[df_plot[param_col] == param_value]
        if subset_param.empty:
            continue

        color = color_map[param_value]
        for nqi in pd.unique(subset_param['nqi']):
            subset = subset_param[subset_param['nqi'] == nqi]
            if subset.empty:
                continue

            ax.scatter(
                subset['noise'],
                subset['mixing_avg'],
                color=[color],
                marker=marker_map[nqi],
                s=50,
                alpha=0.7,
            )
            plotted_any = True

    if not plotted_any:
        print("plot_mixing_by_param: no points to plot; skipping plot")
        plt.close(fig)
        return None, None

    # make y axis log
    ax.set_yscale('log')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Mixing Average')
    ax.set_title(f'Mixing Average vs Noise by {param_col}')
    ax.grid(True, alpha=0.3)

    # Separate legends for color (param_col) and marker (nqi)
    color_handles = [
        Line2D([], [], color=color_map[val], marker='o', linestyle='', markersize=8, label=str(val))
        for val in param_values
    ]
    marker_handles = [
        Line2D([], [], color='gray', marker=marker_map[nqi], linestyle='', markersize=8, label=f"nqi={nqi}")
        for nqi in nqi_values
    ]

    color_legend = ax.legend(handles=color_handles, title=param_col, fontsize=9, framealpha=0.9, loc='upper left')
    ax.add_artist(color_legend)
    ax.legend(handles=marker_handles, title='nqi', fontsize=9, framealpha=0.9, loc='upper right')

    fig.tight_layout()
    output_dir = Path('./results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    for plottype in ['png']:
        filename = f'mixing_by_noise_nqi_{param_col}_{tag}.{plottype}'
        filepath = output_dir / filename
        plt.savefig(filepath)
        print(f"Saved: {filepath}")
    plt.close()
