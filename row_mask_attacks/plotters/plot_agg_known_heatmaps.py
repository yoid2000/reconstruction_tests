from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def plot_agg_known_heatmaps(df: pd.DataFrame):
    required_cols = {'measure', 'noise', 'supp_thresh', 'known_qi_fraction', 'nqi'}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"plot_agg_known_heatmaps: missing columns {missing}; skipping plot")
        return

    output_dir = Path('./results/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_vals = [1, 2, 3]
    supp_vals = [1, 2, 3]
    kqf_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    nqi_vals = sorted(df['nqi'].dropna().unique())

    if len(nqi_vals) == 0:
        print("plot_agg_known_heatmaps: no nqi values found; skipping plot")
        return

    colors = ['#ffffff', '#c7e9c0', '#ffe066', '#fdae6b', '#f7b6d2']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    nrows = len(nqi_vals)
    ncols = len(kqf_vals)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.0 * ncols, 2.6 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    for row_idx, nqi in enumerate(nqi_vals):
        for col_idx, kqf in enumerate(kqf_vals):
            ax = axes[row_idx, col_idx]
            category = np.zeros((len(noise_vals), len(supp_vals)), dtype=int)
            labels = np.full((len(noise_vals), len(supp_vals)), '-', dtype=object)

            for i, noise in enumerate(noise_vals):
                for j, supp in enumerate(supp_vals):
                    subset = df[
                        (df['nqi'] == nqi)
                        & (df['known_qi_fraction'] == kqf)
                        & (df['noise'] == noise)
                        & (df['supp_thresh'] == supp)
                    ]
                    if subset.empty:
                        continue

                    measure = subset['measure'].mean()
                    labels[i, j] = f"{measure:.2f}"
                    if measure >= 0.99:
                        category[i, j] = 4
                    elif measure >= 0.9:
                        category[i, j] = 3
                    elif measure >= 0.75:
                        category[i, j] = 2
                    else:
                        category[i, j] = 1

            ax.imshow(category, cmap=cmap, norm=norm, origin='upper')
            ax.set_xticks(range(len(supp_vals)))
            ax.set_yticks(range(len(noise_vals)))

            if row_idx == nrows - 1:
                ax.set_xticklabels(supp_vals)
                ax.set_xlabel('supp_thresh')
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_yticklabels(noise_vals)
                ax.set_ylabel(f"nqi={nqi}\nnoise")
            else:
                ax.set_yticklabels([])

            if row_idx == 0:
                ax.set_title(f"kqf={kqf}")

            for i in range(len(noise_vals)):
                for j in range(len(supp_vals)):
                    ax.text(j, i, labels[i, j], ha='center', va='center', fontsize=10)

            ax.set_xlim(-0.5, len(supp_vals) - 0.5)
            ax.set_ylim(len(noise_vals) - 0.5, -0.5)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        out_path = output_dir / f'agg_known_heatmaps.{ext}'
        plt.savefig(out_path, dpi=300 if ext == 'png' else None)
    plt.close()
    print(f"Saved: {output_dir / 'agg_known_heatmaps.png'}")
