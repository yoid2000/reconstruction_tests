from pathlib import Path
import matplotlib.pyplot as plt


PLOTS_DIR = Path('./results/plots')
OUTPUT_FILE = PLOTS_DIR / 'x_nqi_mnr3_grouped.png'
FILTER_TOKEN = '___mnr3'
Y_TOKENS = [
    'y_mixing_avg',
    'y_separation_average',
    'y_mix_times_sep',
]


def collect_plots() -> dict:
    groups = {}
    if not PLOTS_DIR.exists():
        print(f"Missing plots directory: {PLOTS_DIR}")
        return groups

    for path in PLOTS_DIR.glob('x_nqi_*.png'):
        name = path.name
        if not name.startswith('x_nqi_'):
            continue
        if FILTER_TOKEN not in name:
            continue
        matched_token = None
        for token in Y_TOKENS:
            if token in name:
                matched_token = token
                break
        if matched_token is None:
            continue
        suffix_idx = name.find('_l_')
        if suffix_idx == -1:
            continue
        suffix = name[suffix_idx:]
        groups.setdefault(suffix, {})
        if matched_token in groups[suffix]:
            print(f"Duplicate plot for {suffix} {matched_token}: {path}")
            continue
        groups[suffix][matched_token] = path
    return groups


def plot_grouped(groups: dict) -> None:
    if not groups:
        print("No matching plots found")
        return

    suffixes = sorted(groups.keys())
    nrows = len(suffixes)
    ncols = len(Y_TOKENS)

    fig_width = 4.5 * ncols
    fig_height = max(3.0, 3.0 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    if nrows == 1:
        axes = [axes]

    for row_idx, suffix in enumerate(suffixes):
        for col_idx, token in enumerate(Y_TOKENS):
            ax = axes[row_idx][col_idx]
            ax.axis('off')
            path = groups[suffix].get(token)
            if path is None:
                ax.text(0.5, 0.5, 'missing', ha='center', va='center', fontsize=10)
                continue
            img = plt.imread(path)
            ax.imshow(img)
            if row_idx == 0:
                ax.set_title(token.replace('y_', ''), fontsize=12)
        axes[row_idx][0].set_ylabel(suffix, rotation=0, labelpad=70, fontsize=8, ha='right', va='center')

    plt.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=200)
    plt.close(fig)
    print(f"Saved grouped plot: {OUTPUT_FILE}")


def main() -> None:
    groups = collect_plots()
    plot_grouped(groups)


if __name__ == '__main__':
    main()
