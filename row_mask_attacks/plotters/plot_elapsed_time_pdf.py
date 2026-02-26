from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_elapsed_time_pdf(df: pd.DataFrame):
    '''
    Create a line plot where the y axis is solver_metrics_runtime, and the x axis is simply
    the index of the sorted runtimes.

    However, limit the datapoints to those rows where measure > = 0.9.

    The y axis should be on a log scale.

    The plot should be saved to elapsed_time_pdf.png and elapsed_time_pdf.pdf in the results/plots/ directory.
    '''
    output_dir = Path('./results/plots')
    required_cols = {'solver_metrics_runtime', 'measure', 'final_attack'}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"plot_elapsed_time_pdf: missing columns {missing}; skipping plot")
        return

    df_plot = df[df['final_attack'] == True].copy()
    string = '(final attacks only)'
    #df_plot = df
    #string = ''

    runtimes_high = df_plot[df_plot['measure'] >= 0.99]['solver_metrics_runtime'].dropna().sort_values().reset_index(drop=True)
    runtimes_mid = df_plot[(df_plot['measure'] >= 0.9) & (df_plot['measure'] < 0.99)]['solver_metrics_runtime'].dropna().sort_values().reset_index(drop=True)
    runtimes_low = df_plot[df_plot['measure'] < 0.9]['solver_metrics_runtime'].dropna().sort_values().reset_index(drop=True)

    if runtimes_high.empty and runtimes_mid.empty and runtimes_low.empty:
        print("plot_elapsed_time_pdf: no runtimes to plot after filtering; skipping plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4, 3))

    def normalized_x(n: int) -> list:
        if n <= 1:
            return [0.0] * n
        return [i / (n - 1) for i in range(n)]

    colors = {
        'high': '#d62728',  # red
        'mid': '#2ca02c',   # green
        'low': '#f1c40f',   # yellow
    }

    if not runtimes_high.empty:
        x_high = normalized_x(len(runtimes_high))
        y_high = runtimes_high.values
        (line_high,) = plt.plot(
            x_high,
            y_high,
            linewidth=1,
            label='accuracy >= 0.99',
            color=colors['high'],
        )
        plt.scatter(
            [x_high[0], x_high[-1]],
            [y_high[0], y_high[-1]],
            s=12,
            color=line_high.get_color(),
            zorder=3,
        )
    if not runtimes_mid.empty:
        x_mid = normalized_x(len(runtimes_mid))
        y_mid = runtimes_mid.values
        (line_mid,) = plt.plot(
            x_mid,
            y_mid,
            linewidth=1,
            label='accuracy >= 0.9 and < 0.99',
            color=colors['mid'],
        )
        plt.scatter(
            [x_mid[0], x_mid[-1]],
            [y_mid[0], y_mid[-1]],
            s=12,
            color=line_mid.get_color(),
            zorder=3,
        )
    if not runtimes_low.empty:
        x_low = normalized_x(len(runtimes_low))
        y_low = runtimes_low.values
        (line_low,) = plt.plot(
            x_low,
            y_low,
            linewidth=1,
            label='accuracy < 0.9',
            color=colors['low'],
        )
        plt.scatter(
            [x_low[0], x_low[-1]],
            [y_low[0], y_low[-1]],
            s=12,
            color=line_low.get_color(),
            zorder=3,
        )

    seconds_per_week = 7 * 24 * 60 * 60
    seconds_per_day = 24 * 60 * 60
    plt.axhline(seconds_per_week, linestyle='--', color='gray', linewidth=1)
    plt.axhline(seconds_per_day, linestyle='--', color='gray', linewidth=1)

    plt.xlabel(f'Sorted runtime {string}')
    plt.ylabel('Solver runtime (seconds)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.xticks([])
    plt.legend(fontsize=8)
    # Annotate threshold lines without adding them to the legend
    x_text = 0.98
    plt.text(x_text, seconds_per_week, '1 week', ha='right', va='bottom', fontsize=8, color='gray')
    plt.text(x_text, seconds_per_day, '1 day', ha='right', va='bottom', fontsize=8, color='gray')
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        out_path = output_dir / f'elapsed_time_pdf.{ext}'
        plt.savefig(out_path, dpi=300 if ext == 'png' else None)
    plt.close()
    print(f"Saved: {output_dir / 'elapsed_time_pdf.png'}")
