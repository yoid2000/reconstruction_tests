import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_measure_by_nqi(
    df: pd.DataFrame,
    output_dir: Path,
    target_accuracy: float = 0.99,
    measure_col: str = 'measure',
    metric_label: str = 'Measure (Accuracy)',
    output_stem: str = 'measure_by_nqi',
):
    """Create scatterplot of metric values by nqi with target percentages."""
    plt.figure(figsize=(10, 6))

    nqi_values = sorted(df['nqi'].unique())
    scatter = plt.scatter(
        df['nqi'],
        df[measure_col],
        c=df['noise'],
        cmap='viridis',
        alpha=0.6,
        s=50,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label('Noise')

    for nqi in nqi_values:
        df_nqi = df[df['nqi'] == nqi]
        total = len(df_nqi)
        if total > 0:
            achieved = len(df_nqi[df_nqi[measure_col] >= target_accuracy])
            percentage = (achieved / total) * 100
            plt.text(
                nqi,
                1.05,
                f'{percentage:.1f}%',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
            )

    plt.xlabel('NQI')
    plt.ylabel(metric_label)
    plt.ylim(0, 1.15)
    plt.title(
        f'{metric_label} by NQI (colored by Noise)\n'
        f'(percentages show % achieving >= {target_accuracy})'
    )
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / f'{output_stem}.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / f'{output_stem}.png'}")
