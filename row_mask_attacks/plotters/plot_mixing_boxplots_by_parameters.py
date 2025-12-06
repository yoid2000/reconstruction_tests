import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_mixing_boxplots_by_parameters(df: pd.DataFrame, output_dir: Path):
    """Create 4 subplots with boxplots of mixing_avg grouped by each parameter.
    
    Args:
        df: DataFrame with complete jobs (measure >= target_accuracy)
        output_dir: Directory to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Boxplot by noise
    ax = axes[0, 0]
    noise_sorted = sorted(df['noise'].unique())
    df_sorted = df.copy()
    df_sorted['noise'] = pd.Categorical(df_sorted['noise'], categories=noise_sorted, ordered=True)
    df_sorted.boxplot(column='mixing_avg', by='noise', ax=ax)
    ax.set_title('Mixing Average by Noise')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Mixing Average')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')  # Remove default title
    
    # Subplot 2: Boxplot by nunique
    ax = axes[0, 1]
    nunique_sorted = sorted(df['nunique'].unique())
    df_sorted = df.copy()
    df_sorted['nunique'] = pd.Categorical(df_sorted['nunique'], categories=nunique_sorted, ordered=True)
    df_sorted.boxplot(column='mixing_avg', by='nunique', ax=ax)
    ax.set_title('Mixing Average by Number of Unique Values')
    ax.set_xlabel('Number of Unique Values')
    ax.set_ylabel('Mixing Average')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    # Subplot 3: Boxplot by nrows
    ax = axes[1, 0]
    nrows_sorted = sorted(df['nrows'].unique())
    df_sorted = df.copy()
    df_sorted['nrows'] = pd.Categorical(df_sorted['nrows'], categories=nrows_sorted, ordered=True)
    df_sorted.boxplot(column='mixing_avg', by='nrows', ax=ax)
    ax.set_title('Mixing Average by Number of Rows')
    ax.set_xlabel('Number of Rows')
    ax.set_ylabel('Mixing Average')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    # Subplot 4: Boxplot by mask_size
    ax = axes[1, 1]
    mask_size_sorted = sorted(df['mask_size'].unique())
    df_sorted = df.copy()
    df_sorted['mask_size'] = pd.Categorical(df_sorted['mask_size'], categories=mask_size_sorted, ordered=True)
    df_sorted.boxplot(column='mixing_avg', by='mask_size', ax=ax)
    ax.set_title('Mixing Average by Mask Size')
    ax.set_xlabel('Mask Size')
    ax.set_ylabel('Mixing Average')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mixing_avg_boxplots_by_parameters.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'mixing_avg_boxplots_by_parameters.png'}")
