import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_boxplots_by_parameters_nqi(df: pd.DataFrame, output_dir: Path):
    """Create 4 subplots with boxplots of num_samples grouped by each parameter (using nqi instead of mask_size).
    
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
    df_sorted.boxplot(column='num_samples', by='noise', ax=ax)
    ax.set_title('Number of Samples by Noise')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Number of Samples')
    ax.get_figure().suptitle('')
    
    # Subplot 2: Boxplot by nunique
    ax = axes[0, 1]
    nunique_sorted = sorted(df['nunique'].unique())
    df_sorted = df.copy()
    df_sorted['nunique'] = pd.Categorical(df_sorted['nunique'], categories=nunique_sorted, ordered=True)
    df_sorted.boxplot(column='num_samples', by='nunique', ax=ax)
    ax.set_title('Number of Samples by Number of Unique Values')
    ax.set_xlabel('Number of Unique Values')
    ax.set_ylabel('Number of Samples')
    ax.get_figure().suptitle('')
    
    # Subplot 3: Boxplot by nrows
    ax = axes[1, 0]
    nrows_sorted = sorted(df['nrows'].unique())
    df_sorted = df.copy()
    df_sorted['nrows'] = pd.Categorical(df_sorted['nrows'], categories=nrows_sorted, ordered=True)
    df_sorted.boxplot(column='num_samples', by='nrows', ax=ax)
    ax.set_title('Number of Samples by Number of Rows')
    ax.set_xlabel('Number of Rows')
    ax.set_ylabel('Number of Samples')
    ax.get_figure().suptitle('')
    
    # Subplot 4: Boxplot by nqi
    ax = axes[1, 1]
    nqi_sorted = sorted(df['nqi'].unique())
    df_sorted = df.copy()
    df_sorted['nqi'] = pd.Categorical(df_sorted['nqi'], categories=nqi_sorted, ordered=True)
    df_sorted.boxplot(column='num_samples', by='nqi', ax=ax)
    ax.set_title('Number of Samples by NQI')
    ax.set_xlabel('NQI')
    ax.set_ylabel('Number of Samples')
    ax.get_figure().suptitle('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'num_samples_boxplots_by_parameters_nqi.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'num_samples_boxplots_by_parameters_nqi.png'}")
