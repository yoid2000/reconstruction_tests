import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from experiments import read_experiments

def plot_mixing_vs_samples(df: pd.DataFrame, output_dir: Path):
    """Create scatterplot of mixing_avg vs num_samples.
    
    Args:
        df: DataFrame with complete jobs (measure >= target_accuracy)
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot with colors based on noise
    scatter = plt.scatter(df['num_samples'], df['mixing_avg'], 
                         c=df['noise'], cmap='viridis', 
                         alpha=0.6, s=50)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Noise')
    
    plt.yscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Mixing Average')
    plt.title('Mixing Average vs Number of Samples (colored by Noise)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'mixing_avg_vs_num_samples.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'mixing_avg_vs_num_samples.png'}")

def plot_boxplots_by_parameters(df: pd.DataFrame, output_dir: Path):
    """Create 4 subplots with boxplots of num_samples grouped by each parameter.
    
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
    ax.get_figure().suptitle('')  # Remove default title
    
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
    
    # Subplot 4: Boxplot by mask_size
    ax = axes[1, 1]
    mask_size_sorted = sorted(df['mask_size'].unique())
    df_sorted = df.copy()
    df_sorted['mask_size'] = pd.Categorical(df_sorted['mask_size'], categories=mask_size_sorted, ordered=True)
    df_sorted.boxplot(column='num_samples', by='mask_size', ax=ax)
    ax.set_title('Number of Samples by Mask Size')
    ax.set_xlabel('Mask Size')
    ax.set_ylabel('Number of Samples')
    ax.get_figure().suptitle('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'num_samples_boxplots_by_parameters.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'num_samples_boxplots_by_parameters.png'}")

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

def plot_elapsed_boxplots_by_parameters(df: pd.DataFrame, output_dir: Path):
    """Create 4 subplots with boxplots of elapsed_time grouped by each parameter.
    
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
    df_sorted.boxplot(column='elapsed_time', by='noise', ax=ax)
    ax.set_title('Elapsed Time by Noise')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Elapsed Time (seconds)')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')  # Remove default title
    
    # Subplot 2: Boxplot by nunique
    ax = axes[0, 1]
    nunique_sorted = sorted(df['nunique'].unique())
    df_sorted = df.copy()
    df_sorted['nunique'] = pd.Categorical(df_sorted['nunique'], categories=nunique_sorted, ordered=True)
    df_sorted.boxplot(column='elapsed_time', by='nunique', ax=ax)
    ax.set_title('Elapsed Time by Number of Unique Values')
    ax.set_xlabel('Number of Unique Values')
    ax.set_ylabel('Elapsed Time (seconds)')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    # Subplot 3: Boxplot by nrows
    ax = axes[1, 0]
    nrows_sorted = sorted(df['nrows'].unique())
    df_sorted = df.copy()
    df_sorted['nrows'] = pd.Categorical(df_sorted['nrows'], categories=nrows_sorted, ordered=True)
    df_sorted.boxplot(column='elapsed_time', by='nrows', ax=ax)
    ax.set_title('Elapsed Time by Number of Rows')
    ax.set_xlabel('Number of Rows')
    ax.set_ylabel('Elapsed Time (seconds)')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    # Subplot 4: Boxplot by mask_size
    ax = axes[1, 1]
    mask_size_sorted = sorted(df['mask_size'].unique())
    df_sorted = df.copy()
    df_sorted['mask_size'] = pd.Categorical(df_sorted['mask_size'], categories=mask_size_sorted, ordered=True)
    df_sorted.boxplot(column='elapsed_time', by='mask_size', ax=ax)
    ax.set_title('Elapsed Time by Mask Size')
    ax.set_xlabel('Mask Size')
    ax.set_ylabel('Elapsed Time (seconds)')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'elapsed_time_boxplots_by_parameters.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'elapsed_time_boxplots_by_parameters.png'}")

def plot_mixing_vs_noise_by_mask_size(df: pd.DataFrame, output_dir: Path):
    """Create line plot of mixing_avg vs noise, with lines for each mask_size.
    
    Args:
        df: DataFrame with complete jobs (measure >= target_accuracy)
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get unique mask sizes and sort them
    mask_sizes = sorted(df['mask_size'].unique())
    
    # For each mask size, compute average mixing_avg for each noise level
    for mf in mask_sizes:
        df_mf = df[df['mask_size'] == mf]
        
        # Group by noise and compute mean, min, max mixing_avg
        grouped = df_mf.groupby('noise')['mixing_avg'].agg(['mean', 'min', 'max']).sort_index()
        
        # Calculate error bars (distance from mean to min/max)
        yerr_lower = grouped['mean'] - grouped['min']
        yerr_upper = grouped['max'] - grouped['mean']
        yerr = [yerr_lower.values, yerr_upper.values]
        
        # Plot line with error bars
        plt.errorbar(grouped.index, grouped['mean'].values, yerr=yerr,
                    marker='o', label=f'mask_size={mf}', linewidth=2,
                    capsize=5, capthick=2)
    
    plt.xlabel('Noise')
    plt.ylabel('Mixing Average')
    plt.yscale('log')
    plt.title('Average Mixing vs Noise by Mask Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'mixing_vs_noise_by_mask_size.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'mixing_vs_noise_by_mask_size.png'}")

def plot_num_samples_vs_noise_by_mask_size(df: pd.DataFrame, output_dir: Path):
    """Create line plot of num_samples vs noise, with lines for each mask_size.
    
    Args:
        df: DataFrame with complete jobs (measure >= target_accuracy)
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get unique mask sizes and sort them
    mask_sizes = sorted(df['mask_size'].unique())
    
    # For each mask size, compute average num_samples for each noise level
    for mf in mask_sizes:
        df_mf = df[df['mask_size'] == mf]
        
        # Group by noise and compute mean, min, max num_samples
        grouped = df_mf.groupby('noise')['num_samples'].agg(['mean', 'min', 'max']).sort_index()
        
        # Calculate error bars (distance from mean to min/max)
        yerr_lower = grouped['mean'] - grouped['min']
        yerr_upper = grouped['max'] - grouped['mean']
        yerr = [yerr_lower.values, yerr_upper.values]
        
        # Plot line with error bars
        plt.errorbar(grouped.index, grouped['mean'].values, yerr=yerr,
                    marker='o', label=f'mask_size={mf}', linewidth=2,
                    capsize=5, capthick=2)
    
    plt.xlabel('Noise')
    plt.ylabel('Number of Samples')
    plt.yscale('log')
    plt.title('Average Number of Samples vs Noise by Mask Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'num_samples_vs_noise_by_mask_size.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'num_samples_vs_noise_by_mask_size.png'}")

def plot_elapsed_time_vs_noise_by_mask_size(df: pd.DataFrame, output_dir: Path):
    """Create line plot of elapsed_time vs noise, with lines for each mask_size.
    
    Args:
        df: DataFrame with complete jobs (measure >= target_accuracy)
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get unique mask sizes and sort them
    mask_sizes = sorted(df['mask_size'].unique())
    
    # For each mask size, compute average elapsed_time for each noise level
    for mf in mask_sizes:
        df_mf = df[df['mask_size'] == mf]
        
        # Group by noise and compute mean, min, max elapsed_time
        grouped = df_mf.groupby('noise')['elapsed_time'].agg(['mean', 'min', 'max']).sort_index()
        
        # Calculate error bars (distance from mean to min/max)
        yerr_lower = grouped['mean'] - grouped['min']
        yerr_upper = grouped['max'] - grouped['mean']
        yerr = [yerr_lower.values, yerr_upper.values]
        
        # Plot line with error bars
        plt.errorbar(grouped.index, grouped['mean'].values, yerr=yerr,
                    marker='o', label=f'mask_size={mf}', linewidth=2,
                    capsize=5, capthick=2)
    
    plt.xlabel('Noise')
    plt.ylabel('Elapsed Time (seconds)')
    plt.yscale('log')
    plt.title('Average Elapsed Time vs Noise by Mask Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'elapsed_time_vs_noise_by_mask_size.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'elapsed_time_vs_noise_by_mask_size.png'}")

def plot_mixing_vs_noise_by_nqi(df: pd.DataFrame, output_dir: Path):
    """Create line plot of mixing_avg vs noise, with lines for each nqi.
    
    Args:
        df: DataFrame with complete jobs (measure >= target_accuracy)
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get unique nqi values and sort them
    nqi_values = sorted(df['nqi'].unique())
    
    # For each nqi, compute average mixing_avg for each noise level
    for nqi in nqi_values:
        df_nqi = df[df['nqi'] == nqi]
        
        # Group by noise and compute mean, min, max mixing_avg
        grouped = df_nqi.groupby('noise')['mixing_avg'].agg(['mean', 'min', 'max']).sort_index()
        
        # Calculate error bars (distance from mean to min/max)
        yerr_lower = grouped['mean'] - grouped['min']
        yerr_upper = grouped['max'] - grouped['mean']
        yerr = [yerr_lower.values, yerr_upper.values]
        
        # Plot line with error bars
        plt.errorbar(grouped.index, grouped['mean'].values, yerr=yerr,
                    marker='o', label=f'nqi={nqi}', linewidth=2,
                    capsize=5, capthick=2)
    
    plt.xlabel('Noise')
    plt.ylabel('Mixing Average')
    plt.yscale('log')
    plt.title('Average Mixing vs Noise by NQI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'mixing_vs_noise_by_nqi.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'mixing_vs_noise_by_nqi.png'}")

def plot_num_samples_vs_noise_by_nqi(df: pd.DataFrame, output_dir: Path):
    """Create line plot of num_samples vs noise, with lines for each nqi.
    
    Args:
        df: DataFrame with complete jobs (measure >= target_accuracy)
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get unique nqi values and sort them
    nqi_values = sorted(df['nqi'].unique())
    
    # For each nqi, compute average num_samples for each noise level
    for nqi in nqi_values:
        df_nqi = df[df['nqi'] == nqi]
        
        # Group by noise and compute mean, min, max num_samples
        grouped = df_nqi.groupby('noise')['num_samples'].agg(['mean', 'min', 'max']).sort_index()
        
        # Calculate error bars (distance from mean to min/max)
        yerr_lower = grouped['mean'] - grouped['min']
        yerr_upper = grouped['max'] - grouped['mean']
        yerr = [yerr_lower.values, yerr_upper.values]
        
        # Plot line with error bars
        plt.errorbar(grouped.index, grouped['mean'].values, yerr=yerr,
                    marker='o', label=f'nqi={nqi}', linewidth=2,
                    capsize=5, capthick=2)
    
    plt.xlabel('Noise')
    plt.ylabel('Number of Samples')
    plt.yscale('log')
    plt.title('Average Number of Samples vs Noise by NQI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'num_samples_vs_noise_by_nqi.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'num_samples_vs_noise_by_nqi.png'}")

def plot_elapsed_time_vs_noise_by_nqi(df: pd.DataFrame, output_dir: Path):
    """Create line plot of elapsed_time vs noise, with lines for each nqi.
    
    Args:
        df: DataFrame with complete jobs (measure >= target_accuracy)
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get unique nqi values and sort them
    nqi_values = sorted(df['nqi'].unique())
    
    # For each nqi, compute average elapsed_time for each noise level
    for nqi in nqi_values:
        df_nqi = df[df['nqi'] == nqi]
        
        # Group by noise and compute mean, min, max elapsed_time
        grouped = df_nqi.groupby('noise')['elapsed_time'].agg(['mean', 'min', 'max']).sort_index()
        
        # Calculate error bars (distance from mean to min/max)
        yerr_lower = grouped['mean'] - grouped['min']
        yerr_upper = grouped['max'] - grouped['mean']
        yerr = [yerr_lower.values, yerr_upper.values]
        
        # Plot line with error bars
        plt.errorbar(grouped.index, grouped['mean'].values, yerr=yerr,
                    marker='o', label=f'nqi={nqi}', linewidth=2,
                    capsize=5, capthick=2)
    
    plt.xlabel('Noise')
    plt.ylabel('Elapsed Time (seconds)')
    plt.yscale('log')
    plt.title('Average Elapsed Time vs Noise by NQI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'elapsed_time_vs_noise_by_nqi.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'elapsed_time_vs_noise_by_nqi.png'}")

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

def plot_mixing_boxplots_by_parameters_nqi(df: pd.DataFrame, output_dir: Path):
    """Create 4 subplots with boxplots of mixing_avg grouped by each parameter (using nqi instead of mask_size).
    
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
    
    # Subplot 4: Boxplot by nqi
    ax = axes[1, 1]
    nqi_sorted = sorted(df['nqi'].unique())
    df_sorted = df.copy()
    df_sorted['nqi'] = pd.Categorical(df_sorted['nqi'], categories=nqi_sorted, ordered=True)
    df_sorted.boxplot(column='mixing_avg', by='nqi', ax=ax)
    ax.set_title('Mixing Average by NQI')
    ax.set_xlabel('NQI')
    ax.set_ylabel('Mixing Average')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mixing_avg_boxplots_by_parameters_nqi.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'mixing_avg_boxplots_by_parameters_nqi.png'}")

def plot_elapsed_boxplots_by_parameters_nqi(df: pd.DataFrame, output_dir: Path):
    """Create 4 subplots with boxplots of elapsed_time grouped by each parameter (using nqi instead of mask_size).
    
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
    df_sorted.boxplot(column='elapsed_time', by='noise', ax=ax)
    ax.set_title('Elapsed Time by Noise')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Elapsed Time (seconds)')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')  # Remove default title
    
    # Subplot 2: Boxplot by nunique
    ax = axes[0, 1]
    nunique_sorted = sorted(df['nunique'].unique())
    df_sorted = df.copy()
    df_sorted['nunique'] = pd.Categorical(df_sorted['nunique'], categories=nunique_sorted, ordered=True)
    df_sorted.boxplot(column='elapsed_time', by='nunique', ax=ax)
    ax.set_title('Elapsed Time by Number of Unique Values')
    ax.set_xlabel('Number of Unique Values')
    ax.set_ylabel('Elapsed Time (seconds)')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    # Subplot 3: Boxplot by nrows
    ax = axes[1, 0]
    nrows_sorted = sorted(df['nrows'].unique())
    df_sorted = df.copy()
    df_sorted['nrows'] = pd.Categorical(df_sorted['nrows'], categories=nrows_sorted, ordered=True)
    df_sorted.boxplot(column='elapsed_time', by='nrows', ax=ax)
    ax.set_title('Elapsed Time by Number of Rows')
    ax.set_xlabel('Number of Rows')
    ax.set_ylabel('Elapsed Time (seconds)')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    # Subplot 4: Boxplot by nqi
    ax = axes[1, 1]
    nqi_sorted = sorted(df['nqi'].unique())
    df_sorted = df.copy()
    df_sorted['nqi'] = pd.Categorical(df_sorted['nqi'], categories=nqi_sorted, ordered=True)
    df_sorted.boxplot(column='elapsed_time', by='nqi', ax=ax)
    ax.set_title('Elapsed Time by NQI')
    ax.set_xlabel('NQI')
    ax.set_ylabel('Elapsed Time (seconds)')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'elapsed_time_boxplots_by_parameters_nqi.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'elapsed_time_boxplots_by_parameters_nqi.png'}")

def plot_measure_by_nqi(df: pd.DataFrame, output_dir: Path, target_accuracy: float = 0.99):
    """Create scatterplot of measure by nqi with target accuracy percentages.
    
    Args:
        df: DataFrame with all jobs (complete and incomplete)
        output_dir: Directory to save plot
        target_accuracy: Target accuracy threshold (default: 0.99)
    """
    plt.figure(figsize=(10, 6))
    
    # Get unique nqi values and sort them
    nqi_values = sorted(df['nqi'].unique())
    
    # Create scatter plot with colors based on noise
    scatter = plt.scatter(df['nqi'], df['measure'], 
                         c=df['noise'], cmap='viridis', 
                         alpha=0.6, s=50)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Noise')
    
    # Calculate and annotate percentages for each nqi value
    for nqi in nqi_values:
        df_nqi = df[df['nqi'] == nqi]
        total = len(df_nqi)
        if total > 0:
            achieved = len(df_nqi[df_nqi['measure'] >= target_accuracy])
            percentage = (achieved / total) * 100
            
            # Position text above the highest point for this nqi
            y_pos = 1.05
            plt.text(nqi, y_pos, f'{percentage:.1f}%', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('NQI')
    plt.ylabel('Measure (Accuracy)')
    plt.ylim(0, 1.15)
    plt.title(f'Accuracy by NQI (colored by Noise)\n(percentages show % achieving ≥ {target_accuracy})')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'measure_by_nqi.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'measure_by_nqi.png'}")

def analyze_single_parameter_variation(df: pd.DataFrame, experiments: list, exp_group: str):
    """Analyze how results vary when only one parameter changes.
    
    Args:
        df: DataFrame for this experiment group
        experiments: List of all experiments
        exp_group: Name of the experiment group
    """
    # Find the experiment definition for this group
    exp_def = None
    for exp in experiments:
        if exp['experiment_group'] == exp_group:
            exp_def = exp
            break
    
    if exp_def is None:
        return
    
    # Find parameters that vary (have more than one value)
    varying_params = []
    for param, values in exp_def.items():
        if param in ['experiment_group', 'dont_run']:
            continue
        if isinstance(values, list) and len(values) > 1:
            varying_params.append(param)
    
    # Only proceed if exactly one parameter varies
    if len(varying_params) != 1:
        return
    
    varying_param = varying_params[0]
    param_values = sorted(exp_def[varying_param])
    
    print(f"\n{'='*80}")
    print(f"SINGLE PARAMETER VARIATION ANALYSIS: {varying_param}")
    print(f"{'='*80}")
    print(f"\nParameter '{varying_param}' varies across values: {param_values}")
    
    # Result columns to analyze
    result_cols = ['num_samples', 'num_equations', 'measure', 'num_suppressed',
                   'mixing_avg', 'mixing_min', 'mixing_max', 'mixing_median', 'elapsed_time', 'solver_metrics_runtime', 'solver_metrics_simplex_iterations', 'solver_metrics_num_vars', 'solver_metrics_num_constraints']
    
    # Filter to columns that exist in the dataframe
    result_cols = [col for col in result_cols if col in df.columns]
    
    print(f"\nShowing how results vary with {varying_param}:")
    
    # Group by result column instead of parameter value
    print(f"\n{'-'*80}")
    for col in result_cols:
        print(f"\n{col}:")
        print(f"  {varying_param:15s}", end="")
        for param_value in param_values:
            print(f" | {str(param_value):15s}", end="")
        print()
        print("  " + "-" * (15 + len(param_values) * 18))
        
        # Collect statistics for this result column across all parameter values
        stats_by_param = {}
        for param_value in param_values:
            df_subset = df[df[varying_param] == param_value]
            values = df_subset[col].dropna()
            
            if len(values) == 0:
                stats_by_param[param_value] = None
            elif len(values) == 1:
                stats_by_param[param_value] = {
                    'value': values.iloc[0],
                    'single': True
                }
            else:
                stats_by_param[param_value] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'single': False
                }
        
        # Print mean values
        print(f"  {'mean':15s}", end="")
        for param_value in param_values:
            stats = stats_by_param[param_value]
            if stats is None:
                print(f" | {'N/A':15s}", end="")
            elif stats['single']:
                print(f" | {stats['value']:15.4f}", end="")
            else:
                print(f" | {stats['mean']:15.4f}", end="")
        print()
        
        # Print std values if any parameter has multiple rows
        has_multiple = any(stats and not stats['single'] for stats in stats_by_param.values())
        if has_multiple:
            print(f"  {'std':15s}", end="")
            for param_value in param_values:
                stats = stats_by_param[param_value]
                if stats is None or stats['single']:
                    print(f" | {'--':15s}", end="")
                else:
                    print(f" | {stats['std']:15.4f}", end="")
            print()
            
            print(f"  {'min':15s}", end="")
            for param_value in param_values:
                stats = stats_by_param[param_value]
                if stats is None or stats['single']:
                    print(f" | {'--':15s}", end="")
                else:
                    print(f" | {stats['min']:15.4f}", end="")
            print()
            
            print(f"  {'max':15s}", end="")
            for param_value in param_values:
                stats = stats_by_param[param_value]
                if stats is None or stats['single']:
                    print(f" | {'--':15s}", end="")
                else:
                    print(f" | {stats['max']:15.4f}", end="")
            print()
    
    print(f"\n{'-'*80}")

def get_experiment_dataframes(experiments, df):
    """Group dataframe rows by experiment parameters.
    
    Args:
        experiments: List of experiment definitions from read_experiments()
        df: Full dataframe with all results
    
    Returns:
        Dict mapping experiment_group name to filtered dataframe
    """
    result = {}
    
    for exp in experiments:
        exp_group = exp['experiment_group']
        
        # Start with all rows - use df.index to ensure alignment
        mask = pd.Series([True] * len(df), index=df.index)
        
        # Filter by each parameter
        for param, values in exp.items():
            if param in ['experiment_group', 'solve_type', 'dont_run', 'seed', ]:
                continue
            
            if param in df.columns:
                # Row must have value in the experiment's value list
                mask = mask & df[param].isin(values)
        
        result[exp_group] = df[mask].copy()
        
    return result

def group_by_experiment_parameters(df_final):
    
    # Group data by key columns before reading experiments
    grouping_cols = ['solve_type', 'nrows', 'mask_size', 'nunique', 'noise', 'nqi', 'vals_per_qi', 'max_samples', 'target_accuracy', 'min_num_rows']
    
    # Filter to only columns that exist in the dataframe
    grouping_cols_present = [col for col in grouping_cols if col in df_final.columns]
    
    print(f"\nGrouping by columns: {grouping_cols_present}")
    
    # Identify numeric and non-numeric columns
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df_final.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Exclude grouping columns from aggregation
    numeric_cols_to_agg = [col for col in numeric_cols if col not in grouping_cols_present]
    non_numeric_cols_to_agg = [col for col in non_numeric_cols if col not in grouping_cols_present]
    
    # Create aggregation dictionary
    agg_dict = {}
    # Mean for numeric columns
    for col in numeric_cols_to_agg:
        agg_dict[col] = 'mean'
    # First for non-numeric columns
    for col in non_numeric_cols_to_agg:
        agg_dict[col] = 'first'
    
    # Group and aggregate
    df_grouped = df_final.groupby(grouping_cols_present, dropna=False).agg(agg_dict).reset_index()

    print(f"Grouped dataframe shape: {df_grouped.shape}")
    print(f"Original dataframe rows: {len(df_final)}")
    print(f"Grouped dataframe rows: {len(df_grouped)}")
    print(f"Numeric columns averaged: {len(numeric_cols_to_agg)}")
    print(f"Non-numeric columns (first value): {len(non_numeric_cols_to_agg)}")

    return df_grouped

def plot_by_x_y_lines(df: pd.DataFrame, x_col: str, y_col: str, lines_col: str, thresh: float = 0.95, thresh_direction: str = 'lowest', output_dir: Path = None):
    """Plot lowest/highest y value where measure >= threshold for each (x, lines) pair.
    
    For each pair of values (x, l) in x_col and lines_col, find the lowest or highest value y 
    of y_col where measure >= thresh. Connect points with the same lines_col value.
    When no data exceeds threshold for an x value, display "None" above that x-tick.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis (find lowest/highest value where measure >= thresh)
        lines_col: Column name for line grouping
        thresh: Threshold for measure column (default: 0.95)
        thresh_direction: 'lowest' or 'highest' - which y value to select (default: 'lowest')
        output_dir: Directory to save plot (default: results/row_mask_attacks/plots)
    """
    if output_dir is None:
        output_dir = Path('./results/row_mask_attacks/plots')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if thresh_direction not in ['lowest', 'highest']:
        raise ValueError(f"thresh_direction must be 'lowest' or 'highest', got '{thresh_direction}'")
    
    # Get unique values for x and lines
    x_values = sorted(df[x_col].unique())
    line_values = sorted(df[lines_col].unique())
    
    # For each (x, line) pair, find the lowest/highest y value where measure >= thresh
    plot_data = []
    x_values_with_none = set()  # Track x values where ALL lines have no valid data
    
    for x_val in x_values:
        has_any_valid_data = False
        
        for line_val in line_values:
            # Filter to this (x, line) combination
            subset = df[(df[x_col] == x_val) & (df[lines_col] == line_val)]
            
            if len(subset) == 0:
                continue
            
            # Find rows where measure >= thresh
            valid_rows = subset[subset['measure'] >= thresh]
            
            if len(valid_rows) > 0:
                # Get the lowest or highest y value from rows that meet threshold
                if thresh_direction == 'lowest':
                    y_val = valid_rows[y_col].min()
                else:  # highest
                    y_val = valid_rows[y_col].max()
                
                plot_data.append({
                    'x': x_val,
                    'y': y_val,
                    'line': line_val
                })
                has_any_valid_data = True
        
        # If no line had valid data for this x value, mark it
        if not has_any_valid_data:
            x_values_with_none.add(x_val)
    
    if len(plot_data) == 0 and len(x_values_with_none) == 0:
        print(f"No data points found")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a line for each line_val
    for line_val in line_values:
        line_data = [p for p in plot_data if p['line'] == line_val]
        
        if len(line_data) > 0:
            # Sort by x value
            line_data = sorted(line_data, key=lambda p: p['x'])
            
            x_vals = [p['x'] for p in line_data]
            y_vals = [p['y'] for p in line_data]
            
            # Plot line with markers
            ax.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=8, 
                    label=f'{lines_col}={line_val}')
    
    # Determine positioning for "None" labels
    if len(plot_data) > 0:
        y_values = [p['y'] for p in plot_data]
        y_min_data = min(y_values)
        y_max_data = max(y_values)
        
        # Position "None" at the minimum y value
        none_y_pos = y_min_data
        
        # Check if we should use log scale
        if all(y > 0 for y in y_values) and y_max_data / y_min_data > 10:
            ax.set_yscale('log')
    else:
        # No valid data points, only "None" labels
        none_y_pos = 0.5
        ax.set_ylim(0, 1)
    
    # Add "None" labels for x values with no valid data
    for x_val in x_values_with_none:
        ax.text(x_val, none_y_pos, 'None', 
                ha='center', va='bottom', fontsize=10, 
                style='italic', color='red', fontweight='bold')
    
    # Set x-axis to show all x values and add margin on the left
    ax.set_xticks(x_values)
    if len(x_values) > 1:
        x_range = x_values[-1] - x_values[0]
        left_margin = x_range * 0.05  # 5% margin on the left
        ax.set_xlim(left=x_values[0] - left_margin)
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(f'{y_col} ({thresh_direction} where measure >= {thresh})')
    ax.set_title(f'{thresh_direction.capitalize()} {y_col} (measure ≥ {thresh}) by {x_col} and {lines_col}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'plot_by_x_{x_col}_y_{y_col}_l_{lines_col}.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Saved: {filepath}")

def analyze():
    """Read result.parquet and analyze correlations with num_samples."""
    
    # Read the parquet file
    parquet_path = Path('./results/row_mask_attacks/result.parquet')
    
    if not parquet_path.exists():
        print(f"File {parquet_path} does not exist")
        print("Please run gather.py first")
        return
    
    df_all = pd.read_parquet(parquet_path)

    cols_to_fill = [{'col': 'seed', 'value': -1},
                    {'col': 'solver_metrics_skipped_constraints', 'value': -1},
                   ]
    for item in cols_to_fill:
        col = item['col']
        value = item['value']
        if col in df_all.columns:
            df_all[col] = df_all[col].fillna(value)

    # check if any columns have NaN values, and if so print the column names and quit
    nan_columns = df_all.columns[df_all.isna().any()].tolist()
    if len(nan_columns) > 0:
        print(f"Columns with NaN values: {nan_columns}")
        print("Please clean the data before analysis")
        return
    else:
        print("No NaN values found in dataframe")

    # Make df_final, which removes rows where final_attack is False
    df_final = df_all[df_all['final_attack'] == True].copy()
    
    print(f"Loaded {len(df_all)} rows from {parquet_path}")
    print(f"\nColumns: {list(df_all.columns)}")
    print(f"\nDataFrame shape: {df_all.shape}")
    
    # Remove unfinished jobs
    unfinished = df_final[df_final['finished'] == False]
    print(f"\nRemoving {len(unfinished)} unfinished jobs (finished==False)")
    df_final = df_final[df_final['finished'] == True].copy()
    print(f"Remaining rows: {len(df_final)}")

    df_grouped = group_by_experiment_parameters(df_final)
    
    # print first row of df_grouped using to_string to show all columns
    print(f"\nFirst row of grouped dataframe:\n{df_grouped.iloc[0].to_string()}\n")

    # Read experiments and group dataframes
    experiments = read_experiments()
    exp_dataframes = get_experiment_dataframes(experiments, df_grouped)
    
    print(f"\nExperiment groups:")
    for exp_group, exp_df in exp_dataframes.items():
        print(f"  {exp_group}: {len(exp_df)} rows")
    
    # Analyze each experiment group
    for exp_group, exp_df in exp_dataframes.items():
        if len(exp_df) == 0:
            print(f"\nSkipping {exp_group}: no data")
            continue
        
        # Determine analysis type based on experiment group name
        if exp_group == 'pure_dinur_basics':
            do_pure_dinur_basic_analysis(exp_df, experiments, exp_group)
        elif exp_group == 'agg_dinur_basics':
            do_agg_dinur_basic_analysis(exp_df, experiments, exp_group)
        elif exp_group == 'agg_dinur_explore_vals_per_qi_nrows':
            do_agg_dinur_explore_vals_per_qi_analysis(exp_df, experiments, exp_group)
        elif exp_group == 'agg_dinur_x_nqi_y_noise_lines_nrows':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='nrows', thresh=0.95)
        else:
            # Generic analysis for other experiment groups
            print(f"\n\n{'='*80}")
            print(f"ANALYSIS FOR {exp_group} EXPERIMENT GROUP")
            print(f"{'='*80}")
            analyze_single_parameter_variation(exp_df, experiments, exp_group)

def do_analysis_by_x_y_lines(df: pd.DataFrame, x_col: str, y_col: str, lines_col: str, thresh: float = 0.95):
    print(f"\n\nANALYSIS BY X={x_col}, Y={y_col}, LINES={lines_col}, THRESH={thresh}")
    print("=" * 80)
    # sort by x_col, then y_col, then lines_col, and display x_col, y_col, lines_col, and measure
    df_sorted = df.sort_values(by=['measure', x_col, y_col, lines_col])
    print(df_sorted[[x_col, y_col, lines_col, 'measure']].to_string())

    plot_by_x_y_lines(df, x_col=x_col, y_col=y_col, lines_col=lines_col, thresh_direction="highest", thresh=thresh)


def do_pure_dinur_basic_analysis(df, experiments=None, exp_group=None):
    print("\n\nANALYSIS FOR pure_dinur_basics EXPERIMENT GROUP")
    
    # Check for single parameter variation
    if experiments is not None and exp_group is not None:
        analyze_single_parameter_variation(df, experiments, exp_group)
    
    # Check if num_samples exists
    if 'num_samples' not in df.columns:
        print("\nError: num_samples column not found")
        return
    
    # Check for incomplete jobs (measure < target_accuracy)
    if 'measure' in df.columns and 'target_accuracy' in df.columns:
        incomplete = df[df['measure'] < df['target_accuracy']]
        print(f"\n\nINCOMPLETE JOBS (measure < target_accuracy):")
        print("=" * 80)
        print(f"Number of incomplete jobs: {len(incomplete)}")
        
        if len(incomplete) > 0:
            print("\nDetails of incomplete jobs:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            for idx, row in incomplete.iterrows():
                print("\n" + "-" * 80)
                print(f"Row {idx}:")
                for col in df.columns:
                    print(f"  {col:20s}: {row[col]}")
            print("-" * 80)
        print("\n")
        
        # Create dataframe with only complete jobs
        df_complete = df[df['measure'] >= df['target_accuracy']].copy()
        print(f"Complete jobs (measure >= target_accuracy): {len(df_complete)}")
    else:
        df_complete = df.copy()
        print("Warning: measure or target_accuracy column not found, using all data")
    
    # Create plots directory
    plots_dir = Path('./results/row_mask_attacks/plots')
    plots_dir.mkdir(exist_ok=True)
    print(f"\nPlots directory: {plots_dir}")
    
    # Generate plots using complete jobs only
    if len(df_complete) > 0:
        print("\nGenerating plots...")
        plot_mixing_vs_samples(df_complete, plots_dir)
        plot_boxplots_by_parameters(df_complete, plots_dir)
        plot_mixing_boxplots_by_parameters(df_complete, plots_dir)
        plot_elapsed_boxplots_by_parameters(df_complete, plots_dir)
        plot_mixing_vs_noise_by_mask_size(df_complete, plots_dir)
        plot_num_samples_vs_noise_by_mask_size(df_complete, plots_dir)
        plot_elapsed_time_vs_noise_by_mask_size(df_complete, plots_dir)
        
        print("Plots generated successfully\n")
    else:
        print("Warning: No complete jobs to plot\n")
    
    # Get numeric columns (exclude filename and num_samples itself)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'num_samples' in numeric_cols:
        numeric_cols.remove('num_samples')
    
    print(f"\n\nAnalyzing correlations with num_samples for {len(numeric_cols)} numeric columns")
    print("=" * 80)
    
    # Calculate correlations
    correlations = []
    
    for col in numeric_cols:
        # Remove rows with NaN in either column
        valid_data = df[[col, 'num_samples']].dropna()
        
        if len(valid_data) < 2:
            print(f"\nSkipping {col}: insufficient data ({len(valid_data)} rows)")
            continue
        
        # Skip columns with constant values
        if valid_data[col].nunique() == 1 or valid_data['num_samples'].nunique() == 1:
            print(f"\nSkipping {col}: constant values")
            continue
        
        # Calculate Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(valid_data[col], valid_data['num_samples'])
        
        # Calculate Spearman correlation (rank-based, handles non-linear monotonic relationships)
        spearman_r, spearman_p = stats.spearmanr(valid_data[col], valid_data['num_samples'])
        
        correlations.append({
            'column': col,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'abs_pearson_r': abs(pearson_r),
            'abs_spearman_r': abs(spearman_r),
            'n_samples': len(valid_data)
        })
    
    # Create correlation dataframe and sort by absolute Pearson correlation
    corr_df = pd.DataFrame(correlations).sort_values('abs_pearson_r', ascending=False)
    
    # Print strong correlations (|r| > 0.3)
    print("\n\nSTRONG CORRELATIONS (|Pearson r| > 0.3):")
    print("=" * 80)
    
    strong_corr = corr_df[corr_df['abs_pearson_r'] > 0.3]
    
    if len(strong_corr) == 0:
        print("No strong correlations found")
    else:
        for _, row in strong_corr.iterrows():
            print(f"\n{row['column']}:")
            print(f"  Pearson correlation:  r = {row['pearson_r']:7.4f}, p = {row['pearson_p']:.4e} (n={row['n_samples']})")
            print(f"  Spearman correlation: r = {row['spearman_r']:7.4f}, p = {row['spearman_p']:.4e}")
            
            # Interpret the correlation
            abs_r = row['abs_pearson_r']
            direction = "positive" if row['pearson_r'] > 0 else "negative"
            if abs_r > 0.7:
                strength = "very strong"
            elif abs_r > 0.5:
                strength = "strong"
            elif abs_r > 0.3:
                strength = "moderate"
            else:
                strength = "weak"
            
            print(f"  Interpretation: {strength} {direction} correlation")
            
            # Show some statistics
            col_data = df[row['column']].dropna()
            print(f"  {row['column']} range: [{col_data.min()}, {col_data.max()}]")
            print(f"  {row['column']} mean: {col_data.mean():.4f}, std: {col_data.std():.4f}")
    
    # Print moderate correlations (0.2 < |r| <= 0.3)
    print("\n\nMODERATE CORRELATIONS (0.2 < |Pearson r| <= 0.3):")
    print("=" * 80)
    
    moderate_corr = corr_df[(corr_df['abs_pearson_r'] > 0.2) & (corr_df['abs_pearson_r'] <= 0.3)]
    
    if len(moderate_corr) == 0:
        print("No moderate correlations found")
    else:
        for _, row in moderate_corr.iterrows():
            direction = "positive" if row['pearson_r'] > 0 else "negative"
            print(f"{row['column']:20s}: r = {row['pearson_r']:7.4f} ({direction}), p = {row['pearson_p']:.4e}")
    
    # Print all correlations summary
    print("\n\nALL CORRELATIONS (sorted by |Pearson r|):")
    print("=" * 80)
    print(corr_df[['column', 'pearson_r', 'pearson_p', 'spearman_r', 'n_samples']].to_string(index=False))
    
    # Summary statistics
    print("\n\nSUMMARY:")
    print("=" * 80)
    print(f"Total numeric columns analyzed: {len(correlations)}")
    print(f"Strong correlations (|r| > 0.3): {len(strong_corr)}")
    print(f"Moderate correlations (0.2 < |r| <= 0.3): {len(moderate_corr)}")
    print(f"Weak correlations (|r| <= 0.2): {len(corr_df[corr_df['abs_pearson_r'] <= 0.2])}")
    
    # Basic statistics on num_samples
    print(f"\nnum_samples statistics:")
    print(f"  Range: [{df['num_samples'].min()}, {df['num_samples'].max()}]")
    print(f"  Mean: {df['num_samples'].mean():.2f}")
    print(f"  Median: {df['num_samples'].median():.2f}")
    print(f"  Std: {df['num_samples'].std():.2f}")

def do_agg_dinur_basic_analysis(df, experiments=None, exp_group=None):
    print("\n\nANALYSIS FOR aggregated_dinur_basics EXPERIMENT GROUP")
    
    # Check for single parameter variation
    if experiments is not None and exp_group is not None:
        analyze_single_parameter_variation(df, experiments, exp_group)
    
    # Check if num_samples exists
    if 'num_samples' not in df.columns:
        print("\nError: num_samples column not found")
        return
    
    # Check for incomplete jobs (measure < target_accuracy)
    if 'measure' in df.columns and 'target_accuracy' in df.columns:
        incomplete = df[df['measure'] < df['target_accuracy']]
        print(f"\n\nINCOMPLETE JOBS (measure < target_accuracy):")
        print("=" * 80)
        print(f"Number of incomplete jobs: {len(incomplete)}")
        
        if len(incomplete) > 0:
            print("\nDetails of incomplete jobs:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            for idx, row in incomplete.iterrows():
                print("\n" + "-" * 80)
                print(f"Row {idx}:")
                for col in df.columns:
                    print(f"  {col:20s}: {row[col]}")
            print("-" * 80)
        print("\n")
        
        # Create dataframe with only complete jobs
        df_complete = df[df['measure'] >= df['target_accuracy']].copy()
        print(f"Complete jobs (measure >= target_accuracy): {len(df_complete)}")
        
        # Get target accuracy for plotting
        target_accuracy = df['target_accuracy'].iloc[0] if len(df) > 0 else 0.99
    else:
        df_complete = df.copy()
        target_accuracy = 0.99
        print("Warning: measure or target_accuracy column not found, using all data")
    
    # Create plots directory
    plots_dir = Path('./results/row_mask_attacks/plots_agg')
    plots_dir.mkdir(exist_ok=True)
    print(f"\nPlots directory: {plots_dir}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot measure by nqi (uses all data, not just complete)
    if 'measure' in df.columns and 'nqi' in df.columns:
        plot_measure_by_nqi(df, plots_dir, target_accuracy)
    
    # Generate other plots using complete jobs only
    if len(df_complete) > 0:
        plot_mixing_vs_samples(df_complete, plots_dir)
        plot_boxplots_by_parameters_nqi(df_complete, plots_dir)
        plot_mixing_boxplots_by_parameters_nqi(df_complete, plots_dir)
        plot_elapsed_boxplots_by_parameters_nqi(df_complete, plots_dir)
        plot_mixing_vs_noise_by_nqi(df_complete, plots_dir)
        plot_num_samples_vs_noise_by_nqi(df_complete, plots_dir)
        plot_elapsed_time_vs_noise_by_nqi(df_complete, plots_dir)
        
        print("Plots generated successfully\n")
    else:
        print("Warning: No complete jobs to plot\n")
    
    # Get numeric columns (exclude filename and num_samples itself)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'num_samples' in numeric_cols:
        numeric_cols.remove('num_samples')
    
    print(f"\n\nAnalyzing correlations with num_samples for {len(numeric_cols)} numeric columns")
    print("=" * 80)
    
    # Calculate correlations
    correlations = []
    
    for col in numeric_cols:
        # Remove rows with NaN in either column
        valid_data = df[[col, 'num_samples']].dropna()
        
        if len(valid_data) < 2:
            print(f"\nSkipping {col}: insufficient data ({len(valid_data)} rows)")
            continue
        
        # Skip columns with constant values
        if valid_data[col].nunique() == 1 or valid_data['num_samples'].nunique() == 1:
            print(f"\nSkipping {col}: constant values")
            continue
        
        # Calculate Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(valid_data[col], valid_data['num_samples'])
        
        # Calculate Spearman correlation (rank-based, handles non-linear monotonic relationships)
        spearman_r, spearman_p = stats.spearmanr(valid_data[col], valid_data['num_samples'])
        
        correlations.append({
            'column': col,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'abs_pearson_r': abs(pearson_r),
            'abs_spearman_r': abs(spearman_r),
            'n_samples': len(valid_data)
        })
    
    # Create correlation dataframe and sort by absolute Pearson correlation
    corr_df = pd.DataFrame(correlations).sort_values('abs_pearson_r', ascending=False)
    
    # Print strong correlations (|r| > 0.3)
    print("\n\nSTRONG CORRELATIONS (|Pearson r| > 0.3):")
    print("=" * 80)
    
    strong_corr = corr_df[corr_df['abs_pearson_r'] > 0.3]
    
    if len(strong_corr) == 0:
        print("No strong correlations found")
    else:
        for _, row in strong_corr.iterrows():
            print(f"\n{row['column']}:")
            print(f"  Pearson correlation:  r = {row['pearson_r']:7.4f}, p = {row['pearson_p']:.4e} (n={row['n_samples']})")
            print(f"  Spearman correlation: r = {row['spearman_r']:7.4f}, p = {row['spearman_p']:.4e}")
            
            # Interpret the correlation
            abs_r = row['abs_pearson_r']
            direction = "positive" if row['pearson_r'] > 0 else "negative"
            if abs_r > 0.7:
                strength = "very strong"
            elif abs_r > 0.5:
                strength = "strong"
            elif abs_r > 0.3:
                strength = "moderate"
            else:
                strength = "weak"
            
            print(f"  Interpretation: {strength} {direction} correlation")
            
            # Show some statistics
            col_data = df[row['column']].dropna()
            print(f"  {row['column']} range: [{col_data.min()}, {col_data.max()}]")
            print(f"  {row['column']} mean: {col_data.mean():.4f}, std: {col_data.std():.4f}")
    
    # Print moderate correlations (0.2 < |r| <= 0.3)
    print("\n\nMODERATE CORRELATIONS (0.2 < |Pearson r| <= 0.3):")
    print("=" * 80)
    
    moderate_corr = corr_df[(corr_df['abs_pearson_r'] > 0.2) & (corr_df['abs_pearson_r'] <= 0.3)]
    
    if len(moderate_corr) == 0:
        print("No moderate correlations found")
    else:
        for _, row in moderate_corr.iterrows():
            direction = "positive" if row['pearson_r'] > 0 else "negative"
            print(f"{row['column']:20s}: r = {row['pearson_r']:7.4f} ({direction}), p = {row['pearson_p']:.4e}")
    
    # Print all correlations summary
    print("\n\nALL CORRELATIONS (sorted by |Pearson r|):")
    print("=" * 80)
    print(corr_df[['column', 'pearson_r', 'pearson_p', 'spearman_r', 'n_samples']].to_string(index=False))
    
    # Summary statistics
    print("\n\nSUMMARY:")
    print("=" * 80)
    print(f"Total numeric columns analyzed: {len(correlations)}")
    print(f"Strong correlations (|r| > 0.3): {len(strong_corr)}")
    print(f"Moderate correlations (0.2 < |r| <= 0.3): {len(moderate_corr)}")
    print(f"Weak correlations (|r| <= 0.2): {len(corr_df[corr_df['abs_pearson_r'] <= 0.2])}")
    
    # Basic statistics on num_samples
    print(f"\nnum_samples statistics:")
    print(f"  Range: [{df['num_samples'].min()}, {df['num_samples'].max()}]")
    print(f"  Mean: {df['num_samples'].mean():.2f}")
    print(f"  Median: {df['num_samples'].median():.2f}")
    print(f"  Std: {df['num_samples'].std():.2f}")

def do_agg_dinur_explore_vals_per_qi_analysis(df, experiments=None, exp_group=None):
    """Analyze agg_dinur_explore_vals_per_qi results with text tables."""
    print("\n\nANALYSIS FOR agg_dinur_explore_vals_per_qi EXPERIMENT GROUP")
    print("=" * 80)
    
    if len(df) == 0:
        print("No results found for this experiment group")
        return
    
    # Get unique values for table axes
    noise_vals = sorted(df['noise'].unique())
    vpq_vals = sorted(df['vals_per_qi'].unique())
    nrows_vals = sorted(df['nrows'].unique())
    
    print(f"\nNoise levels: {noise_vals}")
    print(f"Vals_per_qi values: {vpq_vals}")
    print(f"Nrows values: {nrows_vals}")
    print(f"Total rows: {len(df)}")
    
    # Create tables for each nrows value
    for nrows_val in [100, 200]:
        df_nrows = df[df['nrows'] == nrows_val]
        
        if len(df_nrows) == 0:
            print(f"\nNo data for nrows={nrows_val}")
            continue
        
        print("\n" + "="*80)
        print(f"Table: Measure (Accuracy) for nrows={nrows_val}")
        print("="*80)
        
        # Create header
        header = f"{'vals_per_qi':>12}"
        for noise in noise_vals:
            header += f" | {str(noise):>10}"
        print(header)
        print("-" * len(header))
        
        # Create rows
        for vpq in vpq_vals:
            row = f"{vpq:>12}"
            for noise in noise_vals:
                # Find the specific row
                match = df_nrows[(df_nrows['vals_per_qi'] == vpq) & (df_nrows['noise'] == noise)]
                if len(match) == 1 and 'measure' in match.columns:
                    value = match['measure'].iloc[0]
                    row += f" | {value:>10.4f}"
                else:
                    row += f" | {'---':>10}"
            print(row)
    
    print("="*80 + "\n")
    
    # Check for single parameter variation
    if experiments is not None and exp_group is not None:
        analyze_single_parameter_variation(df, experiments, exp_group)

if __name__ == "__main__":
    analyze()