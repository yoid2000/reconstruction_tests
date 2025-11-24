import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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

def analyze_correlations():
    """Read result.parquet and analyze correlations with num_samples."""
    
    # Read the parquet file
    parquet_path = Path('./results/row_mask_attacks/result.parquet')
    
    if not parquet_path.exists():
        print(f"File {parquet_path} does not exist")
        print("Please run gather.py first")
        return
    
    df = pd.read_parquet(parquet_path)
    
    print(f"Loaded {len(df)} rows from {parquet_path}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nDataFrame shape: {df.shape}")
    
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

if __name__ == '__main__':
    analyze_correlations()
