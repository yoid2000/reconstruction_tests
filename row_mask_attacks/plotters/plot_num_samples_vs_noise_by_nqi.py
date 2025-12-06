import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_num_samples_vs_noise_by_nqi(df: pd.DataFrame, output_dir: Path):
    """Create line plot of num_samples vs noise, with lines for each nqi.
    
    Args:
        df: DataFrame with complete jobs (measure >= target_accuracy)
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get unique nqi values and sort them
    nqi_values = sorted(df['nqi'].unique())
    
    # Define marker styles to cycle through
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    
    # For each nqi, compute average num_samples for each noise level
    for idx, nqi in enumerate(nqi_values):
        df_nqi = df[df['nqi'] == nqi]
        
        # Group by noise and compute mean, min, max num_samples
        grouped = df_nqi.groupby('noise')['num_samples'].agg(['mean', 'min', 'max']).sort_index()
        
        # Calculate error bars (distance from mean to min/max)
        yerr_lower = grouped['mean'] - grouped['min']
        yerr_upper = grouped['max'] - grouped['mean']
        yerr = [yerr_lower.values, yerr_upper.values]
        
        # Plot line with error bars
        plt.errorbar(grouped.index, grouped['mean'].values, yerr=yerr,
                    marker=markers[idx % len(markers)], label=f'nqi={nqi}', linewidth=2,
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
