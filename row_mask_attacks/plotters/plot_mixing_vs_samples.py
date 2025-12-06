import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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
