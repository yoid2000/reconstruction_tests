import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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
