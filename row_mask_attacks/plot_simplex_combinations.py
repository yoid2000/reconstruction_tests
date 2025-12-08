"""Generate 3x3 subplot scatterplots of simplex iterations vs mathematical combinations of vars/constraints."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_simplex_combinations():
    """Create 3x3 subplot scatterplots for simplex iterations analysis."""
    
    # Load gathered results
    results_file = Path('./results/row_mask_attacks/result.parquet')
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run gather.py first to create the parquet file.")
        return
    
    df = pd.read_parquet(results_file)
    
    # Filter to Gurobi solver with valid simplex iterations
    gurobi_df = df[df['solver_metrics_solver'] == 'gurobi'].copy()
    gurobi_df = gurobi_df.dropna(subset=['solver_metrics_simplex_iterations', 
                                          'solver_metrics_num_vars', 
                                          'solver_metrics_num_constrs'])
    
    if len(gurobi_df) == 0:
        print("No valid Gurobi data with simplex iterations found!")
        return
    
    print(f"Analyzing {len(gurobi_df)} Gurobi results")
    
    # Create output directory
    plots_dir = Path('./results/row_mask_attacks/simplex_combinations')
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract data
    nv = gurobi_df['solver_metrics_num_vars'].values
    nc = gurobi_df['solver_metrics_num_constrs'].values
    si = gurobi_df['solver_metrics_simplex_iterations'].values
    
    # Calculate mathematical combinations
    combinations = {
        'nc³': nc ** 3,
        'nv*nc³': nv * (nc ** 3),
        'nv²*nc³': (nv ** 2) * (nc ** 3),
        'nv²': nv ** 2,
        'nc²': nc ** 2,
        'nv×nc': nv * nc,
        'nv²×nc': (nv ** 2) * nc,
        'nv×nc²': nv * (nc ** 2),
        'nv²×nc²': (nv ** 2) * (nc ** 2),
    }
    
    # Color schemes: column name and colormap
    color_schemes = [
        ('noise', 'viridis', 'Noise'),
        ('nrows', 'plasma', 'Number of Rows'),
        ('nunique', 'cividis', 'Distinct Target Values'),
        ('min_num_rows', 'coolwarm', 'Suppress Threshold'),
    ]
    
    # Create 4 plots (one for each color scheme)
    for col_name, cmap, label in color_schemes:
        if col_name not in gurobi_df.columns:
            print(f"Column '{col_name}' not found, skipping...")
            continue
        
        color_values = gurobi_df[col_name].values
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Simplex Iterations vs Problem Complexity (Colored by {label})', 
                     fontsize=16, fontweight='bold')
        
        for idx, (combo_name, combo_values) in enumerate(combinations.items()):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Create scatter plot
            scatter = ax.scatter(combo_values, si, c=color_values, cmap=cmap, 
                                alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
            
            # Set log scale for both axes
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Labels and grid
            ax.set_xlabel(combo_name, fontsize=10, fontweight='bold')
            ax.set_ylabel('Simplex Iterations', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Calculate and display correlation
            valid_mask = (combo_values > 0) & (si > 0)
            if valid_mask.sum() > 1:
                log_combo = np.log10(combo_values[valid_mask])
                log_si = np.log10(si[valid_mask])
                correlation = np.corrcoef(log_combo, log_si)[0, 1]
                ax.text(0.05, 0.95, f'ρ={correlation:.3f}', 
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add colorbar
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label(label, fontsize=12, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 0.92, 0.96])
        
        # Save plot
        plot_file = plots_dir / f'simplex_combinations_by_{col_name}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_file}")
    
    print(f"\nAll plots saved to: {plots_dir}")
    
    # Print summary statistics
    print("\n=== CORRELATION SUMMARY ===\n")
    print(f"{'Combination':<15} {'Log-Log Correlation':>20}")
    print("-" * 37)
    
    for combo_name, combo_values in combinations.items():
        valid_mask = (combo_values > 0) & (si > 0)
        if valid_mask.sum() > 1:
            log_combo = np.log10(combo_values[valid_mask])
            log_si = np.log10(si[valid_mask])
            correlation = np.corrcoef(log_combo, log_si)[0, 1]
            print(f"{combo_name:<15} {correlation:>20.4f}")

if __name__ == '__main__':
    plot_simplex_combinations()
