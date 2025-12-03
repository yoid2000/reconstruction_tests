"""Example script showing how to analyze solver metrics from gathered results."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy import stats

def analyze_solver_metrics():
    """Analyze solver performance metrics from gathered results."""
    
    # Load gathered results
    results_file = Path('./results/row_mask_attacks/result.parquet')
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run gather.py first to create the parquet file.")
        return
    
    df = pd.read_parquet(results_file)
    
    print(f"Loaded {len(df)} results")
    print(f"\nColumns: {list(df.columns)}")
    
    # Check if solver_metrics columns exist
    solver_cols = [col for col in df.columns if 'solver_metrics_' in col]
    if not solver_cols:
        print("\nNo solver_metrics columns found!")
        print("This may be an older dataset without solver metrics.")
        return
    
    print(f"\nFound {len(solver_cols)} solver metric columns:")
    for col in sorted(solver_cols):
        print(f"  - {col}")
    
    # Basic statistics
    print("\n=== SOLVER METRICS SUMMARY ===\n")
    
    if 'solver_metrics_solver' in df.columns:
        print("Solver distribution:")
        print(df['solver_metrics_solver'].value_counts())
        print()
    
    if 'solver_metrics_runtime' in df.columns:
        print("Runtime statistics (seconds):")
        print(df['solver_metrics_runtime'].describe())
        print()
    
    if 'solver_metrics_status_string' in df.columns:
        print("Solver status distribution:")
        print(df['solver_metrics_status_string'].value_counts())
        print()
    
    # Create output directory for plots
    plots_dir = Path('./results/row_mask_attacks/solver_analysis')
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot 1: Runtime vs Number of Equations
    if 'solver_metrics_runtime' in df.columns and 'num_equations' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Separate by solver if available
        if 'solver_metrics_solver' in df.columns:
            for solver in df['solver_metrics_solver'].dropna().unique():
                solver_df = df[df['solver_metrics_solver'] == solver]
                plt.scatter(solver_df['num_equations'], solver_df['solver_metrics_runtime'],
                           alpha=0.6, label=solver, s=50)
            plt.legend()
        else:
            plt.scatter(df['num_equations'], df['solver_metrics_runtime'], alpha=0.6, s=50)
        
        plt.xlabel('Number of Equations')
        plt.ylabel('Solver Runtime (seconds)')
        plt.title('Solver Runtime vs Problem Size')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plot_file = plots_dir / 'runtime_vs_equations.png'
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Saved plot: {plot_file}")
    
    # Plot 2: Runtime vs Number of Variables
    if 'solver_metrics_runtime' in df.columns and 'solver_metrics_num_vars' in df.columns:
        plt.figure(figsize=(10, 6))
        
        if 'solver_metrics_solver' in df.columns:
            for solver in df['solver_metrics_solver'].dropna().unique():
                solver_df = df[df['solver_metrics_solver'] == solver]
                plt.scatter(solver_df['solver_metrics_num_vars'], solver_df['solver_metrics_runtime'],
                           alpha=0.6, label=solver, s=50)
            plt.legend()
        else:
            plt.scatter(df['solver_metrics_num_vars'], df['solver_metrics_runtime'], alpha=0.6, s=50)
        
        plt.xlabel('Number of Variables')
        plt.ylabel('Solver Runtime (seconds)')
        plt.title('Solver Runtime vs Number of Variables')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plot_file = plots_dir / 'runtime_vs_variables.png'
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Saved plot: {plot_file}")
    
    # Plot 3: Solver comparison boxplot
    if 'solver_metrics_solver' in df.columns and 'solver_metrics_runtime' in df.columns:
        if df['solver_metrics_solver'].nunique() > 1:
            plt.figure(figsize=(8, 6))
            df.boxplot(column='solver_metrics_runtime', by='solver_metrics_solver')
            plt.ylabel('Runtime (seconds)')
            plt.xlabel('Solver')
            plt.title('Solver Runtime Comparison')
            plt.yscale('log')
            plt.suptitle('')  # Remove default title
            plt.tight_layout()
            plot_file = plots_dir / 'solver_runtime_comparison.png'
            plt.savefig(plot_file, dpi=300)
            plt.close()
            print(f"Saved plot: {plot_file}")
    
    # Plot 4: Number of branches/iterations (OR-Tools specific)
    if 'solver_metrics_num_branches' in df.columns:
        ortools_df = df[df['solver_metrics_solver'] == 'ortools'].copy()
        if len(ortools_df) > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(ortools_df['num_equations'], ortools_df['solver_metrics_num_branches'],
                       alpha=0.6, s=50, c=ortools_df['solver_metrics_runtime'], cmap='viridis')
            plt.colorbar(label='Runtime (seconds)')
            plt.xlabel('Number of Equations')
            plt.ylabel('Number of Branches')
            plt.title('OR-Tools: Search Tree Size vs Problem Size')
            plt.grid(True, alpha=0.3)
            plt.xscale('log')
            plt.yscale('log')
            plt.tight_layout()
            plot_file = plots_dir / 'ortools_branches_vs_equations.png'
            plt.savefig(plot_file, dpi=300)
            plt.close()
            print(f"Saved plot: {plot_file}")
    
    # Plot 5: Simplex iterations (Gurobi specific)
    if 'solver_metrics_simplex_iterations' in df.columns:
        gurobi_df = df[df['solver_metrics_solver'] == 'gurobi'].copy()
        if len(gurobi_df) > 0 and gurobi_df['solver_metrics_simplex_iterations'].notna().any():
            plt.figure(figsize=(10, 6))
            plt.scatter(gurobi_df['num_equations'], gurobi_df['solver_metrics_simplex_iterations'],
                       alpha=0.6, s=50, c=gurobi_df['solver_metrics_runtime'], cmap='plasma')
            plt.colorbar(label='Runtime (seconds)')
            plt.xlabel('Number of Equations')
            plt.ylabel('Simplex Iterations')
            plt.title('Gurobi: Simplex Iterations vs Problem Size')
            plt.grid(True, alpha=0.3)
            plt.xscale('log')
            plt.yscale('log')
            plt.tight_layout()
            plot_file = plots_dir / 'gurobi_iterations_vs_equations.png'
            plt.savefig(plot_file, dpi=300)
            plt.close()
            print(f"Saved plot: {plot_file}")
    
    # Plot 6: Simplex iterations vs Runtime with correlation
    if 'solver_metrics_simplex_iterations' in df.columns and 'solver_metrics_runtime' in df.columns:
        gurobi_df = df[df['solver_metrics_solver'] == 'gurobi'].copy()
        # Filter out NaN values
        valid_data = gurobi_df[['solver_metrics_simplex_iterations', 'solver_metrics_runtime']].dropna()
        
        if len(valid_data) > 1:
            x = valid_data['solver_metrics_simplex_iterations'].values
            y = valid_data['solver_metrics_runtime'].values
            
            # Calculate correlation on original data
            correlation = np.corrcoef(x, y)[0, 1]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(x, y, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Simplex Iterations')
            plt.ylabel('Runtime (seconds)')
            plt.title(f'Gurobi: Simplex Iterations vs Runtime (Correlation: {correlation:.3f})')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_file = plots_dir / 'simplex_iterations_vs_runtime.png'
            plt.savefig(plot_file, dpi=300)
            plt.close()
            print(f"Saved plot: {plot_file}")
    
    # Summary table
    print("\n=== SOLVER PERFORMANCE BY NOISE LEVEL ===\n")
    if 'noise' in df.columns and 'solver_metrics_runtime' in df.columns:
        summary = df.groupby('noise').agg({
            'solver_metrics_runtime': ['mean', 'median', 'std', 'min', 'max'],
            'num_equations': 'mean',
            'measure': 'mean'
        }).round(3)
        print(summary)
    
    print(f"\n\nAll plots saved to: {plots_dir}")

if __name__ == '__main__':
    analyze_solver_metrics()
