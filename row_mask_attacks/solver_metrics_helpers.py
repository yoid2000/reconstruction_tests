"""
Example functions to add solver metrics analysis to analyze.py

These functions can be integrated into your existing analyze.py to add
solver performance analysis to your current workflow.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_solver_runtime_by_parameters(df: pd.DataFrame, output_dir: Path):
    """Create boxplots of solver runtime grouped by each parameter.
    
    Args:
        df: DataFrame with complete jobs and solver_metrics columns
        output_dir: Directory to save plot
    """
    if 'solver_metrics_runtime' not in df.columns:
        print("No solver_metrics_runtime column found, skipping plot")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Boxplot by noise
    ax = axes[0, 0]
    noise_sorted = sorted(df['noise'].unique())
    df_sorted = df.copy()
    df_sorted['noise'] = pd.Categorical(df_sorted['noise'], categories=noise_sorted, ordered=True)
    df_sorted.boxplot(column='solver_metrics_runtime', by='noise', ax=ax)
    ax.set_title('Solver Runtime by Noise')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Solver Runtime (seconds)')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    # Subplot 2: Boxplot by nunique
    ax = axes[0, 1]
    nunique_sorted = sorted(df['nunique'].unique())
    df_sorted = df.copy()
    df_sorted['nunique'] = pd.Categorical(df_sorted['nunique'], categories=nunique_sorted, ordered=True)
    df_sorted.boxplot(column='solver_metrics_runtime', by='nunique', ax=ax)
    ax.set_title('Solver Runtime by Number of Unique Values')
    ax.set_xlabel('Number of Unique Values')
    ax.set_ylabel('Solver Runtime (seconds)')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    # Subplot 3: Boxplot by nrows
    ax = axes[1, 0]
    nrows_sorted = sorted(df['nrows'].unique())
    df_sorted = df.copy()
    df_sorted['nrows'] = pd.Categorical(df_sorted['nrows'], categories=nrows_sorted, ordered=True)
    df_sorted.boxplot(column='solver_metrics_runtime', by='nrows', ax=ax)
    ax.set_title('Solver Runtime by Number of Rows')
    ax.set_xlabel('Number of Rows')
    ax.set_ylabel('Solver Runtime (seconds)')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    # Subplot 4: Boxplot by mask_size
    ax = axes[1, 1]
    mask_size_sorted = sorted(df['mask_size'].unique())
    df_sorted = df.copy()
    df_sorted['mask_size'] = pd.Categorical(df_sorted['mask_size'], categories=mask_size_sorted, ordered=True)
    df_sorted.boxplot(column='solver_metrics_runtime', by='mask_size', ax=ax)
    ax.set_title('Solver Runtime by Mask Size')
    ax.set_xlabel('Mask Size')
    ax.set_ylabel('Solver Runtime (seconds)')
    ax.set_yscale('log')
    ax.get_figure().suptitle('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'solver_runtime_boxplots_by_parameters.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'solver_runtime_boxplots_by_parameters.png'}")


def plot_solver_efficiency_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot solver efficiency: runtime vs problem complexity.
    
    Args:
        df: DataFrame with complete jobs and solver_metrics columns
        output_dir: Directory to save plots
    """
    if 'solver_metrics_runtime' not in df.columns or 'num_equations' not in df.columns:
        print("Missing required columns for efficiency plot, skipping")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Runtime vs Equations
    ax = axes[0]
    if 'solver_metrics_solver' in df.columns:
        for solver in df['solver_metrics_solver'].dropna().unique():
            solver_df = df[df['solver_metrics_solver'] == solver]
            ax.scatter(solver_df['num_equations'], solver_df['solver_metrics_runtime'],
                      alpha=0.6, label=solver, s=50)
        ax.legend()
    else:
        ax.scatter(df['num_equations'], df['solver_metrics_runtime'], alpha=0.6, s=50)
    
    ax.set_xlabel('Number of Equations')
    ax.set_ylabel('Solver Runtime (seconds)')
    ax.set_title('Solver Runtime vs Problem Size')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Iterations/Branches vs Equations
    ax = axes[1]
    has_data = False
    
    # Try OR-Tools branches
    if 'solver_metrics_num_branches' in df.columns:
        ortools_df = df[df['solver_metrics_solver'] == 'ortools'].dropna(subset=['solver_metrics_num_branches'])
        if len(ortools_df) > 0:
            ax.scatter(ortools_df['num_equations'], ortools_df['solver_metrics_num_branches'],
                      alpha=0.6, label='OR-Tools Branches', s=50)
            has_data = True
    
    # Try Gurobi iterations
    if 'solver_metrics_simplex_iterations' in df.columns:
        gurobi_df = df[df['solver_metrics_solver'] == 'gurobi'].dropna(subset=['solver_metrics_simplex_iterations'])
        if len(gurobi_df) > 0:
            ax.scatter(gurobi_df['num_equations'], gurobi_df['solver_metrics_simplex_iterations'],
                      alpha=0.6, label='Gurobi Iterations', s=50, marker='s')
            has_data = True
    
    if has_data:
        ax.set_xlabel('Number of Equations')
        ax.set_ylabel('Branches/Iterations')
        ax.set_title('Solver Work vs Problem Size')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No iteration/branch data available', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'solver_efficiency_metrics.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'solver_efficiency_metrics.png'}")


def print_solver_statistics(df: pd.DataFrame):
    """Print summary statistics for solver metrics.
    
    Args:
        df: DataFrame with solver_metrics columns
    """
    print("\n" + "="*60)
    print("SOLVER PERFORMANCE STATISTICS")
    print("="*60 + "\n")
    
    if 'solver_metrics_solver' in df.columns:
        print("Solver Distribution:")
        print(df['solver_metrics_solver'].value_counts())
        print()
    
    if 'solver_metrics_runtime' in df.columns:
        print("Runtime Statistics (seconds):")
        print(df['solver_metrics_runtime'].describe())
        print()
        
        # By solver
        if 'solver_metrics_solver' in df.columns:
            print("Runtime by Solver:")
            for solver in df['solver_metrics_solver'].unique():
                if pd.notna(solver):
                    runtime = df[df['solver_metrics_solver'] == solver]['solver_metrics_runtime']
                    print(f"  {solver}: mean={runtime.mean():.3f}s, median={runtime.median():.3f}s, "
                          f"min={runtime.min():.3f}s, max={runtime.max():.3f}s")
            print()
    
    if 'solver_metrics_status_string' in df.columns:
        print("Solver Status Distribution:")
        print(df['solver_metrics_status_string'].value_counts())
        print()
    
    # OR-Tools specific metrics
    if 'solver_metrics_num_branches' in df.columns:
        ortools_df = df[df['solver_metrics_solver'] == 'ortools']
        if len(ortools_df) > 0:
            print("OR-Tools Branch Statistics:")
            print(ortools_df['solver_metrics_num_branches'].describe())
            print()
    
    # Gurobi specific metrics
    if 'solver_metrics_simplex_iterations' in df.columns:
        gurobi_df = df[df['solver_metrics_solver'] == 'gurobi']
        if len(gurobi_df) > 0 and gurobi_df['solver_metrics_simplex_iterations'].notna().any():
            print("Gurobi Iteration Statistics:")
            print(gurobi_df['solver_metrics_simplex_iterations'].describe())
            print()
    
    # Correlation analysis
    if 'solver_metrics_runtime' in df.columns and 'num_equations' in df.columns:
        corr = df[['solver_metrics_runtime', 'num_equations', 'num_samples']].corr()
        print("Correlation Matrix (Runtime, Equations, Samples):")
        print(corr)
        print()


def analyze_solver_scaling(df: pd.DataFrame):
    """Analyze how solver performance scales with problem parameters.
    
    Args:
        df: DataFrame with solver_metrics columns
    """
    print("\n" + "="*60)
    print("SOLVER SCALING ANALYSIS")
    print("="*60 + "\n")
    
    if 'solver_metrics_runtime' not in df.columns or 'num_equations' not in df.columns:
        print("Missing required columns for scaling analysis")
        return
    
    # Group by problem size bins
    df['equation_bins'] = pd.cut(df['num_equations'], bins=5)
    
    scaling = df.groupby('equation_bins').agg({
        'solver_metrics_runtime': ['mean', 'std', 'count'],
        'num_equations': 'mean',
        'measure': 'mean'
    }).round(3)
    
    print("Runtime by Problem Size (Equations):")
    print(scaling)
    print()
    
    # If we have multiple solvers, compare them
    if 'solver_metrics_solver' in df.columns and df['solver_metrics_solver'].nunique() > 1:
        print("Runtime by Solver and Problem Size:")
        pivot = df.pivot_table(
            values='solver_metrics_runtime',
            index='equation_bins',
            columns='solver_metrics_solver',
            aggfunc='mean'
        ).round(3)
        print(pivot)
        print()


# Example of how to integrate into analyze.py:
"""
To add to your analyze.py, you can:

1. Import these functions at the top:
   from solver_metrics_helpers import (
       plot_solver_runtime_by_parameters,
       plot_solver_efficiency_metrics,
       print_solver_statistics,
       analyze_solver_scaling
   )

2. In your analyze() function or wherever you process results:
   
   # Your existing code to load and filter data
   df = pd.read_parquet('./results/row_mask_attacks/result.parquet')
   complete_df = df[df['measure'] >= target_accuracy]
   
   # Add solver metrics analysis
   print_solver_statistics(complete_df)
   analyze_solver_scaling(complete_df)
   
   # Create solver metrics plots
   plots_dir = Path('./results/row_mask_attacks/plots')
   plots_dir.mkdir(exist_ok=True)
   
   plot_solver_runtime_by_parameters(complete_df, plots_dir)
   plot_solver_efficiency_metrics(complete_df, plots_dir)

3. Or create a separate analysis function:
   def analyze_solver_metrics(df, output_dir):
       print_solver_statistics(df)
       analyze_solver_scaling(df)
       plot_solver_runtime_by_parameters(df, output_dir)
       plot_solver_efficiency_metrics(df, output_dir)
"""
