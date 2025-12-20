import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import sys
import pprint as pp
pp = pp.PrettyPrinter(indent=2)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from experiments import read_experiments
from plotters import (
    plot_mixing_vs_samples,
    plot_boxplots_by_parameters,
    plot_mixing_boxplots_by_parameters,
    plot_elapsed_boxplots_by_parameters,
    plot_mixing_vs_noise_by_mask_size,
    plot_num_samples_vs_noise_by_mask_size,
    plot_elapsed_time_vs_noise_by_mask_size,
    plot_mixing_vs_noise_by_nqi,
    plot_num_samples_vs_noise_by_nqi,
    plot_elapsed_time_vs_noise_by_nqi,
    plot_boxplots_by_parameters_nqi,
    plot_mixing_boxplots_by_parameters_nqi,
    plot_elapsed_boxplots_by_parameters_nqi,
    plot_measure_by_nqi,
    plot_by_x_y_lines,
    make_noise_min_num_rows_table,
    plot_mixing_by_measure,
)

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
    pp.pprint(exp_def)
    
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

def group_by_experiment_parameters(df_final, grouping_cols):
    
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

def analyze_seed_effect(df_final: pd.DataFrame, grouping_cols: list):
    """Check whether each parameter grouping has enough seeds for a tight CI on measure.
    
    A "sample" here is a single seed run. We check the number of unique seeds and
    the width of the 95% confidence interval for measure. Groups that have fewer
    than the minimum seeds or a confidence interval wider than the target margin
    are flagged so we know where to collect more runs.
    """
    if 'measure' not in df_final.columns:
        print("\nSkipping seed analysis: 'measure' column missing")
        return
    if 'seed' not in df_final.columns:
        print("\nSkipping seed analysis: 'seed' column missing")
        return

    # Only use grouping columns that exist in the dataframe
    grouping_cols_present = [col for col in grouping_cols if col in df_final.columns]
    if not grouping_cols_present:
        print("\nSkipping seed analysis: no grouping columns present")
        return

    # Heuristics for "enough" samples
    min_seeds = 15  # only consider groups that have at least this many seeds
    abs_margin = 0.05  # absolute margin for CI half-width
    rel_margin = 0.1  # or within 10% of the mean
    alpha = 0.05       # 95% confidence

    print("\nSeed effect / sample adequacy check")
    print(f"  Grouping columns: {grouping_cols_present}")
    print(f"  Criteria: >= {min_seeds} seeds and CI half-width <= max({abs_margin}, {rel_margin} * mean)")

    rows = []
    grouped = df_final.groupby(grouping_cols_present, dropna=False)
    for keys, g in grouped:
        # keys is a scalar when grouping by one column
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(grouping_cols_present, keys))

        measures = g['measure'].dropna()
        n = len(measures)
        if n == 0:
            continue

        mean_val = measures.mean()
        std_val = measures.std(ddof=1) if n > 1 else 0.0

        # 95% CI half-width using t distribution when possible
        if n > 1:
            t_val = stats.t.ppf(1 - alpha / 2, df=n - 1)
            ci_half = t_val * std_val / np.sqrt(n)
        else:
            ci_half = np.nan

        target_margin = max(abs_margin, rel_margin * abs(mean_val))

        # Rough estimate of seeds needed to hit target margin (use normal approx)
        if std_val == 0:
            seeds_needed = min_seeds
        else:
            z_val = stats.norm.ppf(1 - alpha / 2)
            seeds_needed = int(np.ceil((z_val * std_val / target_margin) ** 2))
            seeds_needed = max(seeds_needed, min_seeds)

        seed_count = g['seed'].nunique()
        if seed_count < min_seeds:
            continue

        enough_samples = (seed_count >= min_seeds) and (not np.isnan(ci_half)) and (ci_half <= target_margin)

        rows.append({
            **key_dict,
            'rows': len(g),
            'unique_seeds': seed_count,
            'mean_measure': mean_val,
            'std_measure': std_val,
            'ci_half_width': ci_half,
            'target_margin': target_margin,
            'seeds_needed_est': seeds_needed,
            'enough_samples': enough_samples,
        })

    if not rows:
        print("No data found for seed analysis")
        return

    summary_df = pd.DataFrame(rows)

    not_enough = summary_df[~summary_df['enough_samples']]
    print(f"\nTotal groups analyzed: {len(summary_df)}")
    print(f"Groups with enough samples: {len(summary_df) - len(not_enough)}")
    print(f"Groups needing more seeds: {len(not_enough)}")

    if len(not_enough) > 0:
        # Show the top 10 groups most in need of more seeds (sorted by margin gap)
        not_enough = not_enough.copy()
        not_enough['margin_gap'] = not_enough['ci_half_width'] - not_enough['target_margin']
        cols_to_show = grouping_cols_present + ['unique_seeds', 'seeds_needed_est', 'mean_measure',
                                                'std_measure', 'ci_half_width', 'target_margin', 'margin_gap']
        print("\nGroups needing more samples (top 10):")
        print(not_enough.sort_values(['margin_gap', 'unique_seeds'], ascending=[False, True])[cols_to_show].head(10).to_string(index=False))
    else:
        print("All groups meet the sampling criteria.")

def analyze():
    """Read result.parquet and analyze correlations with num_samples."""
    
    # Read the parquet file
    parquet_path = Path('./results/result.parquet')
    
    if not parquet_path.exists():
        print(f"File {parquet_path} does not exist")
        print("Please run gather.py first")
        return
    
    df_all = pd.read_parquet(parquet_path)
    if 'known_qi_fraction' in df_all.columns:
        # for all rows where solver_type != 'agg_known', set known_qi_fraction to 1.0
        df_all.loc[df_all['solve_type'] != 'agg_known', 'known_qi_fraction'] = 1.0

    cols_to_fill = [{'col': 'seed', 'value': -1},
                    {'col': 'solver_metrics_skipped_constraints', 'value': -1},
                    {'col': 'known_qi_fraction', 'value': 1.0},
                   ]
    for item in cols_to_fill:
        col = item['col']
        value = item['value']
        if col not in df_all.columns:
            df_all[col] = value
        else:
            df_all[col] = df_all[col].fillna(value)

    # check if any columns have NaN values, and if so print the column names and quit
    nan_columns = df_all.columns[df_all.isna().any()].tolist()
    # exclude columns beginning with 'solver_metrics_'
    nan_columns = [col for col in nan_columns if not col.startswith('solver_metrics_')]
    if len(nan_columns) > 0:
        print(f"Columns with NaN values: {nan_columns}")
        print("Please clean the data before analysis")
        return
    else:
        print("No NaN values found in dataframe")

    # Make df_final, which removes rows where final_attack is False
    df_final = df_all[df_all['final_attack'] == True].copy()

    plot_mixing_by_measure(df_all, Path('./results/plots'))
    
    print(f"Loaded {len(df_all)} rows from {parquet_path}")
    print(f"\nColumns: {list(df_all.columns)}")
    print(f"\nDataFrame shape: {df_all.shape}")
    
    # Remove unfinished jobs
    unfinished = df_final[df_final['finished'] == False]
    print(f"\nRemoving {len(unfinished)} unfinished jobs (finished==False)")
    df_final = df_final[df_final['finished'] == True].copy()
    print(f"Remaining rows: {len(df_final)}")

    grouping_cols = ['solve_type', 'nrows', 'mask_size', 'nunique', 'noise', 'nqi', 'vals_per_qi', 'max_samples', 'target_accuracy', 'min_num_rows', 'known_qi_fraction']
    analyze_seed_effect(df_final, grouping_cols)
    df_grouped = group_by_experiment_parameters(df_final, grouping_cols)
    
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
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='nrows', thresh=0.90)
            plot_by_x_y_lines(exp_df, x_col='nqi', y_col='actual_vals_per_qi', lines_col='nrows', thresh_direction="highest", thresh=0.9)
            plot_by_x_y_lines(exp_df, x_col='nqi', y_col='solver_metrics_num_vars', lines_col='nrows', thresh_direction="highest", thresh=0.9)
            plot_by_x_y_lines(exp_df, x_col='nqi', y_col='num_samples', lines_col='nrows', thresh_direction="highest", thresh=0.9)
            plot_by_x_y_lines(exp_df, x_col='nqi', y_col='mixing_avg', lines_col='nrows', thresh_direction="highest", thresh=0.9)
        elif exp_group == 'agg_dinur_x_nqi_y_noise_lines_min_num_rows':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='min_num_rows', thresh=0.90)
        elif exp_group == 'agg_dinur_x_nqi_y_noise_lines_nunique':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='nunique', thresh=0.90)
        elif exp_group == 'agg_dinur_x_nqi_y_noise_lines_vals_per_qi':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='actual_vals_per_qi', thresh=0.90)
        elif exp_group == 'agg_dinur_best_case_nrows_nqi3':
            do_analysis_by_x_y_lines(exp_df, x_col='min_num_rows', y_col='noise', lines_col='nrows', thresh=0.90)
            make_noise_min_num_rows_table(exp_df, "nqi3")
        elif exp_group == 'agg_dinur_best_case_nrows_nqi4':
            do_analysis_by_x_y_lines(exp_df, x_col='min_num_rows', y_col='noise', lines_col='nrows', thresh=0.90)
            make_noise_min_num_rows_table(exp_df, "nqi4")
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
    #print(df_sorted[[x_col, y_col, lines_col, 'measure']].to_string())
    exit_reasons_map = {'target_accuracy': 'targ', 
                    'no_more_qi_subsets': 'no_qi', 
                    'max_samples': 'max', }
    # make a table with x_col as rows, lines_col as columns, and exit_reason as values
    # use the exit_reasons_map to shorten the exit_reason values
    use_noise = 4
    # select rows where noise == use_noise
    df_noise = df[df['noise'] == use_noise]
    table = pd.pivot_table(df_noise, values='exit_reason', index=x_col, columns=lines_col, aggfunc=lambda x: ','.join(sorted(set(exit_reasons_map.get(v, v) for v in x))))
    print(f"\nExit reasons table when noise = {use_noise} (x={x_col}, lines={lines_col}):")
    print(table.to_string())

    for thresh in [0.80, 0.90, 0.95]:
        plot_by_x_y_lines(df, x_col=x_col, y_col=y_col, lines_col=lines_col, thresh_direction="highest", thresh=thresh)
    plot_by_x_y_lines(df, x_col=x_col, y_col='measure', lines_col=lines_col, thresh_direction="highest", thresh=0.9)


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
    plots_dir = Path('./results/plots')
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
    plots_dir = Path('./results/plots_agg')
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
