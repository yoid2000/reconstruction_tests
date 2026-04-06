import argparse
import pandas as pd
import numpy as np
import random
from dataclasses import dataclass
from pathlib import Path
from scipy import stats
import sys
import pprint as pp
pp = pp.PrettyPrinter(indent=2)

grouping_cols = ['max_qi', 'solve_type', 'nrows', 'mask_size', 'nunique', 'noise', 'nqi', 'vals_per_qi', 'max_samples', 'target_accuracy', 'supp_thresh', 'known_qi_fraction', 'corr_strength', 'path_to_dataset', 'target_column']
group_max_cols = ['solver_metrics_runtime']
group_median_cols = ['solver_metrics_runtime']
grouping_cols_seed = grouping_cols + ['seed']


@dataclass(frozen=True)
class MetricConfig:
    metric_col: str
    metric_label: str
    plots_dir: Path
    tables_dir: Path
    agg_plots_dir: Path


ALC_THRESH_MAP = {
    0.95: 0.90,
    0.90: 0.80,
    0.80: 0.60,
    0.75: 0.50,
}

MEASURE_HEATMAP_THRESHOLDS = (0.99, 0.90, 0.75)
ALC_HEATMAP_THRESHOLDS = (0.98, 0.80, 0.50)


def map_metric_threshold(value: float, metric_cfg: MetricConfig) -> float:
    if value is None or pd.isna(value):
        return value
    if metric_cfg.metric_col != 'alc_alc':
        return value
    for src, dst in ALC_THRESH_MAP.items():
        if np.isclose(float(value), src):
            return dst
    return value


def get_metric_configs(df_all: pd.DataFrame):
    configs = [
        MetricConfig(
            metric_col='measure',
            metric_label='Accuracy',
            plots_dir=Path('./results/plots'),
            tables_dir=Path('./results/tables'),
            agg_plots_dir=Path('./results/plots_agg'),
        ),
    ]
    if 'alc_alc' in df_all.columns:
        configs.append(
            MetricConfig(
                metric_col='alc_alc',
                metric_label='ALC',
                plots_dir=Path('./results/plots_alc'),
                tables_dir=Path('./results/tables_alc'),
                agg_plots_dir=Path('./results/plots_agg_alc'),
            )
        )
    else:
        print("\nSkipping ALC analysis pass: 'alc_alc' column missing")
    return configs


def metric_df(df: pd.DataFrame, metric_cfg: MetricConfig) -> pd.DataFrame:
    if metric_cfg.metric_col not in df.columns:
        return None
    df_m = df.copy()
    if metric_cfg.metric_col != 'measure':
        df_m['measure'] = df_m[metric_cfg.metric_col]
        if 'target_accuracy' in df_m.columns:
            df_m['target_accuracy'] = df_m['target_accuracy'].apply(
                lambda x: map_metric_threshold(x, metric_cfg)
            )
    return df_m


def metric_threshold(value: float, metric_cfg: MetricConfig) -> float:
    return map_metric_threshold(value, metric_cfg)


def metric_title(metric_cfg: MetricConfig) -> str:
    if metric_cfg.metric_col == 'measure':
        return "Measure (Accuracy)"
    return metric_cfg.metric_label


def primary_line_threshold(metric_cfg: MetricConfig) -> float:
    return 0.9 if metric_cfg.metric_col == 'measure' else 0.8


def heatmap_thresholds_for_metric(metric_col: str) -> tuple[float, float, float]:
    if metric_col == 'alc_alc':
        return ALC_HEATMAP_THRESHOLDS
    return MEASURE_HEATMAP_THRESHOLDS


def run_plot_by_x_y_lines(
    df: pd.DataFrame,
    metric_cfg: MetricConfig,
    x_col: str,
    y_col: str,
    lines_col: str,
    thresh: float,
    thresh_direction: str = "highest",
    tag: str = "",
    extra_y_cols: list = None,
):
    # For standard line-threshold plots, use 0.9 for Accuracy and 0.8 for ALC.
    if np.isclose(float(thresh), 0.9):
        mapped_thresh = primary_line_threshold(metric_cfg)
    else:
        mapped_thresh = metric_threshold(thresh, metric_cfg)
    plot_by_x_y_lines(
        df,
        x_col=x_col,
        y_col=y_col,
        lines_col=lines_col,
        thresh_direction=thresh_direction,
        thresh=mapped_thresh,
        tag=tag,
        extra_y_cols=extra_y_cols or [],
        output_dir=metric_cfg.plots_dir,
        metric_label=metric_cfg.metric_label,
    )


def run_noise_min_num_rows_table(df: pd.DataFrame, nqi: int, note: str, metric_cfg: MetricConfig, thresh: float = 0.9):
    make_noise_min_num_rows_table(
        df,
        nqi,
        note,
        thresh=metric_threshold(thresh, metric_cfg),
        metric_col='measure',
        metric_label=metric_cfg.metric_label,
        output_dir=metric_cfg.tables_dir,
    )

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
    plot_mixing_by_param,
    plot_elapsed_time_pdf,
    plot_agg_known_heatmaps,
    plot_scatter_acc_cov,
)

def print_experiment_group_results(exp_df, exp_group, metrics):
    if exp_df is None or len(exp_df) == 0:
        print(f"\nSkipping {exp_group}: no data")
        return

    grouping_cols_present = [col for col in grouping_cols if col in exp_df.columns]
    metric_cols_present = [col for col in metrics if col in exp_df.columns]

    if not grouping_cols_present:
        print(f"\nSkipping {exp_group}: no grouping columns present")
        return
    if not metric_cols_present:
        print(f"\nSkipping {exp_group}: none of requested metrics present: {metrics}")
        return

    # Aggregate metrics if multiple rows share the same grouping key
    agg_dict = {}
    for col in metric_cols_present:
        if pd.api.types.is_numeric_dtype(exp_df[col]):
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'first'

    grouped = exp_df.groupby(grouping_cols_present, dropna=False).agg(agg_dict).reset_index()

    if len(grouped) != len(exp_df):
        print(f"\nNote: aggregated {len(exp_df)} rows into {len(grouped)} groups for {exp_group}")

    grouped = grouped.sort_values(by=grouping_cols_present)

    # Only show grouping columns that vary across groups
    varying_group_cols = [
        col for col in grouping_cols_present
        if grouped[col].nunique(dropna=False) > 1
    ]

    print(f"\nResults for experiment group '{exp_group}':")
    for _, row in grouped.iterrows():
        print("-" * 40)
        for col in varying_group_cols:
            print(f"{col}: {row[col]}")
        for col in metric_cols_present:
            print(f"{col}: {row[col]}")

def analyze_single_parameter_variation(df: pd.DataFrame, experiments: list, exp_group: str, metric_cfg: MetricConfig = None):
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
                   'mixing_avg', 'mixing_min', 'mixing_max', 'mixing_median', 'elapsed_time', 'med_solver_metrics_runtime', 'solver_metrics_simplex_iterations', 'solver_metrics_num_vars', 'solver_metrics_num_constraints',
                   'mix_times_sep', 'separation_average'
    ]
    
    # Filter to columns that exist in the dataframe
    result_cols = [col for col in result_cols if col in df.columns]
    
    print(f"\nShowing how results vary with {varying_param}:")
    
    # Group by result column instead of parameter value
    print(f"\n{'-'*80}")
    for col in result_cols:
        col_display = metric_cfg.metric_label if (metric_cfg is not None and col == 'measure') else col
        print(f"\n{col_display}:")
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
    grouped_parts = {}
    
    for exp in experiments:
        exp_group = exp['experiment_group']
        
        # Start with all rows - use df.index to ensure alignment
        mask = pd.Series([True] * len(df), index=df.index)
        print(f"\nFiltering for experiment group '{exp_group}'")
        print(f"Initial mask has {mask.sum()} rows (or {len(df) - mask.sum()} excluded)")

        if 'max_qi' in df.columns and 'max_qi' not in exp:
            default_max_qi = 1000
            mask = mask & (df['max_qi'] == default_max_qi)
            print(f"  Filtering by default max_qi={default_max_qi}")
            print(f"    Mask now has {mask.sum()} rows (or {len(df) - mask.sum()} excluded)")
        
        # Filter by each parameter
        for param, values in exp.items():
            if param in ['experiment_group', 'dont_run', 'seed', 'used_in_paper']:
                continue
            
            if param in df.columns:
                # Row must have value in the experiment's value list
                # check if values is not a list, and if so convert to list
                print(f"  Filtering by parameter '{param}' with values: {values}")
                if not isinstance(values, list):
                    values = [values]
                mask = mask & df[param].isin(values)
                print(f"    Mask now has {mask.sum()} rows (or {len(df) - mask.sum()} excluded)")
        
        grouped_parts.setdefault(exp_group, []).append(df[mask].copy())

    # Combine all parts for the same experiment_group (union of rows)
    result = {}
    for exp_group, parts in grouped_parts.items():
        if not parts:
            result[exp_group] = df.iloc[0:0].copy()
            continue
        combined = pd.concat(parts, axis=0)
        # Ensure union of rows if overlapping filters hit the same row
        combined = combined.loc[~combined.index.duplicated(keep="first")]
        result[exp_group] = combined

    return result

def group_by_experiment_parameters(df_final):
    
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

    # Add max_* columns for selected group_max_cols
    group_max_cols_present = [
        col for col in group_max_cols
        if col in df_final.columns and col not in grouping_cols_present
    ]
    if group_max_cols_present:
        df_max = (
            df_final
            .groupby(grouping_cols_present, dropna=False)[group_max_cols_present]
            .max()
            .reset_index()
            .rename(columns={col: f"max_{col}" for col in group_max_cols_present})
        )
        df_grouped = df_grouped.merge(df_max, on=grouping_cols_present, how='left')

    # Add med_* columns for selected group_median_cols
    group_median_cols_present = [
        col for col in group_median_cols
        if col in df_final.columns and col not in grouping_cols_present
    ]
    if group_median_cols_present:
        df_median = (
            df_final
            .groupby(grouping_cols_present, dropna=False)[group_median_cols_present]
            .median()
            .reset_index()
            .rename(columns={col: f"med_{col}" for col in group_median_cols_present})
        )
        df_grouped = df_grouped.merge(df_median, on=grouping_cols_present, how='left')

    print(f"Grouped dataframe shape: {df_grouped.shape}")
    print(f"Original dataframe rows: {len(df_final)}")
    print(f"Grouped dataframe rows: {len(df_grouped)}")
    print(f"Numeric columns averaged: {len(numeric_cols_to_agg)}")
    print(f"Non-numeric columns (first value): {len(non_numeric_cols_to_agg)}")

    return df_grouped

def analyze_seed_effect(
    df_final: pd.DataFrame,
    grouping_cols: list,
    metric_cfg: MetricConfig,
    write_more_seeds: bool = False,
):
    """Check whether each parameter grouping has enough seeds for a tight CI on the selected metric.
    
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
    min_seeds = 2  # only consider groups that have at least this many seeds
    min_margin = 0.01  # minimum margin for CI half-width
    ci_fraction = 3.0  # require CI half-width to be less than mean_measure / ci_fraction
    alpha = 0.05       # 95% confidence

    print(f"\nSeed effect / sample adequacy check ({metric_cfg.metric_label})")
    print(f"  Grouping columns: {grouping_cols_present}")
    print(f"  Criteria: >= {min_seeds} seeds and CI half-width = < max({min_margin}, (1-mean_measure)/{ci_fraction})")

    rows = []
    grouped = df_final.groupby(grouping_cols_present, dropna=False)
    for keys, g in grouped:
        # keys is a scalar when grouping by one column
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(grouping_cols_present, keys))
        if 'min_num_rows' in g.columns:
            unique_min_rows = g['min_num_rows'].dropna().unique()
            if len(unique_min_rows) == 0:
                key_dict['min_num_rows'] = None
            elif len(unique_min_rows) == 1:
                key_dict['min_num_rows'] = unique_min_rows[0]
            else:
                print(f"Warning: multiple min_num_rows values for group {key_dict}: {unique_min_rows}")
                key_dict['min_num_rows'] = unique_min_rows[0]

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

        target_half_margin = max(min_margin, (1.0-mean_val)/ci_fraction)

        # Rough estimate of seeds needed to hit target margin (use normal approx)
        if std_val == 0:
            seeds_needed = min_seeds
        else:
            z_val = stats.norm.ppf(1 - alpha / 2)
            seeds_needed = int(np.ceil((z_val * std_val / target_half_margin) ** 2))
            seeds_needed = max(seeds_needed, min_seeds)

        seed_count = g['seed'].nunique()
        if seed_count < min_seeds:
            continue

        enough_samples = (seed_count >= min_seeds) and (not np.isnan(ci_half)) and (ci_half <= target_half_margin)

        rows.append({
            **key_dict,
            'rows': len(g),
            'unique_seeds': seed_count,
            'mean_measure': mean_val,
            'std_measure': std_val,
            'ci_half_width': ci_half,
            'target_half_margin': target_half_margin,
            'seeds_needed_est': seeds_needed,
            'enough_samples': enough_samples,
        })

    if not rows:
        print("No data found for seed analysis")
        return

    summary_df = pd.DataFrame(rows)

    not_enough = summary_df[summary_df['enough_samples'].fillna(False).eq(False)]
    print(f"summary_df: {len(summary_df)} groups analyzed, {len(not_enough)} need more samples")
    if write_more_seeds:
        more_seeds_experiments = []
        for _, row in not_enough.iterrows():
            entry = {
                'dont_run': False,
                'used_in_paper': False,
                'experiment_group': 'temp',
                'slurm_run': 0,
            }
            for col in grouping_cols_present:
                if col not in row.index:
                    continue
                value = row[col]
                if pd.isna(value):
                    value = None
                elif hasattr(value, "item"):
                    value = value.item()
                if col == 'solve_type':
                    entry[col] = value
                else:
                    entry[col] = [value]
            if 'min_num_rows' in row.index:
                value = row['min_num_rows']
                if pd.isna(value):
                    value = None
                elif hasattr(value, "item"):
                    value = value.item()
                entry['min_num_rows'] = [value]

            seeds_needed = int(row['seeds_needed_est'])
            seed_pool = list(range(10000, 20001))
            if seeds_needed <= len(seed_pool):
                entry['seed'] = random.sample(seed_pool, seeds_needed)
            else:
                entry['seed'] = [random.randint(10000, 20000) for _ in range(seeds_needed)]

            more_seeds_experiments.append(entry)

        output_path = Path(__file__).parent / "more_seeds_experiments.py"
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("more_seeds_experiments = [\n")
            for entry in more_seeds_experiments:
                handle.write(f"    {entry},\n")
            handle.write("]\n")
        print(f"\nWrote {len(more_seeds_experiments)} new experiments to {output_path}")
    else:
        print("\nSkipping write of more_seeds_experiments.py (use --more_seeds to enable).")
    print(f"\nTotal groups analyzed: {len(summary_df)}")
    print(f"Groups with enough samples: {len(summary_df) - len(not_enough)}")
    print(f"Groups needing more seeds: {len(not_enough)}")
    if len(not_enough) > 0:
        mean_measure_avg = float(not_enough['mean_measure'].mean())
        mean_measure_std = float(not_enough['mean_measure'].std(ddof=1)) if len(not_enough) > 1 else 0.0
        target_half_margin_avg = float(not_enough['target_half_margin'].mean())
        target_half_margin_std = float(not_enough['target_half_margin'].std(ddof=1)) if len(not_enough) > 1 else 0.0
        print(f"\nGroups needing more seeds stats:")
        print(f"  mean_measure avg: {mean_measure_avg:.6f}, std: {mean_measure_std:.6f}")
        print(f"  target_half_margin avg: {target_half_margin_avg:.6f}, std: {target_half_margin_std:.6f}")

    if len(not_enough) > 0:
        # Show the top 100 groups most in need of more seeds (sorted by margin gap)
        not_enough = not_enough.copy()
        not_enough['margin_gap'] = not_enough['ci_half_width'] - not_enough['target_half_margin']
        cols_to_show = grouping_cols_present + ['unique_seeds', 'seeds_needed_est', 'mean_measure',
                                                'std_measure', 'ci_half_width', 'target_half_margin', 'margin_gap']
        print("\nGroups needing more samples (top 100):")
        top_rows = not_enough.sort_values(['margin_gap', 'unique_seeds'], ascending=[False, True])[cols_to_show].head(100)
        for _, row in top_rows.iterrows():
            parts = [f"{col}={row[col]}" for col in cols_to_show]
            print(", ".join(parts))
    else:
        print("All groups meet the sampling criteria.")


def analyze_refinement(df_all: pd.DataFrame, metric_cfg: MetricConfig) -> None:
    """Validate refinement ordering and report num_samples deltas.

    For each group, if multiple rows reach target_accuracy, the smallest
    num_samples among those rows must have final_attack == True.
    """
    required_cols = {'measure', 'target_accuracy', 'num_samples', 'final_attack', 'filename', 'attack_index', 'refine'}
    missing_cols = [col for col in required_cols if col not in df_all.columns]
    if missing_cols:
        print(f"\nSkipping refinement analysis ({metric_cfg.metric_label}): missing columns {missing_cols}")
        return

    grouping_cols_present = [col for col in grouping_cols_seed if col in df_all.columns]
    if not grouping_cols_present:
        print(f"\nSkipping refinement analysis ({metric_cfg.metric_label}): no grouping columns present")
        return

    diffs = []
    num_all_groups = 0
    num_groups_with_multiple_success = 0
    num_groups_with_refine_2 = 0
    num_groups_fixed = 0
    for group_key, group_df in df_all.groupby(grouping_cols_present, dropna=False):
        num_all_groups += 1
        # get the maximum refine value in this group
        max_refine = group_df['refine'].max()
        if max_refine == 2:
            num_groups_with_refine_2 += 1
        success_rows = group_df[group_df['measure'] >= group_df['target_accuracy']]
        if len(success_rows) < 2:
            continue
        num_groups_with_multiple_success += 1

        min_samples = success_rows['num_samples'].min()
        max_samples = success_rows['num_samples'].max()
        diffs.append(max_samples - min_samples)

        min_success = success_rows[success_rows['num_samples'] == min_samples]
        if not min_success['final_attack'].any():
            if max_refine == 2:
                filename = min_success.iloc[0]['filename']
                print(f"\nRefinement check failed for file {filename}:")
                # print all values of filename in group_df
                print(group_df['filename'].unique())
                display_df = group_df[['measure', 'num_samples', 'final_attack', 'attack_index', 'refine']].rename(
                    columns={'measure': metric_cfg.metric_label}
                )
                print(display_df)
                raise ValueError(
                    f"Refinement check failed for group {filename}: min num_samples success is not final_attack"
                )
            else:
                # This can happen if the experiment ended prematurely
                # Set final_attack to True for the min_samples row, and False for all others
                filename = min_success.iloc[0]['filename']
                group_df.loc[min_success.index, 'final_attack'] = True
                other_success = success_rows[success_rows['num_samples'] != min_samples]
                group_df.loc[other_success.index, 'final_attack'] = False
                num_groups_fixed += 1

    if not diffs:
        print(f"\nRefinement analysis ({metric_cfg.metric_label}): no groups with multiple successful rows")
        return

    avg_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    print(f"\nTotal groups analyzed: {num_all_groups}"  )
    print(f"Groups with refine==2: {num_groups_with_refine_2}")
    print(f"Groups with multiple successes: {num_groups_with_multiple_success}")
    print(f"Groups fixed: {num_groups_fixed}")
    print(f"\nRefinement analysis ({metric_cfg.metric_label}): {len(diffs)} groups with multiple successes")
    print(f"  num_samples diff avg: {avg_diff:.2f}")
    print(f"  num_samples diff std: {std_diff:.2f}")

def do_agg_dinur_nrows_high_suppression_analysis(exp_df, exp_group, metric_cfg: MetricConfig):
    print(f"\n\nANALYSIS FOR {exp_group} EXPERIMENT GROUP")
    print("=" * 80)
    # For each distinct value in nrows, print selected metric, num_samples, and med_solver_metrics_runtime
    distinct_min_num_rows = sorted(exp_df['supp_thresh'].dropna().unique())
    distinct_nrows = sorted(exp_df['nrows'].dropna().unique())
    for nrows in distinct_nrows:
        df_nrows = exp_df[exp_df['nrows'] == nrows]
        print(f"\nnrows = {nrows}:")
        display_df = df_nrows[['measure', 'num_samples', 'med_solver_metrics_runtime']].rename(
            columns={'measure': metric_cfg.metric_label}
        )
        print(display_df.to_string(index=False))

def remove_unused_rows(df: pd.DataFrame) -> pd.DataFrame:
    experiments = read_experiments()

    used_mask = pd.Series([False] * len(df), index=df.index)
    for exp in experiments:
        mask = pd.Series([True] * len(df), index=df.index)

        if 'max_qi' in df.columns and 'max_qi' not in exp:
            default_max_qi = 1000
            mask = mask & (df['max_qi'] == default_max_qi)

        for param, values in exp.items():
            if param in ['experiment_group', 'dont_run', 'used_in_paper', 'seed']:
                continue
            if param not in df.columns:
                continue
            if not isinstance(values, list):
                values = [values]
            mask = mask & df[param].isin(values)

        used_mask = used_mask | mask

    df = df.copy()
    df['used_in_experiment'] = used_mask

    num_used = int(used_mask.sum())
    num_removed = len(df) - num_used
    print(f"\nremove_unused_rows: keeping {num_used} rows, removing {num_removed} unused rows")

    return df[df['used_in_experiment']].copy()

def prep_data() -> pd.DataFrame:
    """Read result.parquet and prep"""
    
    # Read the parquet file
    parquet_path = Path('./results/result.parquet')
    
    if not parquet_path.exists():
        print(f"File {parquet_path} does not exist")
        print("Please run gather.py first")
        # throw exception
        raise FileNotFoundError(f"File {parquet_path} does not exist")
    
    df_all = pd.read_parquet(parquet_path)
    # show count of all values for known_qi_fraction column
    print("\nKnown_qi_fraction value counts:")
    print(df_all['known_qi_fraction'].value_counts(dropna=False))
    if 'min_num_rows' in df_all.columns:
        df_all['supp_thresh'] = df_all['min_num_rows'] - 1
        df_all['supp_thresh'] = df_all['supp_thresh'].astype(int)
    if 'known_qi_fraction' in df_all.columns:
        # for all rows where solver_type != 'agg_known', set known_qi_fraction to 1.0
        df_all.loc[df_all['solve_type'] != 'agg_known', 'known_qi_fraction'] = 1.0
    print("\nKnown_qi_fraction value counts:")
    print(df_all['known_qi_fraction'].value_counts(dropna=False))

    cols_to_fill = [{'col': 'seed', 'value': -1},
                    {'col': 'solver_metrics_skipped_constraints', 'value': -1},
                    {'col': 'corr_strength', 'value': 0.0},
                    {'col': 'path_to_dataset', 'value': ""},
                    {'col': 'target_column', 'value': ""},
                    {'col': 'actual_vals_per_qi', 'value': 0},
                    {'col': 'alc_attack_recall', 'value': 1.0},
                    {'col': 'alc_baseline_recall', 'value': 1.0},
                   ]
    for item in cols_to_fill:
        col = item['col']
        value = item['value']
        if col not in df_all.columns:
            df_all[col] = value
        else:
            df_all[col] = df_all[col].fillna(value)

    # For rows where alc_alc is NaN, backfill ALC fields from measure and nunique.
    if 'alc_alc' in df_all.columns:
        nan_mask = df_all['alc_alc'].isna()
        if nan_mask.any():
            alc_cols = [
                'alc_attack_precision',
                'alc_baseline_precision',
                'alc_attack_prc',
                'alc_baseline_prc',
            ]
            for col in alc_cols:
                if col not in df_all.columns:
                    df_all[col] = np.nan

            baseline = 1.0 / df_all.loc[nan_mask, 'nunique'].astype(float)
            attack = df_all.loc[nan_mask, 'measure'].astype(float)

            df_all.loc[nan_mask, 'alc_baseline_prc'] = baseline
            df_all.loc[nan_mask, 'alc_baseline_precision'] = baseline
            df_all.loc[nan_mask, 'alc_attack_prc'] = attack
            df_all.loc[nan_mask, 'alc_attack_precision'] = attack
            df_all.loc[nan_mask, 'alc_alc'] = (
                (df_all.loc[nan_mask, 'alc_attack_prc'] - df_all.loc[nan_mask, 'alc_baseline_prc'])
                / (1.0 - df_all.loc[nan_mask, 'alc_baseline_prc'])
            )

    # check if any columns have NaN values, and if so print the column names and quit
    nan_columns = df_all.columns[df_all.isna().any()].tolist()
    # exclude columns beginning with 'solver_metrics_'
    nan_columns = [col for col in nan_columns if not col.startswith('solver_metrics_')]
    if len(nan_columns) > 0:
        print(f"Columns with NaN values: {nan_columns}")
        print("Please clean the data before analysis")
        raise ValueError(f"Columns with NaN values: {nan_columns}")
    else:
        print("No NaN values found in dataframe")

    # Remove unused rows from df_all
    df_all = remove_unused_rows(df_all)

    # Make a new column mix_sep which is mixing_avg * separation_average
    if 'mixing_avg' in df_all.columns and 'separation_average' in df_all.columns:
        df_all['mix_times_sep'] = df_all['mixing_avg'] * df_all['separation_average']

    print(f"Loaded {len(df_all)} rows from {parquet_path}")
    return df_all

def compare_agg_known_to_agg_row(df_known, df_row, metric_cfg: MetricConfig):
    print(f"\n\nCOMPARE agg_known vs agg_row ({metric_cfg.metric_label})")
    print("=" * 80)

    if df_known is None or len(df_known) == 0:
        print("No agg_known rows to compare")
        return
    if df_row is None or len(df_row) == 0:
        print("No agg_row rows to compare")
        return

    if 'known_qi_fraction' in df_known.columns:
        before = len(df_known)
        df_known = df_known[np.isclose(df_known['known_qi_fraction'], 1.0)].copy()
        print(f"Filtered df_known by known_qi_fraction==1.0: {before} -> {len(df_known)} rows")
    else:
        print("Warning: df_known missing 'known_qi_fraction' column")

    if len(df_known) == 0:
        print("No agg_known rows with known_qi_fraction==1.0")
        return

    if 'solve_type' in df_row.columns:
        df_row = df_row[df_row['solve_type'] == 'agg_row'].copy()
    else:
        print("Warning: df_row missing 'solve_type' column; cannot filter to agg_row")

    if len(df_row) == 0:
        print("No agg_row rows to compare after filtering")
        return

    if 'med_solver_metrics_runtime' in df_row.columns:
        df_row_grouped = df_row.copy()
    else:
        df_row_grouped = group_by_experiment_parameters(df_row)

    compare_groups = [
        'max_qi', 'nrows', 'nunique', 'noise', 'nqi', 'vals_per_qi',
        'max_samples', 'target_accuracy', 'supp_thresh', 'known_qi_fraction'
    ]
    compare_groups_present = [
        col for col in compare_groups
        if col in df_known.columns and col in df_row_grouped.columns
    ]
    drop_cols = [col for col in compare_groups_present if df_row_grouped[col].isna().all()]
    if drop_cols:
        compare_groups_present = [col for col in compare_groups_present if col not in drop_cols]
        print(f"Warning: dropping compare columns with all-NaN agg_row values: {drop_cols}")

    if not compare_groups_present:
        print("No overlapping compare group columns found")
        return

    x_y_group_all = [
        'measure', 'num_samples', 'mixing_avg', 'separation_average',
        'num_suppressed', 'solver_metrics_simplex_iterations',
        'med_solver_metrics_runtime', 'mix_times_sep',
        'solver_metrics_num_vars', 'solver_metrics_num_constrs'
    ]
    x_y_group = [
        col for col in x_y_group_all
        if col in df_known.columns and col in df_row_grouped.columns
    ]
    if not x_y_group:
        print("No overlapping x_y_group columns found for comparison")
        return

    missing_known = [col for col in x_y_group_all if col not in df_known.columns]
    missing_row = [col for col in x_y_group_all if col not in df_row_grouped.columns]
    if missing_known:
        print(f"Warning: df_known missing columns: {missing_known}")
    if missing_row:
        print(f"Warning: df_row missing columns: {missing_row}")

    def summarize_group(group_df: pd.DataFrame, cols: list) -> pd.Series:
        if len(group_df) == 1:
            return group_df.iloc[0][cols]
        summary = {}
        for col in cols:
            if col not in group_df.columns:
                continue
            if pd.api.types.is_numeric_dtype(group_df[col]):
                summary[col] = group_df[col].mean()
            else:
                summary[col] = group_df[col].iloc[0]
        return pd.Series(summary)

    grouped_known = df_known.groupby(compare_groups_present, dropna=False)
    match_count = 0

    for keys, known_group in grouped_known:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(compare_groups_present, keys))

        mask = pd.Series(True, index=df_row_grouped.index)
        for col, value in key_dict.items():
            if pd.isna(value):
                mask &= df_row_grouped[col].isna()
            else:
                mask &= df_row_grouped[col] == value

        row_group = df_row_grouped[mask]

        print("\n" + "-" * 80)
        print("Group:", ", ".join(f"{k}={v}" for k, v in key_dict.items()))

        if len(row_group) == 0:
            print("No matching agg_row group found")
            continue

        if len(row_group) > 1:
            print(f"Warning: multiple agg_row matches ({len(row_group)}), averaging values")
        if len(known_group) > 1:
            print(f"Warning: multiple agg_known rows ({len(known_group)}), averaging values")

        known_vals = summarize_group(known_group, x_y_group)
        row_vals = summarize_group(row_group, x_y_group)

        compare_df = pd.DataFrame({
            'agg_known': known_vals,
            'agg_row': row_vals,
        })
        compare_df = compare_df.rename(index={'measure': metric_cfg.metric_label})
        print(compare_df.to_string(float_format=lambda x: f"{x:0.4f}"))
        match_count += 1

    print(f"\nCompared {match_count} group(s)")

def run_metric_analysis(df_all_raw: pd.DataFrame, more_seeds: bool, metric_cfg: MetricConfig):
    print(f"\n\n{'='*80}")
    print(f"METRIC ANALYSIS: {metric_cfg.metric_label} (column={metric_cfg.metric_col})")
    print(f"{'='*80}")

    df_all = metric_df(df_all_raw, metric_cfg)
    if df_all is None:
        print(f"Skipping metric analysis: column '{metric_cfg.metric_col}' missing")
        return

    if metric_cfg.metric_col == 'measure':
        analyze_refinement(df_all, metric_cfg)
    else:
        print(f"\nSkipping refinement analysis for metric {metric_cfg.metric_label}")

    # Make df_final, which removes rows where final_attack is False
    df_final = df_all[df_all['final_attack'] == True].copy()
    print(f"\nFiltered to final_attack==True: {len(df_all)} rows -> {len(df_final)} rows")

    print(f"\nColumns: {list(df_all.columns)}")
    print(f"\nDataFrame shape: {df_all.shape}")

    # Remove unfinished jobs
    unfinished = df_final[df_final['finished'] == False]
    print(f"\nRemoving {len(unfinished)} unfinished jobs (finished==False)")
    df_final = df_final[df_final['finished'] == True].copy()
    print(f"Remaining rows: {len(df_final)}")

    ##### ChatGPT ######
    alc_cols = sorted([col for col in df_final.columns if col.startswith('alc_')])
    if 'nunique' not in df_final.columns:
        print("\nSkipping example row print: 'nunique' column missing")
    elif 'measure' not in df_final.columns:
        print("\nSkipping example row print: selected metric column missing")
    elif len(alc_cols) == 0:
        print("\nSkipping example row print: no columns starting with 'alc_' found")
    else:
        sample_cols = ['measure'] + alc_cols
        print(f"\nExample rows for columns: {sample_cols}")
        print(f"This is just to validate metric computations ({metric_cfg.metric_label})")
        for nunique_val in [2, 4]:
            subset = df_final[df_final['nunique'] == nunique_val]
            if len(subset) == 0:
                print(f"\nNo rows found where nunique == {nunique_val}")
                continue
            sample_n = min(5, len(subset))
            print(f"\nRandom sample ({sample_n} rows) where nunique == {nunique_val}:")
            print(subset[sample_cols].sample(n=sample_n).to_string(index=False))

    plot_scatter_acc_cov(df_final)
    analyze_seed_effect(df_final, grouping_cols, metric_cfg=metric_cfg, write_more_seeds=more_seeds)
    df_grouped = group_by_experiment_parameters(df_final)
    print("\n nrows value counts:")
    print(df_grouped['nrows'].value_counts(dropna=False))
    print("\n solve_type value counts:")
    print(df_grouped['solve_type'].value_counts(dropna=False))
    print("Columns in df_grouped:")
    print(list(df_grouped.columns))

    # print first row of df_grouped using to_string to show all columns
    print(f"\nFirst row of grouped dataframe:\n{df_grouped.iloc[0].to_string()}\n")

    print(f"\nRelative difference between max_solver_metrics_runtime and med_solver_metrics_runtime where med_solver_metrics_runtime > 60")
    df_runtime_check = df_grouped[df_grouped['med_solver_metrics_runtime'] > 60].copy()
    df_runtime_check['rel_diff'] = (
        (df_runtime_check['max_solver_metrics_runtime'] - df_runtime_check['med_solver_metrics_runtime'])
        / df_runtime_check['med_solver_metrics_runtime']
    )
    print(df_runtime_check['rel_diff'].describe())
    print(f"\nTop 5 cases with largest relative difference:")
    print(df_runtime_check.sort_values('rel_diff', ascending=False).head(5)[['med_solver_metrics_runtime', 'max_solver_metrics_runtime', 'rel_diff']])

    # Read experiments and group dataframes
    experiments = read_experiments()
    exp_dataframes = get_experiment_dataframes(experiments, df_grouped)

    x_y_group = [
        'measure',
        'num_samples',
        'mixing_avg',
        'separation_average',
        'num_suppressed',
        'solver_metrics_simplex_iterations',
        'med_solver_metrics_runtime',
        'mix_times_sep',
        'solver_metrics_num_vars',
        'solver_metrics_num_constrs',
    ]

    print(f"\nExperiment groups:")
    for exp_group, exp_df in exp_dataframes.items():
        print(f"  {exp_group}: {len(exp_df)} rows")

    # Analyze each experiment group
    for exp_group, exp_df in exp_dataframes.items():
        if len(exp_df) == 0:
            print(f"\nSkipping {exp_group}: no data")
            continue

        if exp_group == 'oa_200_low_distortion':
            if metric_cfg.metric_col != 'measure':
                continue
            metrics = ['alc_alc', 'med_solver_metrics_runtime']
            print_experiment_group_results(exp_df, exp_group, metrics)
            high_thresh, medium_thresh, low_thresh = heatmap_thresholds_for_metric('alc_alc')
            plot_agg_known_heatmaps(
                exp_df,
                tag="oa_200_low_distortion",
                attr_name='alc_alc',
                output_dir=metric_cfg.plots_dir,
                high_thresh=high_thresh,
                medium_thresh=medium_thresh,
                low_thresh=low_thresh,
            )
        elif exp_group == 'oa_500_low_distortion':
            if metric_cfg.metric_col != 'measure':
                continue
            metrics = ['alc_alc', 'med_solver_metrics_runtime']
            print_experiment_group_results(exp_df, exp_group, metrics)
            high_thresh, medium_thresh, low_thresh = heatmap_thresholds_for_metric('alc_alc')
            plot_agg_known_heatmaps(
                exp_df,
                tag="oa_500_low_distortion",
                attr_name='alc_alc',
                output_dir=metric_cfg.plots_dir,
                high_thresh=high_thresh,
                medium_thresh=medium_thresh,
                low_thresh=low_thresh,
            )
        elif exp_group == 'pure_dinur_basics':
            do_pure_dinur_basic_analysis(exp_df, metric_cfg, experiments, exp_group)
        elif exp_group == 'agg_dinur_nrows_vals_per_qi':
            for ycol in x_y_group:
                run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='vals_per_qi', y_col=ycol, lines_col='nrows', thresh=0.9, tag="big_nrows")
        elif exp_group == 'agg_dinur_nrows_suppression':
            for ycol in x_y_group:
                run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='supp_thresh', y_col=ycol, lines_col='nrows', thresh=0.9, tag="big_nrows")
        elif exp_group == 'agg_dinur_nrows_high_suppression':
            do_agg_dinur_nrows_high_suppression_analysis(exp_df, exp_group, metric_cfg)
        elif exp_group == 'agg_dinur_basics':
            do_agg_dinur_basic_analysis(exp_df, metric_cfg, experiments, exp_group)
        elif exp_group == 'agg_dinur_explore_vals_per_qi_nrows':
            do_agg_dinur_explore_vals_per_qi_analysis(exp_df, metric_cfg, experiments, exp_group)
        elif exp_group == 'agg_dinur_nrows_low_nqi':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='nrows', metric_cfg=metric_cfg, thresh=0.90, tag="low_nqi")
            run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col='actual_vals_per_qi', lines_col='nrows', thresh=0.9, tag="low_nqi")
            for ycol in x_y_group:
                run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col=ycol, lines_col='nrows', thresh=0.9, tag="low_nqi")
        elif exp_group == 'agg_dinur_x_nqi_y_noise_lines_nrows_mnr5':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='nrows', metric_cfg=metric_cfg, thresh=0.90, tag="mnr5")
            run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col='actual_vals_per_qi', lines_col='nrows', thresh=0.9, tag="mnr5")
            run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col='actual_vals_per_qi', lines_col='nrows', thresh=0.9, tag="mnr5")
            run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col='num_samples', lines_col='nrows', thresh=0.9, tag="mnr5")
            run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col='mixing_avg', lines_col='nrows', thresh=0.9, tag="mnr5")
        elif exp_group == 'agg_dinur_x_nqi_y_noise_lines_nunique_mnr5':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='nunique', metric_cfg=metric_cfg, thresh=0.90, tag="mnr5")
        elif exp_group == 'agg_dinur_x_nqi_y_noise_lines_vals_per_qi_mnr5':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='actual_vals_per_qi', metric_cfg=metric_cfg, thresh=0.90, tag="mnr5")
        elif exp_group == 'agg_dinur_x_nqi_y_noise_lines_nrows_mnr3':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='nrows', metric_cfg=metric_cfg, thresh=0.90, tag="mnr3")
            run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col='actual_vals_per_qi', lines_col='nrows', thresh=0.9, tag="mnr3")
            run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col='actual_vals_per_qi', lines_col='nrows', thresh=0.9, tag="mnr3")
            for ycol in x_y_group:
                print(f"\n################## Plotting {ycol} vs nqi with nrows lines for mnr3")
                run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col=ycol, lines_col='nrows', thresh=0.9, tag="mnr3")
            plot_mixing_by_param(exp_df, param_col='nrows', tag="mnr3", output_dir=metric_cfg.plots_dir)
        elif exp_group == 'agg_dinur_x_nqi_y_stuff_lines_noise_mnr3':
            for ycol in x_y_group:
                run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col=ycol, lines_col='noise', thresh=0.9, tag="mnr3")
        elif exp_group == 'agg_dinur_x_nqi_y_noise_lines_min_num_rows':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='supp_thresh', metric_cfg=metric_cfg, thresh=0.90)
            for ycol in x_y_group:
                run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col=ycol, lines_col='supp_thresh', thresh=0.9, tag="")
            plot_mixing_by_param(exp_df, param_col='supp_thresh', tag="", output_dir=metric_cfg.plots_dir)
        elif exp_group == 'agg_dinur_x_nqi_y_noise_lines_nunique_mnr3':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='nunique', metric_cfg=metric_cfg, thresh=0.90, tag="mnr3")
            for ycol in x_y_group:
                run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col=ycol, lines_col='nunique', thresh=0.9, tag="mnr3")
            plot_mixing_by_param(exp_df, param_col='nunique', tag="mnr3", output_dir=metric_cfg.plots_dir)
        elif exp_group == 'agg_dinur_x_nqi_y_noise_lines_corr_strength_mnr3':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='corr_strength', metric_cfg=metric_cfg, thresh=0.90, tag="mnr3")
            for ycol in x_y_group:
                run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col=ycol, lines_col='corr_strength', thresh=0.9, tag="mnr3")
        elif exp_group == 'agg_dinur_x_nqi_y_noise_lines_max_qi_mnr3':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='max_qi', metric_cfg=metric_cfg, thresh=0.90, tag="mnr3")
            for ycol in x_y_group:
                run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col=ycol, lines_col='max_qi', thresh=0.9, tag="mnr3")
            plot_mixing_by_param(exp_df, param_col='max_qi', tag="mnr3", output_dir=metric_cfg.plots_dir)
        elif exp_group == 'agg_dinur_x_nqi_y_noise_lines_vals_per_qi_mnr3':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='actual_vals_per_qi', metric_cfg=metric_cfg, thresh=0.90, tag="mnr3")
            for ycol in x_y_group:
                run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col=ycol, lines_col='actual_vals_per_qi', thresh=0.9, tag="mnr3")
            plot_mixing_by_param(exp_df, param_col='actual_vals_per_qi', tag="mnr3", output_dir=metric_cfg.plots_dir)
        elif exp_group == 'agg_dinur_best_case_nrows_nqi3':
            do_analysis_by_x_y_lines(exp_df, x_col='supp_thresh', y_col='noise', lines_col='nrows', metric_cfg=metric_cfg, thresh=0.90)
            run_noise_min_num_rows_table(exp_df, 3, "nqi3", metric_cfg)
        elif exp_group == 'agg_dinur_best_case_nrows_nqi4':
            do_analysis_by_x_y_lines(exp_df, x_col='supp_thresh', y_col='noise', lines_col='nrows', metric_cfg=metric_cfg, thresh=0.90)
            run_noise_min_num_rows_table(exp_df, 4, "nqi4", metric_cfg)
        elif exp_group == 'probe_agg_known':
            metrics = ['measure', 'med_solver_metrics_runtime']
            print_experiment_group_results(exp_df, exp_group, metrics)
        elif exp_group == 'agg_known_best':
            metrics = ['measure', 'med_solver_metrics_runtime']
            print_experiment_group_results(exp_df, exp_group, metrics)
            high_thresh, medium_thresh, low_thresh = heatmap_thresholds_for_metric(metric_cfg.metric_col)
            plot_agg_known_heatmaps(
                exp_df,
                tag="agg_known_best",
                attr_name='measure',
                output_dir=metric_cfg.plots_dir,
                high_thresh=high_thresh,
                medium_thresh=medium_thresh,
                low_thresh=low_thresh,
            )
        elif exp_group == 'agg_known_defaults':
            do_analysis_by_x_y_lines(exp_df, x_col='nqi', y_col='noise', lines_col='known_qi_fraction', metric_cfg=metric_cfg, thresh=0.90, tag="mnr3")
            for ycol in x_y_group:
                run_plot_by_x_y_lines(exp_df, metric_cfg, x_col='nqi', y_col=ycol, lines_col='known_qi_fraction', thresh=0.9, tag="mnr3")
        elif exp_group == 'agg_known_compare':
            compare_agg_known_to_agg_row(exp_df, df_final, metric_cfg)
        else:
            print(f"\n\n{'='*80}")
            print(f"ANALYSIS FOR {exp_group} EXPERIMENT GROUP")
            print(f"{'='*80}")
            analyze_single_parameter_variation(exp_df, experiments, exp_group, metric_cfg)

    output_stem = 'mixing_avg_vs_measure_agg_row' if metric_cfg.metric_col == 'measure' else 'mixing_avg_vs_alc_agg_row'
    plot_mixing_by_measure(
        df_all,
        metric_cfg.plots_dir,
        measure_col='measure',
        metric_label=metric_cfg.metric_label,
        output_stem=output_stem,
    )


def analyze(more_seeds: bool = False):
    """Read result.parquet and analyze correlations with num_samples."""
    df_all = prep_data()
    plot_elapsed_time_pdf(df_all)
    metric_configs = get_metric_configs(df_all)
    for cfg in metric_configs:
        run_metric_analysis(df_all, more_seeds, cfg)

def do_analysis_by_x_y_lines(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    lines_col: str,
    metric_cfg: MetricConfig,
    thresh: float = 0.95,
    tag: str = "",
):
    mapped_thresh = metric_threshold(thresh, metric_cfg)
    print(f"\n\nANALYSIS BY X={x_col}, Y={y_col}, LINES={lines_col}, THRESH={mapped_thresh} ({metric_cfg.metric_label})")
    print("=" * 80)
    # sort by x_col, then y_col, then lines_col, and display x_col, y_col, lines_col, and measure
    df_sorted = df.sort_values(by=['measure', x_col, y_col, lines_col])
    #print(df_sorted[[x_col, y_col, lines_col, 'measure']].to_string())
    exit_reasons_map = {'target_accuracy': 'targ', 
                    'no_more_qi_subsets': 'no_qi', 
                    'max_samples': 'max', }
    # make a table with x_col as rows, lines_col as columns, and exit_reason as values
    # use the exit_reasons_map to shorten the exit_reason values
    use_noise = 2
    # select rows where noise == use_noise
    df_noise = df[df['noise'] == use_noise]
    table = pd.pivot_table(df_noise, values='exit_reason', index=x_col, columns=lines_col, aggfunc=lambda x: ','.join(sorted(set(exit_reasons_map.get(v, v) for v in x))))
    print(f"\nExit reasons table when noise = {use_noise} (x={x_col}, lines={lines_col}):")
    print(table.to_string())

    for thresh in [0.80, 0.90, 0.95]:
        run_plot_by_x_y_lines(
            df,
            metric_cfg,
            x_col=x_col,
            y_col=y_col,
            lines_col=lines_col,
            thresh=thresh,
            tag=tag,
            extra_y_cols=['mixing_avg'],
        )
    run_plot_by_x_y_lines(df, metric_cfg, x_col=x_col, y_col='measure', lines_col=lines_col, thresh=0.9, tag=tag)


def do_pure_dinur_basic_analysis(df, metric_cfg: MetricConfig, experiments=None, exp_group=None):
    print(f"\n\nANALYSIS FOR pure_dinur_basics EXPERIMENT GROUP ({metric_cfg.metric_label})")
    
    # Check for single parameter variation
    if experiments is not None and exp_group is not None:
        analyze_single_parameter_variation(df, experiments, exp_group, metric_cfg)
    
    # Check if num_samples exists
    if 'num_samples' not in df.columns:
        print("\nError: num_samples column not found")
        return
    
    # Check for incomplete jobs (metric < target_accuracy)
    if 'measure' in df.columns and 'target_accuracy' in df.columns:
        incomplete = df[df['measure'] < df['target_accuracy']]
        print(f"\n\nINCOMPLETE JOBS ({metric_cfg.metric_label} < target_accuracy):")
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
        print(f"Complete jobs ({metric_cfg.metric_label} >= target_accuracy): {len(df_complete)}")
    else:
        df_complete = df.copy()
        print(f"Warning: selected metric or target_accuracy column not found, using all data ({metric_cfg.metric_label})")
    
    # Create plots directory
    plots_dir = metric_cfg.plots_dir
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
    if not correlations:
        corr_df = pd.DataFrame(columns=[
            'column',
            'pearson_r',
            'pearson_p',
            'spearman_r',
            'spearman_p',
            'abs_pearson_r',
            'abs_spearman_r',
            'n_samples',
        ])
    else:
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

def do_agg_dinur_basic_analysis(df, metric_cfg: MetricConfig, experiments=None, exp_group=None):
    print(f"\n\nANALYSIS FOR aggregated_dinur_basics EXPERIMENT GROUP ({metric_cfg.metric_label})")
    
    # Check for single parameter variation
    if experiments is not None and exp_group is not None:
        analyze_single_parameter_variation(df, experiments, exp_group, metric_cfg)
    
    # Check if num_samples exists
    if 'num_samples' not in df.columns:
        print("\nError: num_samples column not found")
        return
    
    # Check for incomplete jobs (metric < target_accuracy)
    if 'measure' in df.columns and 'target_accuracy' in df.columns:
        incomplete = df[df['measure'] < df['target_accuracy']]
        print(f"\n\nINCOMPLETE JOBS ({metric_cfg.metric_label} < target_accuracy):")
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
        print(f"Complete jobs ({metric_cfg.metric_label} >= target_accuracy): {len(df_complete)}")
        
        # Get target accuracy for plotting
        target_accuracy = df['target_accuracy'].iloc[0] if len(df) > 0 else 0.99
    else:
        df_complete = df.copy()
        target_accuracy = 0.99
        print(f"Warning: selected metric or target_accuracy column not found, using all data ({metric_cfg.metric_label})")
    
    # Create plots directory
    plots_dir = metric_cfg.agg_plots_dir
    plots_dir.mkdir(exist_ok=True)
    print(f"\nPlots directory: {plots_dir}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Plot measure by nqi (uses all data, not just complete)
    if 'measure' in df.columns and 'nqi' in df.columns:
        plot_measure_by_nqi(
            df,
            plots_dir,
            target_accuracy,
            measure_col='measure',
            metric_label=metric_title(metric_cfg),
            output_stem='measure_by_nqi' if metric_cfg.metric_col == 'measure' else 'alc_by_nqi',
        )
    
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
    if not correlations:
        corr_df = pd.DataFrame(columns=[
            'column',
            'pearson_r',
            'pearson_p',
            'spearman_r',
            'spearman_p',
            'abs_pearson_r',
            'abs_spearman_r',
            'n_samples',
        ])
    else:
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

def do_agg_dinur_explore_vals_per_qi_analysis(df, metric_cfg: MetricConfig, experiments=None, exp_group=None):
    """Analyze agg_dinur_explore_vals_per_qi results with text tables."""
    print(f"\n\nANALYSIS FOR agg_dinur_explore_vals_per_qi EXPERIMENT GROUP ({metric_cfg.metric_label})")
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
        print(f"Table: {metric_title(metric_cfg)} for nrows={nrows_val}")
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
        analyze_single_parameter_variation(df, experiments, exp_group, metric_cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results.")
    parser.add_argument(
        "--more_seeds",
        action="store_true",
        help="Write more_seeds_experiments.py based on seed analysis.",
    )
    args = parser.parse_args()
    analyze(more_seeds=args.more_seeds)
