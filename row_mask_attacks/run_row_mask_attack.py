import pandas as pd
from typing import List, Dict, Set
import numpy as np
import json
import os
import sys
import time
from pathlib import Path
import pprint
import argparse
pp = pprint.PrettyPrinter(indent=2)
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from df_builds.build_row_masks import build_row_masks, build_row_masks_qi, get_required_num_distinct
from reconstruct import reconstruct_by_row, measure_by_row, reconstruct_by_aggregate, measure_by_aggregate, reconstruct_by_aggregate_and_known_qi
from compute_separation import compute_separation_metrics

solve_type_map = {
    'pure_row': 'pr',
    'agg_row': 'ar',
    'agg_known': 'ak',
}
        
def generate_filename(params, target_accuracy) -> str:
    """ Generate a filename string based on attack parameters. """
    vals_per_qi = params['vals_per_qi']
    if params['nqi'] > 0:
        new_vals_per_qi = get_required_num_distinct(params['nrows'], params['nqi'])
        # If it so happens that the actual vals_per_qi is more than what is specified,
        # then we pretend that we are on auto-select so that we don't run extra jobs.
        if new_vals_per_qi > params['vals_per_qi']:
            vals_per_qi = 0
    seed_str = f"_s{params['seed']}" if params['seed'] is not None else ""
    # Only include known_qi_fraction in filename for agg_known solve_type
    kqf_str = ""
    if params['solve_type'] == 'agg_known':
        kqf_str = f"_kqf{int(params['known_qi_fraction']*100)}"
    max_qi = params.get('max_qi', 1000)
    max_qi_str = f"_mq{max_qi}" if max_qi != 1000 else ""
    file_name = (f"nr{params['nrows']}_mf{params['mask_size']}_"
                f"nu{params['nunique']}_qi{params['nqi']}_n{params['noise']}_"
                f"mnr{params['min_num_rows']}_vpq{vals_per_qi}{max_qi_str}_"
                f"st{solve_type_map[params['solve_type']]}_"
                f"ms{params['max_samples']}_ta{int(target_accuracy*100)}{kqf_str}{seed_str}")
    return file_name

def mixing_stats(samples: List[Dict]) -> Dict:
    """ Computes mixing statistics for IDs across samples.

    Mixing is a measure of how many times each pair of IDs appears together in the samples.
    
    Args:
        samples: List of dicts, each containing 'ids' (set of integer IDs)
    
    Returns:
        Dict with 'min', 'max', 'avg', 'stddev', 'median' statistics
    """
    # Track how many times each pair of IDs appears together
    from collections import defaultdict
    pair_counts = defaultdict(int)
    
    for sample in samples:
        ids = list(sample['ids'])
        # Count each unique pair in this sample
        for i, id1 in enumerate(ids):
            for id2 in ids[i+1:]:
                # Use sorted tuple to ensure consistent pair representation
                pair = tuple(sorted([id1, id2]))
                pair_counts[pair] += 1
    
    # Get all count values
    counts = list(pair_counts.values())
    
    if len(counts) == 0:
        return {
            'min': 0,
            'max': 0,
            'avg': 0.0,
            'stddev': 0.0,
            'median': 0.0
        }
    
    return {
        'min': int(np.min(counts)),
        'max': int(np.max(counts)),
        'avg': float(np.mean(counts)),
        'stddev': float(np.std(counts)),
        'median': float(np.median(counts))
    }

def get_qi_subset_list(df: pd.DataFrame, min_num_rows: int, target_num_rows: int, max_qi: int = 1000) -> List[Dict]:
    """ Generates list of QI column subsets grouped by qi_cols.
    
    Args:
        df: DataFrame with QI columns (qi0, qi1, ..., qiN)
        min_num_rows: Minimum number of rows in any aggregate (default: 5)
        target_num_rows: Target rows for sorting subsets
        max_qi: Maximum subset size to consider
    
    Returns:
        List of subsets sorted by groups

    Operation:
    get_qi_subset_list() groups subsets by their qi_cols, then splits those groups into “valid” (every subset in the group has num_rows >= target_num_rows) and “invalid”. It sorts valid groups by max_num_rows ascending, then invalid groups by max_num_rows descending, and concatenates valid then invalid. The effect is: first try the most consistently small-but-acceptable groups, and only after that fall back to larger/less-consistent groups, with the biggest invalid groups first.
    """
    import itertools
    
    # Find all QI columns
    all_qi_cols = sorted([col for col in df.columns if col.startswith('qi')])
    
    if len(all_qi_cols) == 0:
        return []
    
    qi_subsets = []
    
    # Iterate through subset sizes from 1 to min(nqi-1, max_qi)
    # (nqi columns would only have 1 row per combination since all combos are unique)
    num_subsets = 0
    max_subset_size = min(len(all_qi_cols) - 1, max_qi)
    if max_subset_size < 1:
        return []
    for subset_size in range(1, max_subset_size + 1):
        if num_subsets >= 20000:
            break
        # Get all combinations of qi columns of this size
        for qi_cols in itertools.combinations(all_qi_cols, subset_size):
            qi_cols = list(qi_cols)
            
            # Get all unique combinations of values for these columns
            qi_combinations = df[qi_cols].drop_duplicates()
            
            # For each combination of values, count matching rows
            for _, combo_row in qi_combinations.iterrows():
                qi_vals = [combo_row[col] for col in qi_cols]
                
                # Create boolean mask for rows matching this combination
                mask = pd.Series([True] * len(df))
                for col, val in zip(qi_cols, qi_vals):
                    mask &= (df[col] == val)
                
                num_rows = mask.sum()
                
                # Only include if meets minimum threshold
                if num_rows >= min_num_rows:
                    qi_subsets.append({
                        'qi_cols': qi_cols,
                        'qi_vals': [int(val) for val in qi_vals],
                        'num_rows': int(num_rows)
                    })
                    num_subsets += 1
    
    # Group subsets by qi_cols
    from collections import defaultdict
    groups_dict = defaultdict(list)
    
    for subset in qi_subsets:
        # Use tuple of qi_cols as key for grouping
        key = tuple(subset['qi_cols'])
        groups_dict[key].append(subset)
    
    # Create qi_groups list
    qi_groups = []
    for qi_cols_tuple, subsets in groups_dict.items():
        num_rows_values = [s['num_rows'] for s in subsets]
        qi_groups.append({
            'qi_cols': list(qi_cols_tuple),
            'subsets': subsets,
            'max_num_rows': max(num_rows_values),
            'min_num_rows': min(num_rows_values)
        })
    
    # Split into two groups based on min_num_rows threshold
    valid_groups = [g for g in qi_groups if g['min_num_rows'] >= target_num_rows]
    invalid_groups = [g for g in qi_groups if g['min_num_rows'] < target_num_rows]
    
    # Sort valid groups by max_num_rows (ascending)
    valid_groups.sort(key=lambda x: x['max_num_rows'])
    
    # Sort invalid groups by max_num_rows (descending)
    invalid_groups.sort(key=lambda x: x['max_num_rows'], reverse=True)
    
    # Flatten groups into sorted_qi_subsets
    sorted_qi_subsets = []
    for group in valid_groups:
        sorted_qi_subsets.extend(group['subsets'])
    for group in invalid_groups:
        sorted_qi_subsets.extend(group['subsets'])
    
    return sorted_qi_subsets

def get_qi_subsets_mask(df: pd.DataFrame, qi_subsets: List[Dict], index: int) -> Set[int]:
    """ Returns set of IDs matching the QI subset at the specified index.
    
    Args:
        df: DataFrame with 'id' column and QI columns
        qi_subsets: List of QI subsets from get_qi_subset_list()
        index: Index into qi_subsets
    
    Returns:
        Set of ID values matching the QI columns and values at the specified index
    """
    if index < 0 or index >= len(qi_subsets):
        raise ValueError(f"Index {index} out of range [0, {len(qi_subsets)-1}]")
    
    subset = qi_subsets[index]
    qi_cols = subset['qi_cols']
    qi_vals = subset['qi_vals']
    
    # Create boolean mask for rows matching this combination
    mask = pd.Series([True] * len(df))
    for col, val in zip(qi_cols, qi_vals):
        mask &= (df[col] == val)
    
    # Return set of matching IDs
    return set(df[mask]['id'].values)

def prior_job_results(file_path: Path) -> Dict:
    """Load prior job results from file if it exists.
    
    Args:
        file_path: Path to JSON file with previous results
    
    Returns:
        List of attack results if file exists, None otherwise
    """
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading prior results from {file_path}: {e}")
        return None

def initialize_samples(df, mask_size, nunique, noise):
    # Create initial bins where each ID appears in exactly one bin
    # This avoids the issue of some IDs never being sampled
    all_ids = list(df['id'].values)
    np.random.shuffle(all_ids)
    
    num_bins = len(all_ids) // mask_size
    remainder = len(all_ids) % mask_size
    
    # First, create all bins with their IDs
    bins = []
    for i in range(num_bins):
        bin_ids = set(all_ids[i * mask_size:(i + 1) * mask_size])
        bins.append(bin_ids)
    
    # Distribute remaining IDs among existing bins
    if remainder > 0:
        remaining_ids = all_ids[num_bins * mask_size:]
        for i, remaining_id in enumerate(remaining_ids):
            bin_idx = i % num_bins
            bins[bin_idx].add(remaining_id)
    
    # Now loop through bins to add noise and create samples
    initial_samples = []
    for bin_ids in bins:
        # Get exact counts for each value in this bin
        bin_df = df[df['id'].isin(bin_ids)]
        exact_counts = bin_df['val'].value_counts().to_dict()
        
        # Add noise to counts
        noisy_counts = []
        for val in range(nunique):
            exact_count = exact_counts.get(val, 0)
            noise_delta = np.random.randint(-noise, noise + 1)
            noisy_count = max(0, exact_count + noise_delta)
            noisy_counts.append({'val': val, 'count': noisy_count})
        
        initial_samples.append({
            'ids': bin_ids,
            'noisy_counts': noisy_counts
        })
    return initial_samples

def initialize_qi_samples(df: pd.DataFrame, nunique: int, noise: int, qi_subsets: List[Dict]) -> tuple[List[Dict], int]:
    """Create initial samples from QI subsets ensuring all IDs appear at once.
    
    Args:
        df: DataFrame with 'id' and 'val' columns
        nunique: Number of unique values
        noise: Noise bound for counts (±noise)
        qi_subsets: List of QI subsets from get_qi_subset_list()
    
    Returns:
        Tuple of (initial_samples, next_qi_index)
        - initial_samples: List of sample dicts
        - next_qi_index: Index of next unused subset in qi_subsets
    """
    initial_samples = []
    all_ids = set(df['id'].values)
    covered_ids = set()
    qi_index = 0
    
    # Loop through qi_subsets until all IDs are covered
    while len(covered_ids) < len(all_ids) and qi_index < len(qi_subsets):
        # Get masked IDs for this subset
        masked_ids = get_qi_subsets_mask(df, qi_subsets, qi_index)
        qi_cols = qi_subsets[qi_index]['qi_cols']
        qi_vals = qi_subsets[qi_index]['qi_vals']
        
        # Get exact counts for each value in the masked subset
        masked_df = df[df['id'].isin(masked_ids)]
        exact_counts = masked_df['val'].value_counts().to_dict()
        
        # Add noise to counts
        noisy_counts = []
        for val in range(nunique):
            exact_count = exact_counts.get(val, 0)
            noise_delta = np.random.randint(-noise, noise + 1)
            noisy_count = max(0, exact_count + noise_delta)
            noisy_counts.append({'val': val, 'count': noisy_count})
        
        # Add sample
        initial_samples.append({
            'ids': masked_ids,
            'qi_cols': qi_cols,
            'qi_vals': qi_vals,
            'noisy_counts': noisy_counts
        })
        
        # Update covered IDs
        covered_ids.update(masked_ids)
        qi_index += 1
    
    return initial_samples, qi_index

def get_best_refine(attack_results: List[Dict]) -> int:
    """Returns the highest integer refine value from attack_results."""
    best_refine = -1
    for entry in attack_results:
        refine = entry.get('refine')
        if isinstance(refine, int) and refine > best_refine:
            best_refine = refine
    return best_refine

def attack_loop(nrows: int, 
                nunique: int, 
                mask_size: int, 
                noise: int,
                nqi: int = 3,
                target_accuracy: float = 0.99,
                min_num_rows: int = 3,
                vals_per_qi: int = 2,
                max_samples: int = 20000,
                solve_type: str = 'agg_known',
                known_qi_fraction: float = 1.0,
                max_qi: int = 1000,
                max_refine: int = 2,
                seed: int = None,
                output_file: Path = None,
                cur_attack_results: List[Dict] = None) -> dict:
    """ Runs an iterative attack loop to reconstruct values from noisy samples.
    
    Args:
        nrows: Number of rows in the dataframe
        nunique: Number of unique values
        mask_size: Number of rows in each random sample (pure Dinur style only)
        noise: Noise bound for counts (±noise)
        nqi: Number of quasi-identifier columns
        vals_per_qi: Number of distinct values per QI column (default: 0, means auto compute)
        target_accuracy: Target accuracy to stop early (default: 0.99)
        min_num_rows: Minimum number of rows in any aggregate (default: 3)
        known_qi_fraction: Fraction of rows with known QI values (default: 0.0, range: 0.0-1.0)
        max_qi: Maximum subset size to consider for aggregate queries (default: 1000)
        max_refine: Maximum number of refinement iterations (default: 2)
        seed: Random seed for reproducibility (default: None)
        output_file: Path to JSON file to save results incrementally (default: None)
        cur_attack_results: Previous attack results to resume from (default: None)
    
    Returns:
        dict
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Check if we're resuming from prior results
    if cur_attack_results is not None and len(cur_attack_results) > 0:
        last_result = cur_attack_results[-1]
        
        # If already achieved target accuracy, do nothing
        refine_count = get_best_refine(cur_attack_results)
        if refine_count >= max_refine:
            print(f"Prior results already achieved target accuracy: {last_result['measure']:.4f}")
            return {}

        # Resume from prior results
        print(f"Resuming from prior results with {last_result['num_samples']} samples, accuracy: {last_result['measure']:.4f}")
        results = cur_attack_results
        if last_result['measure'] >= target_accuracy:
            current_num_samples = last_result['num_samples'] -1  # -1 for the all-IDs sample
        else:
            current_num_samples = last_result['num_samples'] * 2
    else:
        # Starting fresh
        results = []
        current_num_samples = 0
        refine_count = 0
    
    save_dict = {}
    # Start timing
    start_time = time.time()
    has_target_accuracy = False

    def get_refine_bounds():
        success_entries = [entry for entry in results if entry['measure'] >= target_accuracy]
        if not success_entries:
            return None, None
        best_success = min(success_entries, key=lambda entry: entry['num_samples'])
        failure_entries = [
            entry for entry in results
            if entry['measure'] < target_accuracy and entry['num_samples'] < best_success['num_samples']
        ]
        if not failure_entries:
            return None, None
        best_failure = max(failure_entries, key=lambda entry: entry['num_samples'])
        return best_failure, best_success
    
    actual_vals_per_qi = None
    # Build the ground truth dataframe
    if solve_type == 'pure_row':
        df = build_row_masks(nrows=nrows, nunique=nunique)
    else:
        min_vals_per_qi = get_required_num_distinct(nrows, nqi)
        # If it so happens that the actual vals_per_qi is more than what is specified,
        # then we pretend that we are on auto-select so that we don't run extra jobs.
        if min_vals_per_qi > vals_per_qi:
            vals_per_qi = 0
            actual_vals_per_qi = min_vals_per_qi
        else:
            actual_vals_per_qi = vals_per_qi
        df = build_row_masks_qi(nrows=nrows, nunique=nunique, nqi=nqi, vals_per_qi=vals_per_qi)
    
    # Generate complete_known_qi_rows if known_qi_fraction > 0
    complete_known_qi_rows = []
    if known_qi_fraction > 0.0 and solve_type == 'agg_known':
        num_known_qi_rows = int(round(len(df) * known_qi_fraction))
        if num_known_qi_rows > 0:
            # Select random rows
            known_indices = np.random.choice(len(df), size=num_known_qi_rows, replace=False)
            all_qi_cols = [col for col in df.columns if col.startswith('qi')]
            
            # Extract QI columns only (no IDs)
            for idx in known_indices:
                row = df.iloc[idx]
                known_qi_row = {col: int(row[col]) for col in all_qi_cols}
                complete_known_qi_rows.append(known_qi_row)

    
    print(f"Total known QI rows: {len(complete_known_qi_rows)}")
    working_samples = []
    num_masked = None
    qi_index = 0
    qi_subsets = []
    all_qi_cols = [col for col in df.columns if col.startswith('qi')]
    if solve_type in ['agg_row', 'agg_known']:
        qi_subsets = get_qi_subset_list(df, min_num_rows, int(round(min_num_rows * nunique * 1.5)), max_qi)
        working_samples, qi_index = initialize_qi_samples(df, nunique, noise, qi_subsets)
        print(f"Total QI subsets available: {len(qi_subsets)}. qi_index {qi_index}.")
    else:
        working_samples = initialize_samples(df, mask_size, nunique, noise)
        print(f"start with {len(working_samples)} initial samples")
        num_masked = mask_size
    if current_num_samples == 0:
        current_num_samples = len(working_samples)
    
    num_suppressed = 0
    while True:
        current_refine = refine_count
        # Start with initial binned samples, if any
        samples = working_samples.copy()
        avg_num_masked = 0

        if current_num_samples < len(working_samples):
            samples = working_samples[:current_num_samples]
        else:
            for _ in range(current_num_samples - len(working_samples)):
                # Select random subset of IDs
                qi_cols = []
                qi_vals = []
                if solve_type == 'pure_row':
                    masked_ids = set(np.random.choice(df['id'].values, size=num_masked, replace=False))
                else:
                    if qi_index >= len(qi_subsets):
                        print(f"Exhausted QI subsets at index {qi_index}")
                        break
                    masked_ids = get_qi_subsets_mask(df, qi_subsets, qi_index)
                    avg_num_masked += len(masked_ids)
                    qi_cols = qi_subsets[qi_index]['qi_cols']
                    qi_vals = qi_subsets[qi_index]['qi_vals']
                    qi_index += 1
                
                # Get exact counts for each value in the masked subset
                masked_df = df[df['id'].isin(masked_ids)]
                exact_counts = masked_df['val'].value_counts().to_dict()
                
                # Add noise to counts
                noisy_counts = []
                for val in range(nunique):
                    exact_count = exact_counts.get(val, 0)
                    if exact_count < min_num_rows:
                        num_suppressed += 1
                        continue
                    noise_delta = np.random.randint(-noise, noise + 1)
                    noisy_count = max(0, exact_count + noise_delta)
                    noisy_counts.append({'val': val, 'count': noisy_count})
                
                if len(noisy_counts) == 0:
                    # No counts above min_num_rows, skip this sample
                    continue

                # Add sample
                samples.append({
                    'ids': masked_ids,
                    'qi_cols': qi_cols,            # for agg_known attacks
                    'qi_vals': qi_vals,            # for agg_known attacks
                    'noisy_counts': noisy_counts
                })
        
        # Reconstruct and measure
        print(f"Begin {solve_type} reconstruction with {len(samples)} samples\n    (current_num_samples={current_num_samples}, working_samples={len(working_samples)}, qi_index={qi_index}, num_suppressed={num_suppressed})")
        qi_match_accuracy = 0.0
        if solve_type in ['pure_row', 'agg_row']:
            reconstructed, num_equations, solver_metrics = reconstruct_by_row(samples, noise, seed)
            accuracy = measure_by_row(df, reconstructed)
        elif solve_type == 'agg_known':
            if known_qi_fraction == 1.0:
                reconstructed, num_equations, solver_metrics = reconstruct_by_row(samples, noise, seed)
                accuracy = measure_by_row(df, reconstructed)
                qi_match_accuracy = 1.0
            else:
                # Filter complete_known_qi_rows to only those appearing in at least one sample
                known_qi_rows = []
                for known_qi_row in complete_known_qi_rows:
                    # Check if this known_qi_row matches any sample
                    for sample in samples:
                        if 'qi_cols' in sample and 'qi_vals' in sample:
                            # Check if any qi_cols in the sample match the known_qi_row
                            match = False
                            for col, val in zip(sample['qi_cols'], sample['qi_vals']):
                                if known_qi_row.get(col) == val:
                                    match = True
                                    break
                            if match:
                                # This known_qi_row is covered by at least one sample
                                known_qi_rows.append(known_qi_row)
                                break
                
                if (len(known_qi_rows) != len(complete_known_qi_rows)):
                    # throw exception
                    print("Samples:")
                    pp.pprint(samples)
                    print("Complete known QI rows:")
                    pp.pprint(complete_known_qi_rows)
                    print("Filtered known QI rows:")
                    pp.pprint(known_qi_rows)
                    raise ValueError(f"Known QI rows used in reconstruction ({len(known_qi_rows)}) does not match total known QI rows ({len(complete_known_qi_rows)})")
                reconstructed, num_equations, solver_metrics = reconstruct_by_aggregate_and_known_qi(samples, noise, nrows, all_qi_cols, complete_known_qi_rows, seed)
                accuracy_measure = measure_by_aggregate(df, reconstructed)
                accuracy = accuracy_measure['qi_and_val_match']
                qi_match_accuracy = accuracy_measure['qi_match']
        else:
            raise ValueError(f"Unsupported solve_type: {solve_type}")
        mixing = mixing_stats(samples)
        sep = compute_separation_metrics(samples)
        if accuracy >= target_accuracy:
            has_target_accuracy = True

        if solve_type in ['agg_row', 'agg_known']:
            num_masked = int(avg_num_masked / (len(samples) - len(working_samples))) if (len(samples) - len(working_samples)) > 0 else 0
            working_samples = samples.copy()
        
        # Record results
        current_result = {
            'num_samples': len(samples),
            'num_equations': num_equations,
            'measure': accuracy,
            'qi_match_measure': qi_match_accuracy,
            'mixing': mixing,
            'actual_num_rows': num_masked,
            'solver_metrics': solver_metrics,
            'separation': sep,
            'refine': current_refine,
        }
        results.append(current_result)
        if current_refine > 0:
            best_failure, best_success = get_refine_bounds()
            if best_failure is None or best_success is None:
                raise ValueError("Refinement requires both success and failure results.")
            remainder = [entry for entry in results if entry is not best_failure and entry is not best_success]
            results[:] = remainder + [best_failure, best_success]
        pp.pprint(current_result)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        finished = False
        exit_reason = ''

        # Check stopping conditions
        if nqi > 0 and qi_index >= len(qi_subsets):
            print("Exit loop: No more QI subsets to use")
            finished = True
            exit_reason = 'no_more_qi_subsets'

        if not finished:
            if has_target_accuracy:
                best_failure, best_success = get_refine_bounds()
                can_refine = (
                    best_failure is not None
                    and best_success is not None
                    and current_refine < max_refine
                )
                if can_refine:
                    if best_failure['num_samples'] >= best_success['num_samples']:
                        print("Exit loop: Refinement interval invalid (failure >= success)")
                        finished = True
                        exit_reason = 'target_accuracy'
                    else:
                        midpoint = int((best_failure['num_samples'] + best_success['num_samples']) / 2)
                        print(f"Refining: New midpoint = {midpoint} (between {best_failure['num_samples']} and {best_success['num_samples']})")
                        print(f"Initial samples count: {len(working_samples)}")
                        if midpoint <= best_failure['num_samples'] or midpoint >= best_success['num_samples']:
                            print("Exit loop: Refinement interval collapsed")
                            finished = True
                            exit_reason = 'target_accuracy'
                        else:
                            refine_count = current_refine + 1
                            current_num_samples = midpoint
                else:
                    print(f"Exit loop: Target accuracy {target_accuracy} achieved: {accuracy:.4f}")
                    finished = True
                    exit_reason = 'target_accuracy'
            else:
                next_samples = len(samples) * 2
                if nqi == 0 and next_samples > max_samples:
                    print(f"Exit loop: Reached max samples limit: {max_samples}")
                    exit_reason = 'max_samples'
                    finished = True
                else:
                    current_num_samples = next_samples
                    refine_count = 0

        # Save results incrementally if output file is provided
        save_dict = {
            'solve_type': solve_type,
            'nrows': nrows,
            'mask_size': mask_size,
            'nunique': nunique,
            'noise': noise,
            'nqi': nqi,
            'vals_per_qi': vals_per_qi,
            'actual_vals_per_qi': actual_vals_per_qi,
            'known_qi_fraction': known_qi_fraction,
            'max_qi': max_qi,
            'seed': seed,
            'max_samples': max_samples,
            'target_accuracy': target_accuracy,
            'min_num_rows': min_num_rows,
            'elapsed_time': elapsed_time,
            'finished': finished,
            'exit_reason': exit_reason,
            'num_suppressed': num_suppressed,
            'attack_results': results,
        }
        if output_file is not None:
            with open(output_file, 'w') as f:
                json.dump(save_dict, f, indent=2)
            
        if finished:
            break
    return save_dict
        

def main():
    """Main function to run parameter sweep experiments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run row mask attack experiments')
    parser.add_argument('job_num', type=int, nargs='?', default=None,
                       help='Job number to run from parameter combinations')
    parser.add_argument('--solve_type', type=str, default=None,
                       help='Type of solve: pure_row, agg_row, agg_known')
    parser.add_argument('--nrows', type=int, default=None,
                       help='Number of rows')
    parser.add_argument('--mask_size', type=int, default=None,
                       help='Number of rows in each random sample')
    parser.add_argument('--nunique', type=int, default=None,
                       help='Number of unique values')
    parser.add_argument('--noise', type=int, default=None,
                       help='Noise bound')
    parser.add_argument('--nqi', type=int, default=None,
                       help='Number of quasi-identifier columns')
    parser.add_argument('--min_num_rows', type=int, default=None,
                       help='Minimum number of rows in any aggregate')
    parser.add_argument('--vals_per_qi', type=int, default=None,
                       help='Number of distinct values per QI column')
    parser.add_argument('--known_qi_fraction', type=float, default=None,
                       help='Fraction of rows with known QI values (0.0-1.0)')
    parser.add_argument('--max_qi', type=int, default=None,
                       help='Maximum subset size for aggregate queries')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use before quitting')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create directories
    results_dir = Path('./results/files')
    attack_results_dir = results_dir
    slurm_out_dir = Path('./slurm_out')
    results_dir.mkdir(exist_ok=True)
    attack_results_dir.mkdir(exist_ok=True)
    slurm_out_dir.mkdir(exist_ok=True)
    
    # Read in the experiments data structure from experiments.py
    from experiments import read_experiments
    experiments = read_experiments(include_more_seeds_experiments=True)
    
    # Fixed parameters
    max_samples = 20000
    target_accuracy = 0.99
    
    # Defaults
    defaults = {
        'solve_type': 'agg_known',
        'nrows': 100,
        'mask_size': 20,
        'nunique': 2,
        'noise': 2,
        'nqi': 3,
        'min_num_rows': 3,
        'vals_per_qi': 2,
        'known_qi_fraction': 1.0,
        'max_qi': 1000,
        'max_samples': max_samples,
        'seed': None,
    }
    
    # Check if any individual parameters were provided
    individual_params_provided = any([
        args.solve_type is not None,
        args.nrows is not None,
        args.mask_size is not None,
        args.nunique is not None,
        args.noise is not None,
        args.nqi is not None,
        args.min_num_rows is not None,
        args.vals_per_qi is not None,
        args.known_qi_fraction is not None,
        args.max_qi is not None,
        args.max_samples is not None,
        args.seed is not None
    ])
    
    if individual_params_provided:
        # Use command line parameters, falling back to defaults
        params = {
            'nrows': args.nrows if args.nrows is not None else defaults['nrows'],
            'solve_type': args.solve_type if args.solve_type is not None else defaults['solve_type'],
            'mask_size': args.mask_size if args.mask_size is not None else defaults['mask_size'],
            'nunique': args.nunique if args.nunique is not None else defaults['nunique'],
            'noise': args.noise if args.noise is not None else defaults['noise'],
            'nqi': args.nqi if args.nqi is not None else defaults['nqi'],
            'min_num_rows': args.min_num_rows if args.min_num_rows is not None else defaults['min_num_rows'],
            'vals_per_qi': args.vals_per_qi if args.vals_per_qi is not None else defaults['vals_per_qi'],
            'known_qi_fraction': args.known_qi_fraction if args.known_qi_fraction is not None else defaults['known_qi_fraction'],
            'max_qi': args.max_qi if args.max_qi is not None else defaults['max_qi'],
            'max_samples': args.max_samples if args.max_samples is not None else defaults['max_samples'],
            'seed': args.seed if args.seed is not None else defaults['seed'],
        }

        file_name = generate_filename(params, target_accuracy)
        file_path = attack_results_dir / f"{file_name}.json"
        
        # Load prior results if they exist
        cur_attack_results = prior_job_results(file_path)
        cur_attack_results_list = None
        if cur_attack_results is not None:
            cur_attack_results_list = cur_attack_results['attack_results']
            if cur_attack_results['finished'] is True:
                print(f"Attack already finished for parameters: {params}. Results in {file_path}")
                return
        
        # Run attack_loop
        print(f"Running with parameters: {params}")
        
        attack_loop(
            nrows=params['nrows'],
            nunique=params['nunique'],
            mask_size=params['mask_size'],
            noise=params['noise'],
            nqi=params['nqi'],
            max_samples=max_samples,
            target_accuracy=target_accuracy,
            min_num_rows=params['min_num_rows'],
            vals_per_qi=params['vals_per_qi'],
            known_qi_fraction=params['known_qi_fraction'],
            max_qi=params['max_qi'],
            solve_type=params['solve_type'],
            seed=params['seed'],
            output_file=file_path,
            cur_attack_results=cur_attack_results_list,
        )
        
        # Read back the saved file to get the final elapsed time
        with open(file_path, 'r') as f:
            final_results = json.load(f)
        
        print("Parameters:")
        pp.pprint(params)
        print(f"Results saved to {file_path}")
        print(f"Elapsed time: {final_results['elapsed_time']:.2f} seconds")
        print(f"Final accuracy: {final_results['attack_results'][-1]['measure']:.4f}")
        print(f"Samples used: {final_results['attack_results'][-1]['num_samples']}")
        
        return
    
    # Generate test parameter combinations
    max_time_minutes = 60 * 2     # We'll set slurm to this
    # 2 minutes for overhead, convert to seconds, then divide by 20 to safely allow for multiple runs
    time_include_threshold_seconds =  ((max_time_minutes-2) * 60) / 20
    test_params = []
    
    seen = set()

    finished_param_keys = set()
    result_parquet = Path('./results/result.parquet')
    if result_parquet.exists():
        try:
            results_df = pd.read_parquet(result_parquet)
        except Exception as e:
            print(f"Warning: could not read {result_parquet}: {e}")
        else:
            param_cols = [
                'nrows',
                'solve_type',
                'mask_size',
                'nunique',
                'noise',
                'nqi',
                'min_num_rows',
                'vals_per_qi',
                'known_qi_fraction',
                'max_qi',
                'max_samples',
                'seed',
                'target_accuracy',
            ]
            missing_cols = [col for col in param_cols + ['finished'] if col not in results_df.columns]
            if 'max_qi' in missing_cols:
                results_df['max_qi'] = defaults['max_qi']
                missing_cols.remove('max_qi')
            if missing_cols:
                print(f"Warning: {result_parquet} missing columns {missing_cols}; skipping finished filter.")
            else:
                int_cols = {
                    'nrows',
                    'mask_size',
                    'nunique',
                    'noise',
                    'nqi',
                    'min_num_rows',
                    'vals_per_qi',
                    'max_qi',
                    'max_samples',
                    'seed',
                }
                results_df.loc[results_df['solve_type'] != 'agg_known', 'known_qi_fraction'] = 1.0
                finished_df = results_df[results_df['finished'] == True]
                print(f"Found {len(finished_df)} finished jobs in prior results.")
                overtime_df = pd.DataFrame()
                if 'final_attack' in results_df.columns and 'solver_metrics_runtime' in results_df.columns:
                    overtime_df = results_df[
                        (results_df['finished'] == False)
                        & (results_df['final_attack'] == True)
                        & (results_df['solver_metrics_runtime'] > time_include_threshold_seconds)
                    ]
                print(f"Including {len(overtime_df)} additional jobs to filter based on time threshold of {time_include_threshold_seconds} seconds.")
                combined_df = pd.concat([finished_df, overtime_df], ignore_index=True)
                for _, row in combined_df.iterrows():
                    row_params = {}
                    for col in param_cols:
                        val = row[col]
                        if pd.isna(val):
                            row_params[col] = defaults.get(col)
                        elif col in int_cols:
                            row_params[col] = int(val)
                        elif col in ['known_qi_fraction', 'target_accuracy']:
                            row_params[col] = float(val)
                        else:
                            row_params[col] = val
                    file_name = generate_filename(row_params, row_params['target_accuracy'])
                    finished_param_keys.add(file_name)
    
    num_finished_jobs = 0
    for key in finished_param_keys:
        #print(f"Finished: {key}")
        pass
    for exp in experiments:
        if exp['dont_run'] is True:
            continue
        # Get seed list from experiment, default to [None] if not specified
        seed_list = exp.get('seed', [None])
        # Get known_qi_fraction list from experiment, default to [1.0] if not specified
        known_qi_fraction_list = exp.get('known_qi_fraction', [1.0])
        max_qi_list = exp.get('max_qi', [defaults['max_qi']])
        for nrows in exp['nrows']:
            for mask_size in exp['mask_size']:
                for nunique in exp['nunique']:
                    for noise in exp['noise']:
                        for nqi in exp['nqi']:
                            for max_qi in max_qi_list:
                                for min_num_rows in exp['min_num_rows']:
                                    for vals_per_qi in exp['vals_per_qi']:
                                        for known_qi_fraction in known_qi_fraction_list:
                                            for seed in seed_list:
                                                params = {
                                                    'nrows': nrows,
                                                    'solve_type': exp['solve_type'],
                                                    'mask_size': mask_size,
                                                    'nunique': nunique,
                                                    'noise': noise,
                                                    'nqi': nqi,
                                                    'min_num_rows': min_num_rows,
                                                    'vals_per_qi': vals_per_qi,
                                                    'known_qi_fraction': known_qi_fraction,
                                                    'max_qi': max_qi,
                                                    'max_samples': max_samples,
                                                    'seed': seed,
                                                }
                                                key = generate_filename(params, target_accuracy)
                                                if key in finished_param_keys:
                                                    num_finished_jobs += 1
                                                    #print(f"Matched: {key}")
                                                    continue
                                                if key not in seen:
                                                    seen.add(key)
                                                    #print(f"Adding: {key}")
                                                    test_params.append(params)
    
    # If no job_num, just print all combinations
    if args.job_num is None:
        for i, params in enumerate(test_params):
            print(f"Job {i}: {params}")
        
        # Create run.slurm file
        num_jobs = len(test_params) - 1  # Array range is 0 to num_jobs-1
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name=recon_test
#SBATCH --output=/INS/syndiffix/work/paul/github/reconstruction_tests/row_mask_attacks/slurm_out/job_%A_%a.out
#SBATCH --time={max_time_minutes}
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{num_jobs}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
source /INS/syndiffix/work/paul/github/reconstruction_tests/.venv/bin/activate
python /INS/syndiffix/work/paul/github/reconstruction_tests/row_mask_attacks/run_row_mask_attack.py $arrayNum
"""
        
        slurm_file = Path(__file__).parent / 'run.slurm'
        with open(slurm_file, 'w') as f:
            f.write(slurm_content)
        
        print(f"\nSLURM file created: {slurm_file}")
        print(f"Total jobs: {len(test_params)} (array: 0-{num_jobs}) out of which {num_finished_jobs} are already finished.")
        return
    
    # Check if job_num is valid
    if args.job_num < 0 or args.job_num >= len(test_params):
        print(f"Error: job_num {args.job_num} out of range [0, {len(test_params)-1}]")
        return
    
    # Get parameters for this job
    params = test_params[args.job_num]
    
    # Generate filename
    file_name = generate_filename(params, target_accuracy)
    file_path = attack_results_dir / f"{file_name}.json"
    
    # Load prior results if they exist
    cur_attack_results = prior_job_results(file_path)
    cur_attack_results_list = None
    if cur_attack_results is not None:
        cur_attack_results_list = cur_attack_results['attack_results']
        if cur_attack_results['finished'] is True:
            print(f"Attack already finished for parameters: {params}. Results in {file_path}")
            return
    
    # Run attack_loop
    print(f"Running job {args.job_num}: {params}")
    
    attack_loop(
        nrows=params['nrows'],
        nunique=params['nunique'],
        mask_size=params['mask_size'],
        noise=params['noise'],
        nqi=params['nqi'],
        target_accuracy=target_accuracy,
        min_num_rows=params['min_num_rows'],
        vals_per_qi=params['vals_per_qi'],
        known_qi_fraction=params['known_qi_fraction'],
        max_qi=params['max_qi'],
        max_samples=params['max_samples'],
        solve_type=params['solve_type'],
        seed=params['seed'],
        output_file=file_path,
        cur_attack_results=cur_attack_results_list,
    )
    
    # Read back the saved file to get the final elapsed time
    with open(file_path, 'r') as f:
        final_results = json.load(f)
    
    print("Parmeters:")
    pp.pprint(params)
    print(f"Results saved to {file_path}")
    print(f"Elapsed time: {final_results['elapsed_time']:.2f} seconds")
    print(f"Final accuracy: {final_results['attack_results'][-1]['measure']:.4f}")
    print(f"Samples used: {final_results['attack_results'][-1]['num_samples']}")

if __name__ == '__main__':
    main()
