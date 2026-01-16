import pandas as pd
from typing import List, Dict, Set
import numpy as np
import argparse
import json
import sys
from pathlib import Path
import pprint
pp = pprint.PrettyPrinter(indent=2)
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from df_builds.build_row_masks import build_row_masks, build_row_masks_qi, get_required_num_distinct
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
    file_name = (f"nr{params['nrows']}_mf{params['mask_size']}_"
                f"nu{params['nunique']}_qi{params['nqi']}_n{params['noise']}_"
                f"mnr{params['min_num_rows']}_vpq{vals_per_qi}_"
                f"st{solve_type_map[params['solve_type']]}_"
                f"ms{params['max_samples']}_ta{int(target_accuracy*100)}{kqf_str}{seed_str}")
    return file_name

def get_qi_subset_list(df: pd.DataFrame, min_num_rows: int, target_num_rows: int) -> List[Dict]:
    """ Generates list of QI column subsets grouped by qi_cols.
    
    Args:
        df: DataFrame with QI columns (qi0, qi1, ..., qiN)
        min_num_rows: Minimum number of rows in any aggregate (default: 5)
        target_num_rows: Target rows for sorting subsets
    
    Returns:
        List of subsets sorted by groups
    """
    import itertools
    
    # Find all QI columns
    all_qi_cols = sorted([col for col in df.columns if col.startswith('qi')])
    
    if len(all_qi_cols) == 0:
        return []
    
    qi_subsets = []
    
    # Iterate through subset sizes from 1 to nqi-1
    # (nqi columns would only have 1 row per combination since all combos are unique)
    num_subsets = 0
    for subset_size in range(1, len(all_qi_cols)):
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

def add_additional_samples(samples: List[Dict], df: pd.DataFrame, nunique: int, noise: int, num_new_samples: int, solve_type: str, qi_subsets: List[Dict], qi_index: int, num_masked: int, min_num_rows: int) -> List[Dict]:        
    for _ in range(num_new_samples):
        # Select random subset of IDs
        qi_cols = []
        qi_vals = []
        if solve_type == 'pure_row':
            masked_ids = set(np.random.choice(df['id'].values, size=num_masked, replace=False))
        else:
            if qi_index >= len(qi_subsets):
                break
            masked_ids = get_qi_subsets_mask(df, qi_subsets, qi_index)
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
    return samples
        
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
                seed: int = None,
                cur_attack_results: List[Dict] = None) -> List[Dict]:
    """ Runs an iterative attack loop to compute separation from noisy samples.
    
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
        seed: Random seed for reproducibility (default: None)
        cur_attack_results: Previous attack results to update (required)
    
    Returns:
        List[Dict]
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    if cur_attack_results is None or len(cur_attack_results) == 0:
        raise ValueError("cur_attack_results is required.")
    results = cur_attack_results
    current_num_samples = 10
    result_index = 0
    # Build the ground truth dataframe
    if solve_type == 'pure_row':
        df = build_row_masks(nrows=nrows, nunique=nunique)
    else:
        min_vals_per_qi = get_required_num_distinct(nrows, nqi)
        # If it so happens that the actual vals_per_qi is more than what is specified,
        # then we pretend that we are on auto-select so that we don't run extra jobs.
        if min_vals_per_qi > vals_per_qi:
            vals_per_qi = 0
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
    initial_samples = []
    num_masked = 0        # used only with pure_row attacks
    qi_index = 0
    qi_subsets = []
    all_qi_cols = [col for col in df.columns if col.startswith('qi')]
    if solve_type in ['agg_row', 'agg_known']:
        qi_subsets = get_qi_subset_list(df, min_num_rows, int(round(min_num_rows * nunique * 1.5)))
        initial_samples, qi_index = initialize_qi_samples(df, nunique, noise, qi_subsets)
        print(f"Total QI subsets available: {len(qi_subsets)}. qi_index {qi_index}.")
    else:
        initial_samples = initialize_samples(df, mask_size, nunique, noise)
        print(f"start with {len(initial_samples)} initial samples")
        num_masked = mask_size
    
    num_suppressed = 0
    while True:
        if result_index >= len(results):
            # This can happen when the original attack never finished
            print(f"Reached end of results at index {result_index}, stopping loop.")
            return results
        # Start with initial binned samples, if any
        samples = initial_samples.copy()
        avg_num_masked = 0
        
        for _ in range(current_num_samples):
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
        
        # Compute separation using stored metrics
        print(f"Begin {solve_type} separation update with {len(samples)} samples\n    (current_num_samples={current_num_samples}, initial_samples={len(initial_samples)}, qi_index={qi_index}, num_suppressed={num_suppressed}, result_index={result_index})")
        result_entry = results[result_index]
        print("Using result entry:")
        pp.pprint(result_entry)
        if not isinstance(result_entry, dict):
            raise ValueError(f"attack_results entry {result_index} is not a dict.")
        expected_keys = [
            'num_samples',
            'num_equations',
            'measure',
            'qi_match_measure',
            'mixing',
            'actual_num_rows',
            'solver_metrics',
        ]
        missing_keys = [key for key in expected_keys if key not in result_entry]
        if missing_keys:
            raise ValueError(f"attack_results entry {result_index} missing keys: {missing_keys}")
        expected_num_samples = result_entry['num_samples']


        samples_copy = samples.copy()
        if len(samples_copy) < expected_num_samples:
            samples_copy = add_additional_samples(samples_copy, df, nunique, noise, expected_num_samples - len(samples_copy), solve_type, qi_subsets, qi_index, num_masked, min_num_rows)
        elif len(samples_copy) > expected_num_samples:
            samples_copy = samples_copy[:expected_num_samples]
        accuracy = result_entry['measure']
        # make a deep copy of samples
        sep = compute_separation_metrics(samples_copy)

        if solve_type in ['agg_row', 'agg_known']:
            num_masked = int(avg_num_masked / (len(samples) - len(initial_samples))) if (len(samples) - len(initial_samples)) > 0 else 0
            initial_samples = samples.copy()
        
        # Record results
        results[result_index]['separation'] = sep
        result_index += 1
        
        finished = False
        exit_reason = ''

        # Check stopping conditions
        if accuracy >= target_accuracy:
            print(f"Exit loop: Target accuracy {target_accuracy} achieved: {accuracy:.4f}")
            finished = True
            exit_reason = 'target_accuracy'

        if nqi > 0 and qi_index >= len(qi_subsets):
            print("Exit loop: No more QI subsets to use")
            finished = True
            exit_reason = 'no_more_qi_subsets'
        
        # Double the number of samples for next iteration
        current_num_samples *= 2
        
        # Check if we would exceed max_samples
        if nqi == 0 and current_num_samples + len(initial_samples) > max_samples:
            print(f"Exit loop: Reached max samples limit: {max_samples}")
            exit_reason = 'max_samples'
            finished = True

        if finished:
            break
    if result_index != len(results):
        raise ValueError(f"Loop ended after {result_index} iterations, expected {len(results)}.")
    return results
        

def main():
    """Update separation metrics for existing results files."""
    parser = argparse.ArgumentParser(description='Update separation metrics for results files')
    parser.add_argument('file_index', type=int, nargs='?', default=None,
                       help='Index into all_files.json to process a single file')
    args = parser.parse_args()

    results_dir = Path('./results/files')
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    if args.file_index is not None:
        all_files_path = Path(__file__).parent / 'all_files.json'
        if not all_files_path.exists():
            print(f"Missing all_files.json at {all_files_path}")
            return
        try:
            with open(all_files_path, 'r') as f:
                all_files = json.load(f)
        except Exception as e:
            print(f"Error reading {all_files_path}: {e}")
            return
        if not isinstance(all_files, list):
            print(f"Expected a list in {all_files_path}")
            return
        if args.file_index < 0 or args.file_index >= len(all_files):
            print(f"file_index {args.file_index} out of range [0, {len(all_files) - 1}]")
            return
        file_name = all_files[args.file_index]
        json_files = [results_dir / file_name]
        if not json_files[0].exists():
            print(f"Results file not found: {json_files[0]}")
            return
    else:
        json_files = sorted(results_dir.glob('*.json'))
    if not json_files:
        print(f"No results files found under {results_dir}")
        return

    required_keys = [
        'solve_type',
        'nrows',
        'mask_size',
        'nunique',
        'noise',
        'nqi',
        'vals_per_qi',
        'known_qi_fraction',
        'max_samples',
        'target_accuracy',
        'min_num_rows',
        'seed',
        'attack_results',
    ]

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                save_dict = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        print("====================== read file ======================")
        pp.pprint(save_dict)
        print("=======================================================")
        if 'known_qi_fraction' not in save_dict:
            # For backward compatibility, set to 0.0 if missing
            print(f"{file_path}: adding missing known_qi_fraction=0.0")
            save_dict['known_qi_fraction'] = 0.0
        if 'seed' not in save_dict:
            # For backward compatibility, set to None if missing
            print(f"{file_path}: adding missing seed=None")
            save_dict['seed'] = None
        missing_keys = [key for key in required_keys if key not in save_dict]
        if missing_keys:
            raise ValueError(f"{file_path} missing required keys: {missing_keys}")

        attack_results = save_dict['attack_results']
        if not isinstance(attack_results, list) or len(attack_results) == 0:
            raise ValueError(f"{file_path} has no attack_results to update.")

        if any(isinstance(entry, dict) and 'separation' in entry for entry in attack_results):
            print(f"{file_path}: separation already present; doing nothing.")
            continue

        print(f"Updating separation for {file_path}")
        updated_results = attack_loop(
            nrows=save_dict['nrows'],
            nunique=save_dict['nunique'],
            mask_size=save_dict['mask_size'],
            noise=save_dict['noise'],
            nqi=save_dict['nqi'],
            target_accuracy=save_dict['target_accuracy'],
            min_num_rows=save_dict['min_num_rows'],
            vals_per_qi=save_dict['vals_per_qi'],
            known_qi_fraction=save_dict['known_qi_fraction'],
            max_samples=save_dict['max_samples'],
            solve_type=save_dict['solve_type'],
            seed=save_dict['seed'],
            cur_attack_results=attack_results,
        )

        save_dict['attack_results'] = updated_results
        with open(file_path, 'w') as f:
            json.dump(save_dict, f, indent=2)

        print(f"Updated {file_path}")

if __name__ == '__main__':
    main()
