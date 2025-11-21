import pandas as pd
from typing import List, Dict, Set
from ortools.sat.python import cp_model
import numpy as np
import json
import os
import sys
import time
from pathlib import Path
import pprint
import argparse
pp = pprint.PrettyPrinter(indent=2)

# Try to import Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from df_builds.build_row_masks import build_row_masks, build_row_masks_qi

def check_gurobi_available() -> bool:
    """Check if Gurobi is available and can actually be used.
    
    Returns:
        True if Gurobi can be used, False otherwise
    """
    if not GUROBI_AVAILABLE:
        return False
    
    try:
        # Try to create a simple model to verify license
        test_model = gp.Model("test")
        test_model.dispose()
        return True
    except Exception as e:
        print(f"Gurobi import succeeded but cannot create model: {e}")
        return False

def reconstruct(samples: List[Dict], noise: int) -> List[Dict]:
    """ Reconstructs the value associated with each ID from noisy count samples.
    
    Args:
        samples: List of dicts, each containing:
            - 'ids': set of integer IDs
            - 'noisy_counts': list of dicts with 'val' (int) and 'count' (int)
        noise: Integer representing the noise bound (±noise from true count)
    
    Returns:
        List of dicts with 'id' (int) and 'val' (int)
    """
    # Collect all unique IDs and values
    all_ids = set()
    all_vals = set()
    for sample in samples:
        all_ids.update(sample['ids'])
        for count_info in sample['noisy_counts']:
            all_vals.add(count_info['val'])
    
    all_ids = sorted(all_ids)
    all_vals = sorted(all_vals)
    
    if check_gurobi_available():
        # Use Gurobi solver
        print("Using Gurobi solver")
        model = gp.Model("reconstruct")
        model.setParam('OutputFlag', 0)  # Suppress output
        
        # Create binary variables: x[id][val] = 1 if id has value val
        x = {}
        for id in all_ids:
            x[id] = {}
            for val in all_vals:
                x[id][val] = model.addVar(vtype=GRB.BINARY, name=f'x_{id}_{val}')
        
        # Constraint: Each ID must have exactly one value
        for id in all_ids:
            model.addConstr(gp.quicksum(x[id][val] for val in all_vals) == 1)
        
        # Constraint: For each sample, the counts must be within noise bounds
        for sample in samples:
            ids = sample['ids']
            noisy_counts = sample['noisy_counts']
            
            for count_info in noisy_counts:
                val = count_info['val']
                noisy_count = count_info['count']
                
                # True count for this value in this sample
                true_count = gp.quicksum(x[id][val] for id in ids if id in all_ids)
                
                # Constraint: noisy_count - noise <= true_count <= noisy_count + noise
                model.addConstr(true_count >= noisy_count - noise)
                model.addConstr(true_count <= noisy_count + noise)
        
        # Solve the model
        model.optimize()
        
        # Extract solution
        result = []
        if model.status == GRB.OPTIMAL:
            for id in all_ids:
                for val in all_vals:
                    if x[id][val].X > 0.5:  # Binary variable is 1
                        result.append({'id': id, 'val': val})
                        break
    else:
        # Use OR-Tools solver
        print("Using OR-Tools CP-SAT solver")
        model = cp_model.CpModel()
        
        # Create binary variables: x[id][val] = 1 if id has value val
        x = {}
        for id in all_ids:
            x[id] = {}
            for val in all_vals:
                x[id][val] = model.NewBoolVar(f'x_{id}_{val}')
        
        # Constraint: Each ID must have exactly one value
        for id in all_ids:
            model.Add(sum(x[id][val] for val in all_vals) == 1)
        
        # Constraint: For each sample, the counts must be within noise bounds
        for sample in samples:
            ids = sample['ids']
            noisy_counts = sample['noisy_counts']
            
            for count_info in noisy_counts:
                val = count_info['val']
                noisy_count = count_info['count']
                
                # True count for this value in this sample
                true_count = sum(x[id][val] for id in ids if id in all_ids)
                
                # Constraint: noisy_count - noise <= true_count <= noisy_count + noise
                model.Add(true_count >= noisy_count - noise)
                model.Add(true_count <= noisy_count + noise)
        
        # Solve the model
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # Extract solution
        result = []
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for id in all_ids:
                for val in all_vals:
                    if solver.Value(x[id][val]) == 1:
                        result.append({'id': id, 'val': val})
                        break
    
    return result

def measure(df: pd.DataFrame, reconstructed: List[Dict]) -> float:
    """ Measures the accuracy of reconstruction.
    
    Args:
        df: DataFrame with columns 'id' and 'val' (ground truth)
        reconstructed: Output from reconstruct(), list of dicts with 'id' and 'val'
    
    Returns:
        Fraction of correct assignments (float between 0 and 1)
    """
    # Convert reconstructed to dict for easy lookup
    recon_dict = {item['id']: item['val'] for item in reconstructed}
    
    # Count correct assignments
    correct = 0
    total = 0
    
    for _, row in df.iterrows():
        id = row['id']
        true_val = row['val']
        
        if id in recon_dict:
            total += 1
            if recon_dict[id] == true_val:
                correct += 1
    
    return correct / total if total > 0 else 0.0

def mixing_stats(samples: List[Dict]) -> Dict:
    """ Computes mixing statistics for IDs across samples.
    
    For each ID pair that shares at least one sample, counts how many times they share a sample.
    Returns min, max, avg, stddev, and median of these counts.
    
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

def get_qi_subset_list(df: pd.DataFrame, min_num_rows: int) -> List[Dict]:
    """ Generates list of QI column subsets with their values and matching row counts.
    
    Args:
        df: DataFrame with QI columns (qi0, qi1, ..., qiN)
        min_num_rows: Minimum number of rows in any aggregate (default: 5)
    
    Returns:
        List of dicts with 'qi_cols', 'qi_vals', and 'num_rows', sorted by num_rows descending
    """
    import itertools
    
    # Find all QI columns
    all_qi_cols = sorted([col for col in df.columns if col.startswith('qi')])
    
    if len(all_qi_cols) == 0:
        return []
    
    qi_subsets = []
    
    # Iterate through subset sizes from 1 to nqi-1
    # (nqi columns would only have 1 row per combination since all combos are unique)
    for subset_size in range(1, len(all_qi_cols)):
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
    
    # Sort by num_rows descending
    qi_subsets.sort(key=lambda x: x['num_rows'], reverse=True)
    
    return qi_subsets

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

def prior_job_results(file_path: Path) -> List[Dict]:
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
        return data.get('attack_results')
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

def attack_loop(nrows: int, 
                nunique: int, 
                mask_size: int, 
                noise: int,
                nqi: int = 0,
                max_samples: int = 20000,
                target_accuracy: float = 0.99,
                min_num_rows: int = 5,
                output_file: Path = None,
                cur_attack_results: List[Dict] = None) -> None:
    """ Runs an iterative attack loop to reconstruct values from noisy samples.
    
    Args:
        nrows: Number of rows in the dataframe
        nunique: Number of unique values
        mask_size: Number of rows in each random sample
        noise: Noise bound for counts (±noise)
        nqi: Number of quasi-identifier columns
        max_samples: Maximum number of samples to generate (default: 20000)
        target_accuracy: Target accuracy to stop early (default: 0.99)
        min_num_rows: Minimum number of rows in any aggregate (default: 5)
        output_file: Path to JSON file to save results incrementally (default: None)
        cur_attack_results: Previous attack results to resume from (default: None)
    
    Returns:
        None
    """
    # Check if we're resuming from prior results
    if cur_attack_results is not None and len(cur_attack_results) > 0:
        last_result = cur_attack_results[-1]
        
        # If already achieved target accuracy, do nothing
        if last_result['measure'] >= target_accuracy:
            print(f"Prior results already achieved target accuracy: {last_result['measure']:.4f}")
            return
        
        # Resume from prior results
        print(f"Resuming from prior results with {last_result['num_samples']} samples, accuracy: {last_result['measure']:.4f}")
        results = cur_attack_results
        current_num_samples = last_result['num_samples'] - 1  # -1 for the all-IDs sample
    else:
        # Starting fresh
        results = []
        current_num_samples = 10
    
    # Start timing
    start_time = time.time()
    
    # Build the ground truth dataframe
    if nqi == 0:
        df = build_row_masks(nrows=nrows, nunique=nunique)
    else:
        df = build_row_masks_qi(nrows=nrows, nunique=nunique, nqi=nqi)
    
    initial_samples = []
    num_masked = None
    qi_subsets = []
    if nqi > 0:
        qi_subsets = get_qi_subset_list(df, min_num_rows)
        pp.pprint(qi_subsets)
        print(f"Total QI subsets available: {len(qi_subsets)}")
    else:
        initial_samples = initialize_samples(df, mask_size, nunique, noise)
        print(f"start with {len(initial_samples)} initial samples")
        num_masked = mask_size
    
    while True:
        # Start with initial binned samples, if any
        samples = initial_samples.copy()
        qi_index = 0
        avg_num_masked = 0
        
        for _ in range(current_num_samples):
            # Select random subset of IDs
            if nqi == 0:
                masked_ids = set(np.random.choice(df['id'].values, size=num_masked, replace=False))
            else:
                if qi_index >= len(qi_subsets):
                    break
                masked_ids = get_qi_subsets_mask(df, qi_subsets, qi_index)
                avg_num_masked += len(masked_ids)
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
                'noisy_counts': noisy_counts
            })
        
        # Reconstruct and measure
        reconstructed = reconstruct(samples, noise)
        accuracy = measure(df, reconstructed)
        mixing = mixing_stats(samples)

        if nqi > 0:
            num_masked = int(avg_num_masked / (len(samples) - len(initial_samples))) if (len(samples) - len(initial_samples)) > 0 else 0
        
        # Record results
        results.append({
            'num_samples': len(samples),
            'measure': accuracy,
            'mixing': mixing,
            'actual_num_rows': num_masked,
        })
        pp.pprint(results[-1])
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Save results incrementally if output file is provided
        if output_file is not None:
            save_dict = {
                'nrows': nrows,
                'mask_size': mask_size,
                'nunique': nunique,
                'noise': noise,
                'nqi': nqi,
                'max_samples': max_samples,
                'target_accuracy': target_accuracy,
                'min_num_rows': min_num_rows,
                'elapsed_time': elapsed_time,
                'attack_results': results,
            }
            with open(output_file, 'w') as f:
                json.dump(save_dict, f, indent=2)
        
        # Check stopping conditions
        if accuracy >= target_accuracy:
            print(f"Exit loop: Target accuracy {target_accuracy} achieved: {accuracy:.4f}")
            break

        if nqi > 0 and current_num_samples >= len(qi_subsets):
            print("Exit loop: No more QI subsets to use")
            break
        
        # Double the number of samples for next iteration
        current_num_samples *= 2
        
        # Check if we would exceed max_samples
        if current_num_samples + len(initial_samples) > max_samples:
            print(f"Exit loop: Reached max samples limit: {max_samples}")
            break

def main():
    """Main function to run parameter sweep experiments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run row mask attack experiments')
    parser.add_argument('job_num', type=int, nargs='?', default=None,
                       help='Job number to run from parameter combinations')
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
    
    args = parser.parse_args()
    
    # Create directories
    results_dir = Path('./results')
    attack_results_dir = results_dir / 'row_mask_attacks'
    slurm_out_dir = Path('./slurm_out')
    results_dir.mkdir(exist_ok=True)
    attack_results_dir.mkdir(exist_ok=True)
    slurm_out_dir.mkdir(exist_ok=True)
    
    # Define parameter ranges
    nrows_values = [200, 300]
    mask_size_values = [20, 25, 30]
    nunique_values = [2]
    noise_values = [2, 6, 10, 14]
    nqi_values = [0]
    
    # Fixed parameters
    max_samples = 20000
    target_accuracy = 0.99
    min_num_rows = 5
    
    # Defaults
    defaults = {
        'nrows': 200,
        'mask_size': 20,
        'nunique': 2,
        'noise': 2,
        'nqi': 0,
    }
    
    # Check if any individual parameters were provided
    individual_params_provided = any([
        args.nrows is not None,
        args.mask_size is not None,
        args.nunique is not None,
        args.noise is not None,
        args.nqi is not None
    ])
    
    if individual_params_provided:
        # Use command line parameters, falling back to defaults
        params = {
            'nrows': args.nrows if args.nrows is not None else defaults['nrows'],
            'mask_size': args.mask_size if args.mask_size is not None else defaults['mask_size'],
            'nunique': args.nunique if args.nunique is not None else defaults['nunique'],
            'noise': args.noise if args.noise is not None else defaults['noise'],
            'nqi': args.nqi if args.nqi is not None else defaults['nqi'],
        }
        
        # Generate filename
        file_name = (f"nr{params['nrows']}_mf{params['mask_size']}_"
                    f"nu{params['nunique']}_qi{params['nqi']}_n{params['noise']}_"
                    f"ms{max_samples}_ta{int(target_accuracy*100)}")
        
        file_path = attack_results_dir / f"{file_name}.json"
        
        # Load prior results if they exist
        cur_attack_results = prior_job_results(file_path)
        
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
            min_num_rows=min_num_rows,
            output_file=file_path,
            cur_attack_results=cur_attack_results,
        )
        
        # Read back the saved file to get the final elapsed time
        with open(file_path, 'r') as f:
            final_results = json.load(f)
        
        print(f"Results saved to {file_path}")
        print(f"Elapsed time: {final_results['elapsed_time']:.2f} seconds")
        print(f"Final accuracy: {final_results['attack_results'][-1]['measure']:.4f}")
        print(f"Samples used: {final_results['attack_results'][-1]['num_samples']}")
        
        return
    
    # Generate test parameter combinations
    test_params = []
    
    # First pass: vary one parameter at a time
    for nrows in nrows_values:
        test_params.append({
            'nrows': nrows,
            'mask_size': defaults['mask_size'],
            'nunique': defaults['nunique'],
            'noise': defaults['noise'],
            'nqi': defaults['nqi'],
        })
    
    for mask_size in mask_size_values:
        test_params.append({
            'nrows': defaults['nrows'],
            'mask_size': mask_size,
            'nunique': defaults['nunique'],
            'noise': defaults['noise'],
            'nqi': defaults['nqi'],
        })
    
    for nunique in nunique_values:
        test_params.append({
            'nrows': defaults['nrows'],
            'mask_size': defaults['mask_size'],
            'nunique': nunique,
            'noise': defaults['noise'],
            'nqi': defaults['nqi'],
        })
    
    for noise in noise_values:
        test_params.append({
            'nrows': defaults['nrows'],
            'mask_size': defaults['mask_size'],
            'nunique': defaults['nunique'],
            'noise': noise,
            'nqi': defaults['nqi'],
        })

    for nqi in nqi_values:
        test_params.append({
            'nrows': defaults['nrows'],
            'mask_size': defaults['mask_size'],
            'nunique': defaults['nunique'],
            'noise': defaults['noise'],
            'nqi': nqi,
        })
    
    # Remove duplicates from first pass
    seen = set()
    unique_first_pass = []
    for params in test_params:
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            unique_first_pass.append(params)
    
    # Second pass: all combinations not in first pass
    for nrows in nrows_values:
        for mask_size in mask_size_values:
            for nunique in nunique_values:
                for noise in noise_values:
                    for nqi in nqi_values:
                        params = {
                            'nrows': nrows,
                            'mask_size': mask_size,
                            'nunique': nunique,
                            'noise': noise,
                            'nqi': nqi,
                        }
                        key = tuple(sorted(params.items()))
                        if key not in seen:
                            seen.add(key)
                            test_params.append(params)
    
    # If no job_num, just print all combinations
    if args.job_num is None:
        for i, params in enumerate(test_params):
            print(f"Job {i}: {params}")
        
        # Create run.slurm file
        num_jobs = len(test_params) - 1  # Array range is 0 to num_jobs-1
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name=recon_test
#SBATCH --output=/INS/syndiffix/work/paul/github/reconstruction_tests/row_mask_attacks/slurm_out/out.%a.out
#SBATCH --time=7-00:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-{num_jobs}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
source /INS/syndiffix/work/paul/github/reconstruction_tests/.venv/bin/activate
python /INS/syndiffix/work/paul/github/reconstruction_tests/row_mask_attacks/run_row_mask_attack.py $arrayNum
"""
        
        slurm_file = Path(__file__).parent / 'run.slurm'
        with open(slurm_file, 'w') as f:
            f.write(slurm_content)
        
        print(f"\nSLURM file created: {slurm_file}")
        print(f"Total jobs: {len(test_params)} (array: 0-{num_jobs})")
        return
    
    # Check if job_num is valid
    if args.job_num < 0 or args.job_num >= len(test_params):
        print(f"Error: job_num {args.job_num} out of range [0, {len(test_params)-1}]")
        return
    
    # Get parameters for this job
    params = test_params[args.job_num]
    
    # Generate filename
    file_name = (f"nr{params['nrows']}_mf{params['mask_size']}_"
                    f"nu{params['nunique']}_qi{params['nqi']}_n{params['noise']}_"
                    f"ms{max_samples}_ta{int(target_accuracy*100)}")
        
    file_path = attack_results_dir / f"{file_name}.json"
    
    # Load prior results if they exist
    cur_attack_results = prior_job_results(file_path)
    
    # Run attack_loop
    print(f"Running job {args.job_num}: {params}")
    
    attack_loop(
        nrows=params['nrows'],
        nunique=params['nunique'],
        mask_size=params['mask_size'],
        noise=params['noise'],
        nqi=params['nqi'],
        max_samples=max_samples,
        target_accuracy=target_accuracy,
        min_num_rows=min_num_rows,
        output_file=file_path,
        cur_attack_results=cur_attack_results,
    )
    
    # Read back the saved file to get the final elapsed time
    with open(file_path, 'r') as f:
        final_results = json.load(f)
    
    print(f"Results saved to {file_path}")
    print(f"Elapsed time: {final_results['elapsed_time']:.2f} seconds")
    print(f"Final accuracy: {final_results['attack_results'][-1]['measure']:.4f}")
    print(f"Samples used: {final_results['attack_results'][-1]['num_samples']}")

if __name__ == '__main__':
    main()
