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
from df_builds.build_row_masks import build_row_masks

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

def attack_loop(nrows: int, 
                nunique: int, 
                mask_fraction: float, 
                noise: int,
                max_samples: int = 20000,
                target_accuracy: float = 0.99,
                min_num_rows: int = 5,
                output_file: Path = None) -> List[Dict]:
    """ Runs an iterative attack loop to reconstruct values from noisy samples.
    
    Args:
        nrows: Number of rows in the dataframe
        nunique: Number of unique values
        mask_fraction: Fraction of rows to sample (between 0 and 1)
        noise: Noise bound for counts (±noise)
        max_samples: Maximum number of samples to generate (default: 20000)
        target_accuracy: Target accuracy to stop early (default: 0.99)
        min_num_rows: Minimum number of rows to mask in each sample (default: 5)
        output_file: Path to JSON file to save results incrementally (default: None)
    
    Returns:
        List of dicts with 'num_samples' and 'measure' for each loop iteration
    """
    # Start timing
    start_time = time.time()
    
    # Build the ground truth dataframe
    df = build_row_masks(nrows=nrows, nunique=nunique)
    
    # Initialize with a sample containing all IDs
    all_ids = set(df['id'].values)
    exact_counts = df['val'].value_counts().to_dict()
    
    noisy_counts = []
    for val in range(nunique):
        exact_count = exact_counts.get(val, 0)
        noise_delta = np.random.randint(-noise, noise + 1)
        noisy_count = max(0, exact_count + noise_delta)
        noisy_counts.append({'val': val, 'count': noisy_count})
    
    samples = [{
        'ids': all_ids,
        'noisy_counts': noisy_counts
    }]
    
    results = []
    num_masked = max(min_num_rows, int(nrows * mask_fraction))
    
    # Start with 10 samples, then double each iteration
    current_num_samples = 10
    
    while True:
        # Generate new set of samples (replace existing samples, keep the all-IDs sample)
        samples = [samples[0]]  # Keep the first sample with all IDs
        
        for _ in range(current_num_samples):
            # Select random subset of IDs
            masked_ids = set(np.random.choice(df['id'].values, size=num_masked, replace=False))
            
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
            samples.append({
                'ids': masked_ids,
                'noisy_counts': noisy_counts
            })
        
        # Reconstruct and measure
        reconstructed = reconstruct(samples, noise)
        accuracy = measure(df, reconstructed)
        mixing = mixing_stats(samples)
        
        # Record results
        results.append({
            'num_samples': len(samples),
            'measure': accuracy,
            'mixing': mixing,
            'actual_num_rows': num_masked,
        })
        pp.pprint(results)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Save results incrementally if output file is provided
        if output_file is not None:
            save_dict = {
                'nrows': nrows,
                'mask_fraction': mask_fraction,
                'nunique': nunique,
                'noise': noise,
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
            break
        
        # Double the number of samples for next iteration
        current_num_samples *= 2
        
        # Check if we would exceed max_samples
        if current_num_samples + 1 > max_samples:  # +1 for the all-IDs sample
            break
    
    return results

def main():
    """Main function to run parameter sweep experiments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run row mask attack experiments')
    parser.add_argument('job_num', type=int, nargs='?', default=None,
                       help='Job number to run from parameter combinations')
    parser.add_argument('--nrows', type=int, default=None,
                       help='Number of rows')
    parser.add_argument('--mask_fraction', type=float, default=None,
                       help='Mask fraction (0-1)')
    parser.add_argument('--nunique', type=int, default=None,
                       help='Number of unique values')
    parser.add_argument('--noise', type=int, default=None,
                       help='Noise bound')
    
    args = parser.parse_args()
    
    # Create directories
    results_dir = Path('./results')
    attack_results_dir = results_dir / 'row_mask_attacks'
    slurm_out_dir = Path('./slurm_out')
    results_dir.mkdir(exist_ok=True)
    attack_results_dir.mkdir(exist_ok=True)
    slurm_out_dir.mkdir(exist_ok=True)
    
    # Define parameter ranges
    nrows_values = [100, 150]
    mask_fraction_values = [0.5, 0.1, 0.05]
    nunique_values = [2, 4, 8]
    noise_values = [2, 4, 6, 8, 10, 12, 14, 16]
    
    # Fixed parameters
    max_samples = 20000
    target_accuracy = 0.99
    min_num_rows = 5
    
    # Defaults
    defaults = {
        'nrows': 100,
        'mask_fraction': 0.5,
        'nunique': 2,
        'noise': 2
    }
    
    # Check if any individual parameters were provided
    individual_params_provided = any([
        args.nrows is not None,
        args.mask_fraction is not None,
        args.nunique is not None,
        args.noise is not None
    ])
    
    if individual_params_provided:
        # Use command line parameters, falling back to defaults
        params = {
            'nrows': args.nrows if args.nrows is not None else defaults['nrows'],
            'mask_fraction': args.mask_fraction if args.mask_fraction is not None else defaults['mask_fraction'],
            'nunique': args.nunique if args.nunique is not None else defaults['nunique'],
            'noise': args.noise if args.noise is not None else defaults['noise']
        }
        
        # Generate filename
        file_name = (f"nr{params['nrows']}_mf{int(params['mask_fraction']*100)}_"
                     f"nu{params['nunique']}_n{params['noise']}_"
                     f"ms{max_samples}_ta{int(target_accuracy*100)}")
        
        file_path = attack_results_dir / f"{file_name}.json"
        
        # Check if file already exists
        if file_path.exists():
            print(f"File {file_path} already exists. Exiting.")
            sys.exit(0)
        
        # Run attack_loop
        print(f"Running with parameters: {params}")
        
        attack_results = attack_loop(
            nrows=params['nrows'],
            nunique=params['nunique'],
            mask_fraction=params['mask_fraction'],
            noise=params['noise'],
            max_samples=max_samples,
            target_accuracy=target_accuracy,
            min_num_rows=min_num_rows,
            output_file=file_path,
        )
        
        # Read back the saved file to get the final elapsed time
        with open(file_path, 'r') as f:
            final_results = json.load(f)
        
        print(f"Results saved to {file_path}")
        print(f"Elapsed time: {final_results['elapsed_time']:.2f} seconds")
        print(f"Final accuracy: {attack_results[-1]['measure']:.4f}")
        print(f"Samples used: {attack_results[-1]['num_samples']}")
        
        return
    
    # Generate test parameter combinations
    test_params = []
    
    # First pass: vary one parameter at a time
    for nrows in nrows_values:
        test_params.append({
            'nrows': nrows,
            'mask_fraction': defaults['mask_fraction'],
            'nunique': defaults['nunique'],
            'noise': defaults['noise']
        })
    
    for mask_fraction in mask_fraction_values:
        test_params.append({
            'nrows': defaults['nrows'],
            'mask_fraction': mask_fraction,
            'nunique': defaults['nunique'],
            'noise': defaults['noise']
        })
    
    for nunique in nunique_values:
        test_params.append({
            'nrows': defaults['nrows'],
            'mask_fraction': defaults['mask_fraction'],
            'nunique': nunique,
            'noise': defaults['noise']
        })
    
    for noise in noise_values:
        test_params.append({
            'nrows': defaults['nrows'],
            'mask_fraction': defaults['mask_fraction'],
            'nunique': defaults['nunique'],
            'noise': noise
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
        for mask_fraction in mask_fraction_values:
            for nunique in nunique_values:
                for noise in noise_values:
                    params = {
                        'nrows': nrows,
                        'mask_fraction': mask_fraction,
                        'nunique': nunique,
                        'noise': noise
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
#SBATCH --time=12:00:00
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
    file_name = (f"nr{params['nrows']}_mf{int(params['mask_fraction']*100)}_"
                 f"nu{params['nunique']}_n{params['noise']}_"
                 f"ms{max_samples}_ta{int(target_accuracy*100)}")
    
    file_path = attack_results_dir / f"{file_name}.json"
    
    # Check if file already exists
    if file_path.exists():
        print(f"File {file_path} already exists. Exiting.")
        sys.exit(0)
    
    # Run attack_loop
    print(f"Running job {args.job_num}: {params}")
    
    attack_results = attack_loop(
        nrows=params['nrows'],
        nunique=params['nunique'],
        mask_fraction=params['mask_fraction'],
        noise=params['noise'],
        max_samples=max_samples,
        target_accuracy=target_accuracy,
        min_num_rows=min_num_rows,
        output_file=file_path,
    )
    
    # Read back the saved file to get the final elapsed time
    with open(file_path, 'r') as f:
        final_results = json.load(f)
    
    print(f"Results saved to {file_path}")
    print(f"Elapsed time: {final_results['elapsed_time']:.2f} seconds")
    print(f"Final accuracy: {attack_results[-1]['measure']:.4f}")
    print(f"Samples used: {attack_results[-1]['num_samples']}")

if __name__ == '__main__':
    main()
