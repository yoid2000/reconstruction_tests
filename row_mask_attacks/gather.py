import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

def gather_results() -> pd.DataFrame:
    """Read all JSON result files and consolidate into a DataFrame.
    
    Returns:
        DataFrame with all results and parameters
    """
    results_dir = Path('./results/row_mask_attacks')
    
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist")
        return pd.DataFrame()
    
    # Find all JSON files
    json_files = list(results_dir.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(json_files)} JSON files")
    
    # Collect data from each file
    data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
            
            # Extract top-level parameters
            row = {
                'filename': json_file.name,
                'nrows': result.get('nrows'),
                'mask_size': result.get('mask_size'),
                'nunique': result.get('nunique'),
                'noise': result.get('noise'),
                'max_samples': result.get('max_samples'),
                'batch_size': result.get('batch_size'),
                'target_accuracy': result.get('target_accuracy'),
                'elapsed_time': result.get('elapsed_time'),
            }
            
            # Extract from last element of attack_results
            attack_results = result.get('attack_results', [])
            if attack_results:
                last_result = attack_results[-1]
                row['num_samples'] = last_result.get('num_samples')
                row['actual_num_rows'] = last_result.get('actual_num_rows')
                row['measure'] = last_result.get('measure')
                
                # Extract mixing statistics
                mixing = last_result.get('mixing', {})
                row['mixing_min'] = mixing.get('min')
                row['mixing_max'] = mixing.get('max')
                row['mixing_avg'] = mixing.get('avg')
                row['mixing_stddev'] = mixing.get('stddev')
                row['mixing_median'] = mixing.get('median')
            else:
                # No attack results
                row['num_samples'] = None
                row['actual_num_rows'] = None
                row['measure'] = None
                row['mixing_min'] = None
                row['mixing_max'] = None
                row['mixing_avg'] = None
                row['mixing_stddev'] = None
                row['mixing_median'] = None
            
            data.append(row)
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def main():
    """Main function to gather results and save to parquet."""
    # Gather results
    df = gather_results()
    
    if df.empty:
        print("No data collected")
        return
    
    # Save to parquet
    output_path = Path('./results/row_mask_attacks/result.parquet')
    df.to_parquet(output_path, index=False)
    
    print(f"\nSaved {len(df)} results to {output_path}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

if __name__ == '__main__':
    main()
