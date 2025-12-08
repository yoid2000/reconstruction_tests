import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
sys.path.insert(0, str(Path(__file__).parent.parent))
from df_builds.build_row_masks import get_required_num_distinct

def gather_results() -> pd.DataFrame:
    """Read all JSON result files and consolidate into a DataFrame.
    
    Returns:
        DataFrame with all results and parameters
    """
    results_dir = Path('./results/files')
    
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
            if 'actual_vals_per_qi' not in result:
                if result['vals_per_qi'] == 0:
                    actual_vals_per_qi = get_required_num_distinct(result['nrows'], result['nqi'])
                else:
                    actual_vals_per_qi = result['vals_per_qi']
                result['actual_vals_per_qi'] = actual_vals_per_qi
            
            # Get top-level metadata (everything except attack_results)
            base_data = {'filename': json_file.name}
            for key, value in result.items():
                if key != 'attack_results':
                    base_data[key] = value
            
            # Process each entry in attack_results
            attack_results = result.get('attack_results', [])
            if not attack_results:
                # If no attack_results, create one row with base data
                row = base_data.copy()
                row['final_attack'] = True
                row['attack_index'] = 0
                data.append(row)
            else:
                # Create one row per attack_results entry
                for idx, attack_result in enumerate(attack_results):
                    row = base_data.copy()
                    
                    # Add attack_index and final_attack columns
                    row['attack_index'] = idx
                    row['final_attack'] = (idx == len(attack_results) - 1)
                    
                    # Add all keys from this attack result
                    for key, value in attack_result.items():
                        # Handle nested dicts like 'mixing' and 'solver_metrics'
                        if isinstance(value, dict):
                            # Flatten nested dict with prefix
                            for nested_key, nested_value in value.items():
                                row[f'{key}_{nested_key}'] = nested_value
                        else:
                            row[key] = value
                    
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
    output_path = Path('./results/result.parquet')
    df.to_parquet(output_path, index=False)
    
    print(f"\nSaved {len(df)} results to {output_path}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

if __name__ == '__main__':
    main()