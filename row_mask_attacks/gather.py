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
            
            # Start with filename
            row = {'filename': json_file.name}
            
            # Add all top-level keys except 'attack_results'
            for key, value in result.items():
                if key != 'attack_results':
                    row[key] = value
            
            # Extract from last element of attack_results
            attack_results = result.get('attack_results', [])
            if attack_results:
                last_result = attack_results[-1]
                
                # Add all keys from last attack result
                for key, value in last_result.items():
                    # Handle nested dicts like 'mixing'
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
    output_path = Path('./results/row_mask_attacks/result.parquet')
    df.to_parquet(output_path, index=False)
    
    print(f"\nSaved {len(df)} results to {output_path}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

if __name__ == '__main__':
    main()
