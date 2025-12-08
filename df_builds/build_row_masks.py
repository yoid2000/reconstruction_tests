import pandas as pd
import numpy as np
import itertools
import math

def build_row_masks(nrows: int = 1024,
                    nunique: int = 2) -> pd.DataFrame:
    """ Builds a dataframe with nrows rows and two columns, 'id' and 'val'.
        The 'id' column contains unique identifiers. The 'val' column contains
        nunique distinct values, which are randomly assigned to each row.
    """
    df = pd.DataFrame({
        'id': range(nrows),
        'val': np.random.randint(0, nunique, size=nrows)
    })
    return df

def check_masks_qi(df: pd.DataFrame) -> None:
    """ Validates that all combinations of QI columns are unique.
    
    Args:
        df: DataFrame with QI columns (qi0, qi1, ..., qiN)
    
    Raises:
        ValueError: If any QI combination is duplicated
    """
    # Find all QI columns
    qi_cols = [col for col in df.columns if col.startswith('qi')]
    
    if len(qi_cols) == 0:
        raise ValueError("No QI columns found in dataframe")
    
    # Check for duplicate combinations
    qi_combinations = df[qi_cols]
    duplicates = qi_combinations.duplicated()
    
    if duplicates.any():
        num_duplicates = duplicates.sum()
        duplicate_rows = df[duplicates][qi_cols]
        raise ValueError(
            f"Found {num_duplicates} duplicate QI combinations. "
            f"First few duplicates:\n{duplicate_rows.head()}"
        )

def get_required_num_distinct(nrows: int, nqi: int) -> int:
    """ Calculates the minimum number of distinct values needed per QI column
        to ensure all nrows have unique combinations.
    
    Args:
        nrows: Number of rows in the dataframe
        nqi: Number of quasi-identifier columns
    
    Returns:
        Minimum number of distinct values per QI column
    """
    if nqi == 0:
        return 0
    else:
        return math.ceil(nrows ** (1.0 / nqi))

def build_row_masks_qi(nrows: int = 1024,
                       nunique: int = 2,
                       nqi: int = 3,
                       vals_per_qi = 0) -> pd.DataFrame:
    """ Builds a dataframe with nrows rows and 2+nqi columns.
    
    Args:
        nrows: Number of rows in the dataframe
        nunique: Number of unique values for the 'val' column
        nqi: Number of quasi-identifier columns
    
    Returns:
        DataFrame with columns 'id', 'val', and qi0, qi1, ..., qi(nqi-1)
    """
    # Calculate minimum number of distinct values needed per QI column
    # We need n_distinct^nqi >= nrows
    # So n_distinct = ceil(nrows^(1/nqi))
    n_distinct = get_required_num_distinct(nrows, nqi)

    if vals_per_qi >= n_distinct:
        n_distinct = vals_per_qi
    
    # Calculate length of all possible combinations
    len_all_combinations = n_distinct ** nqi
    
    # Check if we can afford to generate all combinations
    if len_all_combinations < 10 * nrows:
        # Generate all possible combinations
        possible_values = list(range(n_distinct))
        all_combinations = list(itertools.product(possible_values, repeat=nqi))
        
        # Randomly select nrows combinations
        if len(all_combinations) < nrows:
            raise ValueError(f"Not enough combinations: {len(all_combinations)} < {nrows}")
        
        selected_indices = np.random.choice(len(all_combinations), size=nrows, replace=False)
        selected_combinations = [all_combinations[i] for i in selected_indices]
    else:
        # Too many combinations to enumerate, sample randomly
        selected_combinations = []
        seen_combinations = set()
        
        while len(selected_combinations) < nrows:
            # Randomly generate a combination
            combo = tuple(np.random.randint(0, n_distinct, size=nqi))
            
            # Check if we've already selected this combination
            if combo not in seen_combinations:
                seen_combinations.add(combo)
                selected_combinations.append(combo)
    
    # Build the dataframe
    df = pd.DataFrame({
        'id': range(nrows),
        'val': np.random.randint(0, nunique, size=nrows)
    })
    
    # Add QI columns
    for qi_idx in range(nqi):
        col_name = f'qi{qi_idx}'
        df[col_name] = [combo[qi_idx] for combo in selected_combinations]
    
    # Validate that all QI combinations are unique
    check_masks_qi(df)
    
    return df