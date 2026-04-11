import pandas as pd
from typing import Any, List, Dict, Set, Optional
import numpy as np
import gc
import json
import hashlib
import os
import re
import sys
import time
from pathlib import Path
import pprint
import argparse
from functools import partial
from itertools import product
from anonymity_loss_coefficient import brm_attack_simple

pp = pprint.PrettyPrinter(indent=2)
print = partial(print, flush=True)
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from df_builds.build_row_masks import build_row_masks, build_row_masks_qi, get_required_num_distinct
from reconstruct import reconstruct_by_row, measure_by_row, measure_by_aggregate, reconstruct_by_aggregate_and_known_qi
from compute_separation import compute_separation_metrics

solve_type_map = {
    'pure_row': 'pr',
    'agg_row': 'ar',
    'agg_known': 'ak',
}

DEFAULT_USE_OBJECTIVE = True
DEFAULT_TIME_LIMIT_SECONDS = (3 * 24 * 60 * 60)  # 3 days in seconds
DEFAULT_SLACK_LIMIT_MULTIPLE = 2
DEFAULT_SLACK_LIMIT_MIN = 10


def _get_process_memory_mb() -> Optional[float]:
    """Best-effort current process RSS in MB."""
    try:
        import psutil  # type: ignore
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
    except Exception:
        pass

    status_path = "/proc/self/status"
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            rss_kb = float(parts[1])
                            return rss_kb / 1024.0
        except Exception:
            pass
    return None


def _print_memory_usage(label: str) -> None:
    mem_mb = _get_process_memory_mb()
    if mem_mb is None:
        print(f"{label}: memory usage unavailable")
    else:
        print(f"{label}: memory usage {mem_mb:.2f} MB")


def _sanitize_filename_part(value: str) -> str:
    """Keep only filename-safe characters and collapse others to underscores."""
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value))
    cleaned = cleaned.strip('._-')
    return cleaned or "x"


def _is_missing_dataset_arg(value: Any) -> bool:
    """Treat None or empty/whitespace string as missing dataset arg."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False

def _resolve_dataset_path(path_to_dataset: str) -> Path:
    dataset_path = Path(path_to_dataset)
    if dataset_path.suffix.lower() != '.parquet':
        raise ValueError(f"path_to_dataset must point to a .parquet file, got: {path_to_dataset}")
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    return dataset_path

def get_dataset_generation_params(path_to_dataset: str, target_column: str) -> Dict[str, int]:
    """Infer nrows, nunique, and nqi from raw parquet dataset metadata/content."""
    dataset_path = _resolve_dataset_path(path_to_dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    source_df = pd.read_parquet(dataset_path)
    if target_column not in source_df.columns:
        raise ValueError(
            f"target_column '{target_column}' is not in dataset columns: {list(source_df.columns)}"
        )
    return {
        'nrows': int(len(source_df)),
        'nunique': int(source_df[target_column].nunique(dropna=False)),
        'nqi': int(max(0, len(source_df.columns) - 1)),
    }
        
def generate_filename(params, target_accuracy) -> str:
    """ Generate a filename string based on attack parameters. """
    slack_limit_multiple = int(params.get('slack_limit_multiple', DEFAULT_SLACK_LIMIT_MULTIPLE))
    slack_limit_min = int(params.get('slack_limit_min', DEFAULT_SLACK_LIMIT_MIN))
    slack_str = f"_slm{slack_limit_multiple}_sln{slack_limit_min}"

    path_to_dataset = params.get('path_to_dataset')
    target_column = params.get('target_column')
    if not _is_missing_dataset_arg(path_to_dataset) and not _is_missing_dataset_arg(target_column):
        dataset_stem = _sanitize_filename_part(Path(path_to_dataset).stem)
        target_label = _sanitize_filename_part(target_column)
        hash_payload = json.dumps(
            {
                'path_to_dataset': path_to_dataset,
                'target_column': target_column,
            },
            sort_keys=True,
        )
        dataset_hash = hashlib.md5(hash_payload.encode('utf-8')).hexdigest()[:8]
        seed_str = f"_s{params['seed']}" if params['seed'] is not None else ""
        kqf_str = ""
        if params['solve_type'] == 'agg_known':
            kqf_str = f"_kqf{int(params['known_qi_fraction']*100)}"
        max_qi = params.get('max_qi', 1000)
        max_qi_str = f"_mq{max_qi}" if max_qi != 1000 else ""
        return (
            f"ds{dataset_stem}_tc{target_label}_h{dataset_hash}_"
            f"mf{params['mask_size']}_n{params['noise']}_mnr{params['min_num_rows']}{max_qi_str}_"
            f"st{solve_type_map[params['solve_type']]}_"
            f"ms{params['max_samples']}_ta{int(target_accuracy*100)}{kqf_str}{seed_str}{slack_str}"
        )

    vals_per_qi = params['vals_per_qi']
    corr_strength = params.get('corr_strength', 0.0)
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
    corr_strength_str = ""
    if corr_strength and corr_strength > 0.0:
        corr_strength_str = f"_cs{int(round(corr_strength * 100))}"
    file_name = (f"nr{params['nrows']}_mf{params['mask_size']}_"
                f"nu{params['nunique']}_qi{params['nqi']}_n{params['noise']}_"
                f"mnr{params['min_num_rows']}_vpq{vals_per_qi}{max_qi_str}{corr_strength_str}_"
                f"st{solve_type_map[params['solve_type']]}_"
                f"ms{params['max_samples']}_ta{int(target_accuracy*100)}{kqf_str}{seed_str}{slack_str}")
    return file_name

def get_and_process_data(path_to_dataset: str,
                         target_column: str) -> pd.DataFrame:
    """Load and process an external parquet dataset into id/val/qi* attack format."""
    dataset_path = _resolve_dataset_path(path_to_dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    source_df = pd.read_parquet(dataset_path)
    if target_column not in source_df.columns:
        raise ValueError(
            f"target_column '{target_column}' is not in dataset columns: {list(source_df.columns)}"
        )
    qi_source_columns = [col for col in source_df.columns if col != target_column]

    selected_columns = [target_column] + qi_source_columns
    df = source_df[selected_columns].copy()

    rename_map = {target_column: 'val'}
    for idx, col_name in enumerate(qi_source_columns, start=1):
        rename_map[col_name] = f'qi{idx}'
    df = df.rename(columns=rename_map)

    value_columns = ['val'] + [f'qi{idx}' for idx in range(1, len(qi_source_columns) + 1)]
    for col in value_columns:
        try:
            codes, _ = pd.factorize(df[col], sort=True)
        except TypeError:
            # Mixed incomparable dtypes can fail with sort=True; preserve deterministic first-seen order.
            codes, _ = pd.factorize(df[col], sort=False)
        if np.any(codes < 0):
            next_code = int(codes[codes >= 0].max() + 1) if np.any(codes >= 0) else 0
            codes = np.where(codes < 0, next_code, codes)
        df[col] = codes.astype(int)

    df.insert(0, 'id', np.arange(len(df), dtype=int))
    return df


def convert_from_qi_val_tuples(df: pd.DataFrame, reconstructed: List[Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert reconstructed QI/value rows into BRM input dataframes.

    Args:
        df: Original attack dataframe with columns `id`, optional `qi*`, and `val`.
        reconstructed: Rows as dicts containing all `qi*` columns and `val`.

    Returns:
        Tuple `(original, anon)`:
            - `original`: `df` without the `id` column.
            - `anon`: reconstructed dataframe with same columns/order as `original`.
    """
    if 'id' not in df.columns or 'val' not in df.columns:
        raise ValueError("df must contain 'id' and 'val' columns.")

    original = df.drop(columns=['id']).copy()
    qi_cols = [col for col in df.columns if col not in ['id', 'val']]
    tuple_cols = qi_cols + ['val']

    if reconstructed is None:
        reconstructed = []

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(reconstructed):
        if not isinstance(item, dict):
            raise ValueError(f"reconstructed[{idx}] must be a dict, got {type(item).__name__}")
        missing_cols = [col for col in tuple_cols if col not in item]
        if missing_cols:
            raise ValueError(
                f"reconstructed[{idx}] is missing columns {missing_cols}; expected {tuple_cols}"
            )
        rows.append({col: item[col] for col in tuple_cols})

    anon = pd.DataFrame(rows, columns=tuple_cols)
    anon = anon.reindex(columns=original.columns)

    for col in original.columns:
        if col in anon.columns and not anon.empty:
            try:
                anon[col] = anon[col].astype(original[col].dtype, copy=False)
            except (TypeError, ValueError):
                pass

    return original, anon


def convert_from_id_val_tuples(df: pd.DataFrame, reconstructed: List[Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert reconstructed ID/value rows into QI/value rows and then to BRM dataframes.

    Args:
        df: Original attack dataframe with columns `id`, optional `qi*`, and `val`.
        reconstructed: Rows as dicts containing `id` and `val`.

    Returns:
        Tuple `(original, anon)` where `anon` has QI/value rows aligned to ids from `df`.
    """
    if 'id' not in df.columns or 'val' not in df.columns:
        raise ValueError("df must contain 'id' and 'val' columns.")

    original = df.drop(columns=['id']).copy()
    qi_cols = [col for col in df.columns if col not in ['id', 'val']]

    if reconstructed is None:
        reconstructed = []

    id_val_rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(reconstructed):
        if not isinstance(item, dict):
            raise ValueError(f"reconstructed[{idx}] must be a dict, got {type(item).__name__}")
        if 'id' not in item or 'val' not in item:
            raise ValueError("id/val dict rows must contain keys 'id' and 'val'.")
        id_val_rows.append({'id': item['id'], 'val': item['val']})

    if len(id_val_rows) == 0:
        return original, pd.DataFrame(columns=original.columns)

    recon_df = pd.DataFrame(id_val_rows, columns=['id', 'val'])
    id_to_qi_df = df[['id'] + qi_cols].drop_duplicates(subset=['id'])
    merged = recon_df.merge(id_to_qi_df, on='id', how='inner')

    qi_val_rows = merged[qi_cols + ['val']].to_dict('records')
    return convert_from_qi_val_tuples(df, qi_val_rows)

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

def initialize_qi_samples(
    df: pd.DataFrame,
    nunique: int,
    noise: int,
    qi_subsets: List[Dict],
    solve_type: str,
) -> tuple[List[Dict], int]:
    """Create initial samples from QI subsets for aggregate reconstruction.
    
    Args:
        df: DataFrame with 'id' and 'val' columns
        nunique: Number of unique values
        noise: Noise bound for counts (±noise)
        qi_subsets: List of QI subsets from get_qi_subset_list()
        solve_type: Aggregate solve type ('agg_row' or 'agg_known')
    
    Returns:
        Tuple of (initial_samples, next_qi_index)
        - initial_samples: List of sample dicts
        - next_qi_index: Index of next unused subset in qi_subsets
    """
    initial_samples = []
    all_ids = set(df['id'].values)
    covered_ids = set()
    qi_index = 0

    if solve_type == 'agg_known':
        # Prioritize single-column predicates so QI value coverage can be reached quickly.
        one_col_subsets = [subset for subset in qi_subsets if len(subset.get('qi_cols', [])) == 1]
        other_subsets = [subset for subset in qi_subsets if len(subset.get('qi_cols', [])) != 1]
        qi_subsets[:] = one_col_subsets + other_subsets

    all_qi_cols = sorted([col for col in df.columns if col.startswith('qi')])
    all_qi_values = {
        col: set(int(val) for val in df[col].drop_duplicates().tolist())
        for col in all_qi_cols
    }
    covered_qi_values = {col: set() for col in all_qi_cols}

    def add_sample(masked_ids: Set[int], qi_cols: List[str], qi_vals: List[int]) -> None:
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

        # Update covered IDs and covered QI values
        covered_ids.update(masked_ids)
        for col, val in zip(qi_cols, qi_vals):
            covered_qi_values[col].add(int(val))

    # Consume qi_subsets in order until required coverage is met, or subsets are exhausted.
    while qi_index < len(qi_subsets):
        needs_id_coverage = len(covered_ids) < len(all_ids)
        needs_qi_value_coverage = solve_type == 'agg_known' and (len(qi_subsets[qi_index]['qi_cols']) == 1)
        if not (needs_id_coverage or needs_qi_value_coverage):
            break
        masked_ids = get_qi_subsets_mask(df, qi_subsets, qi_index)
        qi_cols = qi_subsets[qi_index]['qi_cols']
        qi_vals = [int(v) for v in qi_subsets[qi_index]['qi_vals']]
        add_sample(masked_ids, qi_cols, qi_vals)
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

def compute_alc_measures(
    df: pd.DataFrame,
    reconstructed: List[Dict[str, Any]],
    target_column: str,
    path_to_dataset: str | None,
    attack_precision: float,
) -> Dict[str, Any]:
    if _is_missing_dataset_arg(path_to_dataset):
        # compute distinct number of values for column val in df
        num_distinct_vals = df['val'].nunique()
        if num_distinct_vals == 1:
            # throw exception
            raise ValueError(f"Should always have more than one distinct target value")
        base_precision = 1/num_distinct_vals
        return {
            'alc': (attack_precision - base_precision) / (1.0 - base_precision),
            'attack_precision': attack_precision,
            'attack_recall': 1.0,
            'attack_prc': attack_precision,
            'baseline_precision': base_precision,
            'baseline_recall': 1.0,
            'baseline_prc': base_precision,
        }

    # At this point, df is the original dataset, but with id, qiX, and val columns.
    # Determine whether reconstructed rows use id/val or qi*/val by inspecting reconstructed itself.
    if len(reconstructed) == 0:
        print("Skipping BRM/ALC evaluation because reconstructed is empty.")
        return {}

    first_row = reconstructed[0]
    if not isinstance(first_row, dict):
        raise ValueError(f"reconstructed rows must be dicts, got {type(first_row).__name__}")
    if 'val' not in first_row:
        raise ValueError("reconstructed rows must contain key 'val'.")

    uses_id_rows = 'id' in first_row
    if uses_id_rows:
        original, anon = convert_from_id_val_tuples(df, reconstructed)
    else:
        original, anon = convert_from_qi_val_tuples(df, reconstructed)

    print(f"Evaluating best row match attack and ALC for target column '{target_column}'")
    brm_results = brm_attack_simple(original, anon, 'val')

    return {
        'alc': brm_results['alc'],
        'attack_precision': brm_results['attack']['precision'],
        'attack_recall': brm_results['attack']['recall'],
        'attack_prc': brm_results['attack']['prc'],
        'baseline_precision': brm_results['baseline']['precision'],
        'baseline_recall': brm_results['baseline']['recall'],
        'baseline_prc': brm_results['baseline']['prc'],
    }

def attack_loop(nrows: int, 
                nunique: int, 
                mask_size: int, 
                noise: int,
                nqi: int = 3,
                target_accuracy: float = 0.99,
                min_num_rows: int = 3,
                vals_per_qi: int = 2,
                corr_strength: float = 0.0,
                max_samples: int = 20000,
                solve_type: str = 'agg_known',
                known_qi_fraction: float = 1.0,
                max_qi: int = 1000,
                max_refine: int = 2,
                seed: int = None,
                use_objective: bool = DEFAULT_USE_OBJECTIVE,
                time_limit_seconds: int = DEFAULT_TIME_LIMIT_SECONDS,
                slack_limit_multiple: int = DEFAULT_SLACK_LIMIT_MULTIPLE,
                slack_limit_min: int = DEFAULT_SLACK_LIMIT_MIN,
                path_to_dataset: str = '',
                target_column: str = '',
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
        corr_strength: Correlation probability for QI pairs (default: 0.0)
        target_accuracy: Target accuracy to stop early (default: 0.99)
        min_num_rows: Minimum number of rows in any aggregate (default: 3)
        known_qi_fraction: Fraction of rows with known QI values (default: 0.0, range: 0.0-1.0)
        max_qi: Maximum subset size to consider for aggregate queries (default: 1000)
        max_refine: Maximum number of refinement iterations (default: 2)
        seed: Random seed for reproducibility (default: None)
        use_objective: If True, use slack-minimization objective in reconstruction.
        time_limit_seconds: Solver time limit in seconds for each reconstruction call.
        slack_limit_multiple: Per-side slack upper bound multiplier on noise.
        slack_limit_min: Minimum slack limit (default: 10)
        path_to_dataset: Relative or absolute path to .parquet dataset (default: '')
        target_column: Target column name in dataset to map to 'val' (default: '')
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
    path_missing = _is_missing_dataset_arg(path_to_dataset)
    target_missing = _is_missing_dataset_arg(target_column)
    has_dataset_inputs = not path_missing or not target_missing
    if has_dataset_inputs and (path_missing != target_missing):
        raise ValueError("Both path_to_dataset and target_column must be provided together.")
    if path_missing and target_missing:
        path_to_dataset = ''
        target_column = ''

    # Build the ground truth dataframe, either synthetic or from external parquet.
    if not path_missing and not target_missing:
        dataset_params = get_dataset_generation_params(path_to_dataset, target_column)
        nrows = dataset_params['nrows']
        nunique = dataset_params['nunique']
        nqi = dataset_params['nqi']
        df = get_and_process_data(path_to_dataset, target_column)
        print(
            f"Using dataset mode: {path_to_dataset} (target={target_column})"
        )
        print(
            "Ignoring synthetic data args: "
            "nrows, nunique, nqi, vals_per_qi, corr_strength"
        )
    elif solve_type == 'pure_row':
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
        df = build_row_masks_qi(
            nrows=nrows,
            nunique=nunique,
            nqi=nqi,
            vals_per_qi=vals_per_qi,
            corr_strength=corr_strength,
        )
    
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
        working_samples, qi_index = initialize_qi_samples(df, nunique, noise, qi_subsets, solve_type)
        print(f"Total QI subsets available: {len(qi_subsets)}. qi_index {qi_index}.")
    else:
        working_samples = initialize_samples(df, mask_size, nunique, noise)
        print(f"start with {len(working_samples)} initial samples")
        num_masked = mask_size
    if current_num_samples == 0:
        current_num_samples = len(working_samples)
    
    num_suppressed = 0

    def add_aggregate_sample_once(samples: List[Dict], qi_index: int, avg_num_masked: float) -> tuple[bool, int, float]:
        """Try exactly one subset index and advance qi_index by one.

        Returns:
            (added, next_qi_index, updated_avg_num_masked)
            - added: True if a sample was appended, False if subset produced no usable noisy counts.
        """
        nonlocal num_suppressed
        if qi_index >= len(qi_subsets):
            return False, qi_index, avg_num_masked

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
            return False, qi_index, avg_num_masked

        samples.append({
            'ids': masked_ids,
            'qi_cols': qi_cols,            # for agg_known attacks
            'qi_vals': qi_vals,            # for agg_known attacks
            'noisy_counts': noisy_counts
        })
        return True, qi_index, avg_num_masked

    def add_next_aggregate_sample(samples: List[Dict], qi_index: int, avg_num_masked: float) -> tuple[bool, int, float]:
        """Append the next usable aggregate sample from qi_subsets."""
        while qi_index < len(qi_subsets):
            added, qi_index, avg_num_masked = add_aggregate_sample_once(samples, qi_index, avg_num_masked)
            if added:
                return True, qi_index, avg_num_masked
        return False, qi_index, avg_num_masked

    def choose_best_subset_index_for_missing(qi_index: int, missing_qi_cols: Set[str]) -> tuple[Optional[int], int]:
        """Pick remaining subset index that covers the most currently-missing QI columns."""
        best_index = None
        best_gain = 0
        for candidate_index in range(qi_index, len(qi_subsets)):
            candidate_cols = set(qi_subsets[candidate_index]['qi_cols'])
            gain = len(candidate_cols & missing_qi_cols)
            if gain > best_gain:
                best_gain = gain
                best_index = candidate_index
                if gain == len(missing_qi_cols):
                    break
        return best_index, best_gain

    def get_qi_coverage(samples: List[Dict]) -> tuple[set[str], List[str]]:
        """Return covered QI columns and missing QI columns for current samples."""
        covered_qi_cols = set()
        for sample in samples:
            covered_qi_cols.update(sample.get('qi_cols', []))
        missing_qi_cols = [col for col in all_qi_cols if col not in covered_qi_cols]
        return covered_qi_cols, missing_qi_cols

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
                    added, qi_index, avg_num_masked = add_next_aggregate_sample(samples, qi_index, avg_num_masked)
                    if not added:
                        print(f"Exhausted QI subsets at index {qi_index}")
                        break
                    continue
                
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

        # For aggregate-known reconstruction, ensure sampled predicates touch all QI columns.
        if solve_type == 'agg_known' and known_qi_fraction != 1.0:
            covered_qi_cols, missing_qi_cols = get_qi_coverage(samples)
            print(
                "QI coverage check before solve: "
                f"covered={len(covered_qi_cols)}/{len(all_qi_cols)}, "
                f"missing={len(missing_qi_cols)}, qi_index={qi_index}/{len(qi_subsets)}"
            )
            if len(missing_qi_cols) > 0:
                print(
                    f"QI coverage incomplete before solve ({len(missing_qi_cols)} missing columns). "
                    "Adding more aggregate samples."
                )
                print(f"Missing QI columns: {missing_qi_cols}")
            coverage_samples_added = 0
            while len(missing_qi_cols) > 0:
                missing_set = set(missing_qi_cols)
                best_index, best_gain = choose_best_subset_index_for_missing(qi_index, missing_set)
                if best_index is None or best_gain == 0:
                    print(
                        "No remaining QI subset can cover missing columns; proceeding to solver. "
                        f"missing={len(missing_qi_cols)}, qi_index={qi_index}/{len(qi_subsets)}"
                    )
                    print(f"Still missing QI columns: {missing_qi_cols}")
                    break

                if best_index != qi_index:
                    qi_subsets[qi_index], qi_subsets[best_index] = qi_subsets[best_index], qi_subsets[qi_index]

                selected_cols = qi_subsets[qi_index]['qi_cols']
                print(
                    "Selected targeted coverage subset "
                    f"at qi_index={qi_index} covering up to {best_gain} missing columns: "
                    f"qi_cols={selected_cols}"
                )

                added, qi_index, avg_num_masked = add_aggregate_sample_once(samples, qi_index, avg_num_masked)
                if not added:
                    print(
                        "Targeted subset produced no usable noisy counts (suppressed); "
                        f"qi_index advanced to {qi_index}/{len(qi_subsets)}"
                    )
                    continue

                coverage_samples_added += 1
                new_sample = samples[-1]
                print(
                    "Added coverage sample "
                    f"#{coverage_samples_added} using subset_index={qi_index - 1}: "
                    f"qi_cols={new_sample.get('qi_cols', [])}, "
                    f"matched_rows={len(new_sample.get('ids', []))}, "
                    f"noisy_count_bins={len(new_sample.get('noisy_counts', []))}"
                )
                pp.pprint(new_sample)
                covered_qi_cols, missing_qi_cols = get_qi_coverage(samples)
                print(
                    "QI coverage progress: "
                    f"covered={len(covered_qi_cols)}/{len(all_qi_cols)}, "
                    f"missing={len(missing_qi_cols)}, qi_index={qi_index}/{len(qi_subsets)}"
                )
            if len(missing_qi_cols) == 0:
                print(
                    "QI coverage satisfied before solve. "
                    f"Coverage samples added in this pass: {coverage_samples_added}."
                )
        
        # Reconstruct and measure
        print(f"Begin {solve_type} reconstruction with {len(samples)} samples\n    (current_num_samples={current_num_samples}, working_samples={len(working_samples)}, qi_index={qi_index}, num_suppressed={num_suppressed})")
        qi_match_accuracy = 0.0
        if solve_type in ['pure_row', 'agg_row']:
            reconstructed, num_equations, solver_metrics = reconstruct_by_row(
                samples,
                noise,
                seed,
                use_objective=use_objective,
                time_limit_seconds=time_limit_seconds,
                slack_limit_multiple=slack_limit_multiple,
                slack_limit_min=slack_limit_min,
            )
            accuracy = measure_by_row(df, reconstructed)
        elif solve_type == 'agg_known':
            if known_qi_fraction == 1.0:
                reconstructed, num_equations, solver_metrics = reconstruct_by_row(
                    samples,
                    noise,
                    seed,
                    use_objective=use_objective,
                    time_limit_seconds=time_limit_seconds,
                    slack_limit_multiple=slack_limit_multiple,
                    slack_limit_min=slack_limit_min,
                )
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
                reconstructed, num_equations, solver_metrics = reconstruct_by_aggregate_and_known_qi(
                    samples,
                    noise,
                    nrows,
                    all_qi_cols,
                    complete_known_qi_rows,
                    seed,
                    use_objective=use_objective,
                    time_limit_seconds=time_limit_seconds,
                    slack_limit_multiple=slack_limit_multiple,
                    slack_limit_min=slack_limit_min,
                )
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
        
        alc_result = compute_alc_measures(df, reconstructed, target_column, path_to_dataset, accuracy)
        # Record results
        current_result = {
            'num_samples': len(samples),
            'num_equations': num_equations,
            'measure': accuracy,
            'qi_match_measure': qi_match_accuracy,
            'alc': alc_result,
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
        solver_status = solver_metrics.get('status')
        solver_status_string = solver_metrics.get('status_string', 'UNKNOWN')

        # Check stopping conditions
        if solver_status == 3:
            print(
                "Exit loop: Solver returned infeasible "
                f"(status={solver_status}, {solver_status_string})"
            )
            finished = True
            exit_reason = 'solution_infeasible'
        elif solver_status == 9:
            print(
                "Exit loop: Solver returned time_limit "
                f"(status={solver_status}, {solver_status_string})"
            )
            finished = True
            exit_reason = 'out_of_bounds_solution'
        elif solve_type in ['agg_row', 'agg_known'] and qi_index >= len(qi_subsets):
            print("Exit loop: No more QI subsets to use")
            finished = True
            exit_reason = 'no_more_qi_subsets'
        else:
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
                if solve_type == 'pure_row' and next_samples > max_samples:
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
            'corr_strength': corr_strength,
            'actual_vals_per_qi': actual_vals_per_qi,
            'known_qi_fraction': known_qi_fraction,
            'max_qi': max_qi,
            'seed': seed,
            'use_objective': use_objective,
            'time_limit_seconds': time_limit_seconds,
            'slack_limit_multiple': slack_limit_multiple,
            'slack_limit_min': slack_limit_min,
            'path_to_dataset': path_to_dataset,
            'target_column': target_column,
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

        # Encourage prompt memory reclamation between repeated solver calls.
        _print_memory_usage("Before attack_loop iteration cleanup")
        reconstructed = None
        solver_metrics = None
        samples = None
        mixing = None
        sep = None
        current_result = None
        alc_result = None
        collected = gc.collect()
        _print_memory_usage(f"After attack_loop iteration cleanup (gc_collected={collected})")
            
        if finished:
            break
    return save_dict
        

def main():
    """Main function to run parameter sweep experiments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run row mask attack experiments')
    parser.add_argument('--job_num', type=int, default=None,
                       help='Job number to run from parameter combinations')
    parser.add_argument('--slurm_run', type=int, default=None,
                       help='Only use the experiment with this slurm_run number')
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
    parser.add_argument('--corr_strength', type=float, default=None,
                       help='Correlation probability for QI pairs (0.0-1.0)')
    parser.add_argument('--known_qi_fraction', type=float, default=None,
                       help='Fraction of rows with known QI values (0.0-1.0)')
    parser.add_argument('--max_qi', type=int, default=None,
                       help='Maximum subset size for aggregate queries')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use before quitting')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument(
        '--use_objective',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Whether to use slack-minimization objective in reconstruction',
    )
    parser.add_argument('--time_limit_seconds', type=int, default=None,
                       help='Solver time limit in seconds')
    parser.add_argument('--slack_limit_multiple', type=int, default=None,
                       help='Per-side slack upper bound multiplier on noise')
    parser.add_argument('--slack_limit_min', type=int, default=None,
                       help='Minimum slack limit')
    parser.add_argument('--path_to_dataset', type=str, default=None,
                       help='Path to a .parquet dataset, relative to current working directory')
    parser.add_argument('--target_column', type=str, default=None,
                       help='Dataset column to use as target and rename to val')
    
    args = parser.parse_args()
    job_num = args.job_num
    
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
    if args.slurm_run is not None:
        experiments = [
            exp for exp in experiments
            if exp.get('slurm_run', args.slurm_run) == args.slurm_run
        ]
        if len(experiments) == 0:
            print(f"No experiments found with slurm_run={args.slurm_run}")
            return
    
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
        'corr_strength': 0.0,
        'known_qi_fraction': 1.0,
        'max_qi': 1000,
        'max_samples': max_samples,
        'seed': None,
        'use_objective': DEFAULT_USE_OBJECTIVE,
        'time_limit_seconds': DEFAULT_TIME_LIMIT_SECONDS,
        'slack_limit_multiple': DEFAULT_SLACK_LIMIT_MULTIPLE,
        'slack_limit_min': DEFAULT_SLACK_LIMIT_MIN,
        'path_to_dataset': '',
        'target_column': '',
    }

    dataset_param_cache: Dict[tuple[str, str], Dict[str, int]] = {}

    def get_dataset_params_cached(path_to_dataset: str, target_column: str) -> Dict[str, int]:
        cache_key = (path_to_dataset, target_column)
        if cache_key not in dataset_param_cache:
            dataset_param_cache[cache_key] = get_dataset_generation_params(path_to_dataset, target_column)
        return dataset_param_cache[cache_key]
    
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
        args.corr_strength is not None,
        args.known_qi_fraction is not None,
        args.max_qi is not None,
        args.max_samples is not None,
        args.seed is not None,
        args.use_objective is not None,
        args.time_limit_seconds is not None,
        args.slack_limit_multiple is not None,
        args.slack_limit_min is not None,
        args.path_to_dataset is not None,
        args.target_column is not None,
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
            'corr_strength': args.corr_strength if args.corr_strength is not None else defaults['corr_strength'],
            'known_qi_fraction': args.known_qi_fraction if args.known_qi_fraction is not None else defaults['known_qi_fraction'],
            'max_qi': args.max_qi if args.max_qi is not None else defaults['max_qi'],
            'max_samples': args.max_samples if args.max_samples is not None else defaults['max_samples'],
            'seed': args.seed if args.seed is not None else defaults['seed'],
            'use_objective': args.use_objective if args.use_objective is not None else defaults['use_objective'],
            'time_limit_seconds': args.time_limit_seconds if args.time_limit_seconds is not None else defaults['time_limit_seconds'],
            'slack_limit_multiple': args.slack_limit_multiple if args.slack_limit_multiple is not None else defaults['slack_limit_multiple'],
            'slack_limit_min': args.slack_limit_min if args.slack_limit_min is not None else defaults['slack_limit_min'],
            'path_to_dataset': args.path_to_dataset if args.path_to_dataset is not None else defaults['path_to_dataset'],
            'target_column': args.target_column if args.target_column is not None else defaults['target_column'],
        }
        path_missing = _is_missing_dataset_arg(params['path_to_dataset'])
        target_missing = _is_missing_dataset_arg(params['target_column'])
        if path_missing != target_missing:
            raise ValueError("path_to_dataset and target_column must either both be provided or both be omitted.")
        if path_missing and target_missing:
            params['path_to_dataset'] = ''
            params['target_column'] = ''
        if not path_missing:
            dataset_params = get_dataset_params_cached(params['path_to_dataset'], params['target_column'])
            params['nrows'] = dataset_params['nrows']
            params['nunique'] = dataset_params['nunique']
            params['nqi'] = dataset_params['nqi']
            # These are synthetic-generation knobs and should revert to defaults in dataset mode.
            params['vals_per_qi'] = defaults['vals_per_qi']
            params['corr_strength'] = defaults['corr_strength']

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
            corr_strength=params['corr_strength'],
            known_qi_fraction=params['known_qi_fraction'],
            max_qi=params['max_qi'],
            solve_type=params['solve_type'],
            seed=params['seed'],
            use_objective=params['use_objective'],
            time_limit_seconds=params['time_limit_seconds'],
            slack_limit_multiple=params['slack_limit_multiple'],
            path_to_dataset=params['path_to_dataset'],
            target_column=params['target_column'],
            slack_limit_min=params['slack_limit_min'],
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
    max_time_minutes = int((60*24) * (24/24) * 7)   # We'll set slurm to this
    time_include_threshold_seconds = int((60*60*24) * (24/24) * 7)
    max_memory = '20G'
    test_params = []
    
    seen = set()

    finished_param_keys = set()
    finished_param_keys_from_finished_true = set()
    finished_param_keys_from_overtime = set()
    finished_filter_applied = False
    finished_filter_missing_cols: list[str] = []
    overtime_rule_available = False
    source_finished_rows = 0
    source_overtime_rows = 0
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
                'corr_strength',
                'known_qi_fraction',
                'max_qi',
                'max_samples',
                'seed',
                'use_objective',
                'time_limit_seconds',
                'slack_limit_multiple',
                'slack_limit_min',
                'path_to_dataset',
                'target_column',
                'target_accuracy',
            ]
            missing_cols = [col for col in param_cols + ['finished'] if col not in results_df.columns]
            if 'max_qi' in missing_cols:
                results_df['max_qi'] = defaults['max_qi']
                missing_cols.remove('max_qi')
            if 'corr_strength' in missing_cols:
                results_df['corr_strength'] = defaults['corr_strength']
                missing_cols.remove('corr_strength')
            if 'path_to_dataset' in missing_cols:
                results_df['path_to_dataset'] = defaults['path_to_dataset']
                missing_cols.remove('path_to_dataset')
            if 'target_column' in missing_cols:
                results_df['target_column'] = defaults['target_column']
                missing_cols.remove('target_column')
            if 'use_objective' in missing_cols:
                results_df['use_objective'] = defaults['use_objective']
                missing_cols.remove('use_objective')
            if 'time_limit_seconds' in missing_cols:
                results_df['time_limit_seconds'] = defaults['time_limit_seconds']
                missing_cols.remove('time_limit_seconds')
            if 'slack_limit_multiple' in missing_cols:
                results_df['slack_limit_multiple'] = defaults['slack_limit_multiple']
                missing_cols.remove('slack_limit_multiple')
            if 'slack_limit_min' in missing_cols:
                results_df['slack_limit_min'] = defaults['slack_limit_min']
                missing_cols.remove('slack_limit_min')
            if missing_cols:
                print(f"Warning: {result_parquet} missing columns {missing_cols}; skipping finished filter.")
                finished_filter_missing_cols = list(missing_cols)
            else:
                finished_filter_applied = True
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
                    'time_limit_seconds',
                    'slack_limit_multiple',
                    'slack_limit_min',
                    'seed',
                }
                results_df.loc[results_df['solve_type'] != 'agg_known', 'known_qi_fraction'] = 1.0
                finished_df = results_df[results_df['finished'] == True]
                print(f"Found {len(finished_df)} finished jobs in prior results.")
                source_finished_rows = len(finished_df)
                overtime_df = pd.DataFrame()
                overtime_rule_available = 'solver_metrics_runtime' in results_df.columns
                if overtime_rule_available:
                    overtime_df = results_df[
                        (results_df['finished'] == False)
                        & (results_df['solver_metrics_runtime'] > time_include_threshold_seconds)
                    ]
                    source_overtime_rows = len(overtime_df)
                print(f"Including {len(overtime_df)} additional jobs to filter based on time threshold of {time_include_threshold_seconds} seconds.")

                def is_missing_value(value):
                    if isinstance(value, (list, dict, tuple, set)):
                        return False
                    try:
                        return bool(pd.isna(value))
                    except (TypeError, ValueError):
                        return False

                def row_to_filename(row: pd.Series) -> str:
                    row_params = {}
                    for col in param_cols:
                        val = row[col]
                        if is_missing_value(val):
                            row_params[col] = defaults.get(col)
                        elif col in int_cols:
                            row_params[col] = int(val)
                        elif col in ['known_qi_fraction', 'target_accuracy']:
                            row_params[col] = float(val)
                        elif col == 'corr_strength':
                            row_params[col] = float(val)
                        elif col == 'use_objective':
                            row_params[col] = bool(val)
                        else:
                            row_params[col] = val
                    return generate_filename(row_params, row_params['target_accuracy'])

                for _, row in finished_df.iterrows():
                    file_name = row_to_filename(row)
                    finished_param_keys_from_finished_true.add(file_name)
                for _, row in overtime_df.iterrows():
                    file_name = row_to_filename(row)
                    finished_param_keys_from_overtime.add(file_name)

                finished_param_keys = (
                    finished_param_keys_from_finished_true
                    | finished_param_keys_from_overtime
                )
    
    num_finished_jobs = 0
    num_finished_jobs_from_finished_true = 0
    num_finished_jobs_from_overtime = 0
    num_finished_jobs_from_both = 0
    for key in finished_param_keys:
        #print(f"Finished: {key}")
        pass

    def normalize_grid_value(value):
        if isinstance(value, list):
            return value
        return [value]

    for exp in experiments:
        if exp['dont_run'] is True:
            continue
        # Get seed list from experiment, default to [None] if not specified
        seed_list = normalize_grid_value(exp.get('seed', [None]))
        # Get known_qi_fraction list from experiment, default to [1.0] if not specified
        known_qi_fraction_list = normalize_grid_value(exp.get('known_qi_fraction', [1.0]))
        use_objective_list = normalize_grid_value(exp.get('use_objective', [defaults['use_objective']]))
        time_limit_seconds_list = normalize_grid_value(exp.get('time_limit_seconds', [defaults['time_limit_seconds']]))
        slack_limit_multiple_list = normalize_grid_value(exp.get('slack_limit_multiple', [defaults['slack_limit_multiple']]))
        slack_limit_min_list = normalize_grid_value(exp.get('slack_limit_min', [defaults['slack_limit_min']]))
        path_to_dataset_list = [
            '' if _is_missing_dataset_arg(path) else str(path)
            for path in normalize_grid_value(exp.get('path_to_dataset', [defaults['path_to_dataset']]))
        ]
        target_column_list = [
            '' if _is_missing_dataset_arg(target) else str(target)
            for target in normalize_grid_value(exp.get('target_column', [defaults['target_column']]))
        ]
        has_dataset_mode = any(not _is_missing_dataset_arg(path) for path in path_to_dataset_list)
        if has_dataset_mode:
            inferred_nrows = set()
            inferred_nunique = set()
            inferred_nqi = set()
            for path_to_dataset in path_to_dataset_list:
                for target_column in target_column_list:
                    path_missing = _is_missing_dataset_arg(path_to_dataset)
                    target_missing = _is_missing_dataset_arg(target_column)
                    if path_missing and target_missing:
                        continue
                    if path_missing != target_missing:
                        continue
                    dataset_params = get_dataset_params_cached(path_to_dataset, target_column)
                    inferred_nrows.add(dataset_params['nrows'])
                    inferred_nunique.add(dataset_params['nunique'])
                    inferred_nqi.add(dataset_params['nqi'])
            if len(inferred_nrows) == 0:
                raise ValueError("Experiment has path_to_dataset entries but no valid target_column entries.")
            # Fill experiment structure from parquet metadata for dataset-backed runs.
            exp['nrows'] = sorted(inferred_nrows)
            exp['nunique'] = sorted(inferred_nunique)
            exp['nqi'] = sorted(inferred_nqi)
            exp['vals_per_qi'] = [defaults['vals_per_qi']]
            exp['corr_strength'] = [defaults['corr_strength']]

        max_qi_list = normalize_grid_value(exp.get('max_qi', [defaults['max_qi']]))
        corr_strength_list = normalize_grid_value(exp.get('corr_strength', [defaults['corr_strength']]))

        valid_dataset_pairs = []
        for path_to_dataset in path_to_dataset_list:
            for target_column in target_column_list:
                path_missing = _is_missing_dataset_arg(path_to_dataset)
                target_missing = _is_missing_dataset_arg(target_column)
                if path_missing != target_missing:
                    continue
                valid_dataset_pairs.append((path_to_dataset, target_column, path_missing))

        grid_axes = [
            exp['nrows'],
            exp['mask_size'],
            exp['nunique'],
            exp['noise'],
            exp['nqi'],
            max_qi_list,
            exp['min_num_rows'],
            exp['vals_per_qi'],
            corr_strength_list,
            known_qi_fraction_list,
            use_objective_list,
            time_limit_seconds_list,
            slack_limit_multiple_list,
            slack_limit_min_list,
            valid_dataset_pairs,
            seed_list,
        ]

        for (
            nrows,
            mask_size,
            nunique,
            noise,
            nqi,
            max_qi,
            min_num_rows,
            vals_per_qi,
            corr_strength,
            known_qi_fraction,
            use_objective,
            time_limit_seconds,
            slack_limit_multiple,
            slack_limit_min,
            dataset_pair,
            seed,
        ) in product(*grid_axes):
            path_to_dataset, target_column, path_missing = dataset_pair
            effective_nrows = nrows
            effective_nunique = nunique
            effective_nqi = nqi
            effective_vals_per_qi = vals_per_qi
            effective_corr_strength = corr_strength
            if not path_missing:
                dataset_params = get_dataset_params_cached(path_to_dataset, target_column)
                effective_nrows = dataset_params['nrows']
                effective_nunique = dataset_params['nunique']
                effective_nqi = dataset_params['nqi']
                # Revert synthetic generation knobs to defaults.
                effective_vals_per_qi = defaults['vals_per_qi']
                effective_corr_strength = defaults['corr_strength']
            params = {
                'nrows': effective_nrows,
                'solve_type': exp['solve_type'],
                'mask_size': mask_size,
                'nunique': effective_nunique,
                'noise': noise,
                'nqi': effective_nqi,
                'min_num_rows': min_num_rows,
                'vals_per_qi': effective_vals_per_qi,
                'corr_strength': effective_corr_strength,
                'known_qi_fraction': known_qi_fraction,
                'max_qi': max_qi,
                'max_samples': max_samples,
                'seed': seed,
                'use_objective': bool(use_objective),
                'time_limit_seconds': int(time_limit_seconds),
                'slack_limit_multiple': int(slack_limit_multiple),
                'slack_limit_min': int(slack_limit_min),
                'path_to_dataset': path_to_dataset,
                'target_column': target_column,
            }
            key = generate_filename(params, target_accuracy)
            if key in finished_param_keys:
                num_finished_jobs += 1
                in_finished_true = key in finished_param_keys_from_finished_true
                in_overtime = key in finished_param_keys_from_overtime
                if in_finished_true and in_overtime:
                    num_finished_jobs_from_both += 1
                elif in_finished_true:
                    num_finished_jobs_from_finished_true += 1
                elif in_overtime:
                    num_finished_jobs_from_overtime += 1
                continue
            if key not in seen:
                seen.add(key)
                test_params.append(params)
    
    # If no job_num, just print all combinations
    if job_num is None:
        for i, params in enumerate(test_params):
            print(f"Job {i}: {params}")
        
        # Create run.slurm file
        num_jobs = len(test_params) - 1  # Array range is 0 to num_jobs-1
        slurm_content = f"""#!/bin/bash
#SBATCH --job-name=recon_test
#SBATCH --output=/INS/syndiffix/work/paul/github/reconstruction_tests/row_mask_attacks/slurm_out/job_{args.slurm_run}_%A_%a.out
#SBATCH --time={max_time_minutes}
#SBATCH --mem={max_memory}
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{num_jobs}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
source /INS/syndiffix/work/paul/github/reconstruction_tests/.venv/bin/activate
python /INS/syndiffix/work/paul/github/reconstruction_tests/row_mask_attacks/run_row_mask_attack.py --job_num $arrayNum
"""
        slurm_file_name = 'run.slurm'
        if args.slurm_run is not None:
            slurm_file_name = f"run_{args.slurm_run}.slurm"
            slurm_content = slurm_content.replace(
                "#SBATCH --job-name=recon_test",
                f"#SBATCH --job-name={args.slurm_run}_recon_test",
            )
            slurm_content = slurm_content.replace(
                "run_row_mask_attack.py --job_num $arrayNum",
                f"run_row_mask_attack.py --slurm_run {args.slurm_run} --job_num $arrayNum",
            )
        slurm_file = Path(__file__).parent / slurm_file_name
        with open(slurm_file, 'w') as f:
            f.write(slurm_content)
        
        print(f"\nSLURM file created: {slurm_file}")
        print(f"Total jobs: {len(test_params)} (array: 0-{num_jobs}). {num_finished_jobs} are already finished.")
        print("Finished-job filter summary:")
        if not result_parquet.exists():
            print(f"  No prior results file found at {result_parquet}.")
        elif not finished_filter_applied:
            if finished_filter_missing_cols:
                print(
                    "  Filter skipped because result.parquet is missing required columns: "
                    f"{finished_filter_missing_cols}"
                )
            else:
                print("  Filter not applied (could not read or parse prior results).")
        else:
            overlap_keys = (
                finished_param_keys_from_finished_true
                & finished_param_keys_from_overtime
            )
            print(
                "  Source condition A (finished==True): "
                f"{source_finished_rows} rows -> {len(finished_param_keys_from_finished_true)} unique file keys"
            )
            if overtime_rule_available:
                print(
                    "  Source condition B (finished==False, "
                    f"solver_metrics_runtime>{time_include_threshold_seconds}): "
                    f"{source_overtime_rows} rows -> {len(finished_param_keys_from_overtime)} unique file keys"
                )
            else:
                print(
                    "  Source condition B unavailable: requires columns "
                    "'solver_metrics_runtime'"
                )
            print(f"  Overlap between A and B: {len(overlap_keys)} unique file keys")
            print(
                "  In current parameter grid, skipped as already finished: "
                f"{num_finished_jobs} total"
            )
            print(
                f"    A only: {num_finished_jobs_from_finished_true}, "
                f"B only: {num_finished_jobs_from_overtime}, "
                f"A and B: {num_finished_jobs_from_both}"
            )
        return
    
    # Check if job_num is valid
    if job_num < 0 or job_num >= len(test_params):
        print(f"Error: job_num {job_num} out of range [0, {len(test_params)-1}]")
        return
    
    # Get parameters for this job
    params = test_params[job_num]
    
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
    print(f"Running job {job_num}: {params}")
    
    attack_loop(
        nrows=params['nrows'],
        nunique=params['nunique'],
        mask_size=params['mask_size'],
        noise=params['noise'],
        nqi=params['nqi'],
        target_accuracy=target_accuracy,
        min_num_rows=params['min_num_rows'],
        vals_per_qi=params['vals_per_qi'],
        corr_strength=params['corr_strength'],
        known_qi_fraction=params['known_qi_fraction'],
        max_qi=params['max_qi'],
        max_samples=params['max_samples'],
        solve_type=params['solve_type'],
        seed=params['seed'],
        use_objective=params['use_objective'],
        time_limit_seconds=params['time_limit_seconds'],
        slack_limit_multiple=params['slack_limit_multiple'],
        slack_limit_min=params['slack_limit_min'],
        path_to_dataset=params.get('path_to_dataset'),
        target_column=params.get('target_column'),
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
