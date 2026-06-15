from __future__ import annotations

import pandas as pd
from typing import Any, List, Dict, Set, Optional
import numpy as np
import gc
import json
import hashlib
import math
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
from slurm_manager.core import RunManifestBuilder, init_payload, clean_payload


pp = pprint.PrettyPrinter(indent=2)
print = partial(print, flush=True)
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from df_builds.build_row_masks import build_row_masks_qi, get_required_num_distinct
from reconstruct import reconstruct_by_row, measure_by_row, measure_by_aggregate, reconstruct_by_aggregate_and_known_qi
from compute_separation import compute_separation_metrics

solve_type_map = {
    'agg_row': 'ar',
    'agg_known': 'ak',
}

DEFAULT_USE_OBJECTIVE = True
DEFAULT_TIME_LIMIT_SECONDS = (3 * 24 * 60 * 60)  # 3 days in seconds
DEFAULT_SLACK_LIMIT_MULTIPLE = 2
DEFAULT_SLACK_LIMIT_MIN = 10
DEFAULT_MAX_NUM_CONTINGENCY_TABLES = 100000


def parse_args() -> argparse.Namespace:
    """Parse the arguments supplied by the generated sbatch script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        required=True,
        help="Manifest JSON created by the conductor for this sbatch array.",
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory where this program writes its result JSON file.",
    )
    parser.add_argument(
        "--job_num",
        required=True,
        type=int,
        help="Manifest entry number for this array task.",
    )
    return parser.parse_args()


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


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict values into underscore-joined keys."""
    flattened = {}
    for key, value in data.items():
        flat_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, flat_key))
        else:
            flattened[flat_key] = value
    return flattened

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
        max_num_contingency_tables = params.get(
            'max_num_contingency_tables',
            DEFAULT_MAX_NUM_CONTINGENCY_TABLES,
        )
        max_num_contingency_tables_str = (
            f"_mct{max_num_contingency_tables}"
            if max_num_contingency_tables != DEFAULT_MAX_NUM_CONTINGENCY_TABLES
            else ""
        )
        return (
            f"ds{dataset_stem}_tc{target_label}_h{dataset_hash}_"
            f"mf{params['mask_size']}_n{params['noise']}_mnr{params['min_num_rows']}"
            f"{max_qi_str}{max_num_contingency_tables_str}_"
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
    max_num_contingency_tables = params.get(
        'max_num_contingency_tables',
        DEFAULT_MAX_NUM_CONTINGENCY_TABLES,
    )
    max_num_contingency_tables_str = (
        f"_mct{max_num_contingency_tables}"
        if max_num_contingency_tables != DEFAULT_MAX_NUM_CONTINGENCY_TABLES
        else ""
    )
    corr_strength_str = ""
    if corr_strength and corr_strength > 0.0:
        corr_strength_str = f"_cs{int(round(corr_strength * 100))}"
    file_name = (f"nr{params['nrows']}_mf{params['mask_size']}_"
                f"nu{params['nunique']}_qi{params['nqi']}_n{params['noise']}_"
                f"mnr{params['min_num_rows']}_vpq{vals_per_qi}"
                f"{max_qi_str}{max_num_contingency_tables_str}{corr_strength_str}_"
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

def contingency_table_columns(df: pd.DataFrame, num_contingency_tables: int) -> List[List[str]]:
    """Return QI column groups ordered by distinct value-vector count."""
    import itertools

    if num_contingency_tables < 1:
        return []

    qi_cols = sorted([col for col in df.columns if col.startswith('qi')])
    candidates = []
    for subset_size in range(1, len(qi_cols) + 1):
        for cols_tuple in itertools.combinations(qi_cols, subset_size):
            cols = list(cols_tuple)
            num_distinct_vectors = int(df[cols].drop_duplicates().shape[0])
            candidates.append((num_distinct_vectors, subset_size, cols))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return [cols for _, _, cols in candidates[:num_contingency_tables]]

def get_qi_subset_list(
    df: pd.DataFrame,
    min_num_rows: int,
    target_num_rows: int,
    max_qi: int = 1000,
    max_num_contingency_tables: int = DEFAULT_MAX_NUM_CONTINGENCY_TABLES,
    contingency_tables: Optional[List[List[str]]] = None,
) -> List[Dict]:
    """ Generates list of QI column subsets grouped by qi_cols.
    
    Args:
        df: DataFrame with QI columns (qi0, qi1, ..., qiN)
        min_num_rows: Minimum number of rows in any aggregate (default: 5)
        target_num_rows: Retained for compatibility; no longer used for sorting.
        max_qi: Maximum subset size to consider
        max_num_contingency_tables: Maximum contingency tables to generate when not supplied
        contingency_tables: Optional ordered QI column groups to use as contingency tables
    
    Returns:
        List of subsets in the order their column groups appear in contingency_tables.
    """
    # Find all QI columns
    all_qi_cols = sorted([col for col in df.columns if col.startswith('qi')])
    
    if len(all_qi_cols) == 0:
        return []
    
    qi_subsets = []
    
    # Iterate through selected contingency tables.
    max_subset_size = min(len(all_qi_cols), max_qi)
    if max_subset_size < 1 or max_num_contingency_tables < 1:
        return []
    if contingency_tables is None:
        contingency_tables = contingency_table_columns(df, max_num_contingency_tables)
    for qi_cols in contingency_tables:
        if len(qi_cols) > max_subset_size:
            continue

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

def create_qi_samples(
    df: pd.DataFrame,
    nunique: int,
    noise: int,
    qi_subsets: List[Dict],
) -> List[Dict]:
    """Create samples for every QI subset in order.
    
    Args:
        df: DataFrame with 'id' and 'val' columns
        nunique: Number of unique values
        noise: Noise bound for counts (±noise)
        qi_subsets: List of QI subsets from get_qi_subset_list()
    
    Returns:
        List of sample dicts
    """
    samples = []

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
        samples.append({
            'ids': masked_ids,
            'qi_cols': qi_cols,
            'qi_vals': qi_vals,
            'noisy_counts': noisy_counts
        })

    for qi_index in range(len(qi_subsets)):
        masked_ids = get_qi_subsets_mask(df, qi_subsets, qi_index)
        qi_cols = qi_subsets[qi_index]['qi_cols']
        qi_vals = [int(v) for v in qi_subsets[qi_index]['qi_vals']]
        add_sample(masked_ids, qi_cols, qi_vals)

    return samples

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
                max_num_contingency_tables: int = DEFAULT_MAX_NUM_CONTINGENCY_TABLES,
                max_refine: int = 2,
                seed: int = None,
                use_objective: bool = DEFAULT_USE_OBJECTIVE,
                time_limit_seconds: int = DEFAULT_TIME_LIMIT_SECONDS,
                slack_limit_multiple: int = DEFAULT_SLACK_LIMIT_MULTIPLE,
                slack_limit_min: int = DEFAULT_SLACK_LIMIT_MIN,
                path_to_dataset: str = '',
                target_column: str = '',
                ) -> dict:
    """Run one reconstruction from the initial noisy samples.
    
    Args:
        nrows: Number of rows in the dataframe
        nunique: Number of unique values
        mask_size: Retained for filename compatibility; aggregate attacks use QI predicates.
        noise: Noise bound for counts (±noise)
        nqi: Number of quasi-identifier columns
        vals_per_qi: Number of distinct values per QI column (default: 0, means auto compute)
        corr_strength: Correlation probability for QI pairs (default: 0.0)
        target_accuracy: Target accuracy retained in result metadata (default: 0.99)
        min_num_rows: Minimum number of rows in any aggregate (default: 3)
        known_qi_fraction: Fraction of rows with known QI values (default: 0.0, range: 0.0-1.0)
        max_qi: Maximum subset size to consider for aggregate queries (default: 1000)
        max_num_contingency_tables: Maximum qualifying contingency tables to generate.
        max_refine: Retained for compatibility; no refinement is performed.
        seed: Random seed for reproducibility (default: None)
        use_objective: If True, use slack-minimization objective in reconstruction.
        time_limit_seconds: Solver time limit in seconds for each reconstruction call.
        slack_limit_multiple: Per-side slack upper bound multiplier on noise.
        slack_limit_min: Minimum slack limit (default: 10)
        path_to_dataset: Relative or absolute path to .parquet dataset (default: '')
        target_column: Target column name in dataset to map to 'val' (default: '')
    
    Returns:
        dict
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    save_dict = {}
    if solve_type not in ['agg_row', 'agg_known']:
        raise ValueError(f"Unsupported solve_type: {solve_type}")
    
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
    all_qi_cols = [col for col in df.columns if col.startswith('qi')]
    max_available_contingency_tables = sum(
        math.comb(len(all_qi_cols), subset_size)
        for subset_size in range(1, len(all_qi_cols) + 1)
    )
    num_contingency_tables = min(
        max(0, int(max_num_contingency_tables)),
        max_available_contingency_tables,
    )
    contingency_tables = contingency_table_columns(df, num_contingency_tables)
    qi_subsets = get_qi_subset_list(
        df,
        min_num_rows,
        int(round(min_num_rows * nunique * 1.5)),
        max_qi,
        max_num_contingency_tables,
        contingency_tables,
    )
    samples = create_qi_samples(df, nunique, noise, qi_subsets)
    print(f"Total QI subsets available: {len(qi_subsets)}. Created {len(samples)} samples.")
    
    num_suppressed = 0
    
    # Start timing
    start_time = time.time()

    # Reconstruct and measure once.
    print(f"Begin {solve_type} reconstruction with {len(samples)} samples\n    (qi_subsets={len(qi_subsets)}, num_suppressed={num_suppressed})")
    qi_match_accuracy = 0.0
    if solve_type == 'agg_row':
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

    actual_num_rows = int(np.mean([len(sample['ids']) for sample in samples])) if len(samples) > 0 else 0
    
    alc_result = compute_alc_measures(df, reconstructed, target_column, path_to_dataset, accuracy)
    # Record results
    result = {
        'num_samples': len(samples),
        'num_equations': num_equations,
        'accuracy': accuracy,
        'qi_match_accuracy': qi_match_accuracy,
        'alc': alc_result,
        'mixing': mixing,
        'actual_num_rows': actual_num_rows,
        'solver_metrics': solver_metrics,
        'separation': sep,
        'refine': 0,
    }
    pp.pprint(result)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    finished = True
    solver_status = solver_metrics.get('status')
    solver_status_string = solver_metrics.get('status_string', 'UNKNOWN')
    print(
        "Single reconstruction finished: "
        f"solver_status={solver_status}, "
        f"solver_status_string={solver_status_string}, "
        f"accuracy={accuracy:.4f}"
    )

    save_dict = {
        'actual_vals_per_qi': actual_vals_per_qi,
        'elapsed_time': elapsed_time,
        'finished': finished,
        'num_suppressed': num_suppressed,
    }
    save_dict.update(result)

    return flatten_dict(save_dict)
        

def run_experiment(parameters: dict[str, object], seed: int) -> dict[str, object]:
    """Run one experiment and return result fields to append to the payload.

    ``parameters`` contains the unprefixed parameters from the jobs JSON file.
    For example, if jobs.json contains ``{"sleep_seconds": 10, "alpha": 0.2}``,
    then ``parameters["sleep_seconds"]`` and ``parameters["alpha"]`` are
    available here.

    ``seed`` is the conductor-assigned run seed. Use it to make randomized work
    deterministic and reproducible.

    Return only experiment result fields from this function. The base payload
    already includes metadata and parameters via ``init_payload(entry)``.
    """
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
        'max_num_contingency_tables': DEFAULT_MAX_NUM_CONTINGENCY_TABLES,
        'max_refine': 2,
        'max_samples': 20000,
        'target_accuracy': 0.99,
        'use_objective': DEFAULT_USE_OBJECTIVE,
        'time_limit_seconds': DEFAULT_TIME_LIMIT_SECONDS,
        'slack_limit_multiple': DEFAULT_SLACK_LIMIT_MULTIPLE,
        'slack_limit_min': DEFAULT_SLACK_LIMIT_MIN,
        'path_to_dataset': '',
        'target_column': '',
    }

    attack_parameters = {
        key: parameters.get(key, default_value)
        for key, default_value in defaults.items()
    }
    attack_parameters['seed'] = seed

    print("Running attack with parameters:")
    pp.pprint(attack_parameters)

    result = attack_loop(**attack_parameters)
    result["experiment_finished"] = True
    pp.pprint(result)
    return result


def write_result_json(
    *,
    results_dir: str | Path,
    experiment_id: str,
    seed: int,
    payload: dict[str, object],
) -> Path:
    """Write the result JSON file where the conductor expects to find it."""
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / f"{experiment_id}_{seed}.json"
    result_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return result_path


def main() -> None:
    args = parse_args()

    # The manifest maps each SLURM array task to one experiment_id, one seed,
    # and the parameter dictionary from jobs.json.
    entries = RunManifestBuilder.load_manifest(args.manifest)
    entry = entries[args.job_num]

    # Base payload fields:
    #   m__experiment_id: conductor experiment identity
    #   m__seed: conductor run seed
    #   p__<name>: each jobs.json parameter, prefixed to avoid name collisions
    payload = init_payload(entry)
    payload_clean = clean_payload(payload)

    # Add experiment-specific result fields without prefixes. The conductor
    # also adds its own c__ fields after reading this JSON and before writing
    # results.parquet.
    payload.update(run_experiment(parameters=entry.parameters, seed=entry.seed))

    write_result_json(
        results_dir=args.results_dir,
        experiment_id=entry.experiment_id,
        seed=entry.seed,
        payload=payload,
    )


if __name__ == '__main__':
    main()
