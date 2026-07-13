from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import pprint
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from anonymity_loss_coefficient import BrmAttack


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROW_MASK_DIR = PROJECT_ROOT / "row_mask_attacks"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROW_MASK_DIR))

from reconstruct import (  # noqa: E402
    measure_by_aggregate,
    measure_by_row,
    reconstruct_by_aggregate_and_known_qi,
    reconstruct_by_row,
)


pp = pprint.PrettyPrinter(indent=2)
print = partial(print, flush=True)

DEFAULT_USE_OBJECTIVE = True
DEFAULT_TIME_LIMIT_SECONDS = (3 * 24 * 60 * 60)  # 3 days in seconds
DEFAULT_SLACK_LIMIT_MULTIPLE = 2
DEFAULT_SLACK_LIMIT_MIN = 10
DEFAULT_MAX_NUM_CONTINGENCY_TABLES = 200
DEFAULT_NOISE = 2
DEFAULT_START_QI_NUM = 1

REQUIRED_INFO_COLUMNS = {
    "seed",
    "p__input_path",
    "p__contingency_table",
    "p__output_path",
}

ALC_RESULT_FIELDS = (
    "alc",
    "attack_precision",
    "attack_recall",
    "attack_prc",
    "baseline_precision",
    "baseline_recall",
    "baseline_prc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the row-mask reconstruction attack from synthetic microdata tables."
    )
    parser.add_argument(
        "--info",
        required=True,
        help="Path to the parquet file containing dataset metadata (df_info).",
    )
    parser.add_argument(
        "--max_num_contingency_tables",
        type=int,
        default=DEFAULT_MAX_NUM_CONTINGENCY_TABLES,
        help="Maximum number of synthetic contingency tables to use per seed.",
    )
    parser.add_argument(
        "--known_qi_fraction",
        type=float,
        default=1.0,
        help="Fraction of rows whose full QI values are known to the attacker.",
    )
    parser.add_argument(
        "--noise",
        type=int,
        default=DEFAULT_NOISE,
        help="Noise bound for synthetic contingency-table counts.",
    )
    parser.add_argument(
        "--start_qi_num",
        type=int,
        default=DEFAULT_START_QI_NUM,
        help="Ignore contingency tables with fewer than this many QI columns.",
    )
    parser.add_argument(
        "--results_path",
        required=True,
        help="Path to the parquet file where attack results are written.",
    )
    return parser.parse_args()


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, flat_key))
        else:
            flattened[flat_key] = value
    return flattened


def _get_process_memory_mb() -> Optional[float]:
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
                            return float(parts[1]) / 1024.0
        except Exception:
            pass
    return None


def _print_memory_usage(label: str) -> None:
    mem_mb = _get_process_memory_mb()
    if mem_mb is None:
        print(f"{label}: memory usage unavailable")
    else:
        print(f"{label}: memory usage {mem_mb:.2f} MB")


def _append_seed_to_filename(path: Path, seed: Any) -> Path:
    return path.with_name(f"{path.stem}_{int(seed)}{path.suffix}")


def resolve_existing_path(
    path_str: str,
    *,
    info_path: Optional[Path] = None,
    seed: Any = None,
) -> Path:
    path = Path(path_str)
    base_candidates: list[Path] = []
    if path.is_absolute():
        base_candidates.append(path)
    else:
        base_candidates.extend(
            [
                Path.cwd() / path,
                PROJECT_ROOT / path,
                PROJECT_ROOT / "ctgan_builds" / path,
            ]
        )
        if info_path is not None:
            base_candidates.append(info_path.parent / path)

    for candidate in base_candidates:
        if candidate.exists():
            return candidate.resolve()

    if seed is not None:
        for candidate in base_candidates:
            seeded_candidate = _append_seed_to_filename(candidate, seed)
            if seeded_candidate.exists():
                return seeded_candidate.resolve()

    raise FileNotFoundError(f"Path not found: {path_str}")


def normalize_contingency_table(raw_value: Any) -> tuple[str, ...]:
    if not isinstance(raw_value, str):
        raise ValueError(f"p__contingency_table must be a JSON string, got {type(raw_value).__name__}")
    parsed = json.loads(raw_value)
    if not isinstance(parsed, list) or not all(isinstance(col, str) for col in parsed):
        raise ValueError(f"Invalid contingency table JSON: {raw_value!r}")
    if "val" not in parsed:
        raise ValueError(f"Contingency table must contain 'val': {raw_value!r}")
    if "splitter" not in parsed:
        raise ValueError(f"Contingency table must contain 'splitter': {raw_value!r}")
    return tuple(parsed)


def qi_columns_from_table(contingency_table: Iterable[str]) -> list[str]:
    return [col for col in contingency_table if col not in {"val", "splitter"}]


def load_source_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"id", "val", "splitter"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source dataframe {path} is missing required columns: {sorted(missing)}")
    return df


def load_microdata_dataframe(path: Path, contingency_table: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    expected_columns = list(contingency_table)
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Microdata table {path} is missing columns: {missing}")
    return df[expected_columns].copy()


def canonicalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.reindex(sorted(df.columns), axis=1).copy()
    if len(ordered.columns) > 0:
        ordered = ordered.sort_values(list(ordered.columns), kind="stable")
    return ordered.reset_index(drop=True)


def validate_seed_variation(df_info: pd.DataFrame, *, info_path: Path) -> None:
    print("Validating that different seeds produce different microdata tables")
    grouped = df_info.groupby("contingency_table_key", sort=False)
    for contingency_key, group in grouped:
        distinct_seeds = list(dict.fromkeys(group["seed"].tolist()))
        if len(distinct_seeds) < 2:
            continue

        candidate_rows = group.sort_values(["seed", "_row_order"], kind="stable")
        for left_idx, right_idx in itertools.combinations(candidate_rows.index.tolist(), 2):
            left = candidate_rows.loc[left_idx]
            right = candidate_rows.loc[right_idx]
            if left["seed"] == right["seed"]:
                continue

            contingency_table = tuple(left["contingency_table_list"])
            left_path = resolve_existing_path(
                left["p__output_path"],
                info_path=info_path,
                seed=left["seed"],
            )
            right_path = resolve_existing_path(
                right["p__output_path"],
                info_path=info_path,
                seed=right["seed"],
            )
            left_df = canonicalize_dataframe(load_microdata_dataframe(left_path, contingency_table))
            right_df = canonicalize_dataframe(load_microdata_dataframe(right_path, contingency_table))
            if not left_df.equals(right_df):
                print(
                    "Seed variation validated with "
                    f"contingency_table={list(contingency_table)}, "
                    f"seed_a={left['seed']}, seed_b={right['seed']}"
                )
                return

    raise ValueError(
        "Could not verify seed variation: every same-contingency-table comparison across "
        "different seeds was identical, or no such cross-seed pair was available."
    )


def prepare_info_dataframe(info_path: Path) -> pd.DataFrame:
    df_info = pd.read_parquet(info_path)
    missing_columns = REQUIRED_INFO_COLUMNS - set(df_info.columns)
    if missing_columns:
        raise ValueError(f"df_info is missing required columns: {sorted(missing_columns)}")

    df_info = df_info.copy().reset_index(drop=True)
    df_info["_row_order"] = np.arange(len(df_info), dtype=int)
    df_info["contingency_table_list"] = df_info["p__contingency_table"].map(normalize_contingency_table)
    df_info["contingency_table_key"] = df_info["contingency_table_list"].map(json.dumps)
    df_info["num_qi_columns"] = df_info["contingency_table_list"].map(
        lambda contingency_table: len(qi_columns_from_table(contingency_table))
    )
    return df_info


def build_seed_contexts(
    df_info: pd.DataFrame,
    *,
    max_num_contingency_tables: int,
    start_qi_num: int,
    info_path: Path,
) -> list[dict[str, Any]]:
    if max_num_contingency_tables < 1:
        raise ValueError("max_num_contingency_tables must be at least 1.")
    if start_qi_num < 1:
        raise ValueError("start_qi_num must be at least 1.")

    contexts: list[dict[str, Any]] = []
    for seed, seed_df in df_info.groupby("seed", sort=True):
        input_paths = list(dict.fromkeys(seed_df["p__input_path"].tolist()))
        if len(input_paths) != 1:
            raise ValueError(
                f"Seed {seed} maps to multiple source paths: {input_paths}. "
                "This script expects one source dataframe per seed."
            )

        dedup_seed_df = seed_df.sort_values("_row_order", kind="stable").drop_duplicates(
            subset=["contingency_table_key"],
            keep="first",
        )
        dedup_seed_df = dedup_seed_df[dedup_seed_df["num_qi_columns"] >= start_qi_num].copy()
        dedup_seed_df = dedup_seed_df.sort_values(
            ["num_qi_columns", "_row_order"],
            ascending=[True, True],
            kind="stable",
        )
        selected_seed_df = dedup_seed_df.head(max_num_contingency_tables).copy()
        if len(selected_seed_df) == 0:
            raise ValueError(
                f"Seed {seed} has no contingency tables after selection with start_qi_num={start_qi_num}."
            )

        source_path = resolve_existing_path(input_paths[0], info_path=info_path)
        output_paths = [
            str(resolve_existing_path(path_str, info_path=info_path, seed=seed))
            for path_str in selected_seed_df["p__output_path"].tolist()
        ]
        contingency_tables = [
            list(contingency_table) for contingency_table in selected_seed_df["contingency_table_list"].tolist()
        ]

        contexts.append(
            {
                "seed": seed,
                "source_path": str(source_path),
                "selected_rows": selected_seed_df,
                "output_paths": output_paths,
                "contingency_tables": contingency_tables,
                "num_contingency_tables_available": int(len(dedup_seed_df)),
                "num_contingency_tables_used": int(len(selected_seed_df)),
            }
        )

    return contexts


def get_source_subset(df_source: pd.DataFrame, splitter_value: Any) -> pd.DataFrame:
    subset = df_source[df_source["splitter"] == splitter_value].copy()
    if subset.empty:
        raise ValueError(f"No source rows found for splitter={splitter_value!r}")
    subset = subset.drop(columns=["splitter"])
    return subset


def convert_from_qi_val_tuples(
    df: pd.DataFrame,
    reconstructed: List[Dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = df.drop(columns=["id"]).copy()
    qi_cols = [col for col in df.columns if col not in ["id", "val"]]
    tuple_cols = qi_cols + ["val"]

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(reconstructed or []):
        if not isinstance(item, dict):
            raise ValueError(f"reconstructed[{idx}] must be a dict, got {type(item).__name__}")
        missing_cols = [col for col in tuple_cols if col not in item]
        if missing_cols:
            raise ValueError(
                f"reconstructed[{idx}] is missing columns {missing_cols}; expected {tuple_cols}"
            )
        rows.append({col: item[col] for col in tuple_cols})

    anon = pd.DataFrame(rows, columns=tuple_cols).reindex(columns=original.columns)
    for col in original.columns:
        if col in anon.columns and not anon.empty:
            try:
                anon[col] = anon[col].astype(original[col].dtype, copy=False)
            except (TypeError, ValueError):
                pass
    return original, anon


def convert_from_id_val_tuples(
    df: pd.DataFrame,
    reconstructed: List[Dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = df.drop(columns=["id"]).copy()
    qi_cols = [col for col in df.columns if col not in ["id", "val"]]

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(reconstructed or []):
        if not isinstance(item, dict):
            raise ValueError(f"reconstructed[{idx}] must be a dict, got {type(item).__name__}")
        if "id" not in item or "val" not in item:
            raise ValueError("id/val dict rows must contain keys 'id' and 'val'.")
        rows.append({"id": item["id"], "val": item["val"]})

    if len(rows) == 0:
        return original, pd.DataFrame(columns=original.columns)

    recon_df = pd.DataFrame(rows, columns=["id", "val"])
    id_to_qi_df = df[["id"] + qi_cols].drop_duplicates(subset=["id"])
    merged = recon_df.merge(id_to_qi_df, on="id", how="inner")
    qi_val_rows = merged[qi_cols + ["val"]].to_dict("records")
    return convert_from_qi_val_tuples(df, qi_val_rows)


def compute_alc_measures(
    df: pd.DataFrame,
    reconstructed: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(reconstructed) == 0:
        print("Skipping BRM/ALC evaluation because reconstructed is empty.")
        return {}

    first_row = reconstructed[0]
    if not isinstance(first_row, dict):
        raise ValueError(f"reconstructed rows must be dicts, got {type(first_row).__name__}")
    if "val" not in first_row:
        raise ValueError("reconstructed rows must contain key 'val'.")

    if "id" in first_row:
        original, anon = convert_from_id_val_tuples(df, reconstructed)
    else:
        original, anon = convert_from_qi_val_tuples(df, reconstructed)

    brm_results = run_brm_attack(original, anon, "val")
    return {
        "alc": brm_results["alc"],
        "attack_precision": brm_results["attack"]["precision"],
        "attack_recall": brm_results["attack"]["recall"],
        "attack_prc": brm_results["attack"]["prc"],
        "baseline_precision": brm_results["baseline"]["precision"],
        "baseline_recall": brm_results["baseline"]["recall"],
        "baseline_prc": brm_results["baseline"]["prc"],
    }


def run_brm_attack(
    original: pd.DataFrame,
    anon: pd.DataFrame,
    secret_column: str,
) -> Dict[str, Any]:
    known_columns = [col for col in original.columns if col != secret_column]
    with tempfile.TemporaryDirectory(prefix="microdata_brm_") as temp_dir:
        attack = BrmAttack(
            original,
            anon,
            results_path=temp_dir,
            no_counter=True,
            flush=True,
        )
        attack.run_one_attack(secret_column, known_columns=known_columns)
        data = dict(attack.alcm.halt_info.get("data", {}))

    return {
        "alc": float(data.get("alc", 0.0)),
        "attack": {
            "precision": float(data.get("attack_prec", 0.0)),
            "recall": float(data.get("attack_recall", 0.0)),
            "prc": float(data.get("attack_prc", 0.0)),
        },
        "baseline": {
            "precision": float(data.get("base_prec", 0.0)),
            "recall": float(data.get("base_recall", 0.0)),
            "prc": float(data.get("base_prc", 0.0)),
        },
    }


def flatten_alc_result(alc_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        f"alc_{field_name}": alc_result.get(field_name)
        for field_name in ALC_RESULT_FIELDS
    }


def build_ids_for_qi_values(
    df_source_subset: pd.DataFrame,
    qi_cols: list[str],
    qi_vals: list[int],
) -> set[int]:
    mask = pd.Series(True, index=df_source_subset.index)
    for col, val in zip(qi_cols, qi_vals):
        mask &= df_source_subset[col] == val
    return set(df_source_subset.loc[mask, "id"].astype(int).tolist())


def build_samples_for_splitter_seed(
    df_source_subset: pd.DataFrame,
    selected_seed_rows: pd.DataFrame,
    *,
    splitter_value: Any,
    info_path: Path,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    all_target_vals: set[int] = set(int(val) for val in df_source_subset["val"].unique().tolist())

    for _, info_row in selected_seed_rows.iterrows():
        contingency_table = tuple(info_row["contingency_table_list"])
        qi_cols = qi_columns_from_table(contingency_table)
        output_path = resolve_existing_path(
            info_row["p__output_path"],
            info_path=info_path,
            seed=info_row["seed"],
        )
        microdata_df = load_microdata_dataframe(output_path, contingency_table)
        microdata_subset = microdata_df[microdata_df["splitter"] == splitter_value].copy()
        if microdata_subset.empty:
            print(
                f"Skipping empty microdata subset for splitter={splitter_value!r}, "
                f"seed={info_row['seed']}, path={output_path}"
            )
            continue

        microdata_subset = microdata_subset.drop(columns=["splitter"])
        grouped = (
            microdata_subset.groupby(qi_cols + ["val"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        all_target_vals.update(int(val) for val in grouped["val"].unique().tolist())

        for qi_values, grouped_rows in grouped.groupby(qi_cols, dropna=False, sort=False):
            qi_values_tuple = qi_values if isinstance(qi_values, tuple) else (qi_values,)
            qi_vals = [int(value) for value in qi_values_tuple]
            count_map = {
                int(row["val"]): int(row["count"])
                for _, row in grouped_rows.iterrows()
            }
            sample = {
                "ids": build_ids_for_qi_values(df_source_subset, qi_cols, qi_vals),
                "qi_cols": list(qi_cols),
                "qi_vals": qi_vals,
                "noisy_counts": count_map,
            }
            samples.append(sample)

    ordered_target_vals = sorted(all_target_vals)
    for sample in samples:
        count_map = sample.pop("noisy_counts")
        sample["noisy_counts"] = [
            {"val": int(target_val), "count": int(count_map.get(target_val, 0))}
            for target_val in ordered_target_vals
        ]

    return samples


def build_known_qi_rows(
    df_source_subset: pd.DataFrame,
    *,
    seed: int,
    known_qi_fraction: float,
) -> list[dict[str, int]]:
    if known_qi_fraction <= 0.0:
        return []

    num_known_qi_rows = int(round(len(df_source_subset) * known_qi_fraction))
    if num_known_qi_rows <= 0:
        return []

    all_qi_cols = [col for col in df_source_subset.columns if col.startswith("qi")]
    chosen_indices = np.random.choice(
        len(df_source_subset),
        size=num_known_qi_rows,
        replace=False,
    )
    known_rows: list[dict[str, int]] = []
    for idx in chosen_indices:
        row = df_source_subset.iloc[idx]
        known_rows.append({col: int(row[col]) for col in all_qi_cols})
    return known_rows


def filter_known_qi_rows_against_samples(
    known_qi_rows: list[dict[str, int]],
    samples: list[dict[str, Any]],
) -> list[dict[str, int]]:
    filtered_rows: list[dict[str, int]] = []
    for known_qi_row in known_qi_rows:
        for sample in samples:
            match = False
            for col, val in zip(sample["qi_cols"], sample["qi_vals"]):
                if known_qi_row.get(col) == val:
                    match = True
                    break
            if match:
                filtered_rows.append(known_qi_row)
                break
    return filtered_rows


def run_attack_for_splitter_seed(
    *,
    df_source_subset: pd.DataFrame,
    selected_seed_rows: pd.DataFrame,
    seed: int,
    splitter_value: Any,
    noise: int,
    known_qi_fraction: float,
    info_path: Path,
) -> dict[str, Any]:
    if not 0.0 <= known_qi_fraction <= 1.0:
        raise ValueError("known_qi_fraction must be between 0.0 and 1.0.")
    if noise < 0:
        raise ValueError("noise must be >= 0.")

    np.random.seed(seed)
    all_qi_cols = [col for col in df_source_subset.columns if col.startswith("qi")]
    samples = build_samples_for_splitter_seed(
        df_source_subset,
        selected_seed_rows,
        splitter_value=splitter_value,
        info_path=info_path,
    )
    if len(samples) == 0:
        raise ValueError(f"No usable samples found for splitter={splitter_value!r}, seed={seed}")

    print(
        f"Running reconstruction for splitter={splitter_value!r}, seed={seed}, "
        f"samples={len(samples)}, noise={noise}, known_qi_fraction={known_qi_fraction}"
    )

    start_time = time.time()
    complete_known_qi_rows = build_known_qi_rows(
        df_source_subset,
        seed=seed,
        known_qi_fraction=known_qi_fraction,
    )

    qi_match_accuracy = 0.0
    if known_qi_fraction == 1.0:
        reconstructed, _, solver_metrics = reconstruct_by_row(
            samples,
            noise,
            seed,
            use_objective=DEFAULT_USE_OBJECTIVE,
            time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
            slack_limit_multiple=DEFAULT_SLACK_LIMIT_MULTIPLE,
            slack_limit_min=DEFAULT_SLACK_LIMIT_MIN,
        )
        accuracy = measure_by_row(df_source_subset, reconstructed)
        qi_match_accuracy = 1.0
    else:
        filtered_known_qi_rows = filter_known_qi_rows_against_samples(
            complete_known_qi_rows,
            samples,
        )
        if len(filtered_known_qi_rows) != len(complete_known_qi_rows):
            raise ValueError(
                "Known QI rows covered by samples do not match the number selected for reconstruction: "
                f"{len(filtered_known_qi_rows)} != {len(complete_known_qi_rows)}"
            )
        reconstructed, _, solver_metrics = reconstruct_by_aggregate_and_known_qi(
            samples,
            noise,
            len(df_source_subset),
            all_qi_cols,
            complete_known_qi_rows,
            seed,
            use_objective=DEFAULT_USE_OBJECTIVE,
            time_limit_seconds=DEFAULT_TIME_LIMIT_SECONDS,
            slack_limit_multiple=DEFAULT_SLACK_LIMIT_MULTIPLE,
            slack_limit_min=DEFAULT_SLACK_LIMIT_MIN,
        )
        accuracy_measure = measure_by_aggregate(df_source_subset, reconstructed)
        accuracy = accuracy_measure["qi_and_val_match"]
        qi_match_accuracy = accuracy_measure["qi_match"]

    elapsed_time = time.time() - start_time
    mean_cell_size = int(np.mean([len(sample["ids"]) for sample in samples])) if samples else 0
    alc_result = compute_alc_measures(df_source_subset, reconstructed)

    result = {
        "accuracy": accuracy,
        "qi_match_accuracy": qi_match_accuracy,
        "mean_cell_size": mean_cell_size,
        "solver_metrics": solver_metrics,
        "finished": True,
        "elapsed_time": elapsed_time,
        "noise": noise,
        "num_samples": len(samples),
        "num_suppressed": 0,
        "num_known_qi_rows": len(complete_known_qi_rows),
    }
    result.update(flatten_alc_result(alc_result))
    return flatten_dict(result)


def write_results(results: list[dict[str, Any]], results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(results_path, index=False)


def _normalize_path_for_key(path: Path | str) -> str:
    return str(Path(path).resolve())


def _normalize_value_for_key(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_attack_key(
    *,
    info_path: Path | str,
    max_num_contingency_tables: int,
    start_qi_num: int,
    known_qi_fraction: float,
    noise: int,
    seed: Any,
    splitter: Any,
) -> tuple[str, int, int, float, int, Any, str]:
    normalized_splitter = json.dumps(_normalize_value_for_key(splitter), sort_keys=True)
    return (
        _normalize_path_for_key(info_path),
        int(max_num_contingency_tables),
        int(start_qi_num),
        float(known_qi_fraction),
        int(noise),
        _normalize_value_for_key(seed),
        normalized_splitter,
    )


def load_existing_results(
    results_path: Path,
    *,
    info_path: Path,
) -> tuple[list[dict[str, Any]], set[tuple[str, int, int, float, int, Any, str]]]:
    if not results_path.exists():
        print(f"No existing results parquet at {results_path}")
        return [], set()

    existing_df = pd.read_parquet(results_path)
    existing_results = existing_df.to_dict("records")
    existing_keys: set[tuple[str, int, int, float, int, Any, str]] = set()
    current_info_path = _normalize_path_for_key(info_path)

    for row in existing_results:
        row_info_path = row.get("info_path")
        if pd.isna(row_info_path):
            row_info_path = current_info_path

        max_num_contingency_tables = row.get("max_num_contingency_tables")
        start_qi_num = row.get("start_qi_num")
        known_qi_fraction = row.get("known_qi_fraction")
        noise = row.get("noise")
        seed = row.get("seed")
        splitter = row.get("splitter")
        if pd.isna(noise):
            noise = DEFAULT_NOISE
        if pd.isna(start_qi_num):
            start_qi_num = DEFAULT_START_QI_NUM
        if any(
            pd.isna(value)
            for value in [max_num_contingency_tables, known_qi_fraction, seed, splitter]
        ):
            continue

        existing_keys.add(
            build_attack_key(
                info_path=row_info_path,
                max_num_contingency_tables=int(max_num_contingency_tables),
                start_qi_num=int(start_qi_num),
                known_qi_fraction=float(known_qi_fraction),
                noise=int(noise),
                seed=seed,
                splitter=splitter,
            )
        )

    print(
        f"Loaded {len(existing_results)} existing results from {results_path}; "
        f"indexed {len(existing_keys)} resumable attack keys"
    )
    return existing_results, existing_keys


def main() -> None:
    args = parse_args()
    info_path = resolve_existing_path(args.info)
    results_path = Path(args.results_path)
    if not results_path.is_absolute():
        results_path = Path.cwd() / results_path
    results_path = results_path.resolve()

    df_info = prepare_info_dataframe(info_path)
    validate_seed_variation(df_info, info_path=info_path)
    seed_contexts = build_seed_contexts(
        df_info,
        max_num_contingency_tables=args.max_num_contingency_tables,
        start_qi_num=args.start_qi_num,
        info_path=info_path,
    )

    source_cache: dict[str, pd.DataFrame] = {}
    splitters_by_source: dict[str, list[Any]] = {}
    for context in seed_contexts:
        source_path = context["source_path"]
        if source_path not in source_cache:
            source_df = load_source_dataframe(Path(source_path))
            source_cache[source_path] = source_df
            splitters_by_source[source_path] = sorted(source_df["splitter"].drop_duplicates().tolist())

    all_splitters = sorted(
        {
            splitter
            for splitters in splitters_by_source.values()
            for splitter in splitters
        }
    )

    results, existing_attack_keys = load_existing_results(
        results_path,
        info_path=info_path,
    )
    total_completed = 0
    total_skipped = 0
    for splitter_value in all_splitters:
        print(f"Starting splitter={splitter_value!r}")
        splitter_completed = 0
        splitter_skipped = 0
        for context in seed_contexts:
            source_path = context["source_path"]
            source_df = source_cache[source_path]
            if splitter_value not in splitters_by_source[source_path]:
                continue

            attack_key = build_attack_key(
                info_path=info_path,
                max_num_contingency_tables=args.max_num_contingency_tables,
                start_qi_num=args.start_qi_num,
                known_qi_fraction=args.known_qi_fraction,
                noise=args.noise,
                seed=context["seed"],
                splitter=splitter_value,
            )
            if attack_key in existing_attack_keys:
                splitter_skipped += 1
                total_skipped += 1
                print(
                    f"Status: splitter={splitter_value!r}, seed={context['seed']} skipped "
                    f"(already present). splitter_completed={splitter_completed}, "
                    f"splitter_skipped={splitter_skipped}, total_results={len(results)}"
                )
                continue

            source_subset = get_source_subset(source_df, splitter_value)
            row_result = run_attack_for_splitter_seed(
                df_source_subset=source_subset,
                selected_seed_rows=context["selected_rows"],
                seed=int(context["seed"]),
                splitter_value=splitter_value,
                noise=args.noise,
                known_qi_fraction=args.known_qi_fraction,
                info_path=info_path,
            )
            row_result.update(
                {
                    "info_path": _normalize_path_for_key(info_path),
                    "seed": int(context["seed"]),
                    "splitter": splitter_value,
                    "p__input_path": source_path,
                    "p__output_paths_json": json.dumps(context["output_paths"]),
                    "p__contingency_tables_json": json.dumps(context["contingency_tables"]),
                    "known_qi_fraction": args.known_qi_fraction,
                    "noise": args.noise,
                    "start_qi_num": args.start_qi_num,
                    "max_num_contingency_tables": args.max_num_contingency_tables,
                    "num_contingency_tables_available": context["num_contingency_tables_available"],
                    "num_contingency_tables_used": context["num_contingency_tables_used"],
                    "splitter_num_rows": int(len(source_subset)),
                    "source_num_rows": int(len(source_df)),
                    "source_nqi": int(len([col for col in source_subset.columns if col.startswith("qi")])),
                    "source_nunique": int(source_df["val"].nunique(dropna=False)),
                    "experiment_finished": True,
                }
            )
            results.append(row_result)
            existing_attack_keys.add(attack_key)
            splitter_completed += 1
            total_completed += 1
            write_results(results, results_path)
            _print_memory_usage(
                f"After writing results for splitter={splitter_value!r}, seed={context['seed']}"
            )
            print(
                f"Status: splitter={splitter_value!r}, seed={context['seed']} completed. "
                f"splitter_completed={splitter_completed}, splitter_skipped={splitter_skipped}, "
                f"total_results={len(results)}"
            )
            gc.collect()
        print(
            f"Finished splitter={splitter_value!r}: completed={splitter_completed}, "
            f"skipped={splitter_skipped}, cumulative_completed={total_completed}, "
            f"cumulative_skipped={total_skipped}, total_results={len(results)}"
        )

    print(f"Wrote {len(results)} results to {results_path}")


if __name__ == "__main__":
    main()
