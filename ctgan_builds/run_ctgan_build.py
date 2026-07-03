from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from slurm_manager.core import RunManifestBuilder, init_payload, clean_payload


SCRIPT_DIR = Path(__file__).resolve().parent
CTGAN_CONFIG_FIELDS = (
    "generator_dim",
    "discriminator_dim",
    "embedding_dim",
    "batch_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "discriminator_dropout",
    "gradient_clip",
    "temperature",
)


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


def resolve_local_path(path: str) -> Path:
    local_path = Path(path)
    if local_path.is_absolute():
        return local_path
    return SCRIPT_DIR / local_path


def load_ctgan_config(config_path: str) -> dict[str, Any]:
    resolved_path = resolve_local_path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"CTGAN config file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict) or not isinstance(config.get("ctgan"), dict):
        raise ValueError(f"Expected top-level 'ctgan' mapping in {resolved_path}")

    ctgan_config = config["ctgan"]
    missing_fields = [field for field in CTGAN_CONFIG_FIELDS if field not in ctgan_config]
    if missing_fields:
        raise ValueError(
            f"Missing CTGAN config fields in {resolved_path}: {', '.join(missing_fields)}"
        )

    return ctgan_config


def load_source_dataframe(input_path: str, contingency_table: list[str]) -> pd.DataFrame:
    resolved_path = resolve_local_path(input_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {resolved_path}")

    df_source = pd.read_parquet(resolved_path)
    missing_columns = [column for column in contingency_table if column not in df_source.columns]
    if missing_columns:
        raise ValueError(
            f"Input dataframe is missing contingency columns {missing_columns}: {resolved_path}"
        )

    return df_source[contingency_table].copy()


def build_synthesizer_kwargs(
    ctgan_config: dict[str, Any],
    constructor_parameters: set[str],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    direct_fields = (
        "generator_dim",
        "discriminator_dim",
        "embedding_dim",
        "batch_size",
        "epochs",
        "discriminator_dropout",
        "gradient_clip",
        "temperature",
    )
    for field in direct_fields:
        if field in constructor_parameters:
            value = ctgan_config[field]
            if field in {"generator_dim", "discriminator_dim"}:
                value = tuple(value)
            kwargs[field] = value

    learning_rate = ctgan_config["learning_rate"]
    if "learning_rate" in constructor_parameters:
        kwargs["learning_rate"] = learning_rate
    for alias in ("generator_lr", "discriminator_lr"):
        if alias in constructor_parameters:
            kwargs[alias] = learning_rate

    weight_decay = ctgan_config["weight_decay"]
    if "weight_decay" in constructor_parameters:
        kwargs["weight_decay"] = weight_decay
    for alias in ("generator_decay", "discriminator_decay"):
        if alias in constructor_parameters:
            kwargs[alias] = weight_decay

    if "verbose" in constructor_parameters:
        kwargs["verbose"] = False

    return kwargs


def generate_synthetic_dataframe(
    df_source: pd.DataFrame,
    ctgan_config: dict[str, Any],
) -> pd.DataFrame:
    try:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer
    except ImportError as exc:
        raise ImportError(
            "SDV is required to run CTGAN builds. Install the 'sdv' package in the runtime environment."
        ) from exc

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df_source)

    # These QI columns are finite coded categories, not continuous numerics.
    for column in df_source.columns:
        metadata.update_column(column_name=column, sdtype="categorical")

    constructor_parameters = set(inspect.signature(CTGANSynthesizer.__init__).parameters)
    constructor_parameters.discard("self")
    synthesizer_kwargs = build_synthesizer_kwargs(ctgan_config, constructor_parameters)
    synthesizer = CTGANSynthesizer(metadata, **synthesizer_kwargs)
    synthesizer.fit(df_source)
    return synthesizer.sample(num_rows=len(df_source))


def write_synthetic_dataframe(df_synth: pd.DataFrame, output_path: str) -> Path:
    resolved_path = resolve_local_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    df_synth.to_parquet(resolved_path, index=False)
    return resolved_path


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def run_experiment(parameters: dict[str, object], seed: int) -> dict[str, object]:
    input_path = str(parameters["input_path"])
    output_path = str(parameters["output_path"])
    ctgan_config_path = str(parameters["ctgan_config_path"])
    contingency_table = parameters["contingency_table"]

    if not isinstance(contingency_table, list) or not all(
        isinstance(column, str) for column in contingency_table
    ):
        raise ValueError("contingency_table must be a list of column names.")

    set_random_seed(seed)
    ctgan_config = load_ctgan_config(ctgan_config_path)
    df_source = load_source_dataframe(input_path, contingency_table)

    start_time = time.time()
    df_synth = generate_synthetic_dataframe(df_source, ctgan_config)
    write_synthetic_dataframe(df_synth, output_path)
    elapsed_time = time.time() - start_time

    result: dict[str, object] = {
        "experiment_finished": True,
        "ctgan_elapsed_time": elapsed_time,
    }
    for field in CTGAN_CONFIG_FIELDS:
        result[field] = ctgan_config[field]

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

    sys.path.insert(0, str(Path(__file__).parent.parent))

    entries = RunManifestBuilder.load_manifest(args.manifest)
    entry = entries[args.job_num]

    payload = init_payload(entry)
    payload_clean = clean_payload(payload)

    payload.update(run_experiment(parameters=entry.parameters, seed=entry.seed))

    write_result_json(
        results_dir=args.results_dir,
        experiment_id=entry.experiment_id,
        seed=entry.seed,
        payload=payload,
    )


if __name__ == "__main__":
    main()
