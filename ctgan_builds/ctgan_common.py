from __future__ import annotations

import inspect
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_TABLE_NAME = "table"
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


def parse_contingency_table(value: object) -> list[str]:
    if isinstance(value, list) and all(isinstance(column, str) for column in value):
        return value

    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list) and all(isinstance(column, str) for column in parsed):
            return parsed

    raise ValueError("contingency_table must encode a list of column names.")


def get_metadata_output_path(output_path: str) -> Path:
    return resolve_local_path(output_path).with_suffix(".json")


def get_seeded_output_path(output_path: str, seed: int) -> str:
    output_file = Path(output_path)
    return str(output_file.with_name(f"{output_file.stem}_{seed}{output_file.suffix}"))


def build_synthesizer_kwargs(
    ctgan_config: dict[str, Any],
    constructor_parameters: set[str],
    constructor_signature: inspect.Signature,
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

    pac = 1
    if "pac" in constructor_parameters:
        pac_param = constructor_signature.parameters.get("pac")
        if pac_param is not None and pac_param.default is not inspect._empty:
            pac = int(pac_param.default)

    if "batch_size" in kwargs:
        batch_size = int(kwargs["batch_size"])
        required_multiple = math.lcm(2, max(1, pac))
        adjusted_batch_size = batch_size
        if batch_size % required_multiple != 0:
            adjusted_batch_size = batch_size - (batch_size % required_multiple)
            if adjusted_batch_size < required_multiple:
                adjusted_batch_size = required_multiple
            print(
                "Adjusting CTGAN batch_size for pac compatibility: "
                f"{batch_size} -> {adjusted_batch_size} (pac={pac})"
            )
            kwargs["batch_size"] = adjusted_batch_size

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


def build_metadata(df_source: pd.DataFrame):
    from sdv.metadata import Metadata

    metadata = Metadata.detect_from_dataframes(
        data={SINGLE_TABLE_NAME: df_source},
        infer_keys=None,
    )
    metadata.update_columns(
        column_names=list(df_source.columns),
        sdtype="categorical",
        table_name=SINGLE_TABLE_NAME,
    )
    metadata.validate()
    metadata.validate_table(data=df_source, table_name=SINGLE_TABLE_NAME)
    return metadata


def generate_synthetic_dataframe(
    df_source: pd.DataFrame,
    ctgan_config: dict[str, Any],
) -> tuple[pd.DataFrame, object]:
    try:
        from sdv.single_table import CTGANSynthesizer
    except ImportError as exc:
        raise ImportError(
            "SDV is required to run CTGAN builds. Install the 'sdv' package in the runtime environment."
        ) from exc

    metadata = build_metadata(df_source)
    constructor_signature = inspect.signature(CTGANSynthesizer.__init__)
    constructor_parameters = set(constructor_signature.parameters)
    constructor_parameters.discard("self")
    synthesizer_kwargs = build_synthesizer_kwargs(
        ctgan_config,
        constructor_parameters,
        constructor_signature,
    )
    synthesizer = CTGANSynthesizer(metadata, **synthesizer_kwargs)
    synthesizer.fit(df_source)
    return synthesizer.sample(num_rows=len(df_source)), metadata


def write_synthetic_dataframe(df_synth: pd.DataFrame, output_path: str) -> Path:
    resolved_path = resolve_local_path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    df_synth.to_parquet(resolved_path, index=False)
    return resolved_path


def write_metadata_json(metadata: object, output_path: str) -> Path:
    metadata_path = get_metadata_output_path(output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.save_to_json(metadata_path, mode="overwrite")
    return metadata_path


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
    output_path = get_seeded_output_path(str(parameters["output_path"]), seed)
    ctgan_config_path = str(parameters["ctgan_config_path"])
    contingency_table = parse_contingency_table(parameters["contingency_table"])

    set_random_seed(seed)
    ctgan_config = load_ctgan_config(ctgan_config_path)
    df_source = load_source_dataframe(input_path, contingency_table)

    start_time = time.time()
    df_synth, metadata = generate_synthetic_dataframe(df_source, ctgan_config)
    write_synthetic_dataframe(df_synth, output_path)
    write_metadata_json(metadata, output_path)
    elapsed_time = time.time() - start_time

    result: dict[str, object] = {
        "experiment_finished": True,
        "ctgan_elapsed_time": elapsed_time,
    }
    for field in CTGAN_CONFIG_FIELDS:
        result[field] = ctgan_config[field]

    return result
