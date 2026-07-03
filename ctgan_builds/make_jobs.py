from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.contingency_tables import contingency_table_columns


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CTGAN build jobs for selected contingency tables."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="input_files/nr_150_nu_2_nq_11_vq_2_cs_0p0.parquet",
    )
    parser.add_argument("--output_dir", type=str, default="output_files")
    parser.add_argument("--ctgan_configs_dir", type=str, default="ctgan_configs")
    parser.add_argument("--contingency_files", type=int, default=200)
    parser.add_argument("--jobs_file", type=str, default="jobs.json")
    return parser.parse_args()


def resolve_local_path(path: str) -> Path:
    local_path = Path(path)
    if local_path.is_absolute():
        return local_path
    return SCRIPT_DIR / local_path


def sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    cleaned = cleaned.strip("._-")
    return cleaned or "x"


def qi_columns_filename(qi_columns: Iterable[str]) -> str:
    return "_".join(sanitize_filename_part(col) for col in qi_columns) + ".parquet"


def join_job_path(base: str, *parts: str) -> str:
    separator = "\\" if "\\" in base else "/"
    suffix = separator.join(part.strip("/\\") for part in parts if part != "")
    if not suffix:
        return base
    trimmed_base = base.rstrip("/\\")
    return f"{trimmed_base}{separator}{suffix}"


def build_jobs(args: argparse.Namespace) -> list[dict[str, object]]:
    input_path = resolve_local_path(args.input_path)
    ctgan_configs_dir = resolve_local_path(args.ctgan_configs_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {input_path}")
    if not ctgan_configs_dir.exists():
        raise FileNotFoundError(f"CTGAN configs directory not found: {ctgan_configs_dir}")

    df = pd.read_parquet(input_path)
    contingency_tables = contingency_table_columns(df, args.contingency_files)
    config_paths = sorted(ctgan_configs_dir.glob("*.yaml"))

    file_name = Path(args.input_path).stem
    jobs: list[dict[str, object]] = []
    for config_path in config_paths:
        yaml_name = config_path.stem
        for qi_columns in contingency_tables:
            qi_file_name = qi_columns_filename(qi_columns)
            jobs.append(
                {
                    "input_path": args.input_path,
                    "output_path": join_job_path(
                        args.output_dir,
                        file_name,
                        yaml_name,
                        qi_file_name,
                    ),
                    "ctgan_config_path": join_job_path(
                        args.ctgan_configs_dir,
                        f"{yaml_name}.yaml",
                    ),
                    "contingency_table": json.dumps(qi_columns),
                }
            )

    return jobs


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    jobs_file = resolve_local_path(args.jobs_file)
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    jobs_file.write_text(json.dumps(jobs, indent=4), encoding="utf-8")
    print(f"Wrote {len(jobs)} jobs to {jobs_file}")


if __name__ == "__main__":
    main()
