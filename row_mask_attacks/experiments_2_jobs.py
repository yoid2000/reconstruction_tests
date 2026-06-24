from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from experiments import read_experiments


NOT_JOB_KEYS = {"not_params"}


def _as_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _dedupe_key(job: dict[str, Any]) -> str:
    return json.dumps(job, sort_keys=True, separators=(",", ":"), default=str)


def expand_experiments(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    seen = set()

    for experiment in experiments:
        keys = [key for key in experiment.keys() if key not in NOT_JOB_KEYS]
        value_lists = [_as_values(experiment[key]) for key in keys]
        for values in product(*value_lists):
            job = dict(zip(keys, values))
            dedupe_key = _dedupe_key(job)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            jobs.append(job)

    return jobs


def write_jobs(jobs: list[dict[str, Any]], output_path: Path) -> None:
    output_path.write_text(json.dumps(jobs, indent=4), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand experiments.py definitions into jobs.json."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("jobs.json"),
        help="Output JSON path. Defaults to row_mask_attacks/jobs.json.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include experiments not marked used_in_paper.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = read_experiments(used_only_in_paper=not args.all)
    jobs = expand_experiments(experiments)
    write_jobs(jobs, args.output)
    print(f"Wrote {len(jobs)} unique jobs to {args.output}")


if __name__ == "__main__":
    main()
