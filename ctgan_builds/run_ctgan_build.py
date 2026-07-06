from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slurm_manager.core import RunManifestBuilder, init_payload, clean_payload

try:
    from .ctgan_common import run_experiment
except ImportError:
    from ctgan_common import run_experiment


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
