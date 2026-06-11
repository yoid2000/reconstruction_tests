"""Sample job program called by ``slurm_manager.cli.start_conductor``.

Copy this file and replace the marked sections with experiment-specific code.
The conductor runs this program through an sbatch array and passes:

    --manifest <path to manifest JSON>
    --results_dir <directory where this program must write result JSON>
    --job_num <SLURM array index / manifest entry number>

The result JSON file is the contract between the job and the conductor. A run is
considered successful only when the payload includes ``experiment_finished: true``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from slurm_manager.core import RunManifestBuilder, init_payload


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
    start = time.time()
    # Put experiment-specific logic here.
    elapsed_seconds = round(time.time() - start, 5)

    return {
        # Required by the conductor. Without this exact True value, the run is
        # treated as failed or incomplete.
        "experiment_finished": True,
        # Add result metrics here. These become columns in results.parquet.
        # Keep values flat: strings, numbers, booleans, or null. Avoid nested
        # objects/lists unless ResultIngestor is explicitly extended to support them.
        "elapsed_seconds": elapsed_seconds,
        "sample_metric": 1.0,
        "used_seed": seed,
    }


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


if __name__ == "__main__":
    main()
