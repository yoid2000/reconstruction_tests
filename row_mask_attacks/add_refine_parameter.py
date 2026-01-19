import json
from pathlib import Path


def update_results_file(file_path: Path) -> bool:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as exc:
        print(f"{file_path.name}: failed to read ({exc})")
        return False

    if not isinstance(data, dict):
        print(f"{file_path.name}: json root is not a dict")
        return False

    attack_results = data.get('attack_results')
    if not isinstance(attack_results, list):
        print(f"{file_path.name}: attack_results is not a list")
        return False

    for idx, entry in enumerate(attack_results):
        if not isinstance(entry, dict):
            print(f"{file_path.name}: attack_results[{idx}] is not a dict")
            return False
        entry['refine'] = 0

    if data.get('solve_type') == 'agg_row':
        data['known_qi_fraction'] = 1.0

    finished = data.get('finished', False)
    if finished is True:
        exit_reason = data.get('exit_reason')
        if exit_reason == 'target_accuracy':
            data['finished'] = False

    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        print(f"{file_path.name}: failed to write ({exc})")
        return False

    return True


def main() -> None:
    results_dir = Path('./results/files')
    if not results_dir.exists():
        print(f"Missing results directory: {results_dir}")
        return

    json_files = sorted(results_dir.glob('*.json'))
    if not json_files:
        print(f"No JSON files found under {results_dir}")
        return

    updated = 0
    for file_path in json_files:
        if update_results_file(file_path):
            updated += 1

    print(f"Updated {updated} files")


if __name__ == '__main__':
    main()
