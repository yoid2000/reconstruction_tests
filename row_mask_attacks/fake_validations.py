import json
from pathlib import Path
from typing import Any, List


def _strip_separation(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry
    if 'separation' not in entry:
        return entry
    stripped = dict(entry)
    stripped.pop('separation', None)
    return stripped


def _compare_attack_results(old_results: Any, new_results: Any) -> List[str]:
    reasons = []
    if not isinstance(old_results, list):
        reasons.append("old attack_results is not a list")
        return reasons
    if not isinstance(new_results, list):
        reasons.append("new attack_results is not a list")
        return reasons
    if len(old_results) != len(new_results):
        reasons.append(f"attack_results length mismatch (old={len(old_results)}, new={len(new_results)})")
        return reasons

    for idx, (old_entry, new_entry) in enumerate(zip(old_results, new_results)):
        if not isinstance(new_entry, dict):
            reasons.append(f"attack_results[{idx}] in new is not a dict")
            continue
        if 'separation' not in new_entry:
            reasons.append(f"attack_results[{idx}] missing separation")
        if old_entry != _strip_separation(new_entry):
            reasons.append(f"attack_results[{idx}] values differ from backup (excluding separation)")
    return reasons


def _compare_dicts(old: Any, new: Any) -> List[str]:
    if not isinstance(old, dict):
        return ["old json is not a dict"]
    if not isinstance(new, dict):
        return ["new json is not a dict"]

    reasons = []
    for key, old_value in old.items():
        if key not in new:
            reasons.append(f"missing key '{key}' in new")
            continue
        if key == 'attack_results':
            reasons.extend(_compare_attack_results(old_value, new.get(key)))
            continue
        if new.get(key) != old_value:
            reasons.append(f"value mismatch for key '{key}'")
    return reasons


def main() -> None:
    new_dir = Path('results/files')
    old_dir = Path('results/files_bk')

    if not new_dir.exists():
        print(f"Missing directory: {new_dir}")
        return
    if not old_dir.exists():
        print(f"Missing directory: {old_dir}")
        return

    for new_path in sorted(new_dir.glob('*.json')):
        reasons = []
        old_path = old_dir / new_path.name
        if not old_path.exists():
            print(f"{new_path.name}: missing backup file in {old_dir}")
            continue

        try:
            with open(new_path, 'r') as f:
                new_data = json.load(f)
        except Exception as exc:
            print(f"{new_path.name}: failed to read new file ({exc})")
            continue

        try:
            with open(old_path, 'r') as f:
                old_data = json.load(f)
        except Exception as exc:
            print(f"{new_path.name}: failed to read backup file ({exc})")
            continue

        reasons.extend(_compare_dicts(old_data, new_data))
        if reasons:
            print(f"{new_path.name}: " + "; ".join(reasons))


if __name__ == '__main__':
    main()
