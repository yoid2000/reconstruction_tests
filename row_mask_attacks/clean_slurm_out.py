from pathlib import Path


def should_delete(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def main() -> None:
    slurm_out_dir = Path(__file__).parent / "slurm_out"
    if not slurm_out_dir.exists():
        print(f"Missing directory: {slurm_out_dir}")
        return

    needles = [
        "Attack already finished",
        "Final accuracy",
    ]

    deleted = 0
    scanned = 0
    for path in slurm_out_dir.glob("*.txt"):
        scanned += 1
        try:
            text = path.read_text(errors="ignore")
        except OSError as exc:
            print(f"Failed to read {path}: {exc}")
            continue
        if should_delete(text, needles):
            try:
                path.unlink()
                deleted += 1
            except OSError as exc:
                print(f"Failed to delete {path}: {exc}")

    print(f"Scanned {scanned} files, deleted {deleted} files.")


if __name__ == "__main__":
    main()
