import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "List filenames whose rows all have empty exit_reason. "
            "Use --replace to delete files and rewrite result.parquet."
        )
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Apply cleanup: delete matching files and rewrite result.parquet.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    parquet_path = results_dir / "result.parquet"
    files_dir = results_dir / "files"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet file: {parquet_path}")
    if not files_dir.exists():
        raise FileNotFoundError(f"Missing files directory: {files_dir}")

    df = pd.read_parquet(parquet_path)
    required_cols = {"filename", "exit_reason"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"result.parquet missing required columns: {sorted(missing)}")

    filename_series = df["filename"].astype("string")
    valid_filename_mask = filename_series.notna()
    exit_reason_is_empty = df["exit_reason"].fillna("").astype(str).str.strip().eq("")
    all_rows_empty_exit_reason = (
        exit_reason_is_empty[valid_filename_mask]
        .groupby(filename_series[valid_filename_mask])
        .all()
    )
    target_filenames = sorted(
        all_rows_empty_exit_reason[all_rows_empty_exit_reason].index.astype(str)
    )

    for name in target_filenames:
        print(name)

    if len(target_filenames) == 0:
        print("No cleanup required.")
        print(
            "Number of distinct filenames where all rows have exit_reason=='': 0"
        )
        return
    if not args.replace:
        print("--replace not set. Dry run only: no files deleted and result.parquet unchanged.")
        print(
            "Number of distinct filenames where all rows have exit_reason=='': "
            f"{len(target_filenames)}"
        )
        return

    deleted_files = 0
    missing_files = 0
    delete_failures: list[tuple[str, Exception]] = []
    for filename in target_filenames:
        target_path = files_dir / Path(filename).name
        if not target_path.exists():
            missing_files += 1
            continue
        try:
            target_path.unlink()
            deleted_files += 1
        except OSError as exc:
            delete_failures.append((str(target_path), exc))

    if delete_failures:
        print("Failed to delete one or more files; parquet file was not modified.")
        for path, exc in delete_failures:
            print(f"  {path}: {exc}")
        raise RuntimeError(f"Could not delete {len(delete_failures)} files.")

    rows_before = len(df)
    keep_mask = ~filename_series.isin(target_filenames)
    cleaned_df = df.loc[keep_mask].copy()

    tmp_parquet_path = parquet_path.with_name(f"{parquet_path.stem}.tmp.parquet")
    cleaned_df.to_parquet(tmp_parquet_path, index=False)
    tmp_parquet_path.replace(parquet_path)

    print(f"Files deleted: {deleted_files}")
    print(f"Files already missing: {missing_files}")
    print(f"Rows removed from result.parquet: {rows_before - len(cleaned_df)}")
    print(f"Rows remaining in result.parquet: {len(cleaned_df)}")
    print(
        "Number of distinct filenames where all rows have exit_reason=='': "
        f"{len(target_filenames)}"
    )


if __name__ == "__main__":
    main()
