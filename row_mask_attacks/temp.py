from pathlib import Path

import pandas as pd


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y", "t"})


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    parquet_path = results_dir / "result.parquet"
    files_dir = results_dir / "files"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet file: {parquet_path}")
    if not files_dir.exists():
        raise FileNotFoundError(f"Missing files directory: {files_dir}")

    df = pd.read_parquet(parquet_path)
    required_cols = {"filename", "final_attack"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"result.parquet missing required columns: {sorted(missing)}")

    filename_series = df["filename"].astype("string")
    valid_filename_mask = filename_series.notna()
    final_attack_true = _coerce_bool(df["final_attack"])
    has_true_per_filename = final_attack_true[valid_filename_mask].groupby(filename_series[valid_filename_mask]).any()
    missing_true_filenames = sorted(has_true_per_filename[~has_true_per_filename].index.astype(str))

    print(
        "Number of distinct filenames with no row where final_attack==True: "
        f"{len(missing_true_filenames)}"
    )
    for name in missing_true_filenames:
        print(name)

    if len(missing_true_filenames) == 0:
        print("No cleanup required.")
        return

    deleted_files = 0
    missing_files = 0
    delete_failures: list[tuple[str, Exception]] = []
    for filename in missing_true_filenames:
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
    keep_mask = ~filename_series.isin(missing_true_filenames)
    cleaned_df = df.loc[keep_mask].copy()

    tmp_parquet_path = parquet_path.with_name(f"{parquet_path.stem}.tmp.parquet")
    cleaned_df.to_parquet(tmp_parquet_path, index=False)
    tmp_parquet_path.replace(parquet_path)

    print(f"Files deleted: {deleted_files}")
    print(f"Files already missing: {missing_files}")
    print(f"Rows removed from result.parquet: {rows_before - len(cleaned_df)}")
    print(f"Rows remaining in result.parquet: {len(cleaned_df)}")


if __name__ == "__main__":
    main()
