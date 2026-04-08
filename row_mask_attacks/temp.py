from pathlib import Path

import pandas as pd


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
    required_cols = {"solver_metrics_status", "filename"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"result.parquet is missing required columns: {sorted(missing_cols)}")

    infeasible_mask = pd.to_numeric(df["solver_metrics_status"], errors="coerce") == 3
    rows_to_remove = int(infeasible_mask.sum())
    if rows_to_remove == 0:
        print("No rows with solver_metrics_status == 3. Nothing to do.")
        return

    target_filenames = sorted(set(df.loc[infeasible_mask, "filename"].dropna().astype(str)))

    deleted_files = 0
    missing_files = 0
    delete_failures: list[tuple[str, Exception]] = []
    for file_name in target_filenames:
        target_path = files_dir / Path(file_name).name
        if not target_path.exists():
            missing_files += 1
            continue
        try:
            target_path.unlink()
            deleted_files += 1
        except OSError as exc:
            delete_failures.append((str(target_path), exc))

    if delete_failures:
        print("Failed to delete one or more files; result.parquet was not modified.")
        for path, exc in delete_failures:
            print(f"  {path}: {exc}")
        raise RuntimeError(f"Could not delete {len(delete_failures)} files.")

    df_clean = df.loc[~infeasible_mask].copy()
    tmp_path = parquet_path.with_name(f"{parquet_path.stem}.tmp.parquet")
    df_clean.to_parquet(tmp_path, index=False)
    tmp_path.replace(parquet_path)

    print(f"Rows removed from result.parquet: {rows_to_remove}")
    print(f"Unique target files: {len(target_filenames)}")
    print(f"Files deleted: {deleted_files}")
    print(f"Files already missing: {missing_files}")
    print(f"Rows remaining: {len(df_clean)}")


if __name__ == "__main__":
    main()
