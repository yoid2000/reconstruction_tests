from pathlib import Path

import pandas as pd


def main() -> None:
    parquet_path = Path("results/result.parquet")
    df = pd.read_parquet(parquet_path)

    if "path_to_dataset" not in df.columns:
        raise ValueError("Column 'path_to_dataset' does not exist in results/result.parquet")

    filtered_df = df[df["path_to_dataset"].isna()].copy()

    # remove columns path_to_dataset and target_column
    filtered_df.drop(columns=["path_to_dataset", "target_column"], inplace=True)
    filtered_df.to_parquet(parquet_path, index=False)


if __name__ == "__main__":
    main()
