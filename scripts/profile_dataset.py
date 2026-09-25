from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/processed/transactions_v1.parquet")


def main() -> None:
    print("Loading dataset...")

    df = pd.read_parquet(DATASET_PATH)

    print("\n=== DATASET OVERVIEW ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")

    print("\n=== COLUMNS ===")
    for column in df.columns:
        print(f"- {column}")

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    missing = df.isna().sum()
    print(missing[missing > 0] if missing.any() else "No missing values")

    print("\n=== DUPLICATE ROWS ===")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\n=== UNIQUE VALUES ===")
    for column in [
        "user_id",
        "merchant_id",
        "merchant_category",
        "payment_method",
        "user_segment",
        "device_type",
        "network_quality",
    ]:
        print(f"{column}: {df[column].nunique():,}")

    print("\n=== NUMERIC SUMMARY ===")
    numeric_columns = [
        "amount",
        "retry_count",
        "transaction_velocity",
        "user_method_success_rate",
        "merchant_method_success_rate",
        "failure_probability",
        "payment_success",
    ]
    print(df[numeric_columns].describe().round(4).to_string())

    print("\n=== TARGET DISTRIBUTION ===")
    print(
        df["payment_success"]
        .value_counts()
        .sort_index()
        .rename(index={0: "failure", 1: "success"})
    )

    print("\n=== FAILURE RATE ===")
    print(f"{(1 - df['payment_success'].mean()) * 100:.2f}%")

    print("\n=== PAYMENT METHOD DISTRIBUTION ===")
    print(df["payment_method"].value_counts())

    print("\n=== FAILURE RATE BY PAYMENT METHOD ===")
    failure_by_method = (
        1 - df.groupby("payment_method")["payment_success"].mean()
    )
    print((failure_by_method * 100).round(2).sort_values(ascending=False))

    print("\n=== FAILURE RATE BY NETWORK QUALITY ===")
    failure_by_network = (
        1 - df.groupby("network_quality")["payment_success"].mean()
    )
    print((failure_by_network * 100).round(2).sort_values(ascending=False))

    print("\n=== FAILURE RATE BY USER SEGMENT ===")
    failure_by_segment = (
        1 - df.groupby("user_segment")["payment_success"].mean()
    )
    print((failure_by_segment * 100).round(2).sort_values(ascending=False))

    print("\n=== DATE RANGE ===")
    print(f"Start: {df['timestamp'].min()}")
    print(f"End:   {df['timestamp'].max()}")

    print("\nProfiling complete.")


if __name__ == "__main__":
    main()
    