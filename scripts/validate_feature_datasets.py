from pathlib import Path

import pandas as pd


FEATURE_DIR = Path("data/processed/features")

DATASETS = {
    "train": FEATURE_DIR / "train_features.parquet",
    "validation": FEATURE_DIR / "validation_features.parquet",
    "test": FEATURE_DIR / "test_features.parquet",
}


def main() -> None:
    print("Validating model-ready feature datasets...")

    datasets = {}

    for name, path in DATASETS.items():
        df = pd.read_parquet(path)
        datasets[name] = df

        print(f"\n{name.upper()}:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")

    train_columns = list(datasets["train"].columns)

    # Verify identical feature columns.
    for name, df in datasets.items():
        if list(df.columns) != train_columns:
            raise ValueError(
                f"{name}: feature columns do not match the training dataset."
            )

    # Verify compatible dtypes.
    for column in train_columns:
        train_dtype = datasets["train"][column].dtype

        for name in ("validation", "test"):
            if datasets[name][column].dtype != train_dtype:
                raise ValueError(
                    f"{name}: dtype mismatch for column '{column}'. "
                    f"Expected {train_dtype}, "
                    f"found {datasets[name][column].dtype}."
                )

    # Verify no missing values.
    for name, df in datasets.items():
        if df.isna().any().any():
            raise ValueError(f"{name}: missing values found.")

    # Verify no infinite numeric values.
    for name, df in datasets.items():
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.isin([float("inf"), float("-inf")]).any().any():
            raise ValueError(f"{name}: infinite values found.")

    # Verify leakage columns are absent.
    forbidden_columns = {
        "transaction_id",
        "user_id",
        "merchant_id",
        "timestamp",
        "failure_probability",
        "payment_success",
    }

    leakage_columns = set(train_columns) & forbidden_columns

    if leakage_columns:
        raise ValueError(
            f"Forbidden/leakage columns found: {sorted(leakage_columns)}"
        )

    print("\nFeature columns:")
    for column in train_columns:
        print(f"  - {column}")

    print("\nValidation results:")
    print("  ✓ Feature columns match across train/validation/test.")
    print("  ✓ Feature dtypes match across train/validation/test.")
    print("  ✓ No missing values.")
    print("  ✓ No infinite values.")
    print("  ✓ No identifier, timestamp, target, or leakage columns.")
    print("\nAll feature dataset checks passed.")


if __name__ == "__main__":
    main()