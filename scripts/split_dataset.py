from pathlib import Path

import pandas as pd

from payment_platform.data.splitting import (
    temporal_train_validation_test_split,
)


INPUT_PATH = Path("data/processed/transactions_v1.parquet")

TRAIN_OUTPUT_PATH = Path("data/processed/train.parquet")
VALIDATION_OUTPUT_PATH = Path("data/processed/validation.parquet")
TEST_OUTPUT_PATH = Path("data/processed/test.parquet")

TRAIN_END = "2025-07-01"
VALIDATION_END = "2025-10-01"


def main() -> None:
    print("Loading dataset...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"Input rows: {len(df):,}")

    train, validation, test = temporal_train_validation_test_split(
        df,
        train_end=TRAIN_END,
        validation_end=VALIDATION_END,
    )

    TRAIN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    train.to_parquet(TRAIN_OUTPUT_PATH, index=False)
    validation.to_parquet(VALIDATION_OUTPUT_PATH, index=False)
    test.to_parquet(TEST_OUTPUT_PATH, index=False)

    print("\nTemporal split completed.")
    print("-" * 50)

    print(f"TRAIN:")
    print(f"  Rows: {len(train):,}")
    print(f"  Date range: {train['timestamp'].min()} -> {train['timestamp'].max()}")
    print(f"  Output: {TRAIN_OUTPUT_PATH}")

    print(f"\nVALIDATION:")
    print(f"  Rows: {len(validation):,}")
    print(
        f"  Date range: "
        f"{validation['timestamp'].min()} -> {validation['timestamp'].max()}"
    )
    print(f"  Output: {VALIDATION_OUTPUT_PATH}")

    print(f"\nTEST:")
    print(f"  Rows: {len(test):,}")
    print(f"  Date range: {test['timestamp'].min()} -> {test['timestamp'].max()}")
    print(f"  Output: {TEST_OUTPUT_PATH}")

    print("\nRow-count verification:")
    print(f"  Total split rows: {len(train) + len(validation) + len(test):,}")
    print(f"  Original rows:    {len(df):,}")

    if len(train) + len(validation) + len(test) == len(df):
        print("  ✓ All rows accounted for.")
    else:
        raise ValueError("Split row counts do not match the original dataset.")


if __name__ == "__main__":
    main()