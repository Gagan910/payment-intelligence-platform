from pathlib import Path

import numpy as np
import pandas as pd

from payment_platform.data.features import get_model_features


DATASETS = {
    "train": Path("data/processed/train.parquet"),
    "validation": Path("data/processed/validation.parquet"),
    "test": Path("data/processed/test.parquet"),
}

OUTPUT_DIR = Path("data/processed/features")


def prepare_dataset(name: str, input_path: Path) -> None:
    print(f"\nPreparing {name} dataset...")
    
    df = pd.read_parquet(input_path)

    target = df["payment_success"].copy()
    features = get_model_features(df)

    # Validate feature values.
    if features.isna().any().any():
        raise ValueError(f"{name}: missing values found in model features.")

    numeric_features = features.select_dtypes(include=np.number)

    if np.isinf(numeric_features.to_numpy()).any():
        raise ValueError(f"{name}: infinite values found in numeric features.")

    if len(features) != len(df):
        raise ValueError(f"{name}: feature row count changed.")

    if len(target) != len(df):
        raise ValueError(f"{name}: target row count changed.")

    # Save features and target separately.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_path = OUTPUT_DIR / f"{name}_features.parquet"
    target_path = OUTPUT_DIR / f"{name}_target.parquet"

    features.to_parquet(features_path, index=False)
    target.to_frame().to_parquet(target_path, index=False)

    print(f"  Original rows: {len(df):,}")
    print(f"  Feature rows:  {len(features):,}")
    print(f"  Feature columns: {len(features.columns)}")
    print(f"  Target rows:   {len(target):,}")
    print(f"  Features: {features_path}")
    print(f"  Target:   {target_path}")


def main() -> None:
    print("Preparing model-ready feature datasets...")
    
    for name, input_path in DATASETS.items():
        prepare_dataset(name, input_path)

    print("\nFeature preparation completed successfully.")


if __name__ == "__main__":
    main()
    