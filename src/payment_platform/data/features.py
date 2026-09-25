import numpy as np
import pandas as pd


TARGET_COLUMN = "payment_success"

LEAKAGE_COLUMNS = {
    "failure_probability",
    TARGET_COLUMN,
}

IDENTIFIER_COLUMNS = {
    "transaction_id",
    "user_id",
    "merchant_id",
}

BASE_FEATURE_COLUMNS = [
    "amount",
    "merchant_category",
    "payment_method",
    "user_segment",
    "device_type",
    "network_quality",
    "hour_of_day",
    "day_of_week",
    "retry_count",
    "transaction_velocity",
    "user_method_success_rate",
    "merchant_method_success_rate",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from raw transaction data."""

    required_columns = set(BASE_FEATURE_COLUMNS)

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    features = df.copy()

    # Transaction amount transformation.
    features["log_amount"] = np.log1p(features["amount"])

    # Transaction velocity transformation for models
    # that benefit from a compressed right tail.
    features["log_transaction_velocity"] = np.log1p(
        features["transaction_velocity"]
    )

    # Cyclical encoding of hour of day.
    features["hour_sin"] = np.sin(
        2 * np.pi * features["hour_of_day"] / 24
    )

    features["hour_cos"] = np.cos(
        2 * np.pi * features["hour_of_day"] / 24
    )

    # Cyclical encoding of day of week.
    features["day_sin"] = np.sin(
        2 * np.pi * features["day_of_week"] / 7
    )

    features["day_cos"] = np.cos(
        2 * np.pi * features["day_of_week"] / 7
    )

    return features


def get_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return features that can safely be passed to the ML pipeline."""

    features = create_features(df)

    excluded_columns = (
        LEAKAGE_COLUMNS
        | IDENTIFIER_COLUMNS
        | {"timestamp"}
    )

    model_features = features.drop(
        columns=[
            column
            for column in excluded_columns
            if column in features.columns
        ]
    )

    return model_features