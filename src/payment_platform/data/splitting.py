from typing import Tuple

import pandas as pd


def temporal_train_validation_test_split(
    df: pd.DataFrame,
    train_end: str,
    validation_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split transactions chronologically into train, validation, and test sets.

    train:
        timestamp < train_end

    validation:
        train_end <= timestamp < validation_end

    test:
        timestamp >= validation_end
    """

    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain a 'timestamp' column.")

    data = df.copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    train_boundary = pd.Timestamp(train_end)
    validation_boundary = pd.Timestamp(validation_end)

    if train_boundary >= validation_boundary:
        raise ValueError(
            "train_end must be earlier than validation_end."
        )

    train = data[data["timestamp"] < train_boundary].copy()

    validation = data[
        (data["timestamp"] >= train_boundary)
        & (data["timestamp"] < validation_boundary)
    ].copy()

    test = data[data["timestamp"] >= validation_boundary].copy()

    if len(train) == 0:
        raise ValueError("Training split is empty.")

    if len(validation) == 0:
        raise ValueError("Validation split is empty.")

    if len(test) == 0:
        raise ValueError("Test split is empty.")

    return train, validation, test