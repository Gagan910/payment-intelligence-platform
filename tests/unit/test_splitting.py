import pandas as pd
import pytest

from payment_platform.data.splitting import (
    temporal_train_validation_test_split,
)


def create_test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [
                "txn_1",
                "txn_2",
                "txn_3",
                "txn_4",
                "txn_5",
                "txn_6",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-06-01",
                    "2025-06-30",
                    "2025-07-01",
                    "2025-09-30",
                    "2025-10-01",
                ]
            ),
            "payment_success": [1, 0, 1, 1, 0, 1],
        }
    )


def test_temporal_split_has_no_data_loss():
    df = create_test_data()

    train, validation, test = temporal_train_validation_test_split(
        df,
        train_end="2025-07-01",
        validation_end="2025-10-01",
    )

    assert len(train) + len(validation) + len(test) == len(df)


def test_temporal_split_boundaries():
    df = create_test_data()

    train, validation, test = temporal_train_validation_test_split(
        df,
        train_end="2025-07-01",
        validation_end="2025-10-01",
    )

    assert train["timestamp"].max() < pd.Timestamp("2025-07-01")

    assert validation["timestamp"].min() >= pd.Timestamp("2025-07-01")
    assert validation["timestamp"].max() < pd.Timestamp("2025-10-01")

    assert test["timestamp"].min() >= pd.Timestamp("2025-10-01")


def test_temporal_split_preserves_rows():
    df = create_test_data()

    train, validation, test = temporal_train_validation_test_split(
        df,
        train_end="2025-07-01",
        validation_end="2025-10-01",
    )

    combined_ids = set(train["transaction_id"]) | set(
        validation["transaction_id"]
    ) | set(test["transaction_id"])

    assert combined_ids == set(df["transaction_id"])


def test_invalid_boundaries_raise_error():
    df = create_test_data()

    with pytest.raises(ValueError):
        temporal_train_validation_test_split(
            df,
            train_end="2025-10-01",
            validation_end="2025-07-01",
        )


def test_empty_split_raises_error():
    df = create_test_data()

    with pytest.raises(ValueError):
        temporal_train_validation_test_split(
            df,
            train_end="2023-01-01",
            validation_end="2023-06-01",
        )