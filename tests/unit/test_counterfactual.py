import pandas as pd
import pytest

from payment_platform.recommendation.counterfactual import (
    DEFAULT_PAYMENT_METHODS,
    build_counterfactual_transactions,
)


def test_build_counterfactual_transactions_changes_only_payment_method():
    transaction = pd.Series(
        {
            "transaction_id": "txn_test",
            "timestamp": pd.Timestamp("2025-01-01"),
            "user_id": "user_001",
            "merchant_id": "merchant_001",
            "amount": 500.0,
            "merchant_category": "electronics",
            "payment_method": "upi",
            "user_segment": "regular",
            "device_type": "mobile",
            "network_quality": "good",
            "hour_of_day": 14,
            "day_of_week": 2,
            "retry_count": 0,
            "transaction_velocity": 2,
            "user_method_success_rate": 0.90,
            "merchant_method_success_rate": 0.92,
            "failure_probability": 0.15,
            "payment_success": 1,
        }
    )

    result = build_counterfactual_transactions(transaction)

    assert len(result) == 5
    assert set(result["payment_method"]) == set(DEFAULT_PAYMENT_METHODS)

    unchanged_columns = [
        column
        for column in transaction.index
        if column != "payment_method"
    ]

    for column in unchanged_columns:
        assert result[column].tolist() == [transaction[column]] * 5


def test_build_counterfactual_transactions_supports_custom_methods():
    transaction = pd.Series(
        {
            "payment_method": "upi",
            "amount": 500.0,
        }
    )

    result = build_counterfactual_transactions(
        transaction,
        payment_methods=["upi", "wallet"],
    )

    assert len(result) == 2
    assert result["payment_method"].tolist() == ["upi", "wallet"]


def test_rejects_empty_payment_methods():
    transaction = pd.Series({"payment_method": "upi"})

    with pytest.raises(ValueError, match="cannot be empty"):
        build_counterfactual_transactions(
            transaction,
            payment_methods=[],
        )


def test_rejects_missing_payment_method():
    transaction = pd.Series({"amount": 500.0})

    with pytest.raises(ValueError, match="payment_method"):
        build_counterfactual_transactions(transaction)