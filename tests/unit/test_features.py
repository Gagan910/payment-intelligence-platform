import numpy as np
import pandas as pd

from payment_platform.data.features import (
    create_features,
    get_model_features,
)


def test_create_features_adds_expected_columns():
    df = pd.DataFrame(
        {
            "amount": [100.0, 1000.0],
            "merchant_category": ["food", "electronics"],
            "payment_method": ["upi", "credit_card"],
            "user_segment": ["regular", "new"],
            "device_type": ["mobile", "desktop"],
            "network_quality": ["good", "poor"],
            "hour_of_day": [0, 12],
            "day_of_week": [0, 6],
            "retry_count": [0, 1],
            "transaction_velocity": [1, 4],
            "user_method_success_rate": [0.9, 0.8],
            "merchant_method_success_rate": [0.95, 0.85],
        }
    )

    result = create_features(df)

    expected_columns = {
        "log_amount",
        "log_transaction_velocity",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    }

    assert expected_columns.issubset(result.columns)


def test_log_features_are_correct():
    df = pd.DataFrame(
        {
            "amount": [100.0],
            "merchant_category": ["food"],
            "payment_method": ["upi"],
            "user_segment": ["regular"],
            "device_type": ["mobile"],
            "network_quality": ["good"],
            "hour_of_day": [12],
            "day_of_week": [3],
            "retry_count": [0],
            "transaction_velocity": [4],
            "user_method_success_rate": [0.9],
            "merchant_method_success_rate": [0.95],
        }
    )

    result = create_features(df)

    assert np.isclose(result.loc[0, "log_amount"], np.log1p(100.0))
    assert np.isclose(
        result.loc[0, "log_transaction_velocity"],
        np.log1p(4),
    )


def test_hour_cyclical_encoding():
    df = pd.DataFrame(
        {
            "amount": [100.0, 100.0],
            "merchant_category": ["food", "food"],
            "payment_method": ["upi", "upi"],
            "user_segment": ["regular", "regular"],
            "device_type": ["mobile", "mobile"],
            "network_quality": ["good", "good"],
            "hour_of_day": [0, 6],
            "day_of_week": [0, 0],
            "retry_count": [0, 0],
            "transaction_velocity": [1, 1],
            "user_method_success_rate": [0.9, 0.9],
            "merchant_method_success_rate": [0.95, 0.95],
        }
    )

    result = create_features(df)

    assert np.isclose(result.loc[0, "hour_sin"], 0.0)
    assert np.isclose(result.loc[0, "hour_cos"], 1.0)

    assert np.isclose(result.loc[1, "hour_sin"], 1.0)
    assert np.isclose(result.loc[1, "hour_cos"], 0.0, atol=1e-10)


def test_model_features_exclude_target_and_leakage():
    df = pd.DataFrame(
        {
            "transaction_id": ["txn_1"],
            "timestamp": pd.Timestamp("2024-01-01"),
            "user_id": ["user_1"],
            "merchant_id": ["merchant_1"],
            "amount": [100.0],
            "merchant_category": ["food"],
            "payment_method": ["upi"],
            "user_segment": ["regular"],
            "device_type": ["mobile"],
            "network_quality": ["good"],
            "hour_of_day": [12],
            "day_of_week": [0],
            "retry_count": [0],
            "transaction_velocity": [1],
            "user_method_success_rate": [0.9],
            "merchant_method_success_rate": [0.95],
            "failure_probability": [0.1],
            "payment_success": [1],
        }
    )

    result = get_model_features(df)

    forbidden_columns = {
        "transaction_id",
        "timestamp",
        "user_id",
        "merchant_id",
        "failure_probability",
        "payment_success",
    }

    assert forbidden_columns.isdisjoint(result.columns)


def test_model_features_have_no_missing_values():
    df = pd.DataFrame(
        {
            "amount": [100.0],
            "merchant_category": ["food"],
            "payment_method": ["upi"],
            "user_segment": ["regular"],
            "device_type": ["mobile"],
            "network_quality": ["good"],
            "hour_of_day": [12],
            "day_of_week": [0],
            "retry_count": [0],
            "transaction_velocity": [1],
            "user_method_success_rate": [0.9],
            "merchant_method_success_rate": [0.95],
        }
    )

    result = get_model_features(df)

    assert result.isna().sum().sum() == 0