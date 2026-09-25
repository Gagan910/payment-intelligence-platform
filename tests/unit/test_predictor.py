import pandas as pd
import pytest

from payment_platform.ml.predictor import PaymentPredictor


def make_transaction() -> pd.DataFrame:
    return pd.DataFrame(
        [
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
            }
        ]
    )


def test_predictor_loads_model():
    predictor = PaymentPredictor()

    assert predictor.model is not None
    assert predictor.model_version == "logistic_regression_baseline_v1"


def test_predict_failure_probability_returns_valid_probability():
    predictor = PaymentPredictor()

    transaction = make_transaction()

    probability = predictor.predict_failure_probability(transaction)

    assert 0 <= probability <= 1
    assert isinstance(probability, float)


def test_predictor_returns_failure_probability_not_success_probability():
    predictor = PaymentPredictor()

    transaction = make_transaction()

    failure_probability = predictor.predict_failure_probability(transaction)

    success_probability = float(
        predictor.model.predict_proba(
            __import__(
                "payment_platform.data.features",
                fromlist=["get_model_features"],
            ).get_model_features(transaction)
        )[0, 1]
    )

    assert failure_probability == pytest.approx(
        1.0 - success_probability
    )