import pandas as pd
import pytest

from payment_platform.ml.predictor import PaymentPredictor
from payment_platform.recommendation.engine import RecommendationResult
from payment_platform.services.payment_decision import (
    build_prediction_dataframe,
    build_prediction_input,
    generate_smart_routing_recommendation,
    predict_current_payment_failure_probability,
)


VALID_TRANSACTION_CONTEXT = {
    "transaction_id": "txn_test",
    "amount": 500.0,
    "merchant_category": "electronics",
    "payment_method": "debit_card",
    "user_segment": "regular",
    "device_type": "mobile",
    "network_quality": "poor",
    "hour_of_day": 0,
    "day_of_week": 2,
    "retry_count": 0,
    "transaction_velocity": 2,
    "user_method_success_rate": 0.90,
    "merchant_method_success_rate": 0.92,
}


def test_build_prediction_input_returns_required_fields() -> None:
    result = build_prediction_input(VALID_TRANSACTION_CONTEXT)

    assert result == VALID_TRANSACTION_CONTEXT


def test_build_prediction_dataframe_returns_single_row() -> None:
    result = build_prediction_dataframe(VALID_TRANSACTION_CONTEXT)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 13)
    assert result.iloc[0]["payment_method"] == "debit_card"
    assert result.iloc[0]["network_quality"] == "poor"


def test_build_prediction_input_rejects_missing_fields() -> None:
    incomplete_context = VALID_TRANSACTION_CONTEXT.copy()
    del incomplete_context["merchant_category"]

    with pytest.raises(
        ValueError,
        match="Transaction context is missing required fields",
    ):
        build_prediction_input(incomplete_context)


def test_predict_current_payment_failure_probability() -> None:
    predictor = PaymentPredictor()

    probability = predict_current_payment_failure_probability(
        VALID_TRANSACTION_CONTEXT,
        predictor=predictor,
    )

    assert isinstance(probability, float)
    assert 0 <= probability <= 1


def test_generate_smart_routing_recommendation() -> None:
    predictor = PaymentPredictor()

    result = generate_smart_routing_recommendation(
        VALID_TRANSACTION_CONTEXT,
        predictor=predictor,
    )

    assert isinstance(result, RecommendationResult)
    assert result.current_method == "debit_card"
    assert result.recommended_method == "upi"
    assert result.recommendation_action == "RECOMMEND_ALTERNATIVE"
    assert 0 <= result.current_failure_probability <= 1
    assert 0 <= result.recommended_failure_probability <= 1
    assert result.expected_improvement >= 0.05