from __future__ import annotations

import pandas as pd

from payment_platform.ml.predictor import PaymentPredictor
from payment_platform.recommendation.counterfactual import (
    predict_counterfactual_failure_probabilities,
)
from payment_platform.recommendation.engine import (
    RecommendationResult,
    generate_recommendation,
)


PREDICTION_FIELDS = (
    "transaction_id",
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
)


def build_prediction_input(
    transaction_context: dict,
) -> dict:
    """Build the prediction input from persisted transaction context."""

    missing_fields = [
        field
        for field in PREDICTION_FIELDS
        if field not in transaction_context
    ]

    if missing_fields:
        raise ValueError(
            f"Transaction context is missing required fields: {missing_fields}"
        )

    return {
        field: transaction_context[field]
        for field in PREDICTION_FIELDS
    }


def build_prediction_dataframe(
    transaction_context: dict,
) -> pd.DataFrame:
    """Build a model-ready DataFrame from persisted transaction context."""

    prediction_input = build_prediction_input(transaction_context)

    return pd.DataFrame([prediction_input])


def predict_current_payment_failure_probability(
    transaction_context: dict,
    *,
    predictor: PaymentPredictor,
) -> float:
    """Predict failure probability for the currently selected method."""

    transaction_df = build_prediction_dataframe(transaction_context)

    return predictor.predict_failure_probability(transaction_df)


def generate_smart_routing_recommendation(
    transaction_context: dict,
    *,
    predictor: PaymentPredictor,
) -> RecommendationResult:
    """
    Generate a smart-routing recommendation from persisted transaction context.

    The transaction context is kept fixed while the payment method is varied
    counterfactually across the supported payment methods.
    """

    prediction_input = build_prediction_input(transaction_context)
    transaction_series = pd.Series(prediction_input)

    failure_probabilities = (
        predict_counterfactual_failure_probabilities(
            model=predictor.model,
            transaction=transaction_series,
        )
    )

    return generate_recommendation(
        current_method=str(transaction_context["payment_method"]),
        method_failure_probabilities=failure_probabilities,
    )