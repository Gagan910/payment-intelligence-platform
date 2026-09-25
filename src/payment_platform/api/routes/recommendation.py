from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from payment_platform.api.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
)
from payment_platform.ml.predictor import PaymentPredictor
from payment_platform.recommendation.counterfactual import (
    predict_counterfactual_failure_probabilities,
)
from payment_platform.recommendation.engine import generate_recommendation


router = APIRouter(
    prefix="/recommend",
    tags=["Recommendation"],
)

predictor = PaymentPredictor()


@router.post("", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    transaction = request.model_dump()
    transaction_series = pd.Series(transaction)

    failure_probabilities = predict_counterfactual_failure_probabilities(
        model=predictor.model,
        transaction=transaction_series,
    )

    recommendation = generate_recommendation(
        current_method=request.payment_method,
        method_failure_probabilities=failure_probabilities,
    )

    return RecommendationResponse(
        recommendation_action=recommendation.recommendation_action,
        current_method=recommendation.current_method,
        recommended_method=recommendation.recommended_method,
        current_failure_probability=recommendation.current_failure_probability,
        recommended_failure_probability=(
            recommendation.recommended_failure_probability
        ),
        expected_improvement=recommendation.expected_improvement,
        reason=recommendation.reason,
        model_version=predictor.model_version,
    )