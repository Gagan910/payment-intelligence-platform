from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException

from payment_platform.api.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
)
from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.recommendations import (
    create_recommendation,
)
from payment_platform.ml.predictor import PaymentPredictor
from payment_platform.recommendation.counterfactual import (
    predict_counterfactual_failure_probabilities,
)
from payment_platform.recommendation.engine import generate_recommendation


router = APIRouter(prefix="/recommend", tags=["Recommendation"])

predictor = PaymentPredictor()


@router.post("", response_model=RecommendationResponse)
def recommend(
    request: RecommendationRequest,
) -> RecommendationResponse:
    """Generate and persist a smart payment recommendation."""

    transaction = request.model_dump(exclude={"transaction_id"})
    transaction_series = pd.Series(transaction)

    failure_probabilities = predict_counterfactual_failure_probabilities(
        model=predictor.model,
        transaction=transaction_series,
    )

    recommendation = generate_recommendation(
        current_method=request.payment_method,
        method_failure_probabilities=failure_probabilities,
    )

    recommendation_timestamp = datetime.now(timezone.utc)

    conn = get_connection()

    try:
        try:
            create_recommendation(
                conn,
                transaction_id=request.transaction_id,
                original_method=recommendation.current_method,
                recommended_method=recommendation.recommended_method,
                recommendation_score=(
                    recommendation.recommended_failure_probability
                ),
                reason=recommendation.reason,
                accepted=None,
                timestamp=recommendation_timestamp,
            )
        except Exception as exc:
            conn.rollback()
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Transaction '{request.transaction_id}' "
                    "does not exist."
                ),
            ) from exc
    finally:
        conn.close()

    return RecommendationResponse(
        transaction_id=request.transaction_id,
        recommendation_action=recommendation.recommendation_action,
        current_method=recommendation.current_method,
        recommended_method=recommendation.recommended_method,
        current_failure_probability=(
            recommendation.current_failure_probability
        ),
        recommended_failure_probability=(
            recommendation.recommended_failure_probability
        ),
        expected_improvement=recommendation.expected_improvement,
        reason=recommendation.reason,
        model_version=predictor.model_version,
    )