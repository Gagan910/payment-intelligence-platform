from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException

from payment_platform.api.schemas.recommendation import (
    RecommendationDecisionRequest,
    RecommendationRequest,
    RecommendationResponse,
)
from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.recommendations import (
    create_recommendation,
    get_recommendation,
    update_recommendation_decision,
)
from payment_platform.db.repositories.transactions import get_transaction
from payment_platform.ml.predictor import PaymentPredictor
from payment_platform.recommendation.counterfactual import (
    predict_counterfactual_failure_probabilities,
)
from payment_platform.recommendation.engine import generate_recommendation
from payment_platform.recommendation.selection import select_payment_method
from payment_platform.experiments.events import record_experiment_event


router = APIRouter(prefix="/recommend", tags=["Recommendation"])

predictor = PaymentPredictor()


@router.post("", response_model=RecommendationResponse)
def recommend(
    request: RecommendationRequest,
) -> RecommendationResponse:
    transaction = request.model_dump(exclude={"transaction_id"})
    transaction_series = pd.Series(transaction)

    failure_probabilities = (
        predict_counterfactual_failure_probabilities(
            model=predictor.model,
            transaction=transaction_series,
        )
    )

    recommendation = generate_recommendation(
        current_method=request.payment_method,
        method_failure_probabilities=failure_probabilities,
    )

    recommendation_timestamp = datetime.now(timezone.utc)

    conn = get_connection()

    try:
        try:
            recommendation_id = create_recommendation(
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
        recommendation_id=recommendation_id,
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


@router.post("/decision")
def record_recommendation_decision(
    request: RecommendationDecisionRequest,
) -> dict[str, object]:
    conn = get_connection()

    try:
        recommendation = get_recommendation(
            conn,
            recommendation_id=request.recommendation_id,
        )

        if recommendation is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Recommendation "
                    f"'{request.recommendation_id}' does not exist."
                ),
            )

        transaction = get_transaction(
            conn,
            transaction_id=str(recommendation["transaction_id"]),
        )

        if transaction is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Transaction "
                    f"'{recommendation['transaction_id']}' does not exist."
                ),
            )

        update_recommendation_decision(
            conn,
            recommendation_id=request.recommendation_id,
            accepted=request.accepted,
        )

        current_method = str(recommendation["original_method"])
        recommended_method = recommendation["recommended_method"]

        if recommended_method is not None:
            from payment_platform.recommendation.engine import (
                RecommendationResult,
            )

            recommendation_result = RecommendationResult(
                recommendation_action="RECOMMEND_ALTERNATIVE",
                current_method=current_method,
                recommended_method=str(recommended_method),
                current_failure_probability=0.0,
                recommended_failure_probability=0.0,
                expected_improvement=0.0,
                reason=str(recommendation["reason"]),
            )
        else:
            from payment_platform.recommendation.engine import (
                RecommendationResult,
            )

            recommendation_result = RecommendationResult(
                recommendation_action="KEEP_CURRENT",
                current_method=current_method,
                recommended_method=None,
                current_failure_probability=0.0,
                recommended_failure_probability=None,
                expected_improvement=0.0,
                reason=str(recommendation["reason"]),
            )

        selected_payment_method = select_payment_method(
            recommendation_result,
            accepted=request.accepted,
        )

        record_experiment_event(
            conn,
            experiment_id="payment_routing_v1",
            transaction_id=str(recommendation["transaction_id"]),
            event_type="recommendation_decision",
            metadata={
                "variant": transaction["experiment_variant"],
                "accepted": request.accepted,
                "original_method": current_method,
                "recommended_method": (
                    str(recommended_method)
                    if recommended_method is not None
                    else None
                ),
                "selected_payment_method": selected_payment_method,
            },
            commit=False,
        )

        conn.commit()

        return {
            "recommendation_id": request.recommendation_id,
            "accepted": request.accepted,
            "selected_payment_method": selected_payment_method,
        }

    except HTTPException:
        conn.rollback()
        raise

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()