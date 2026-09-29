from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException

from payment_platform.api.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.predictions import create_prediction
from payment_platform.db.repositories.transactions import (
    get_transaction_context,
)
from payment_platform.ml.predictor import PaymentPredictor


router = APIRouter(prefix="/predict", tags=["Prediction"])

predictor = PaymentPredictor()


@router.post("", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict payment failure probability using persisted transaction context."""

    conn = get_connection()

    try:
        transaction_context = get_transaction_context(
            conn,
            transaction_id=request.transaction_id,
        )

        if transaction_context is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Transaction '{request.transaction_id}' "
                    "does not exist or has no transaction context."
                ),
            )

        transaction_df = pd.DataFrame([transaction_context])

        prediction_started_at = datetime.now(timezone.utc)

        failure_probability = predictor.predict_failure_probability(
            transaction_df
        )

        prediction_completed_at = datetime.now(timezone.utc)

        success_probability = 1.0 - failure_probability

        latency_ms = (
            prediction_completed_at - prediction_started_at
        ).total_seconds() * 1000.0

        create_prediction(
            conn,
            transaction_id=request.transaction_id,
            model_version=predictor.model_version,
            failure_probability=failure_probability,
            success_probability=success_probability,
            prediction_timestamp=prediction_completed_at,
            latency_ms=latency_ms,
        )

    except HTTPException:
        conn.rollback()
        raise

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    return PredictionResponse(
        transaction_id=request.transaction_id,
        failure_probability=failure_probability,
        success_probability=success_probability,
        model_version=predictor.model_version,
    )