import pandas as pd
from fastapi import APIRouter

from payment_platform.api.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from payment_platform.ml.predictor import PaymentPredictor


router = APIRouter(prefix="/predict", tags=["Prediction"])

predictor = PaymentPredictor()


@router.post("", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict payment failure probability for a transaction."""

    transaction = request.model_dump()
    transaction_df = pd.DataFrame([transaction])

    failure_probability = predictor.predict_failure_probability(
        transaction_df
    )

    success_probability = 1.0 - failure_probability

    return PredictionResponse(
        failure_probability=failure_probability,
        success_probability=success_probability,
        model_version=predictor.model_version,
    )