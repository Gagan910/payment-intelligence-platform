from fastapi import APIRouter

from payment_platform.api.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)


router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Prediction endpoint placeholder.

    ML model integration will be added in the next step.
    """
    return PredictionResponse(
        failure_probability=0.0,
        success_probability=1.0,
        model_version="placeholder",
    )