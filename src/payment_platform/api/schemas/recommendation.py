from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    transaction_id: str
    amount: float = Field(gt=0)
    merchant_category: str
    payment_method: str
    user_segment: str
    device_type: str
    network_quality: str
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    retry_count: int = Field(ge=0)
    transaction_velocity: int = Field(ge=0)
    user_method_success_rate: float = Field(ge=0, le=1)
    merchant_method_success_rate: float = Field(ge=0, le=1)


class RecommendationResponse(BaseModel):
    transaction_id: str
    recommendation_action: str
    current_method: str
    recommended_method: str | None
    current_failure_probability: float = Field(ge=0, le=1)
    recommended_failure_probability: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    expected_improvement: float
    reason: str
    model_version: str