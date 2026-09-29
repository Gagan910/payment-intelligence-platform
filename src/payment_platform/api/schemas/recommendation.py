from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    transaction_id: str


class RecommendationResponse(BaseModel):
    recommendation_id: int
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


class RecommendationDecisionRequest(BaseModel):
    recommendation_id: int = Field(gt=0)
    accepted: bool