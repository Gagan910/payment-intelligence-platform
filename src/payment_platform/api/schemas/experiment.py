from pydantic import BaseModel, Field


class ExperimentVariantSummary(BaseModel):
    transaction_count: int = Field(ge=0)
    attempted_transactions: int = Field(ge=0)
    successful_transactions: int = Field(ge=0)
    unresolved_transactions: int = Field(ge=0)
    success_rate: float | None = Field(default=None, ge=0, le=1)


class ExperimentSummaryResponse(BaseModel):
    experiment_id: str
    metric: str
    control: ExperimentVariantSummary
    treatment: ExperimentVariantSummary
    absolute_lift_percentage_points: float | None
    relative_lift_percent: float | None
