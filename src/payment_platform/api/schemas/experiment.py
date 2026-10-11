
from typing import Literal

from pydantic import BaseModel, Field


class ExperimentVariantSummary(BaseModel):
    transaction_count: int = Field(ge=0)
    attempted_transactions: int = Field(ge=0)
    successful_transactions: int = Field(ge=0)
    unresolved_transactions: int = Field(ge=0)

    success_rate: float | None = Field(default=None, ge=0, le=1)
    success_rate_ci_lower: float | None = Field(
        default=None, ge=0, le=1
    )
    success_rate_ci_upper: float | None = Field(
        default=None, ge=0, le=1
    )


class ExperimentSummaryResponse(BaseModel):
    experiment_id: str
    metric: str

    control: ExperimentVariantSummary
    treatment: ExperimentVariantSummary

    absolute_lift_percentage_points: float | None
    relative_lift_percent: float | None

    minimum_sample_size_per_variant: int = Field(default=100, ge=1)
    experiment_status: Literal[
        "insufficient_data",
        "inconclusive",
        "observed_positive_lift",
        "observed_negative_lift",
        "no_observed_lift",
    ]
