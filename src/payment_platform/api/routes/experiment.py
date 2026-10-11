
from math import sqrt
from typing import Literal

from fastapi import APIRouter, HTTPException

from payment_platform.api.schemas.experiment import (
    ExperimentSummaryResponse,
    ExperimentVariantSummary,
)
from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.experiment_analytics import (
    get_experiment_summary,
)


router = APIRouter(prefix="/experiments", tags=["Experiments"])

MINIMUM_SAMPLE_SIZE_PER_VARIANT = 100
WILSON_Z_SCORE = 1.96


def wilson_confidence_interval(
    successes: int,
    attempts: int,
) -> tuple[float | None, float | None]:
    """Calculate the two-sided 95% Wilson score interval for a proportion."""

    if attempts <= 0:
        return None, None

    if not 0 <= successes <= attempts:
        raise ValueError("Successes must be between zero and attempts.")

    proportion = successes / attempts
    z_squared = WILSON_Z_SCORE**2
    denominator = 1 + z_squared / attempts

    center = (
        proportion + z_squared / (2 * attempts)
    ) / denominator

    margin = (
        WILSON_Z_SCORE
        * sqrt(
            proportion * (1 - proportion) / attempts
            + z_squared / (4 * attempts**2)
        )
        / denominator
    )

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return lower, upper


def determine_experiment_status(
    control_rate: float | None,
    treatment_rate: float | None,
    control_attempts: int,
    treatment_attempts: int,
) -> Literal[
    "insufficient_data",
    "inconclusive",
    "observed_positive_lift",
    "observed_negative_lift",
    "no_observed_lift",
]:
    """Describe observed lift without claiming statistical significance."""

    if (
        control_attempts < MINIMUM_SAMPLE_SIZE_PER_VARIANT
        or treatment_attempts < MINIMUM_SAMPLE_SIZE_PER_VARIANT
        or control_rate is None
        or treatment_rate is None
    ):
        return "insufficient_data"

    if treatment_rate > control_rate:
        return "observed_positive_lift"

    if treatment_rate < control_rate:
        return "observed_negative_lift"

    return "no_observed_lift"


@router.get(
    "/{experiment_id}/summary",
    response_model=ExperimentSummaryResponse,
)
def experiment_summary_endpoint(
    experiment_id: str,
) -> ExperimentSummaryResponse:
    """Return transaction-level success metrics for an A/B experiment."""

    if experiment_id != "payment_routing_v1":
        raise HTTPException(
            status_code=404,
            detail="Experiment not found.",
        )

    conn = get_connection()

    try:
        summary = get_experiment_summary(
            conn,
            experiment_id=experiment_id,
        )
    finally:
        conn.close()

    control = summary["control"]
    treatment = summary["treatment"]

    control_attempts = control["attempted_transactions"]
    treatment_attempts = treatment["attempted_transactions"]

    control_rate = (
        control["successful_transactions"] / control_attempts
        if control_attempts > 0
        else None
    )

    treatment_rate = (
        treatment["successful_transactions"] / treatment_attempts
        if treatment_attempts > 0
        else None
    )

    control_ci_lower, control_ci_upper = wilson_confidence_interval(
        control["successful_transactions"],
        control_attempts,
    )

    treatment_ci_lower, treatment_ci_upper = wilson_confidence_interval(
        treatment["successful_transactions"],
        treatment_attempts,
    )

    absolute_lift = (
        (treatment_rate - control_rate) * 100
        if control_rate is not None and treatment_rate is not None
        else None
    )

    relative_lift = (
        ((treatment_rate - control_rate) / control_rate) * 100
        if (
            control_rate is not None
            and treatment_rate is not None
            and control_rate > 0
        )
        else None
    )

    experiment_status = determine_experiment_status(
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        control_attempts=control_attempts,
        treatment_attempts=treatment_attempts,
    )

    return ExperimentSummaryResponse(
        experiment_id=experiment_id,
        metric="attempted_transaction_success_rate",
        control=ExperimentVariantSummary(
            **control,
            success_rate=control_rate,
            success_rate_ci_lower=control_ci_lower,
            success_rate_ci_upper=control_ci_upper,
        ),
        treatment=ExperimentVariantSummary(
            **treatment,
            success_rate=treatment_rate,
            success_rate_ci_lower=treatment_ci_lower,
            success_rate_ci_upper=treatment_ci_upper,
        ),
        absolute_lift_percentage_points=absolute_lift,
        relative_lift_percent=relative_lift,
        minimum_sample_size_per_variant=MINIMUM_SAMPLE_SIZE_PER_VARIANT,
        experiment_status=experiment_status,
    )
