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

    control_rate = (
        control["successful_transactions"]
        / control["attempted_transactions"]
        if control["attempted_transactions"] > 0
        else None
    )

    treatment_rate = (
        treatment["successful_transactions"]
        / treatment["attempted_transactions"]
        if treatment["attempted_transactions"] > 0
        else None
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

    return ExperimentSummaryResponse(
        experiment_id=experiment_id,
        metric="attempted_transaction_success_rate",
        control=ExperimentVariantSummary(
            **control,
            success_rate=control_rate,
        ),
        treatment=ExperimentVariantSummary(
            **treatment,
            success_rate=treatment_rate,
        ),
        absolute_lift_percentage_points=absolute_lift,
        relative_lift_percent=relative_lift,
    )
