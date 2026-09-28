from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException

from payment_platform.api.schemas.payment_attempt import (
    PaymentAttemptRequest,
    PaymentAttemptResponse,
)
from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.payment_attempts import (
    create_payment_attempt,
)
from payment_platform.db.repositories.transactions import (
    get_transaction,
    get_transaction_context,
    update_transaction_status,
)
from payment_platform.experiments.events import record_experiment_event
from payment_platform.simulation.payment import simulate_payment_outcome


router = APIRouter(
    prefix="/payment-attempts",
    tags=["Payment Attempts"],
)


@router.post("", response_model=PaymentAttemptResponse)
def create_payment_attempt_endpoint(
    request: PaymentAttemptRequest,
) -> PaymentAttemptResponse:
    """Create a simulated payment attempt and persist its outcome."""

    started_at = datetime.now(timezone.utc)

    conn = get_connection()

    try:
        transaction = get_transaction(
            conn,
            transaction_id=request.transaction_id,
        )

        if transaction is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Transaction '{request.transaction_id}' "
                    "does not exist."
                ),
            )

        transaction_context_data = get_transaction_context(
            conn,
            transaction_id=request.transaction_id,
            payment_method=request.payment_method,
        )

        if transaction_context_data is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Transaction '{request.transaction_id}' "
                    "or its transaction context does not exist."
                ),
            )

        transaction_context = pd.Series(transaction_context_data)

        _, outcome = simulate_payment_outcome(transaction_context)

        completed_at = datetime.now(timezone.utc)

        attempt_id = create_payment_attempt(
            conn,
            transaction_id=request.transaction_id,
            payment_method=request.payment_method,
            attempt_number=request.attempt_number,
            started_at=started_at,
            completed_at=completed_at,
            outcome=outcome,
            commit=False,
        )

        update_transaction_status(
            conn,
            transaction_id=request.transaction_id,
            status=outcome,
            commit=False,
        )

        record_experiment_event(
            conn,
            experiment_id="payment_routing_v1",
            transaction_id=request.transaction_id,
            event_type="payment_completed",
            metadata={
                "variant": transaction["experiment_variant"],
                "payment_method": request.payment_method,
                "attempt_number": request.attempt_number,
                "outcome": outcome,
            },
            commit=False,
        )

        conn.commit()

    except HTTPException:
        conn.rollback()
        raise

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    return PaymentAttemptResponse(
        attempt_id=str(attempt_id),
        transaction_id=request.transaction_id,
        payment_method=request.payment_method,
        attempt_number=request.attempt_number,
        outcome=outcome,
    )