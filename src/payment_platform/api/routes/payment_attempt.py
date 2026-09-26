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
from payment_platform.db.repositories.transactions import get_transaction
from payment_platform.simulation.payment import simulate_payment_outcome


router = APIRouter(prefix="/payment-attempts", tags=["Payment Attempts"])


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

        transaction_context = pd.Series(
            {
                "amount": transaction["amount"],
                "merchant_category": "electronics",
                "payment_method": request.payment_method,
                "user_segment": "regular",
                "device_type": "mobile",
                "network_quality": "good",
                "hour_of_day": started_at.hour,
                "day_of_week": started_at.weekday(),
                "retry_count": request.attempt_number - 1,
                "transaction_velocity": 1,
                "user_method_success_rate": 0.90,
                "merchant_method_success_rate": 0.92,
            }
        )

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
        )

    finally:
        conn.close()

    return PaymentAttemptResponse(
        attempt_id=str(attempt_id),
        transaction_id=request.transaction_id,
        payment_method=request.payment_method,
        attempt_number=request.attempt_number,
        outcome=outcome,
    )