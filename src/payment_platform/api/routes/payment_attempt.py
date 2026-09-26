from __future__ import annotations

import random
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from payment_platform.api.schemas.payment_attempt import (
    PaymentAttemptRequest,
    PaymentAttemptResponse,
)
from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.payment_attempts import (
    create_payment_attempt,
)


router = APIRouter(prefix="/payment-attempts", tags=["Payment Attempts"])


@router.post("", response_model=PaymentAttemptResponse)
def create_payment_attempt_endpoint(
    request: PaymentAttemptRequest,
) -> PaymentAttemptResponse:
    """Create a simulated payment attempt and persist its outcome."""

    started_at = datetime.now(timezone.utc)

    outcome = random.choice(["success", "failure"])

    completed_at = datetime.now(timezone.utc)

    conn = get_connection()

    try:
        try:
            attempt_id = create_payment_attempt(
                conn,
                transaction_id=request.transaction_id,
                payment_method=request.payment_method,
                attempt_number=request.attempt_number,
                started_at=started_at,
                completed_at=completed_at,
                outcome=outcome,
            )
        except Exception as exc:
            conn.rollback()
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Transaction '{request.transaction_id}' "
                    "does not exist."
                ),
            ) from exc
    finally:
        conn.close()

    return PaymentAttemptResponse(
        attempt_id=str(attempt_id),
        transaction_id=request.transaction_id,
        payment_method=request.payment_method,
        attempt_number=request.attempt_number,
        outcome=outcome,
    )