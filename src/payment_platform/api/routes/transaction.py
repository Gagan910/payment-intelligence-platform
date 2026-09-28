from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from payment_platform.api.schemas.transaction import (
    TransactionRequest,
    TransactionResponse,
)
from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.transactions import create_transaction
from payment_platform.experiments.config import (
    EXPERIMENT_ID,
    TREATMENT_PERCENTAGE,
)
from payment_platform.experiments.service import assign_and_persist_variant

from payment_platform.experiments.events import record_experiment_event

router = APIRouter(prefix="/transactions", tags=["Transactions"])


@router.post("", response_model=TransactionResponse)
def create_transaction_endpoint(
    request: TransactionRequest,
) -> TransactionResponse:
    """Create a transaction and assign its experiment variant server-side."""

    transaction_timestamp = datetime.now(timezone.utc)

    conn = get_connection()

    try:
        experiment_variant = assign_and_persist_variant(
            conn,
            experiment_id=EXPERIMENT_ID,
            subject_id=request.user_id,
            user_id=request.user_id,
            session_id=None,
            treatment_percentage=TREATMENT_PERCENTAGE,
            commit=False,
        )

        create_transaction(
            conn,
            transaction_id=request.transaction_id,
            user_id=request.user_id,
            merchant_id=request.merchant_id,
            amount=request.amount,
            currency=request.currency,
            timestamp=transaction_timestamp,
            selected_payment_method=request.selected_payment_method,
            experiment_variant=experiment_variant,
            status=request.status,
        )

        record_experiment_event(
            conn,
            experiment_id=EXPERIMENT_ID,
            transaction_id=request.transaction_id,
            event_type="checkout_started",
            metadata={
                "variant": experiment_variant,
                "payment_method": request.selected_payment_method,
                "amount": request.amount,
            },
            commit=False,
        )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    return TransactionResponse(
        transaction_id=request.transaction_id,
        status=request.status,
    )