from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from payment_platform.api.schemas.transaction import (
    TransactionRequest,
    TransactionResponse,
)
from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.transactions import create_transaction


router = APIRouter(prefix="/transactions", tags=["Transactions"])


@router.post("", response_model=TransactionResponse)
def create_transaction_endpoint(
    request: TransactionRequest,
) -> TransactionResponse:
    """Create and persist a payment transaction."""

    transaction_timestamp = datetime.now(timezone.utc)

    conn = get_connection()

    try:
        create_transaction(
            conn,
            transaction_id=request.transaction_id,
            user_id=request.user_id,
            merchant_id=request.merchant_id,
            amount=request.amount,
            currency=request.currency,
            timestamp=transaction_timestamp,
            selected_payment_method=request.selected_payment_method,
            experiment_variant=request.experiment_variant,
            status=request.status,
        )
    finally:
        conn.close()

    return TransactionResponse(
        transaction_id=request.transaction_id,
        status=request.status,
    )