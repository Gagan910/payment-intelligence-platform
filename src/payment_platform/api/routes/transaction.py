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
from payment_platform.db.repositories.transaction_context import (
    create_transaction_context,
)
from payment_platform.experiments.events import record_experiment_event
from payment_platform.experiments.service import assign_and_persist_variant


router = APIRouter(prefix="/transactions", tags=["Transactions"])


@router.post("", response_model=TransactionResponse)
def create_transaction_endpoint(
    request: TransactionRequest,
) -> TransactionResponse:
    """Create a simulated checkout transaction atomically."""

    transaction_timestamp = datetime.now(timezone.utc)

    conn = get_connection()

    try:
        # Create the parent user record if it does not already exist.
        # The frontend generates simulated user IDs, so the API owns
        # creation of these synthetic parent records.
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (
                    user_id,
                    user_segment,
                    preferred_payment_method
                )
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING
                """,
                (
                    request.user_id,
                    "regular",
                    request.selected_payment_method,
                ),
            )

            # The current frontend represents a demo merchant without
            # a separate merchant-category field. Use the existing
            # synthetic electronics merchant category for this demo.
            cursor.execute(
                """
                INSERT INTO merchants (
                    merchant_id,
                    merchant_category,
                    merchant_risk_score
                )
                VALUES (%s, %s, %s)
                ON CONFLICT (merchant_id) DO NOTHING
                """,
                (
                    request.merchant_id,
                    "electronics",
                    0.20,
                ),
            )

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
            commit=False,
        )

        create_transaction_context(
            conn,
            transaction_id=request.transaction_id,
            device_type=request.context.device_type,
            network_quality=request.context.network_quality,
            retry_count=request.context.retry_count,
            transaction_velocity=request.context.transaction_velocity,
            user_method_success_rate=(
                request.context.user_method_success_rate
            ),
            merchant_method_success_rate=(
                request.context.merchant_method_success_rate
            ),
            commit=False,
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
        experiment_variant=experiment_variant,
    )