from __future__ import annotations

from datetime import datetime

import psycopg


def create_transaction(
    conn: psycopg.Connection,
    *,
    transaction_id: str,
    user_id: str,
    merchant_id: str,
    amount: float,
    currency: str,
    timestamp: datetime,
    selected_payment_method: str,
    experiment_variant: str | None,
    status: str,
) -> None:
    """Insert a transaction into the database."""

    query = """
        INSERT INTO transactions (
            transaction_id,
            user_id,
            merchant_id,
            amount,
            currency,
            timestamp,
            selected_payment_method,
            experiment_variant,
            status
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                transaction_id,
                user_id,
                merchant_id,
                amount,
                currency,
                timestamp,
                selected_payment_method,
                experiment_variant,
                status,
            ),
        )

    conn.commit()