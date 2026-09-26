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


def get_transaction(
    conn: psycopg.Connection,
    *,
    transaction_id: str,
) -> dict | None:
    """Retrieve a transaction by its transaction ID."""

    query = """
        SELECT
            transaction_id,
            user_id,
            merchant_id,
            amount,
            currency,
            timestamp,
            selected_payment_method,
            experiment_variant,
            status
        FROM transactions
        WHERE transaction_id = %s
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (transaction_id,))
        row = cursor.fetchone()

    if row is None:
        return None

    return {
        "transaction_id": row[0],
        "user_id": row[1],
        "merchant_id": row[2],
        "amount": float(row[3]),
        "currency": row[4],
        "timestamp": row[5],
        "selected_payment_method": row[6],
        "experiment_variant": row[7],
        "status": row[8],
    }