from __future__ import annotations

from datetime import datetime

import psycopg


def create_payment_attempt(
    conn: psycopg.Connection,
    *,
    transaction_id: str,
    payment_method: str,
    attempt_number: int,
    started_at: datetime,
    completed_at: datetime | None,
    outcome: str,
) -> int:
    """Insert a payment attempt and return its database ID."""

    query = """
        INSERT INTO payment_attempts (
            transaction_id,
            payment_method,
            attempt_number,
            started_at,
            completed_at,
            outcome
        )
        VALUES (
            %s, %s, %s, %s, %s, %s
        )
        RETURNING attempt_id
    """

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                transaction_id,
                payment_method,
                attempt_number,
                started_at,
                completed_at,
                outcome,
            ),
        )
        attempt_id = cursor.fetchone()[0]

    conn.commit()

    return int(attempt_id)