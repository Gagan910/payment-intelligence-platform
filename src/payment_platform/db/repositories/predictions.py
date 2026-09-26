from __future__ import annotations

from datetime import datetime

import psycopg


def create_prediction(
    conn: psycopg.Connection,
    *,
    transaction_id: str,
    model_version: str,
    failure_probability: float,
    success_probability: float,
    prediction_timestamp: datetime,
    latency_ms: float | None,
) -> int:
    """Insert a model prediction and return its database ID."""

    query = """
        INSERT INTO predictions (
            transaction_id,
            model_version,
            failure_probability,
            success_probability,
            prediction_timestamp,
            latency_ms
        )
        VALUES (
            %s, %s, %s, %s, %s, %s
        )
        RETURNING prediction_id
    """

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                transaction_id,
                model_version,
                failure_probability,
                success_probability,
                prediction_timestamp,
                latency_ms,
            ),
        )
        prediction_id = cursor.fetchone()[0]

    conn.commit()

    return int(prediction_id)