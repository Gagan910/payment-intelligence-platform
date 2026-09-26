from __future__ import annotations

from datetime import datetime

import psycopg


def create_recommendation(
    conn: psycopg.Connection,
    *,
    transaction_id: str,
    original_method: str,
    recommended_method: str | None,
    recommendation_score: float | None,
    reason: str,
    accepted: bool | None,
    timestamp: datetime,
) -> int:
    """Insert a payment recommendation and return its database ID."""

    query = """
        INSERT INTO recommendations (
            transaction_id,
            original_method,
            recommended_method,
            recommendation_score,
            reason,
            accepted,
            timestamp
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s
        )
        RETURNING recommendation_id
    """

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                transaction_id,
                original_method,
                recommended_method,
                recommendation_score,
                reason,
                accepted,
                timestamp,
            ),
        )
        recommendation_id = cursor.fetchone()[0]

    conn.commit()

    return int(recommendation_id)