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
        VALUES (%s, %s, %s, %s, %s, %s, %s)
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


def get_recommendation(
    conn: psycopg.Connection,
    *,
    recommendation_id: int,
) -> dict[str, object] | None:
    query = """
        SELECT
            recommendation_id,
            transaction_id,
            original_method,
            recommended_method,
            recommendation_score,
            reason,
            accepted,
            timestamp
        FROM recommendations
        WHERE recommendation_id = %s
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (recommendation_id,))
        row = cursor.fetchone()

    if row is None:
        return None

    return {
        "recommendation_id": row[0],
        "transaction_id": row[1],
        "original_method": row[2],
        "recommended_method": row[3],
        "recommendation_score": row[4],
        "reason": row[5],
        "accepted": row[6],
        "timestamp": row[7],
    }


def update_recommendation_decision(
    conn: psycopg.Connection,
    *,
    recommendation_id: int,
    accepted: bool,
) -> None:
    query = """
        UPDATE recommendations
        SET accepted = %s
        WHERE recommendation_id = %s
    """

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                accepted,
                recommendation_id,
            ),
        )

        if cursor.rowcount == 0:
            conn.rollback()
            raise ValueError(
                f"Recommendation '{recommendation_id}' does not exist."
            )

    conn.commit()