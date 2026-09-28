from __future__ import annotations

from datetime import datetime

import psycopg


def create_experiment_assignment(
    conn: psycopg.Connection,
    *,
    experiment_id: str,
    user_id: str | None,
    session_id: str | None,
    variant: str,
    assigned_at: datetime,
    commit: bool = True,
) -> int:
    """Insert an experiment assignment and return its database ID."""

    query = """
        INSERT INTO experiment_assignments (
            experiment_id,
            user_id,
            session_id,
            variant,
            assigned_at
        )
        VALUES (
            %s, %s, %s, %s, %s
        )
        RETURNING assignment_id
    """

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                experiment_id,
                user_id,
                session_id,
                variant,
                assigned_at,
            ),
        )
        assignment_id = cursor.fetchone()[0]

    if commit:
        conn.commit()

    return int(assignment_id)