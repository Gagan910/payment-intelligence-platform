from __future__ import annotations

from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


def create_experiment_event(
    conn: psycopg.Connection,
    *,
    experiment_id: str,
    transaction_id: str | None,
    event_type: str,
    timestamp: datetime,
    metadata: dict[str, Any] | None,
    commit: bool = True,
) -> int:
    """Insert an experiment event and return its database ID."""

    query = """
        INSERT INTO experiment_events (
            experiment_id,
            transaction_id,
            event_type,
            timestamp,
            metadata
        )
        VALUES (
            %s, %s, %s, %s, %s
        )
        RETURNING event_id
    """

    metadata_value = Jsonb(metadata) if metadata is not None else None

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                experiment_id,
                transaction_id,
                event_type,
                timestamp,
                metadata_value,
            ),
        )
        event_id = cursor.fetchone()[0]

    if commit:
        conn.commit()

    return int(event_id)