from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import psycopg

from payment_platform.db.repositories.experiment_events import (
    create_experiment_event,
)


def record_experiment_event(
    conn: psycopg.Connection,
    *,
    experiment_id: str,
    transaction_id: str | None,
    event_type: str,
    metadata: dict[str, Any] | None = None,
    commit: bool = True,
) -> int:
    """Record an event associated with an experiment."""

    if not experiment_id:
        raise ValueError("experiment_id must not be empty")

    if not event_type:
        raise ValueError("event_type must not be empty")

    return create_experiment_event(
        conn,
        experiment_id=experiment_id,
        transaction_id=transaction_id,
        event_type=event_type,
        timestamp=datetime.now(timezone.utc),
        metadata=metadata,
        commit=commit,
    )