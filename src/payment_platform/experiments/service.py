from __future__ import annotations

from datetime import datetime, timezone

import psycopg

from payment_platform.db.repositories.experiment_assignments import (
    create_experiment_assignment,
)
from payment_platform.experiments.assignment import assign_variant


def assign_and_persist_variant(
    conn: psycopg.Connection,
    *,
    experiment_id: str,
    subject_id: str,
    user_id: str | None = None,
    session_id: str | None = None,
    treatment_percentage: int = 50,
    commit: bool = True,
) -> str:
    """Assign a subject to an experiment variant and persist the assignment."""

    variant = assign_variant(
        experiment_id=experiment_id,
        subject_id=subject_id,
        treatment_percentage=treatment_percentage,
    )

    create_experiment_assignment(
        conn,
        experiment_id=experiment_id,
        user_id=user_id,
        session_id=session_id,
        variant=variant,
        assigned_at=datetime.now(timezone.utc),
        commit=commit,
    )

    return variant