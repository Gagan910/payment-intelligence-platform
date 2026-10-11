from __future__ import annotations

import psycopg


def get_experiment_summary(
    conn: psycopg.Connection,
    *,
    experiment_id: str,
) -> dict[str, dict[str, int]]:
    """Return transaction-level outcome counts by experiment variant.

    A transaction is successful if at least one of its payment attempts
    succeeded. Transactions without attempts are counted as unresolved.
    """


    query = """
        WITH experiment_transactions AS (
            SELECT DISTINCT
                t.transaction_id,
                t.experiment_variant AS variant
            FROM transactions AS t
            INNER JOIN experiment_events AS ee
                ON ee.transaction_id = t.transaction_id
               AND ee.experiment_id = %s
               AND ee.event_type = 'checkout_started'
            WHERE t.experiment_variant IN ('control', 'treatment')
        ),
        transaction_outcomes AS (
            SELECT
                et.transaction_id,
                et.variant,
                COUNT(pa.attempt_id) AS attempt_count,
                BOOL_OR(pa.outcome = 'success') AS had_success
            FROM experiment_transactions AS et
            LEFT JOIN payment_attempts AS pa
                ON pa.transaction_id = et.transaction_id
            GROUP BY et.transaction_id, et.variant
        )
        SELECT
            variant,
            COUNT(*) AS transaction_count,
            COUNT(*) FILTER (
                WHERE attempt_count > 0
            ) AS attempted_transactions,
            COUNT(*) FILTER (
                WHERE had_success IS TRUE
            ) AS successful_transactions,
            COUNT(*) FILTER (
                WHERE attempt_count = 0
            ) AS unresolved_transactions
        FROM transaction_outcomes
        GROUP BY variant
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (experiment_id,))
        rows = cursor.fetchall()

    summary: dict[str, dict[str, int]] = {
        "control": {
            "transaction_count": 0,
            "attempted_transactions": 0,
            "successful_transactions": 0,
            "unresolved_transactions": 0,
        },
        "treatment": {
            "transaction_count": 0,
            "attempted_transactions": 0,
            "successful_transactions": 0,
            "unresolved_transactions": 0,
        },
    }

    for row in rows:
        variant = row[0]
        summary[variant] = {
            "transaction_count": int(row[1]),
            "attempted_transactions": int(row[2]),
            "successful_transactions": int(row[3]),
            "unresolved_transactions": int(row[4]),
        }

    return summary
