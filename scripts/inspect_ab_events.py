from __future__ import annotations

import argparse

from payment_platform.db.connection import get_connection


EXPERIMENT_ID = "payment_routing_v1"


def load_run_summary(
    run_id: str,
) -> list[tuple]:
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    t.experiment_variant,
                    r.original_method,
                    r.recommended_method,
                    COUNT(*) AS recommendation_count
                FROM recommendations r
                JOIN transactions t
                    ON t.transaction_id = r.transaction_id
                WHERE
                    t.experiment_variant IS NOT NULL
                    AND LEFT(t.transaction_id, %s) = %s
                GROUP BY
                    t.experiment_variant,
                    r.original_method,
                    r.recommended_method
                ORDER BY
                    t.experiment_variant,
                    recommendation_count DESC
                """,
                (
                    len(run_id),
                    run_id,
                ),
            )

            return cursor.fetchall()

    finally:
        conn.close()


def load_outcome_summary(
    run_id: str,
) -> list[tuple]:
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    t.experiment_variant,
                    r.original_method,
                    r.recommended_method,
                    pa.outcome,
                    COUNT(*) AS transaction_count
                FROM recommendations r
                JOIN transactions t
                    ON t.transaction_id = r.transaction_id
                JOIN payment_attempts pa
                    ON pa.transaction_id = t.transaction_id
                WHERE
                    LEFT(t.transaction_id, %s) = %s
                GROUP BY
                    t.experiment_variant,
                    r.original_method,
                    r.recommended_method,
                    pa.outcome
                ORDER BY
                    t.experiment_variant,
                    r.original_method,
                    r.recommended_method,
                    pa.outcome
                """,
                (
                    len(run_id),
                    run_id,
                ),
            )

            return cursor.fetchall()

    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect recommendation and payment outcome "
            "distribution for an A/B experiment run."
        )
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Experiment run identifier.",
    )

    args = parser.parse_args()

    recommendation_rows = load_run_summary(
        args.run_id
    )

    outcome_rows = load_outcome_summary(
        args.run_id
    )

    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Run: {args.run_id}")
    print()

    print("Recommendation distribution:")
    print()

    if not recommendation_rows:
        print("No recommendation records found.")
    else:
        for row in recommendation_rows:
            (
                variant,
                original_method,
                recommended_method,
                count,
            ) = row

            print(
                f"  variant={variant}"
                f" | original={original_method}"
                f" | recommended={recommended_method}"
                f" | count={count}"
            )

    print()
    print("Payment outcome distribution:")
    print()

    if not outcome_rows:
        print("No payment outcome records found.")
    else:
        for row in outcome_rows:
            (
                variant,
                original_method,
                recommended_method,
                outcome,
                count,
            ) = row

            print(
                f"  variant={variant}"
                f" | original={original_method}"
                f" | recommended={recommended_method}"
                f" | outcome={outcome}"
                f" | count={count}"
            )


if __name__ == "__main__":
    main()