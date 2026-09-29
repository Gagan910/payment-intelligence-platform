from __future__ import annotations

from payment_platform.db.connection import get_connection


RUN_ID = "ab_20260929_165701_51d73117"


def main() -> None:
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    r.original_method,
                    r.recommended_method,
                    r.recommendation_score,
                    r.reason,
                    COUNT(*) AS recommendation_count
                FROM recommendations r
                JOIN transactions t
                    ON t.transaction_id = r.transaction_id
                WHERE LEFT(t.transaction_id, %s) = %s
                GROUP BY
                    r.original_method,
                    r.recommended_method,
                    r.recommendation_score,
                    r.reason
                ORDER BY
                    recommendation_count DESC,
                    r.recommendation_score DESC
                """,
                (len(RUN_ID), RUN_ID),
            )

            print("Recommendation distribution:")

            for row in cursor.fetchall():
                print(row)

    finally:
        conn.close()


if __name__ == "__main__":
    main()