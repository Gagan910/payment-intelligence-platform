
from datetime import datetime, timezone
from uuid import uuid4

from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.experiment_analytics import (
    get_experiment_summary,
)


def test_experiment_summary_handles_duplicate_events_and_multiple_attempts():
    suffix = uuid4().hex[:12]
    experiment_id = f"analytics_test_{suffix}"

    user_id = f"test_user_{suffix}"
    merchant_id = f"test_merchant_{suffix}"

    control_transaction_1 = f"test_ctrl1_{suffix}"
    control_transaction_2 = f"test_ctrl2_{suffix}"
    treatment_transaction = f"test_treat_{suffix}"

    transaction_ids = [
        control_transaction_1,
        control_transaction_2,
        treatment_transaction,
    ]

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (
                    user_id, user_segment, preferred_payment_method
                )
                VALUES (%s, %s, %s)
                """,
                (user_id, "regular", "upi"),
            )

            cursor.execute(
                """
                INSERT INTO merchants (
                    merchant_id, merchant_category, merchant_risk_score
                )
                VALUES (%s, %s, %s)
                """,
                (merchant_id, "grocery", 0.1),
            )

            transactions = [
                (control_transaction_1, "control"),
                (control_transaction_2, "control"),
                (treatment_transaction, "treatment"),
            ]

            for transaction_id, variant in transactions:
                cursor.execute(
                    """
                    INSERT INTO transactions (
                        transaction_id, user_id, merchant_id, amount,
                        currency, timestamp, selected_payment_method,
                        experiment_variant, status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        transaction_id,
                        user_id,
                        merchant_id,
                        100.00,
                        "INR",
                        datetime.now(timezone.utc),
                        "upi",
                        variant,
                        "pending",
                    ),
                )

                cursor.execute(
                    """
                    INSERT INTO experiment_events (
                        experiment_id, transaction_id, event_type
                    )
                    VALUES (%s, %s, %s)
                    """,
                    (experiment_id, transaction_id, "checkout_started"),
                )

            # Duplicate event must not double-count the control transaction.
            cursor.execute(
                """
                INSERT INTO experiment_events (
                    experiment_id, transaction_id, event_type
                )
                VALUES (%s, %s, %s)
                """,
                (
                    experiment_id,
                    control_transaction_1,
                    "checkout_started",
                ),
            )

            now = datetime.now(timezone.utc)

            # Two attempts for one control transaction, one of which succeeds.
            cursor.execute(
                """
                INSERT INTO payment_attempts (
                    transaction_id, payment_method, attempt_number,
                    started_at, completed_at, outcome
                )
                VALUES
                    (%s, %s, %s, %s, %s, %s),
                    (%s, %s, %s, %s, %s, %s),
                    (%s, %s, %s, %s, %s, %s)
                """,
                (
                    control_transaction_1,
                    "upi",
                    1,
                    now,
                    now,
                    "failure",
                    control_transaction_1,
                    "credit_card",
                    2,
                    now,
                    now,
                    "success",
                    treatment_transaction,
                    "upi",
                    1,
                    now,
                    now,
                    "failure",
                ),
            )

        conn.commit()

        summary = get_experiment_summary(
            conn,
            experiment_id=experiment_id,
        )

        control = summary["control"]
        treatment = summary["treatment"]

        # Duplicate checkout events count as one transaction.
        assert control == {
            "transaction_count": 2,
            "attempted_transactions": 1,
            "successful_transactions": 1,
            "unresolved_transactions": 1,
        }

        # The control transaction succeeds if any attempt succeeds.
        assert treatment == {
            "transaction_count": 1,
            "attempted_transactions": 1,
            "successful_transactions": 0,
            "unresolved_transactions": 0,
        }

    finally:
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM experiment_events
                    WHERE transaction_id = ANY(%s)
                    """,
                    (transaction_ids,),
                )

                cursor.execute(
                    """
                    DELETE FROM payment_attempts
                    WHERE transaction_id = ANY(%s)
                    """,
                    (transaction_ids,),
                )

                cursor.execute(
                    """
                    DELETE FROM transactions
                    WHERE transaction_id = ANY(%s)
                    """,
                    (transaction_ids,),
                )

                cursor.execute(
                    "DELETE FROM users WHERE user_id = %s",
                    (user_id,),
                )

                cursor.execute(
                    "DELETE FROM merchants WHERE merchant_id = %s",
                    (merchant_id,),
                )

            conn.commit()
        finally:
            conn.close()
