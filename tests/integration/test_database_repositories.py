from __future__ import annotations

from datetime import datetime, timezone

from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.experiment_assignments import (
    create_experiment_assignment,
)
from payment_platform.db.repositories.experiment_events import (
    create_experiment_event,
)
from payment_platform.db.repositories.payment_attempts import (
    create_payment_attempt,
)
from payment_platform.db.repositories.predictions import create_prediction
from payment_platform.db.repositories.recommendations import (
    create_recommendation,
)
from payment_platform.db.repositories.transactions import (
    create_transaction,
    get_transaction,
)


TEST_USER_ID = "integration_test_user"
TEST_MERCHANT_ID = "integration_test_merchant"
TEST_TRANSACTION_ID = "integration_test_transaction"


def cleanup_test_data() -> None:
    """Remove records created by the integration test."""

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM experiment_events
                WHERE transaction_id = %s
                """,
                (TEST_TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM experiment_assignments
                WHERE user_id = %s
                """,
                (TEST_USER_ID,),
            )

            cursor.execute(
                """
                DELETE FROM payment_attempts
                WHERE transaction_id = %s
                """,
                (TEST_TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM recommendations
                WHERE transaction_id = %s
                """,
                (TEST_TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM predictions
                WHERE transaction_id = %s
                """,
                (TEST_TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM transactions
                WHERE transaction_id = %s
                """,
                (TEST_TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM merchants
                WHERE merchant_id = %s
                """,
                (TEST_MERCHANT_ID,),
            )

            cursor.execute(
                """
                DELETE FROM users
                WHERE user_id = %s
                """,
                (TEST_USER_ID,),
            )

        conn.commit()

    finally:
        conn.close()


def test_database_repositories() -> None:
    """Verify the complete repository persistence flow."""

    cleanup_test_data()

    conn = get_connection()

    try:
        now = datetime.now(timezone.utc)

        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (
                    user_id,
                    user_segment,
                    preferred_payment_method
                )
                VALUES (%s, %s, %s)
                """,
                (TEST_USER_ID, "regular", "upi"),
            )

            cursor.execute(
                """
                INSERT INTO merchants (
                    merchant_id,
                    merchant_category,
                    merchant_risk_score
                )
                VALUES (%s, %s, %s)
                """,
                (TEST_MERCHANT_ID, "electronics", 0.10),
            )

        conn.commit()

        create_transaction(
            conn,
            transaction_id=TEST_TRANSACTION_ID,
            user_id=TEST_USER_ID,
            merchant_id=TEST_MERCHANT_ID,
            amount=500.00,
            currency="INR",
            timestamp=now,
            selected_payment_method="upi",
            experiment_variant="control",
            status="initiated",
        )

        transaction = get_transaction(
            conn,
            transaction_id=TEST_TRANSACTION_ID,
        )

        assert transaction is not None
        assert transaction["transaction_id"] == TEST_TRANSACTION_ID
        assert transaction["user_id"] == TEST_USER_ID
        assert transaction["merchant_id"] == TEST_MERCHANT_ID
        assert transaction["amount"] == 500.00
        assert transaction["currency"] == "INR"
        assert transaction["selected_payment_method"] == "upi"
        assert transaction["experiment_variant"] == "control"
        assert transaction["status"] == "initiated"

        prediction_id = create_prediction(
            conn,
            transaction_id=TEST_TRANSACTION_ID,
            model_version="logistic_regression_baseline_v1",
            failure_probability=0.15,
            success_probability=0.85,
            prediction_timestamp=now,
            latency_ms=12.5,
        )

        recommendation_id = create_recommendation(
            conn,
            transaction_id=TEST_TRANSACTION_ID,
            original_method="upi",
            recommended_method="credit_card",
            recommendation_score=0.08,
            reason="Alternative payment method has lower predicted failure probability.",
            accepted=True,
            timestamp=now,
        )

        attempt_id = create_payment_attempt(
            conn,
            transaction_id=TEST_TRANSACTION_ID,
            payment_method="upi",
            attempt_number=1,
            started_at=now,
            completed_at=now,
            outcome="failure",
        )

        assignment_id = create_experiment_assignment(
            conn,
            experiment_id="smart_routing_ab_v1",
            user_id=TEST_USER_ID,
            session_id="integration_test_session",
            variant="control",
            assigned_at=now,
        )

        event_id = create_experiment_event(
            conn,
            experiment_id="smart_routing_ab_v1",
            transaction_id=TEST_TRANSACTION_ID,
            event_type="recommendation_shown",
            timestamp=now,
            metadata={
                "recommended_method": "credit_card",
                "accepted": True,
            },
        )

        assert prediction_id > 0
        assert recommendation_id > 0
        assert attempt_id > 0
        assert assignment_id > 0
        assert event_id > 0

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM transactions
                WHERE transaction_id = %s
                """,
                (TEST_TRANSACTION_ID,),
            )
            assert cursor.fetchone()[0] == 1

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM predictions
                WHERE transaction_id = %s
                """,
                (TEST_TRANSACTION_ID,),
            )
            assert cursor.fetchone()[0] == 1

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM recommendations
                WHERE transaction_id = %s
                """,
                (TEST_TRANSACTION_ID,),
            )
            assert cursor.fetchone()[0] == 1

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM payment_attempts
                WHERE transaction_id = %s
                """,
                (TEST_TRANSACTION_ID,),
            )
            assert cursor.fetchone()[0] == 1

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM experiment_assignments
                WHERE user_id = %s
                """,
                (TEST_USER_ID,),
            )
            assert cursor.fetchone()[0] == 1

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM experiment_events
                WHERE transaction_id = %s
                """,
                (TEST_TRANSACTION_ID,),
            )
            assert cursor.fetchone()[0] == 1

    finally:
        conn.close()
        cleanup_test_data()