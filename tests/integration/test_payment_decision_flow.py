from fastapi.testclient import TestClient

from payment_platform.api.app import app
from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.transaction_context import (
    create_transaction_context,
)


client = TestClient(app)


TRANSACTION_ID = "txn_payment_decision_flow_test"
USER_ID = "user_payment_decision_flow_test"
MERCHANT_ID = "merchant_payment_decision_flow_test"


TRANSACTION_REQUEST = {
    "transaction_id": TRANSACTION_ID,
    "user_id": USER_ID,
    "merchant_id": MERCHANT_ID,
    "amount": 500,
    "currency": "INR",
    "selected_payment_method": "debit_card",
    "experiment_variant": "control",
    "status": "initiated",
}


PREDICTION_REQUEST = {
    "transaction_id": TRANSACTION_ID,
    "amount": 500,
    "merchant_category": "electronics",
    "payment_method": "debit_card",
    "user_segment": "regular",
    "device_type": "mobile",
    "network_quality": "good",
    "hour_of_day": 14,
    "day_of_week": 2,
    "retry_count": 0,
    "transaction_velocity": 2,
    "user_method_success_rate": 0.90,
    "merchant_method_success_rate": 0.92,
}


RECOMMENDATION_REQUEST = {
    **PREDICTION_REQUEST,
}


def cleanup_test_data() -> None:
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM recommendations
                WHERE transaction_id = %s
                """,
                (TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM predictions
                WHERE transaction_id = %s
                """,
                (TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM payment_attempts
                WHERE transaction_id = %s
                """,
                (TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM transaction_context
                WHERE transaction_id = %s
                """,
                (TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM transactions
                WHERE transaction_id = %s
                """,
                (TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM users
                WHERE user_id = %s
                """,
                (USER_ID,),
            )

            cursor.execute(
                """
                DELETE FROM merchants
                WHERE merchant_id = %s
                """,
                (MERCHANT_ID,),
            )

        conn.commit()
    finally:
        conn.close()


def create_parent_records() -> None:
    conn = get_connection()

    try:
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
                (
                    USER_ID,
                    "regular",
                    "debit_card",
                ),
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
                (
                    MERCHANT_ID,
                    "electronics",
                    0.20,
                ),
            )

        conn.commit()
    finally:
        conn.close()


def test_transaction_prediction_recommendation_flow():
    cleanup_test_data()
    create_parent_records()

    try:
        transaction_response = client.post(
            "/transactions",
            json=TRANSACTION_REQUEST,
        )

        assert transaction_response.status_code == 200

        transaction_data = transaction_response.json()

        assert transaction_data["transaction_id"] == TRANSACTION_ID
        assert transaction_data["status"] == "initiated"

        conn = get_connection()

        try:
            create_transaction_context(
                conn,
                transaction_id=TRANSACTION_ID,
                device_type="mobile",
                network_quality="poor",
                retry_count=0,
                transaction_velocity=2,
                user_method_success_rate=0.90,
                merchant_method_success_rate=0.92,
            )
        finally:
            conn.close()

        prediction_request = {
            **PREDICTION_REQUEST,
            "network_quality": "poor",
            "hour_of_day": 0,
        }

        recommendation_request = {
            **RECOMMENDATION_REQUEST,
            "network_quality": "poor",
            "hour_of_day": 0,
        }

        prediction_response = client.post(
            "/predict",
            json=prediction_request,
        )

        assert prediction_response.status_code == 200

        prediction_data = prediction_response.json()

        assert prediction_data["transaction_id"] == TRANSACTION_ID
        assert 0 <= prediction_data["failure_probability"] <= 1
        assert 0 <= prediction_data["success_probability"] <= 1

        recommendation_response = client.post(
            "/recommend",
            json=recommendation_request,
        )

        assert recommendation_response.status_code == 200

        recommendation_data = recommendation_response.json()

        assert recommendation_data["transaction_id"] == TRANSACTION_ID
        assert recommendation_data["current_method"] == "debit_card"
        assert (
            recommendation_data["recommendation_action"]
            == "RECOMMEND_ALTERNATIVE"
        )
        assert recommendation_data["recommended_method"] == "upi"
        assert 0 <= recommendation_data["current_failure_probability"] <= 1

        if recommendation_data["recommended_failure_probability"] is not None:
            assert (
                0
                <= recommendation_data["recommended_failure_probability"]
                <= 1
            )

        assert recommendation_data["expected_improvement"] >= 0

        recommendation_id = recommendation_data["recommendation_id"]

        decision_response = client.post(
            "/recommend/decision",
            json={
                "recommendation_id": recommendation_id,
                "accepted": True,
            },
        )

        assert decision_response.status_code == 200

        decision_data = decision_response.json()

        assert decision_data["recommendation_id"] == recommendation_id
        assert decision_data["accepted"] is True
        assert decision_data["selected_payment_method"] == "upi"

        selected_payment_method = decision_data["selected_payment_method"]

        payment_attempt_response = client.post(
            "/payment-attempts",
            json={
                "transaction_id": TRANSACTION_ID,
                "payment_method": selected_payment_method,
                "attempt_number": 1,
            },
        )

        assert payment_attempt_response.status_code == 200

        payment_attempt_data = payment_attempt_response.json()

        assert payment_attempt_data["transaction_id"] == TRANSACTION_ID
        assert (
            payment_attempt_data["payment_method"]
            == selected_payment_method
        )
        assert payment_attempt_data["attempt_number"] == 1
        assert payment_attempt_data["outcome"] in {"success", "failure"}

        conn = get_connection()

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM transactions
                    WHERE transaction_id = %s
                    """,
                    (TRANSACTION_ID,),
                )
                transaction_count = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM predictions
                    WHERE transaction_id = %s
                    """,
                    (TRANSACTION_ID,),
                )
                prediction_count = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM recommendations
                    WHERE transaction_id = %s
                    """,
                    (TRANSACTION_ID,),
                )
                recommendation_count = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM payment_attempts
                    WHERE transaction_id = %s
                    """,
                    (TRANSACTION_ID,),
                )
                payment_attempt_count = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT payment_method
                    FROM payment_attempts
                    WHERE transaction_id = %s
                    """,
                    (TRANSACTION_ID,),
                )
                persisted_payment_method = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT status
                    FROM transactions
                    WHERE transaction_id = %s
                    """,
                    (TRANSACTION_ID,),
                )
                transaction_status = cursor.fetchone()[0]

            assert transaction_count == 1
            assert prediction_count == 1
            assert recommendation_count == 1
            assert payment_attempt_count == 1
            assert persisted_payment_method == selected_payment_method
            assert transaction_status == payment_attempt_data["outcome"]

        finally:
            conn.close()

    finally:
        cleanup_test_data()