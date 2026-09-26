from datetime import datetime, timezone

from fastapi.testclient import TestClient

from payment_platform.api.app import app
from payment_platform.db.connection import get_connection


client = TestClient(app)


TRANSACTION_ID = "txn_payment_attempt_api_test"
USER_ID = "user_payment_attempt_api_test"
MERCHANT_ID = "merchant_payment_attempt_api_test"


VALID_REQUEST = {
    "transaction_id": TRANSACTION_ID,
    "payment_method": "debit_card",
    "attempt_number": 1,
}


def create_test_transaction() -> None:
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

            cursor.execute(
                """
                INSERT INTO transactions (
                    transaction_id,
                    user_id,
                    merchant_id,
                    amount,
                    currency,
                    timestamp,
                    selected_payment_method,
                    experiment_variant,
                    status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    TRANSACTION_ID,
                    USER_ID,
                    MERCHANT_ID,
                    500,
                    "INR",
                    datetime.now(timezone.utc),
                    "debit_card",
                    None,
                    "initiated",
                ),
            )

            cursor.execute(
                """
                INSERT INTO transaction_context (
                    transaction_id,
                    device_type,
                    network_quality,
                    retry_count,
                    transaction_velocity,
                    user_method_success_rate,
                    merchant_method_success_rate
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    TRANSACTION_ID,
                    "mobile",
                    "good",
                    0,
                    2,
                    0.90,
                    0.92,
                ),
            )

        conn.commit()

    finally:
        conn.close()


def cleanup_test_data() -> None:
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
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


def test_payment_attempt_api_returns_success_response():
    cleanup_test_data()
    create_test_transaction()

    try:
        response = client.post(
            "/payment-attempts",
            json=VALID_REQUEST,
        )

        assert response.status_code == 200

        data = response.json()

        assert data["transaction_id"] == TRANSACTION_ID
        assert data["payment_method"] == "debit_card"
        assert data["attempt_number"] == 1
        assert data["outcome"] in {"success", "failure"}
        assert data["attempt_id"]

    finally:
        cleanup_test_data()


def test_payment_attempt_api_persists_attempt():
    cleanup_test_data()
    create_test_transaction()

    try:
        response = client.post(
            "/payment-attempts",
            json=VALID_REQUEST,
        )

        assert response.status_code == 200

        conn = get_connection()

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        transaction_id,
                        payment_method,
                        attempt_number,
                        outcome
                    FROM payment_attempts
                    WHERE transaction_id = %s
                    """,
                    (TRANSACTION_ID,),
                )

                row = cursor.fetchone()

            assert row is not None
            assert row[0] == TRANSACTION_ID
            assert row[1] == "debit_card"
            assert row[2] == 1
            assert row[3] in {"success", "failure"}

        finally:
            conn.close()

    finally:
        cleanup_test_data()


def test_payment_attempt_api_rejects_invalid_attempt_number():
    invalid_request = {
        **VALID_REQUEST,
        "attempt_number": 0,
    }

    response = client.post(
        "/payment-attempts",
        json=invalid_request,
    )

    assert response.status_code == 422


def test_payment_attempt_api_rejects_missing_transaction():
    missing_transaction_request = {
        **VALID_REQUEST,
        "transaction_id": "txn_does_not_exist",
    }

    response = client.post(
        "/payment-attempts",
        json=missing_transaction_request,
    )

    assert response.status_code == 404