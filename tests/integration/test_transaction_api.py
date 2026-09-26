from fastapi.testclient import TestClient

from payment_platform.api.app import app
from payment_platform.db.connection import get_connection


client = TestClient(app)

TEST_USER_ID = "api_test_user"
TEST_MERCHANT_ID = "api_test_merchant"
TEST_TRANSACTION_ID = "api_test_transaction"


def setup_test_data() -> None:
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
    finally:
        conn.close()


def cleanup_test_data() -> None:
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
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


def test_create_transaction_api() -> None:
    cleanup_test_data()
    setup_test_data()

    response = client.post(
        "/transactions",
        json={
            "transaction_id": TEST_TRANSACTION_ID,
            "user_id": TEST_USER_ID,
            "merchant_id": TEST_MERCHANT_ID,
            "amount": 500.00,
            "currency": "INR",
            "selected_payment_method": "upi",
            "experiment_variant": "control",
            "status": "initiated",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["transaction_id"] == TEST_TRANSACTION_ID
    assert data["status"] == "initiated"

    conn = get_connection()

    try:
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
    finally:
        conn.close()
        cleanup_test_data()