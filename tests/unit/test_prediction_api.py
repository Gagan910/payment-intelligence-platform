from fastapi.testclient import TestClient

from payment_platform.api.app import app
from payment_platform.db.connection import get_connection


client = TestClient(app)

TEST_USER_ID = "prediction_api_test_user"
TEST_MERCHANT_ID = "prediction_api_test_merchant"
TEST_TRANSACTION_ID = "prediction_api_test_transaction"

VALID_REQUEST = {
    "transaction_id": TEST_TRANSACTION_ID,
}


def setup_test_transaction() -> None:
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
                VALUES (
                    %s, %s, %s, %s, %s, CURRENT_TIMESTAMP,
                    %s, %s, %s
                )
                """,
                (
                    TEST_TRANSACTION_ID,
                    TEST_USER_ID,
                    TEST_MERCHANT_ID,
                    500.00,
                    "INR",
                    "debit_card",
                    "control",
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
                    TEST_TRANSACTION_ID,
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
                DELETE FROM predictions
                WHERE transaction_id = %s
                """,
                (TEST_TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM transaction_context
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


def test_prediction_api_returns_success_response():
    cleanup_test_data()
    setup_test_transaction()

    try:
        response = client.post("/predict", json=VALID_REQUEST)

        assert response.status_code == 200

        data = response.json()

        assert data["transaction_id"] == TEST_TRANSACTION_ID
        assert "failure_probability" in data
        assert "success_probability" in data
        assert "model_version" in data

    finally:
        cleanup_test_data()


def test_prediction_api_probability_values_are_valid():
    cleanup_test_data()
    setup_test_transaction()

    try:
        response = client.post("/predict", json=VALID_REQUEST)

        assert response.status_code == 200

        data = response.json()

        assert 0 <= data["failure_probability"] <= 1
        assert 0 <= data["success_probability"] <= 1

    finally:
        cleanup_test_data()


def test_prediction_api_rejects_missing_transaction_id():
    cleanup_test_data()

    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_prediction_api_rejects_unknown_transaction():
    cleanup_test_data()

    request = {
        "transaction_id": "nonexistent_transaction",
    }

    response = client.post("/predict", json=request)

    assert response.status_code == 404


def test_prediction_api_uses_persisted_transaction_context():
    cleanup_test_data()
    setup_test_transaction()

    try:
        first_response = client.post(
            "/predict",
            json=VALID_REQUEST,
        )

        second_response = client.post(
            "/predict",
            json=VALID_REQUEST,
        )

        assert first_response.status_code == 200
        assert second_response.status_code == 200

        first_data = first_response.json()
        second_data = second_response.json()

        assert (
            second_data["failure_probability"]
            == first_data["failure_probability"]
        )
        assert (
            second_data["success_probability"]
            == first_data["success_probability"]
        )

    finally:
        cleanup_test_data()
        