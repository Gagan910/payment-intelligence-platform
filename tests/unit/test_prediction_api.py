from fastapi.testclient import TestClient

from payment_platform.api.app import app
from payment_platform.db.connection import get_connection


client = TestClient(app)

TEST_USER_ID = "prediction_api_test_user"
TEST_MERCHANT_ID = "prediction_api_test_merchant"
TEST_TRANSACTION_ID = "prediction_api_test_transaction"

VALID_REQUEST = {
    "transaction_id": TEST_TRANSACTION_ID,
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
                    %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s
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


def test_prediction_api_rejects_invalid_amount():
    cleanup_test_data()
    setup_test_transaction()

    try:
        invalid_request = {**VALID_REQUEST, "amount": 0}

        response = client.post("/predict", json=invalid_request)

        assert response.status_code == 422

    finally:
        cleanup_test_data()


def test_prediction_api_rejects_invalid_hour():
    cleanup_test_data()
    setup_test_transaction()

    try:
        invalid_request = {**VALID_REQUEST, "hour_of_day": 24}

        response = client.post("/predict", json=invalid_request)

        assert response.status_code == 422

    finally:
        cleanup_test_data()


def test_prediction_api_rejects_unknown_transaction():
    cleanup_test_data()

    request = {
        **VALID_REQUEST,
        "transaction_id": "nonexistent_transaction",
    }

    response = client.post("/predict", json=request)

    assert response.status_code == 404

def test_prediction_api_uses_persisted_transaction_context():
    cleanup_test_data()
    setup_test_transaction()

    try:
        request_with_conflicting_context = {
            **VALID_REQUEST,
            "network_quality": "poor",
            "hour_of_day": 0,
            "device_type": "tablet",
            "retry_count": 2,
            "transaction_velocity": 10,
            "user_method_success_rate": 0.10,
            "merchant_method_success_rate": 0.10,
        }

        persisted_request = {
            **VALID_REQUEST,
            "network_quality": "good",
            "hour_of_day": 14,
            "device_type": "mobile",
            "retry_count": 0,
            "transaction_velocity": 2,
            "user_method_success_rate": 0.90,
            "merchant_method_success_rate": 0.92,
        }

        persisted_response = client.post(
            "/predict",
            json=persisted_request,
        )

        conflicting_response = client.post(
            "/predict",
            json=request_with_conflicting_context,
        )

        assert persisted_response.status_code == 200
        assert conflicting_response.status_code == 200

        persisted_data = persisted_response.json()
        conflicting_data = conflicting_response.json()

        assert (
            conflicting_data["failure_probability"]
            == persisted_data["failure_probability"]
        )
        assert (
            conflicting_data["success_probability"]
            == persisted_data["success_probability"]
        )

    finally:
        cleanup_test_data()
