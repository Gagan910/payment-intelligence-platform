from datetime import datetime, timezone

from fastapi.testclient import TestClient

from payment_platform.api.app import app
from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.transactions import create_transaction


client = TestClient(app)


VALID_REQUEST = {
    "transaction_id": "txn_recommendation_test",
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
                ON CONFLICT (user_id) DO NOTHING
                """,
                (
                    "user_recommendation_test",
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
                ON CONFLICT (merchant_id) DO NOTHING
                """,
                (
                    "merchant_recommendation_test",
                    "electronics",
                    0.20,
                ),
            )

        conn.commit()

        create_transaction(
            conn,
            transaction_id=VALID_REQUEST["transaction_id"],
            user_id="user_recommendation_test",
            merchant_id="merchant_recommendation_test",
            amount=VALID_REQUEST["amount"],
            currency="INR",
            timestamp=datetime.now(timezone.utc),
            selected_payment_method=VALID_REQUEST["payment_method"],
            experiment_variant=None,
            status="initiated",
        )
    finally:
        conn.close()


def cleanup_test_data() -> None:
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM recommendations
                WHERE transaction_id = %s
                """,
                (VALID_REQUEST["transaction_id"],),
            )

            cursor.execute(
                """
                DELETE FROM transactions
                WHERE transaction_id = %s
                """,
                (VALID_REQUEST["transaction_id"],),
            )

            cursor.execute(
                """
                DELETE FROM users
                WHERE user_id = %s
                """,
                ("user_recommendation_test",),
            )

            cursor.execute(
                """
                DELETE FROM merchants
                WHERE merchant_id = %s
                """,
                ("merchant_recommendation_test",),
            )

        conn.commit()
    finally:
        conn.close()


def test_recommendation_api_returns_success_response():
    cleanup_test_data()
    create_test_transaction()

    try:
        response = client.post("/recommend", json=VALID_REQUEST)

        assert response.status_code == 200

        data = response.json()

        assert data["transaction_id"] == VALID_REQUEST["transaction_id"]
        assert "recommendation_action" in data
        assert "current_method" in data
        assert "recommended_method" in data
        assert "current_failure_probability" in data
        assert "recommended_failure_probability" in data
        assert "expected_improvement" in data
        assert "reason" in data
        assert "model_version" in data
    finally:
        cleanup_test_data()


def test_recommendation_api_preserves_current_method():
    cleanup_test_data()
    create_test_transaction()

    try:
        response = client.post("/recommend", json=VALID_REQUEST)

        assert response.status_code == 200

        data = response.json()

        assert data["current_method"] == "debit_card"
    finally:
        cleanup_test_data()


def test_recommendation_api_probability_values_are_valid():
    cleanup_test_data()
    create_test_transaction()

    try:
        response = client.post("/recommend", json=VALID_REQUEST)

        assert response.status_code == 200

        data = response.json()

        assert 0 <= data["current_failure_probability"] <= 1

        if data["recommended_failure_probability"] is not None:
            assert 0 <= data["recommended_failure_probability"] <= 1

        assert data["expected_improvement"] >= 0
    finally:
        cleanup_test_data()


def test_recommendation_api_rejects_invalid_amount():
    invalid_request = {
        **VALID_REQUEST,
        "amount": 0,
    }

    response = client.post("/recommend", json=invalid_request)

    assert response.status_code == 422


def test_recommendation_api_rejects_invalid_hour():
    invalid_request = {
        **VALID_REQUEST,
        "hour_of_day": 24,
    }

    response = client.post("/recommend", json=invalid_request)

    assert response.status_code == 422