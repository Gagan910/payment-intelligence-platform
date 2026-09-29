from datetime import datetime, timezone

from fastapi.testclient import TestClient

from payment_platform.api.app import app
from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.transaction_context import (
    create_transaction_context,
)
from payment_platform.db.repositories.transactions import create_transaction


client = TestClient(app)


VALID_REQUEST = {
    "transaction_id": "txn_recommendation_test",
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
            amount=500,
            currency="INR",
            timestamp=datetime.now(timezone.utc),
            selected_payment_method="debit_card",
            experiment_variant="treatment",
            status="initiated",
        )

        create_transaction_context(
            conn,
            transaction_id=VALID_REQUEST["transaction_id"],
            device_type="mobile",
            network_quality="good",
            retry_count=0,
            transaction_velocity=2,
            user_method_success_rate=0.90,
            merchant_method_success_rate=0.92,
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
                DELETE FROM transaction_context
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


def create_test_recommendation() -> int:
    response = client.post(
        "/recommend",
        json=VALID_REQUEST,
    )

    assert response.status_code == 200

    return response.json()["recommendation_id"]


def test_recommendation_api_returns_success_response():
    cleanup_test_data()
    create_test_transaction()

    try:
        response = client.post(
            "/recommend",
            json=VALID_REQUEST,
        )

        assert response.status_code == 200

        data = response.json()

        assert data["recommendation_id"] > 0
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
        response = client.post(
            "/recommend",
            json=VALID_REQUEST,
        )

        assert response.status_code == 200
        assert response.json()["current_method"] == "debit_card"

    finally:
        cleanup_test_data()


def test_recommendation_api_probability_values_are_valid():
    cleanup_test_data()
    create_test_transaction()

    try:
        response = client.post(
            "/recommend",
            json=VALID_REQUEST,
        )

        assert response.status_code == 200

        data = response.json()

        assert 0 <= data["current_failure_probability"] <= 1

        if data["recommended_failure_probability"] is not None:
            assert 0 <= data["recommended_failure_probability"] <= 1

        assert data["expected_improvement"] >= 0

    finally:
        cleanup_test_data()


def test_recommendation_api_rejects_missing_transaction_id():
    response = client.post(
        "/recommend",
        json={},
    )

    assert response.status_code == 422


def test_recommendation_api_rejects_unknown_transaction():
    response = client.post(
        "/recommend",
        json={
            "transaction_id": "nonexistent_transaction",
        },
    )

    assert response.status_code == 404


def test_recommendation_decision_api_accepts_recommendation():
    cleanup_test_data()
    create_test_transaction()

    try:
        recommendation_id = create_test_recommendation()

        response = client.post(
            "/recommend/decision",
            json={
                "recommendation_id": recommendation_id,
                "accepted": True,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["recommendation_id"] == recommendation_id
        assert data["accepted"] is True

        conn = get_connection()

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT accepted
                    FROM recommendations
                    WHERE recommendation_id = %s
                    """,
                    (recommendation_id,),
                )

                assert cursor.fetchone()[0] is True

        finally:
            conn.close()

    finally:
        cleanup_test_data()


def test_recommendation_decision_api_rejects_recommendation():
    cleanup_test_data()
    create_test_transaction()

    try:
        recommendation_id = create_test_recommendation()

        response = client.post(
            "/recommend/decision",
            json={
                "recommendation_id": recommendation_id,
                "accepted": False,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["recommendation_id"] == recommendation_id
        assert data["accepted"] is False

        conn = get_connection()

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT accepted
                    FROM recommendations
                    WHERE recommendation_id = %s
                    """,
                    (recommendation_id,),
                )

                assert cursor.fetchone()[0] is False

        finally:
            conn.close()

    finally:
        cleanup_test_data()


def test_recommendation_decision_api_rejects_invalid_recommendation_id():
    response = client.post(
        "/recommend/decision",
        json={
            "recommendation_id": 0,
            "accepted": True,
        },
    )

    assert response.status_code == 422


def test_recommendation_decision_returns_selected_method_when_accepted():
    cleanup_test_data()
    create_test_transaction()

    try:
        recommendation_response = client.post(
            "/recommend",
            json=VALID_REQUEST,
        )

        assert recommendation_response.status_code == 200

        recommendation_data = recommendation_response.json()

        recommendation_id = recommendation_data["recommendation_id"]

        expected_method = (
            recommendation_data["recommended_method"]
            or recommendation_data["current_method"]
        )

        response = client.post(
            "/recommend/decision",
            json={
                "recommendation_id": recommendation_id,
                "accepted": True,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["recommendation_id"] == recommendation_id
        assert data["accepted"] is True
        assert data["selected_payment_method"] == expected_method

    finally:
        cleanup_test_data()


def test_recommendation_decision_returns_current_method_when_rejected():
    cleanup_test_data()
    create_test_transaction()

    try:
        recommendation_id = create_test_recommendation()

        response = client.post(
            "/recommend/decision",
            json={
                "recommendation_id": recommendation_id,
                "accepted": False,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["recommendation_id"] == recommendation_id
        assert data["accepted"] is False
        assert data["selected_payment_method"] == "debit_card"

    finally:
        cleanup_test_data()


def test_recommendation_api_control_variant_keeps_current_method():
    cleanup_test_data()
    create_test_transaction()

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE transactions
                SET experiment_variant = %s
                WHERE transaction_id = %s
                """,
                (
                    "control",
                    VALID_REQUEST["transaction_id"],
                ),
            )

        conn.commit()

    finally:
        conn.close()

    try:
        response = client.post(
            "/recommend",
            json=VALID_REQUEST,
        )

        assert response.status_code == 200

        data = response.json()

        assert data["recommendation_action"] == "KEEP_CURRENT"
        assert data["current_method"] == "debit_card"
        assert data["recommended_method"] is None
        assert data["expected_improvement"] == 0.0
        assert (
            data["reason"]
            == "Control variant: smart routing is not enabled."
        )

    finally:
        cleanup_test_data()
        