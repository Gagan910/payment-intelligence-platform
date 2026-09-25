from fastapi.testclient import TestClient

from payment_platform.api.app import app


client = TestClient(app)


VALID_REQUEST = {
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


def test_recommendation_api_returns_success_response():
    response = client.post("/recommend", json=VALID_REQUEST)

    assert response.status_code == 200

    data = response.json()

    assert "recommendation_action" in data
    assert "current_method" in data
    assert "recommended_method" in data
    assert "current_failure_probability" in data
    assert "recommended_failure_probability" in data
    assert "expected_improvement" in data
    assert "reason" in data
    assert "model_version" in data


def test_recommendation_api_preserves_current_method():
    response = client.post("/recommend", json=VALID_REQUEST)

    assert response.status_code == 200

    data = response.json()

    assert data["current_method"] == "debit_card"


def test_recommendation_api_probability_values_are_valid():
    response = client.post("/recommend", json=VALID_REQUEST)

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["current_failure_probability"] <= 1

    if data["recommended_failure_probability"] is not None:
        assert 0 <= data["recommended_failure_probability"] <= 1

    assert data["expected_improvement"] >= 0


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