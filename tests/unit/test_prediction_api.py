from fastapi.testclient import TestClient

from payment_platform.api.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction_endpoint_accepts_valid_request():
    payload = {
        "amount": 500.0,
        "merchant_category": "electronics",
        "payment_method": "upi",
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

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert 0 <= body["failure_probability"] <= 1
    assert 0 <= body["success_probability"] <= 1
    assert body["model_version"] == "placeholder"


def test_prediction_endpoint_rejects_invalid_amount():
    payload = {
        "amount": -100,
        "merchant_category": "electronics",
        "payment_method": "upi",
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

    response = client.post("/predict", json=payload)

    assert response.status_code == 422