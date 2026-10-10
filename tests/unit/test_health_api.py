from fastapi.testclient import TestClient

from payment_platform.api.app import app


client = TestClient(app)


def test_health_get_returns_healthy_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_health_head_returns_success_without_body() -> None:
    response = client.head("/health")

    assert response.status_code == 200
    assert response.content == b""
