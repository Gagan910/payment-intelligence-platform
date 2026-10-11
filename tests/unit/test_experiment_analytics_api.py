import pytest

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from payment_platform.api.app import app


client = TestClient(app)


def test_experiment_summary_returns_metrics():
    summary = {
        "control": {
            "transaction_count": 10,
            "attempted_transactions": 8,
            "successful_transactions": 6,
            "unresolved_transactions": 2,
        },
        "treatment": {
            "transaction_count": 12,
            "attempted_transactions": 10,
            "successful_transactions": 9,
            "unresolved_transactions": 2,
        },
    }

    mock_conn = MagicMock()

    with (
        patch(
            "payment_platform.api.routes.experiment.get_connection",
            return_value=mock_conn,
        ),
        patch(
            "payment_platform.api.routes.experiment.get_experiment_summary",
            return_value=summary,
        ) as mock_summary,
    ):
        response = client.get(
            "/experiments/payment_routing_v1/summary"
        )

    assert response.status_code == 200
    data = response.json()

    assert data["experiment_id"] == "payment_routing_v1"
    assert data["metric"] == "attempted_transaction_success_rate"

    assert data["control"]["success_rate"] == 0.75
    assert data["treatment"]["success_rate"] == 0.9

    assert data["absolute_lift_percentage_points"] == pytest.approx(15.0)
    assert data["relative_lift_percent"] == pytest.approx(20.0)

    mock_summary.assert_called_once_with(
        mock_conn,
        experiment_id="payment_routing_v1",
    )
    mock_conn.close.assert_called_once()


def test_experiment_summary_returns_null_rates_without_attempts():
    summary = {
        "control": {
            "transaction_count": 2,
            "attempted_transactions": 0,
            "successful_transactions": 0,
            "unresolved_transactions": 2,
        },
        "treatment": {
            "transaction_count": 3,
            "attempted_transactions": 0,
            "successful_transactions": 0,
            "unresolved_transactions": 3,
        },
    }

    with (
        patch(
            "payment_platform.api.routes.experiment.get_connection",
            return_value=MagicMock(),
        ),
        patch(
            "payment_platform.api.routes.experiment.get_experiment_summary",
            return_value=summary,
        ),
    ):
        response = client.get(
            "/experiments/payment_routing_v1/summary"
        )

    assert response.status_code == 200
    data = response.json()

    assert data["control"]["success_rate"] is None
    assert data["treatment"]["success_rate"] is None
    assert data["absolute_lift_percentage_points"] is None
    assert data["relative_lift_percent"] is None


def test_experiment_summary_rejects_unknown_experiment():
    response = client.get(
        "/experiments/unknown_experiment/summary"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found."


def test_experiment_summary_closes_connection_on_repository_error():
    mock_conn = MagicMock()

    with (
        patch(
            "payment_platform.api.routes.experiment.get_connection",
            return_value=mock_conn,
        ),
        patch(
            "payment_platform.api.routes.experiment.get_experiment_summary",
            side_effect=RuntimeError("Simulated database failure"),
        ),
    ):
        test_client = TestClient(app, raise_server_exceptions=False)

        response = test_client.get(
            "/experiments/payment_routing_v1/summary"
        )

    assert response.status_code == 500
    mock_conn.close.assert_called_once()
