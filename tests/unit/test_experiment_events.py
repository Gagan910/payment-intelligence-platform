from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from payment_platform.experiments.events import record_experiment_event


@patch("payment_platform.experiments.events.create_experiment_event")
def test_record_experiment_event_persists_event(
    mock_create_event: Mock,
) -> None:
    conn = Mock()

    mock_create_event.return_value = 42

    event_id = record_experiment_event(
        conn,
        experiment_id="payment_routing_v1",
        transaction_id="txn_001",
        event_type="checkout_started",
        metadata={"amount": 500},
    )

    assert event_id == 42

    mock_create_event.assert_called_once()

    call_kwargs = mock_create_event.call_args.kwargs

    assert call_kwargs["experiment_id"] == "payment_routing_v1"
    assert call_kwargs["transaction_id"] == "txn_001"
    assert call_kwargs["event_type"] == "checkout_started"
    assert call_kwargs["metadata"] == {"amount": 500}
    assert call_kwargs["timestamp"] is not None


@pytest.mark.parametrize(
    "experiment_id,event_type",
    [
        ("", "checkout_started"),
        ("payment_routing_v1", ""),
    ],
)
def test_invalid_event_identifiers_are_rejected(
    experiment_id: str,
    event_type: str,
) -> None:
    conn = Mock()

    with pytest.raises(ValueError):
        record_experiment_event(
            conn,
            experiment_id=experiment_id,
            transaction_id="txn_001",
            event_type=event_type,
        )