from __future__ import annotations

from unittest.mock import Mock, patch

from payment_platform.experiments.service import assign_and_persist_variant


@patch("payment_platform.experiments.service.create_experiment_assignment")
def test_assign_and_persist_variant_persists_assignment(
    mock_create_assignment: Mock,
) -> None:
    conn = Mock()

    variant = assign_and_persist_variant(
        conn,
        experiment_id="payment_routing_v1",
        subject_id="user_001",
        user_id="user_001",
        session_id="session_001",
        treatment_percentage=100,
    )

    assert variant == "treatment"

    mock_create_assignment.assert_called_once()

    call_kwargs = mock_create_assignment.call_args.kwargs

    assert call_kwargs["experiment_id"] == "payment_routing_v1"
    assert call_kwargs["user_id"] == "user_001"
    assert call_kwargs["session_id"] == "session_001"
    assert call_kwargs["variant"] == "treatment"
    assert call_kwargs["assigned_at"] is not None