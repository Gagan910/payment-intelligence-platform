from __future__ import annotations

import pytest

from payment_platform.experiments.assignment import (
    CONTROL_VARIANT,
    TREATMENT_VARIANT,
    assign_variant,
)


def test_assignment_is_deterministic() -> None:
    first = assign_variant(
        experiment_id="payment_routing_v1",
        subject_id="user_001",
    )
    second = assign_variant(
        experiment_id="payment_routing_v1",
        subject_id="user_001",
    )

    assert first == second


def test_assignment_returns_valid_variant() -> None:
    variant = assign_variant(
        experiment_id="payment_routing_v1",
        subject_id="user_001",
    )

    assert variant in {
        CONTROL_VARIANT,
        TREATMENT_VARIANT,
    }


def test_zero_treatment_percentage_assigns_control() -> None:
    variant = assign_variant(
        experiment_id="payment_routing_v1",
        subject_id="user_001",
        treatment_percentage=0,
    )

    assert variant == CONTROL_VARIANT


def test_hundred_percent_treatment_assigns_treatment() -> None:
    variant = assign_variant(
        experiment_id="payment_routing_v1",
        subject_id="user_001",
        treatment_percentage=100,
    )

    assert variant == TREATMENT_VARIANT


@pytest.mark.parametrize(
    "experiment_id,subject_id",
    [
        ("", "user_001"),
        ("payment_routing_v1", ""),
    ],
)
def test_empty_identifiers_are_rejected(
    experiment_id: str,
    subject_id: str,
) -> None:
    with pytest.raises(ValueError):
        assign_variant(
            experiment_id=experiment_id,
            subject_id=subject_id,
        )


@pytest.mark.parametrize("percentage", [-1, 101])
def test_invalid_treatment_percentage_is_rejected(
    percentage: int,
) -> None:
    with pytest.raises(ValueError):
        assign_variant(
            experiment_id="payment_routing_v1",
            subject_id="user_001",
            treatment_percentage=percentage,
        )