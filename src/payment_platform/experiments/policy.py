from __future__ import annotations

from payment_platform.experiments.config import (
    CONTROL_VARIANT,
    TREATMENT_VARIANT,
)


def should_use_smart_routing(experiment_variant: str) -> bool:
    """Return whether smart routing is enabled for an experiment variant."""

    if experiment_variant == CONTROL_VARIANT:
        return False

    if experiment_variant == TREATMENT_VARIANT:
        return True

    raise ValueError(
        f"Unsupported experiment variant: {experiment_variant!r}"
    )