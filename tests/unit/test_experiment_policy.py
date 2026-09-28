import pytest

from payment_platform.experiments.policy import should_use_smart_routing


def test_control_variant_disables_smart_routing() -> None:
    assert should_use_smart_routing("control") is False


def test_treatment_variant_enables_smart_routing() -> None:
    assert should_use_smart_routing("treatment") is True


def test_unknown_variant_raises_error() -> None:
    with pytest.raises(ValueError, match="Unsupported experiment variant"):
        should_use_smart_routing("unknown")