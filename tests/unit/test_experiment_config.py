from payment_platform.experiments.config import (
    CONTROL_VARIANT,
    EXPERIMENT_ID,
    TREATMENT_PERCENTAGE,
    TREATMENT_VARIANT,
)


def test_experiment_configuration() -> None:
    assert EXPERIMENT_ID == "payment_routing_v1"
    assert CONTROL_VARIANT == "control"
    assert TREATMENT_VARIANT == "treatment"
    assert TREATMENT_PERCENTAGE == 50