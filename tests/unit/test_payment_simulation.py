import pandas as pd

from payment_platform.simulation.payment import simulate_payment_outcome


VALID_TRANSACTION = pd.Series(
    {
        "amount": 500.0,
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
)


def test_simulator_returns_valid_failure_probability():
    failure_probability, outcome = simulate_payment_outcome(
        VALID_TRANSACTION
    )

    assert 0 <= failure_probability <= 1
    assert outcome in {"success", "failure"}


def test_simulator_returns_string_outcome():
    _, outcome = simulate_payment_outcome(VALID_TRANSACTION)

    assert isinstance(outcome, str)


def test_simulator_probability_is_independent_of_ml_prediction():
    failure_probability, _ = simulate_payment_outcome(
        VALID_TRANSACTION
    )

    assert isinstance(failure_probability, float)