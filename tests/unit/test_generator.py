from __future__ import annotations

import pandas as pd

from payment_platform.data.generator import (
    calculate_failure_probability,
    generate_transactions,
    simulate_payment_outcomes,
)


def test_generate_transactions_has_expected_shape_and_columns():
    transactions = generate_transactions(
        n_transactions=100,
        seed=42,
    )

    expected_columns = {
        "transaction_id",
        "timestamp",
        "user_id",
        "merchant_id",
        "amount",
        "merchant_category",
        "payment_method",
        "user_segment",
        "device_type",
        "network_quality",
        "hour_of_day",
        "day_of_week",
        "retry_count",
        "transaction_velocity",
        "user_method_success_rate",
        "merchant_method_success_rate",
    }

    assert transactions.shape == (100, 16)
    assert set(transactions.columns) == expected_columns


def test_generate_transactions_is_reproducible():
    first = generate_transactions(
        n_transactions=100,
        seed=42,
    )

    second = generate_transactions(
        n_transactions=100,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)


def test_generate_transactions_contains_all_payment_methods():
    transactions = generate_transactions(
        n_transactions=10_000,
        seed=42,
    )

    payment_methods = set(
        transactions["payment_method"].unique()
    )

    assert payment_methods == {
        "upi",
        "credit_card",
        "debit_card",
        "net_banking",
        "wallet",
    }


def test_failure_probability_is_between_zero_and_one():
    transactions = generate_transactions(
        n_transactions=1_000,
        seed=42,
    )

    probabilities = calculate_failure_probability(
        transactions
    )

    assert (probabilities >= 0.001).all()
    assert (probabilities <= 0.999).all()


def test_payment_outcomes_are_binary():
    transactions = generate_transactions(
        n_transactions=1_000,
        seed=42,
    )

    result = simulate_payment_outcomes(
        transactions,
        seed=42,
    )

    assert set(result["payment_success"].unique()) <= {
        0,
        1,
    }


def test_payment_outcomes_are_reproducible():
    transactions = generate_transactions(
        n_transactions=1_000,
        seed=42,
    )

    first = simulate_payment_outcomes(
        transactions,
        seed=42,
    )

    second = simulate_payment_outcomes(
        transactions,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)


def test_payment_method_changes_failure_probability():
    base = {
        "amount": 500.0,
        "merchant_category": "food",
        "device_type": "mobile",
        "network_quality": "good",
        "retry_count": 0,
        "transaction_velocity": 2,
        "user_method_success_rate": 0.90,
        "merchant_method_success_rate": 0.90,
        "hour_of_day": 14,
    }

    transactions = pd.DataFrame(
        [
            {
                **base,
                "payment_method": "upi",
            },
            {
                **base,
                "payment_method": "credit_card",
            },
            {
                **base,
                "payment_method": "debit_card",
            },
            {
                **base,
                "payment_method": "net_banking",
            },
            {
                **base,
                "payment_method": "wallet",
            },
        ]
    )

    probabilities = calculate_failure_probability(
        transactions
    )

    assert probabilities.nunique() == 5


def test_poor_network_has_higher_failure_probability_than_good_network():
    base = {
        "amount": 500.0,
        "merchant_category": "electronics",
        "device_type": "desktop",
        "payment_method": "upi",
        "retry_count": 0,
        "transaction_velocity": 2,
        "user_method_success_rate": 0.90,
        "merchant_method_success_rate": 0.90,
        "hour_of_day": 14,
    }

    transactions = pd.DataFrame(
        [
            {
                **base,
                "network_quality": "poor",
            },
            {
                **base,
                "network_quality": "good",
            },
        ]
    )

    probabilities = calculate_failure_probability(
        transactions
    )

    assert probabilities.iloc[0] > probabilities.iloc[1]


def test_failure_probability_is_deterministic():
    transactions = generate_transactions(
        n_transactions=100,
        seed=42,
    )

    first = calculate_failure_probability(
        transactions
    )

    second = calculate_failure_probability(
        transactions
    )

    pd.testing.assert_series_equal(
        first,
        second,
    )