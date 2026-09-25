import pytest

from payment_platform.recommendation.engine import (
    DEFAULT_MIN_IMPROVEMENT,
    generate_recommendation,
)


def test_recommends_alternative_when_improvement_meets_threshold():
    probabilities = {
        "upi": 0.10,
        "credit_card": 0.18,
        "debit_card": 0.22,
        "net_banking": 0.30,
        "wallet": 0.20,
    }

    result = generate_recommendation(
        current_method="net_banking",
        method_failure_probabilities=probabilities,
    )

    assert result.recommendation_action == "RECOMMEND_ALTERNATIVE"
    assert result.recommended_method == "upi"
    assert result.current_failure_probability == pytest.approx(0.30)
    assert result.recommended_failure_probability == pytest.approx(0.10)
    assert result.expected_improvement == pytest.approx(0.20)


def test_keeps_current_when_improvement_is_below_threshold():
    probabilities = {
        "upi": 0.18,
        "credit_card": 0.19,
        "debit_card": 0.21,
    }

    result = generate_recommendation(
        current_method="credit_card",
        method_failure_probabilities=probabilities,
    )

    assert result.recommendation_action == "KEEP_CURRENT"
    assert result.recommended_method is None
    assert result.expected_improvement == pytest.approx(0.01)


def test_exact_threshold_triggers_recommendation():
    probabilities = {
        "upi": 0.15,
        "credit_card": 0.20,
        "debit_card": 0.25,
    }

    result = generate_recommendation(
        current_method="credit_card",
        method_failure_probabilities=probabilities,
        min_improvement=0.05,
    )

    assert result.recommendation_action == "RECOMMEND_ALTERNATIVE"
    assert result.recommended_method == "upi"
    assert result.expected_improvement == pytest.approx(0.05)


def test_does_not_recommend_worse_alternative():
    probabilities = {
        "upi": 0.25,
        "credit_card": 0.15,
        "debit_card": 0.20,
    }

    result = generate_recommendation(
        current_method="credit_card",
        method_failure_probabilities=probabilities,
    )

    assert result.recommendation_action == "KEEP_CURRENT"
    assert result.recommended_method is None
    assert result.expected_improvement == pytest.approx(-0.05)


def test_no_recommendation_when_no_alternatives_exist():
    probabilities = {
        "upi": 0.15,
    }

    result = generate_recommendation(
        current_method="upi",
        method_failure_probabilities=probabilities,
    )

    assert result.recommendation_action == "NO_RECOMMENDATION"
    assert result.recommended_method is None
    assert result.expected_improvement == pytest.approx(0.0)


def test_rejects_missing_current_method():
    probabilities = {
        "upi": 0.15,
        "credit_card": 0.20,
    }

    with pytest.raises(ValueError, match="Current method"):
        generate_recommendation(
            current_method="wallet",
            method_failure_probabilities=probabilities,
        )


def test_rejects_invalid_probability():
    probabilities = {
        "upi": 1.2,
        "credit_card": 0.20,
    }

    with pytest.raises(ValueError, match="between 0 and 1"):
        generate_recommendation(
            current_method="credit_card",
            method_failure_probabilities=probabilities,
        )


def test_rejects_invalid_threshold():
    probabilities = {
        "upi": 0.15,
        "credit_card": 0.20,
    }

    with pytest.raises(ValueError, match="min_improvement"):
        generate_recommendation(
            current_method="credit_card",
            method_failure_probabilities=probabilities,
            min_improvement=1.5,
        )


def test_default_threshold_is_five_percentage_points():
    assert DEFAULT_MIN_IMPROVEMENT == pytest.approx(0.05)