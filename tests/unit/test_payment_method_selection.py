from payment_platform.recommendation.engine import RecommendationResult
from payment_platform.recommendation.selection import select_payment_method


def test_accepting_alternative_selects_recommended_method():
    recommendation = RecommendationResult(
        recommendation_action="RECOMMEND_ALTERNATIVE",
        current_method="debit_card",
        recommended_method="upi",
        current_failure_probability=0.30,
        recommended_failure_probability=0.20,
        expected_improvement=0.10,
        reason="Alternative has lower predicted failure probability.",
    )

    selected_method = select_payment_method(
        recommendation,
        accepted=True,
    )

    assert selected_method == "upi"


def test_rejecting_alternative_keeps_current_method():
    recommendation = RecommendationResult(
        recommendation_action="RECOMMEND_ALTERNATIVE",
        current_method="debit_card",
        recommended_method="upi",
        current_failure_probability=0.30,
        recommended_failure_probability=0.20,
        expected_improvement=0.10,
        reason="Alternative has lower predicted failure probability.",
    )

    selected_method = select_payment_method(
        recommendation,
        accepted=False,
    )

    assert selected_method == "debit_card"


def test_accepting_keep_current_keeps_current_method():
    recommendation = RecommendationResult(
        recommendation_action="KEEP_CURRENT",
        current_method="debit_card",
        recommended_method=None,
        current_failure_probability=0.30,
        recommended_failure_probability=0.27,
        expected_improvement=0.03,
        reason="No alternative provides the required improvement.",
    )

    selected_method = select_payment_method(
        recommendation,
        accepted=True,
    )

    assert selected_method == "debit_card"


def test_accepting_no_recommendation_keeps_current_method():
    recommendation = RecommendationResult(
        recommendation_action="NO_RECOMMENDATION",
        current_method="debit_card",
        recommended_method=None,
        current_failure_probability=0.30,
        recommended_failure_probability=None,
        expected_improvement=0.0,
        reason="No alternative payment methods are available.",
    )

    selected_method = select_payment_method(
        recommendation,
        accepted=True,
    )

    assert selected_method == "debit_card"