from __future__ import annotations

from payment_platform.recommendation.engine import RecommendationResult


def select_payment_method(
    recommendation: RecommendationResult,
    *,
    accepted: bool,
) -> str:
    """
    Select the payment method that should be attempted.

    If the recommendation is accepted and an alternative was recommended,
    use the recommended method. Otherwise, keep the current method.
    """

    if (
        accepted
        and recommendation.recommendation_action == "RECOMMEND_ALTERNATIVE"
        and recommendation.recommended_method is not None
    ):
        return recommendation.recommended_method

    return recommendation.current_method