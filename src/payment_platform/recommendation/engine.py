from dataclasses import dataclass
from typing import Mapping


DEFAULT_MIN_IMPROVEMENT = 0.05


@dataclass(frozen=True)
class RecommendationResult:
    """Result produced by the smart routing decision engine."""

    recommendation_action: str
    current_method: str
    recommended_method: str | None
    current_failure_probability: float
    recommended_failure_probability: float | None
    expected_improvement: float
    reason: str


def generate_recommendation(
    current_method: str,
    method_failure_probabilities: Mapping[str, float],
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
) -> RecommendationResult:
    """
    Generate a payment-method recommendation from predicted failure probabilities.

    The recommendation is based on the reduction in predicted failure probability
    relative to the currently selected payment method.

    Parameters
    ----------
    current_method:
        Payment method currently selected by the user.

    method_failure_probabilities:
        Mapping of payment method -> predicted failure probability.

    min_improvement:
        Minimum reduction in predicted failure probability required before
        recommending an alternative method.

    Returns
    -------
    RecommendationResult
        Structured routing recommendation.
    """
    if not method_failure_probabilities:
        raise ValueError("method_failure_probabilities cannot be empty.")

    if current_method not in method_failure_probabilities:
        raise ValueError(
            f"Current method '{current_method}' is missing from "
            "method_failure_probabilities."
        )

    if not 0 <= min_improvement <= 1:
        raise ValueError("min_improvement must be between 0 and 1.")

    for method, probability in method_failure_probabilities.items():
        if not 0 <= probability <= 1:
            raise ValueError(
                f"Failure probability for '{method}' must be between 0 and 1."
            )

    current_failure_probability = method_failure_probabilities[current_method]

    alternative_methods = {
        method: probability
        for method, probability in method_failure_probabilities.items()
        if method != current_method
    }

    if not alternative_methods:
        return RecommendationResult(
            recommendation_action="NO_RECOMMENDATION",
            current_method=current_method,
            recommended_method=None,
            current_failure_probability=current_failure_probability,
            recommended_failure_probability=None,
            expected_improvement=0.0,
            reason="No alternative payment methods are available.",
        )

    recommended_method = min(
        alternative_methods,
        key=alternative_methods.get,
    )

    recommended_failure_probability = alternative_methods[recommended_method]

    expected_improvement = (
        current_failure_probability - recommended_failure_probability
    )

    if expected_improvement >= min_improvement:
        return RecommendationResult(
            recommendation_action="RECOMMEND_ALTERNATIVE",
            current_method=current_method,
            recommended_method=recommended_method,
            current_failure_probability=current_failure_probability,
            recommended_failure_probability=recommended_failure_probability,
            expected_improvement=expected_improvement,
            reason=(
                f"Predicted failure probability decreases by "
                f"{expected_improvement:.2%} when switching from "
                f"{current_method} to {recommended_method}."
            ),
        )

    return RecommendationResult(
        recommendation_action="KEEP_CURRENT",
        current_method=current_method,
        recommended_method=None,
        current_failure_probability=current_failure_probability,
        recommended_failure_probability=recommended_failure_probability,
        expected_improvement=expected_improvement,
        reason=(
            f"No alternative provides the required "
            f"{min_improvement:.2%} minimum improvement."
        ),
    )