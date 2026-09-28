from __future__ import annotations

import pandas as pd

from payment_platform.ml.predictor import PaymentPredictor
from payment_platform.recommendation.counterfactual import (
    predict_counterfactual_failure_probabilities,
)
from payment_platform.recommendation.engine import generate_recommendation


predictor = PaymentPredictor()


BASE_REQUEST = {
    "amount": 500,
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


def main() -> None:
    variations = []

    for network_quality in (
        "poor",
        "average",
        "good",
        "excellent",
    ):
        for hour_of_day in range(24):
            request = BASE_REQUEST.copy()
            request["network_quality"] = network_quality
            request["hour_of_day"] = hour_of_day
            variations.append(request)

    for request in variations:
        transaction = pd.Series(request)

        probabilities = predict_counterfactual_failure_probabilities(
            model=predictor.model,
            transaction=transaction,
        )

        recommendation = generate_recommendation(
            current_method=request["payment_method"],
            method_failure_probabilities=probabilities,
        )

        if recommendation.recommendation_action == "RECOMMEND_ALTERNATIVE":
            print("FOUND TEST CASE")
            print(request)
            print()
            print("Failure probabilities:")
            print(probabilities)
            print()
            print("Recommendation:")
            print(recommendation)
            return

    print("No RECOMMEND_ALTERNATIVE test case found.")


if __name__ == "__main__":
    main()