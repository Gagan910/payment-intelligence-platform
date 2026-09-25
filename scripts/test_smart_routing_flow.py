from pathlib import Path

import joblib
import pandas as pd

from payment_platform.recommendation.counterfactual import (
    DEFAULT_PAYMENT_METHODS,
    predict_counterfactual_failure_probabilities,
)
from payment_platform.recommendation.engine import generate_recommendation


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_DATA = PROJECT_ROOT / "data" / "processed" / "validation.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib"


def main() -> None:
    validation = pd.read_parquet(VALIDATION_DATA)
    model = joblib.load(MODEL_PATH)

    transaction = validation.iloc[0]

    probabilities = predict_counterfactual_failure_probabilities(
        model=model,
        transaction=transaction,
        payment_methods=DEFAULT_PAYMENT_METHODS,
    )

    result = generate_recommendation(
        current_method=transaction["payment_method"],
        method_failure_probabilities=probabilities,
    )

    print("SMART ROUTING INTEGRATION TEST")
    print("=" * 40)

    print(f"Transaction ID: {transaction['transaction_id']}")
    print(f"Current method: {transaction['payment_method']}")
    print()

    print("Predicted failure probabilities:")
    for method, probability in probabilities.items():
        print(f"  {method:15s}: {probability:.4f}")

    print()

    print("Recommendation:")
    print(f"  Action:              {result.recommendation_action}")
    print(f"  Current method:      {result.current_method}")
    print(f"  Recommended method:  {result.recommended_method}")
    print(
        f"  Current failure:     "
        f"{result.current_failure_probability:.4f}"
    )
    print(
        f"  Recommended failure: "
        f"{result.recommended_failure_probability:.4f}"
        if result.recommended_failure_probability is not None
        else "  Recommended failure: None"
    )
    print(f"  Improvement:         {result.expected_improvement:.4f}")
    print(f"  Reason:              {result.reason}")


if __name__ == "__main__":
    main()