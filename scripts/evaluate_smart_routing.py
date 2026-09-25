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

    results = []

    for _, transaction in validation.iterrows():
        probabilities = predict_counterfactual_failure_probabilities(
            model=model,
            transaction=transaction,
            payment_methods=DEFAULT_PAYMENT_METHODS,
        )

        recommendation = generate_recommendation(
            current_method=transaction["payment_method"],
            method_failure_probabilities=probabilities,
        )

        results.append(
            {
                "transaction_id": transaction["transaction_id"],
                "current_method": recommendation.current_method,
                "recommendation_action": recommendation.recommendation_action,
                "recommended_method": recommendation.recommended_method,
                "current_failure_probability": (
                    recommendation.current_failure_probability
                ),
                "recommended_failure_probability": (
                    recommendation.recommended_failure_probability
                ),
                "expected_improvement": recommendation.expected_improvement,
            }
        )

    results_df = pd.DataFrame(results)

    total = len(results_df)

    action_counts = results_df["recommendation_action"].value_counts()

    recommendation_count = int(
        action_counts.get("RECOMMEND_ALTERNATIVE", 0)
    )
    keep_count = int(action_counts.get("KEEP_CURRENT", 0))
    no_recommendation_count = int(
        action_counts.get("NO_RECOMMENDATION", 0)
    )

    print("SMART ROUTING VALIDATION REPORT")
    print("=" * 50)
    print(f"Validation transactions:       {total:,}")
    print(
        f"Recommendations:               "
        f"{recommendation_count:,} "
        f"({recommendation_count / total:.2%})"
    )
    print(
        f"Keep current:                  "
        f"{keep_count:,} "
        f"({keep_count / total:.2%})"
    )
    print(
        f"No recommendation:             "
        f"{no_recommendation_count:,} "
        f"({no_recommendation_count / total:.2%})"
    )

    recommended = results_df[
        results_df["recommendation_action"] == "RECOMMEND_ALTERNATIVE"
    ]

    if not recommended.empty:
        print()
        print("Recommended transactions:")
        print(
            f"Mean predicted improvement:    "
            f"{recommended['expected_improvement'].mean():.4f}"
        )
        print(
            f"Median predicted improvement:  "
            f"{recommended['expected_improvement'].median():.4f}"
        )
        print(
            f"Minimum predicted improvement: "
            f"{recommended['expected_improvement'].min():.4f}"
        )
        print(
            f"Maximum predicted improvement: "
            f"{recommended['expected_improvement'].max():.4f}"
        )

    print()
    print("Recommendation rate by current method:")
    print("-" * 50)

    method_summary = (
        results_df.groupby("current_method")
        .agg(
            transactions=("transaction_id", "count"),
            recommendations=(
                "recommendation_action",
                lambda x: (x == "RECOMMEND_ALTERNATIVE").sum(),
            ),
        )
        .reset_index()
    )

    method_summary["recommendation_rate"] = (
        method_summary["recommendations"]
        / method_summary["transactions"]
    )

    print(method_summary.to_string(index=False))


if __name__ == "__main__":
    main()