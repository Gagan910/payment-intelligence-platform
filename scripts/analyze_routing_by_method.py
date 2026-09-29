from pathlib import Path

import joblib
import pandas as pd

from payment_platform.data.features import get_model_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "validation.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib"

PAYMENT_METHODS = [
    "upi",
    "credit_card",
    "debit_card",
    "net_banking",
    "wallet",
]

THRESHOLDS = [
    0.02,
    0.05,
    0.08,
]


def generate_counterfactual_predictions(
    df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    candidate_frames = []

    for method in PAYMENT_METHODS:
        candidates = df.copy()
        candidates["payment_method"] = method
        candidates["candidate_method"] = method
        candidate_frames.append(candidates)

    all_candidates = pd.concat(
        candidate_frames,
        ignore_index=True,
    )

    X_candidates = get_model_features(all_candidates)

    probabilities = model.predict_proba(X_candidates)[:, 1]

    all_candidates["predicted_success_probability"] = probabilities

    probability_table = (
        all_candidates
        .pivot(
            index="transaction_id",
            columns="candidate_method",
            values="predicted_success_probability",
        )
        .reset_index()
    )

    original_methods = (
        df[["transaction_id", "payment_method"]]
        .rename(
            columns={
                "payment_method": "current_method",
            }
        )
    )

    probability_table = probability_table.merge(
        original_methods,
        on="transaction_id",
        how="left",
    )

    probability_table["current_success_probability"] = (
        probability_table.apply(
            lambda row: row[row["current_method"]],
            axis=1,
        )
    )

    best_methods = []
    best_probabilities = []

    for _, row in probability_table.iterrows():
        alternatives = {
            method: row[method]
            for method in PAYMENT_METHODS
            if method != row["current_method"]
        }

        best_method = max(
            alternatives,
            key=alternatives.get,
        )

        best_methods.append(best_method)
        best_probabilities.append(
            alternatives[best_method]
        )

    probability_table["best_alternative_method"] = best_methods

    probability_table["best_alternative_success_probability"] = (
        best_probabilities
    )

    probability_table["potential_improvement"] = (
        probability_table["best_alternative_success_probability"]
        - probability_table["current_success_probability"]
    )

    return probability_table


def main() -> None:
    print("Loading validation data...")
    df = pd.read_parquet(DATA_PATH)

    print(f"Validation rows: {len(df):,}")

    print("\nLoading Logistic Regression model...")
    model = joblib.load(MODEL_PATH)

    print("\nGenerating counterfactual predictions...")
    routing_data = generate_counterfactual_predictions(
        df,
        model,
    )

    routing_data = routing_data.merge(
        df[["transaction_id", "payment_success"]],
        on="transaction_id",
        how="left",
    )

    print("\nRouting Opportunity by Current Payment Method")
    print("=" * 100)

    for threshold in THRESHOLDS:
        print(
            f"\nMinimum predicted improvement: "
            f"{threshold * 100:.0f} percentage points"
        )
        print("-" * 100)

        rows = []

        for method in PAYMENT_METHODS:
            method_data = routing_data[
                routing_data["current_method"] == method
            ]

            transaction_count = len(method_data)

            actual_failures = (
                method_data["payment_success"] == 0
            ).sum()

            recommended = (
                method_data["potential_improvement"]
                >= threshold
            )

            recommendation_count = recommended.sum()

            recommendation_rate = (
                recommendation_count / transaction_count
                if transaction_count > 0
                else 0.0
            )

            recommended_failures = (
                recommended
                & (method_data["payment_success"] == 0)
            ).sum()

            failure_rate = (
                actual_failures / transaction_count
                if transaction_count > 0
                else 0.0
            )

            average_improvement = (
                method_data.loc[
                    recommended,
                    "potential_improvement",
                ].mean()
                if recommendation_count > 0
                else 0.0
            )

            rows.append(
                {
                    "current_method": method,
                    "transactions": transaction_count,
                    "actual_failures": actual_failures,
                    "actual_failure_rate": failure_rate,
                    "recommendations": recommendation_count,
                    "recommendation_rate": recommendation_rate,
                    "recommended_failures": recommended_failures,
                    "avg_predicted_improvement": average_improvement,
                }
            )

        results = pd.DataFrame(rows)

        display_results = results.copy()

        display_results["actual_failure_rate"] *= 100
        display_results["recommendation_rate"] *= 100
        display_results["avg_predicted_improvement"] *= 100

        display_results = display_results.rename(
            columns={
                "actual_failure_rate": "failure_rate_percent",
                "recommendation_rate": "recommendation_rate_percent",
                "avg_predicted_improvement": (
                    "avg_predicted_improvement_percentage_points"
                ),
            }
        )

        print(
            display_results.to_string(
                index=False,
                float_format=lambda value: f"{value:.2f}",
            )
        )

        recommended_routes = routing_data[
            routing_data["potential_improvement"] >= threshold
        ].copy()

        route_counts = (
            recommended_routes
            .groupby(
                ["current_method", "best_alternative_method"]
            )
            .size()
            .reset_index(name="recommendations")
            .sort_values(
                "recommendations",
                ascending=False,
            )
        )

        print("\nRecommended Route Distribution")
        print("-" * 100)

        if route_counts.empty:
            print("No recommendations at this threshold.")
        else:
            print(
                route_counts.to_string(
                    index=False,
                )
            )

if __name__ == "__main__":
    main()
    