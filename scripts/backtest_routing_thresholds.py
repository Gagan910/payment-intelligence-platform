from payment_platform.data.generator import calculate_failure_probability

from pathlib import Path

import joblib
import pandas as pd

from payment_platform.data.features import get_model_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib"

PAYMENT_METHODS = [
    "upi",
    "credit_card",
    "debit_card",
    "net_banking",
    "wallet",
]

THRESHOLDS = [
    0.05
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
        best_probabilities.append(alternatives[best_method])

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


    # Calculate simulator-ground-truth risk for each candidate method.
    candidate_frames = []

    for method in PAYMENT_METHODS:
        candidates = df.copy()
        candidates["payment_method"] = method
        candidates["candidate_method"] = method
        candidate_frames.append(candidates)

    all_candidates = pd.concat(candidate_frames, ignore_index=True)

    all_candidates["ground_truth_failure_probability"] = (
        calculate_failure_probability(all_candidates)
    )

    ground_truth_table = (
        all_candidates.pivot(
            index="transaction_id",
            columns="candidate_method",
            values="ground_truth_failure_probability",
        )
        .reset_index()
    )

    ground_truth_table = ground_truth_table.merge(
        df[["transaction_id", "payment_method"]].rename(
            columns={"payment_method": "current_method"}
        ),
        on="transaction_id",
        how="left",
    )

    ground_truth_table["current_failure_risk"] = (
        ground_truth_table.apply(
            lambda row: row[row["current_method"]],
            axis=1,
        )
    )

    ground_truth_table["oracle_alternative_failure_risk"] = (
        ground_truth_table.apply(
            lambda row: min(
                row[method]
                for method in PAYMENT_METHODS
                if method != row["current_method"]
            ),
            axis=1,
        )
    )

    ground_truth_table["model_selected_failure_risk"] = (
        ground_truth_table.apply(
            lambda row: row[
                routing_data.loc[
                    routing_data["transaction_id"] == row["transaction_id"],
                    "best_alternative_method",
                ].iloc[0]
            ],
            axis=1,
        )
    )

    routing_data = routing_data.merge(
        ground_truth_table[
            [
                "transaction_id",
                "current_failure_risk",
                "oracle_alternative_failure_risk",
                "model_selected_failure_risk",
            ]
        ],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    total_transactions = len(routing_data)
    total_failures = (routing_data["payment_success"] == 0).sum()

    print("\nRouting Threshold Backtest")
    print("=" * 80)

    results = []

    for threshold in THRESHOLDS:
        recommended = (
            routing_data["potential_improvement"] >= threshold
        )

        # Baseline: keep the current payment method for every transaction.
        baseline_expected_failure_rate = (
            routing_data["current_failure_risk"].mean()
        )

        # Model policy: switch only recommended transactions.
        model_policy_risk = routing_data[
            "current_failure_risk"
        ].where(
            ~recommended,
            routing_data["model_selected_failure_risk"],
        )

        model_policy_expected_failure_rate = model_policy_risk.mean()

        # Oracle policy: switch when ground-truth risk reduction
        # meets the same threshold.
        oracle_improvement = (
            routing_data["current_failure_risk"]
            - routing_data["oracle_alternative_failure_risk"]
        )

        oracle_recommended = oracle_improvement >= threshold

        oracle_policy_risk = routing_data[
            "current_failure_risk"
        ].where(
            ~oracle_recommended,
            routing_data["oracle_alternative_failure_risk"],
        )

        oracle_policy_expected_failure_rate = oracle_policy_risk.mean()

        model_risk_reduction = (
            baseline_expected_failure_rate
            - model_policy_expected_failure_rate
        )

        oracle_risk_reduction = (
            baseline_expected_failure_rate
            - oracle_policy_expected_failure_rate
        )

        recommended_count = recommended.sum()

        recommended_rate = (
            recommended_count / total_transactions
        )

        recommended_failures = (
            recommended
            & (routing_data["payment_success"] == 0)
        ).sum()

        failure_opportunity_rate = (
            recommended_failures / total_failures
            if total_failures > 0
            else 0.0
        )

        average_predicted_improvement = (
            routing_data.loc[
                recommended,
                "potential_improvement",
            ].mean()
            if recommended_count > 0
            else 0.0
        )

        recommended_data = routing_data.loc[recommended]

        mean_current_risk = (
            recommended_data["current_failure_risk"].mean()
            if recommended_count > 0
            else 0.0
        )

        mean_model_risk = (
            recommended_data["model_selected_failure_risk"].mean()
            if recommended_count > 0
            else 0.0
        )

        mean_oracle_risk = (
            recommended_data["oracle_alternative_failure_risk"].mean()
            if recommended_count > 0
            else 0.0
        )

        results.append(
            {
                "threshold": threshold,
                "recommendations": recommended_count,
                "recommendation_rate": recommended_rate,
                "recommended_failures": recommended_failures,
                "failure_opportunity_rate": failure_opportunity_rate,
                "avg_predicted_improvement": average_predicted_improvement,
                "mean_current_failure_risk": mean_current_risk,
                "mean_model_failure_risk": mean_model_risk,
                "mean_oracle_failure_risk": mean_oracle_risk,
                "baseline_expected_failure_rate": baseline_expected_failure_rate,
                "model_policy_expected_failure_rate": model_policy_expected_failure_rate,
                "model_risk_reduction": model_risk_reduction,
                "oracle_policy_expected_failure_rate": oracle_policy_expected_failure_rate,
                "oracle_risk_reduction": oracle_risk_reduction,
            }
        )

    results_df = pd.DataFrame(results)

    display_df = results_df.copy()

    display_df["threshold"] = (
        display_df["threshold"] * 100
    )

    display_df["recommendation_rate"] = (
        display_df["recommendation_rate"] * 100
    )

    display_df["failure_opportunity_rate"] = (
        display_df["failure_opportunity_rate"] * 100
    )

    display_df["avg_predicted_improvement"] = (
        display_df["avg_predicted_improvement"] * 100
    )

    for column in [
        "mean_current_failure_risk",
        "mean_model_failure_risk",
        "mean_oracle_failure_risk",
        "baseline_expected_failure_rate",
        "model_policy_expected_failure_rate",
        "model_risk_reduction",
        "oracle_policy_expected_failure_rate",
        "oracle_risk_reduction",
    ]:
        display_df[column] *= 100

    display_df = display_df.rename(
        columns={
            "threshold": "threshold_percentage_points",
            "recommendation_rate": "recommendation_rate_percent",
            "failure_opportunity_rate": "failure_opportunity_rate_percent",
            "avg_predicted_improvement": "avg_predicted_improvement_percentage_points",
            "mean_current_failure_risk": "mean_current_failure_risk_percent",
            "mean_model_failure_risk": "mean_model_failure_risk_percent",
            "mean_oracle_failure_risk": "mean_oracle_failure_risk_percent",
            "baseline_expected_failure_rate": "baseline_expected_failure_rate_percent",
            "model_policy_expected_failure_rate": "model_policy_expected_failure_rate_percent",
            "model_risk_reduction": "model_risk_reduction_percentage_points",
            "oracle_policy_expected_failure_rate": "oracle_policy_expected_failure_rate_percent",
            "oracle_risk_reduction": "oracle_risk_reduction_percentage_points",
        }
    )

    print(
        display_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print("\nTest population:")
    print(f"Total transactions: {total_transactions:,}")
    print(f"Total actual failures: {total_failures:,}")
    print("Threshold: 5 percentage points")
    print("Evaluation data: October-December 2025")


if __name__ == "__main__":
    main()