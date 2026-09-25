from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from payment_platform.data.features import get_model_features
from payment_platform.data.generator import calculate_failure_probability


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

RANDOM_SEED = 42


def generate_model_predictions(
    df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """Generate ML predictions for every counterfactual method."""

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

    all_candidates["predicted_success_probability"] = (
        model.predict_proba(X_candidates)[:, 1]
    )

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

    probability_table["best_alternative_method"] = (
        best_methods
    )

    probability_table["best_alternative_success_probability"] = (
        best_probabilities
    )

    probability_table["potential_improvement"] = (
        probability_table["best_alternative_success_probability"]
        - probability_table["current_success_probability"]
    )

    return probability_table


def generate_ground_truth_counterfactuals(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate synthetic ground-truth failure probabilities
    for every alternative payment method.

    This uses the existing frozen ground-truth mechanism.
    """

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

    all_candidates["ground_truth_failure_probability"] = (
        calculate_failure_probability(all_candidates)
    )

    all_candidates["ground_truth_success_probability"] = (
        1.0
        - all_candidates["ground_truth_failure_probability"]
    )

    probability_table = (
        all_candidates
        .pivot(
            index="transaction_id",
            columns="candidate_method",
            values="ground_truth_success_probability",
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

    probability_table["ground_truth_best_alternative_method"] = (
        best_methods
    )

    probability_table[
        "ground_truth_best_alternative_success_probability"
    ] = best_probabilities

    probability_table[
        "ground_truth_current_success_probability"
    ] = probability_table.apply(
        lambda row: row[row["current_method"]],
        axis=1,
    )

    return probability_table


def simulate_alternative_outcomes(
    routing_data: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Independently simulate the outcome of the model-selected
    alternative using the synthetic ground-truth probability.

    The random outcome is independent of the ML prediction.
    """

    routing_data = routing_data.copy()

    random_values = rng.random(len(routing_data))

    routing_data["simulated_alternative_success"] = (
        random_values
        < routing_data[
            "ground_truth_best_alternative_success_probability"
        ].to_numpy()
    ).astype(int)

    return routing_data


def main() -> None:
    print("Loading validation data...")

    df = pd.read_parquet(DATA_PATH)

    print(f"Validation rows: {len(df):,}")

    print("\nLoading Logistic Regression model...")

    model = joblib.load(MODEL_PATH)

    print("\nGenerating ML counterfactual predictions...")

    model_predictions = generate_model_predictions(
        df,
        model,
    )

    print("\nGenerating ground-truth counterfactual probabilities...")

    ground_truth_predictions = generate_ground_truth_counterfactuals(
        df,
    )

    routing_data = model_predictions.merge(
        ground_truth_predictions[
            [
                "transaction_id",
                "ground_truth_current_success_probability",
                "ground_truth_best_alternative_method",
                "ground_truth_best_alternative_success_probability",
            ]
        ],
        on="transaction_id",
        how="left",
    )

    routing_data = routing_data.merge(
        df[
            [
                "transaction_id",
                "payment_success",
            ]
        ],
        on="transaction_id",
        how="left",
    )

    rng = np.random.default_rng(RANDOM_SEED)

    routing_data = simulate_alternative_outcomes(
        routing_data,
        rng,
    )

    print("\nSynthetic Routing Recovery Evaluation")
    print("=" * 100)

    total_transactions = len(routing_data)

    original_failures = (
        routing_data["payment_success"] == 0
    ).sum()

    print(f"Total validation transactions: {total_transactions:,}")
    print(f"Original actual failures: {original_failures:,}")

    results = []

    for threshold in THRESHOLDS:
        recommended = (
            routing_data["potential_improvement"]
            >= threshold
        )

        recommended_count = recommended.sum()

        recommended_failures = (
            recommended
            & (routing_data["payment_success"] == 0)
        ).sum()

        simulated_recoveries = (
            recommended
            & (routing_data["payment_success"] == 0)
            & (
                routing_data[
                    "simulated_alternative_success"
                ]
                == 1
            )
        ).sum()

        recovery_rate_among_recommended_failures = (
            simulated_recoveries / recommended_failures
            if recommended_failures > 0
            else 0.0
        )

        recovery_rate_among_all_failures = (
            simulated_recoveries / original_failures
            if original_failures > 0
            else 0.0
        )

        results.append(
            {
                "threshold": threshold,
                "recommendations": recommended_count,
                "recommended_failures": recommended_failures,
                "simulated_recoveries": simulated_recoveries,
                "recovery_rate_among_recommended_failures": (
                    recovery_rate_among_recommended_failures
                ),
                "recovery_rate_among_all_failures": (
                    recovery_rate_among_all_failures
                ),
            }
        )

    results_df = pd.DataFrame(results)

    display_results = results_df.copy()

    display_results["threshold"] *= 100

    display_results[
        "recovery_rate_among_recommended_failures"
    ] *= 100

    display_results[
        "recovery_rate_among_all_failures"
    ] *= 100

    display_results = display_results.rename(
        columns={
            "threshold": "threshold_percentage_points",
            "recovery_rate_among_recommended_failures": (
                "recovery_rate_among_recommended_failures_percent"
            ),
            "recovery_rate_among_all_failures": (
                "recovery_rate_among_all_failures_percent"
            ),
        }
    )

    print(
        display_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )


if __name__ == "__main__":
    main()