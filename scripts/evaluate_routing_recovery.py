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

    routing_data["simulated_alternative_success_probability"] = (

        routing_data.apply(

            lambda row: 1.0 - row[

                f"ground_truth_{row['best_alternative_method']}"

            ],

            axis=1,

        )

    )

    routing_data["simulated_alternative_success"] = (

        random_values

        < routing_data[

            "simulated_alternative_success_probability"

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
    model_predictions = generate_model_predictions(df, model)

    print("\nGenerating ground-truth counterfactual probabilities...")
    ground_truth_predictions = generate_ground_truth_counterfactuals(df)

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
        validate="one_to_one",
    )

    # Calculate synthetic ground-truth failure risk for every payment method.
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
        validate="one_to_one",
    )

    ground_truth_table["current_failure_risk"] = ground_truth_table.apply(
        lambda row: row[row["current_method"]],
        axis=1,
    )

    ground_truth_table = ground_truth_table.rename(
        columns={method: f"ground_truth_{method}" for method in PAYMENT_METHODS}
    )

    routing_data = routing_data.merge(
        ground_truth_table[
            [
                "transaction_id",
                "current_failure_risk",
                *[f"ground_truth_{method}" for method in PAYMENT_METHODS],
            ]
        ],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    routing_data = routing_data.merge(
        df[["transaction_id", "payment_success"]],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    # Compare the ML-selected alternative against the simulator oracle.
    routing_data["oracle_method_match"] = (
        routing_data["best_alternative_method"]
        == routing_data["ground_truth_best_alternative_method"]
    )

    routing_data["model_alternative_failure_probability"] = routing_data.apply(
        lambda row: row[f"ground_truth_{row['best_alternative_method']}"],
        axis=1,
    )
    routing_data["oracle_alternative_failure_probability"] = (
        1.0 - routing_data["ground_truth_best_alternative_success_probability"]
    )
    routing_data["oracle_risk_regret"] = (
        routing_data["model_alternative_failure_probability"]
        - routing_data["oracle_alternative_failure_probability"]
    )

    print("\nMODEL ROUTING VS. SIMULATOR ORACLE")
    print("=" * 60)
    print(f"Oracle method agreement: {routing_data['oracle_method_match'].mean():.2%}")
    print(
        "Mean model-selected alternative failure risk: "
        f"{routing_data['model_alternative_failure_probability'].mean():.4f}"
    )
    print(
        "Mean oracle alternative failure risk: "
        f"{routing_data['oracle_alternative_failure_probability'].mean():.4f}"
    )
    print(f"Mean oracle risk regret: {routing_data['oracle_risk_regret'].mean():.4f}")
    print(f"Median oracle risk regret: {routing_data['oracle_risk_regret'].median():.4f}")

    # Evaluate the current policy at a 5-percentage-point threshold.
    policy_threshold = 0.05
    recommended = routing_data["potential_improvement"] >= policy_threshold
    policy_data = routing_data.loc[recommended].copy()

    policy_data["model_method_failure_risk"] = policy_data.apply(
        lambda row: row[f"ground_truth_{row['best_alternative_method']}"],
        axis=1,
    )
    policy_data["current_method_failure_risk"] = policy_data["current_failure_risk"]
    policy_data["oracle_method_failure_risk"] = (
        1.0 - policy_data["ground_truth_best_alternative_success_probability"]
    )
    policy_data["model_risk_change"] = (
        policy_data["model_method_failure_risk"]
        - policy_data["current_method_failure_risk"]
    )

    print("\nACTUAL POLICY VS. SIMULATOR BENCHMARK (5% THRESHOLD)")
    print("=" * 65)
    print(f"Recommended transactions: {len(policy_data):,}")
    print(
        "Mean current-method failure risk: "
        f"{policy_data['current_method_failure_risk'].mean():.4f}"
    )
    print(
        "Mean model-selected failure risk: "
        f"{policy_data['model_method_failure_risk'].mean():.4f}"
    )
    print(
        "Mean oracle-selected failure risk: "
        f"{policy_data['oracle_method_failure_risk'].mean():.4f}"
    )
    print(f"Mean model-selected risk change: {policy_data['model_risk_change'].mean():+.4f}")
    print(
        "Recommendations with higher model-selected failure risk: "
        f"{(policy_data['model_risk_change'] > 0).mean():.2%}"
    )
    print(
        "Recommendations with equal failure risk: "
        f"{np.isclose(policy_data['model_risk_change'], 0.0).mean():.2%}"
    )
    print(
        "Recommendations with lower model-selected failure risk: "
        f"{(policy_data['model_risk_change'] < 0).mean():.2%}"
    )

    rng = np.random.default_rng(RANDOM_SEED)
    routing_data = simulate_alternative_outcomes(routing_data, rng)

    print("\nSynthetic Routing Recovery Evaluation")
    print("=" * 100)
    
    total_transactions = len(routing_data)
    original_failures = (routing_data["payment_success"] == 0).sum()
    print(f"Total validation transactions: {total_transactions:,}")
    print(f"Original actual failures: {original_failures:,}")

    results = []
    
    for threshold in THRESHOLDS:
        recommended = routing_data["potential_improvement"] >= threshold
        recommended_count = int(recommended.sum())
        recommended_failures = int(
            (recommended & (routing_data["payment_success"] == 0)).sum()
        )
        simulated_recoveries = int(
            (
                recommended
                & (routing_data["payment_success"] == 0)
                & (routing_data["simulated_alternative_success"] == 1)
            ).sum()
        )

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
                "recovery_rate_among_all_failures": recovery_rate_among_all_failures,
            }
        )

    results_df = pd.DataFrame(results)
    
    display_results = results_df.copy()
    
    display_results["threshold"] *= 100
    display_results["recovery_rate_among_recommended_failures"] *= 100
    display_results["recovery_rate_among_all_failures"] *= 100
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
    print(display_results.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

if __name__ == "__main__":
    main()
