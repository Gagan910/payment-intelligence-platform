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


def main() -> None:
    print("Loading validation data...")
    df = pd.read_parquet(DATA_PATH)

    print(f"Validation rows: {len(df):,}")

    print("\nLoading Logistic Regression model...")
    model = joblib.load(MODEL_PATH)

    # Create one counterfactual version of every transaction
    # for each available payment method.
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

    print("\nPreparing counterfactual features...")
    X_candidates = get_model_features(all_candidates)

    print("Generating counterfactual predictions...")
    probabilities = model.predict_proba(X_candidates)[:, 1]

    all_candidates["predicted_success_probability"] = probabilities

    # Convert the candidate predictions into one row per transaction:
    #
    # transaction_id | upi | credit_card | debit_card | ...
    probability_table = (
        all_candidates
        .pivot(
            index="transaction_id",
            columns="candidate_method",
            values="predicted_success_probability",
        )
        .reset_index()
    )

    # Restore the payment method that was actually selected
    # in the original validation transaction.
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

    # Get the model's predicted success probability for the
    # payment method that was actually selected.
    probability_table["current_success_probability"] = (
        probability_table.apply(
            lambda row: row[row["current_method"]],
            axis=1,
        )
    )

    # Find the best alternative while explicitly excluding
    # the currently selected payment method.
    best_alternative_methods = []
    best_alternative_probabilities = []

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

        best_probability = alternatives[best_method]

        best_alternative_methods.append(best_method)
        best_alternative_probabilities.append(best_probability)

    probability_table["best_alternative_method"] = (
        best_alternative_methods
    )

    probability_table["best_alternative_success_probability"] = (
        best_alternative_probabilities
    )

    # Improvement is measured in probability points.
    probability_table["potential_improvement"] = (
        probability_table["best_alternative_success_probability"]
        - probability_table["current_success_probability"]
    )

    print("\nRouting Opportunity Analysis")
    print("=" * 60)

    better_alternative_count = (
        probability_table["potential_improvement"] > 0
    ).sum()

    no_better_alternative_count = (
        probability_table["potential_improvement"] <= 0
    ).sum()

    print(
        "Transactions with a better predicted alternative: "
        f"{better_alternative_count:,}"
    )

    print(
        "Transactions with no better predicted alternative: "
        f"{no_better_alternative_count:,}"
    )

    print("\nPotential improvement distribution:")
    print(
        probability_table["potential_improvement"]
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print("\nPotential improvement in percentage points:")

    improvement_percentage_points = (
        probability_table["potential_improvement"] * 100
    )

    print(
        improvement_percentage_points
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print("\nTop 10 routing opportunities:")

    top_opportunities = probability_table.nlargest(
        10,
        "potential_improvement",
    )

    print(
        top_opportunities[
            [
                "transaction_id",
                "current_method",
                "current_success_probability",
                "best_alternative_method",
                "best_alternative_success_probability",
                "potential_improvement",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()