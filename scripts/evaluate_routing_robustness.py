
from pathlib import Path

import joblib
import pandas as pd

from payment_platform.data.features import get_model_features
from payment_platform.data.generator import (
    calculate_failure_probability,
    generate_transactions,
    simulate_payment_outcomes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

MODELS = {
    "baseline": MODEL_DIR / "logistic_regression_baseline.joblib",
    "candidate": MODEL_DIR / "logistic_regression_no_interactions.joblib",
}

PAYMENT_METHODS = [
    "upi",
    "credit_card",
    "debit_card",
    "net_banking",
    "wallet",
]

N_TRANSACTIONS = 20_000
CONTEXT_SEED = 20261012
OUTCOME_SEED = 20261013
ROUTING_THRESHOLD = 0.05


def predict_best_alternatives(df: pd.DataFrame, model) -> pd.DataFrame:
    """Predict success probability for each alternative payment method."""
    rows = []

    for method in PAYMENT_METHODS:
        candidates = df.copy()
        candidates["payment_method"] = method

        features = get_model_features(candidates)
        success_probability = model.predict_proba(features)[:, 1]

        rows.append(
            pd.DataFrame(
                {
                    "transaction_id": df["transaction_id"].to_numpy(),
                    "candidate_method": method,
                    "predicted_success_probability": success_probability,
                }
            )
        )

    predictions = pd.concat(rows, ignore_index=True)

    prediction_table = predictions.pivot(
        index="transaction_id",
        columns="candidate_method",
        values="predicted_success_probability",
    ).reset_index()

    current_methods = df[
        ["transaction_id", "payment_method"]
    ].rename(columns={"payment_method": "current_method"})

    prediction_table = prediction_table.merge(
        current_methods,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    prediction_table["current_success_probability"] = prediction_table.apply(
        lambda row: row[row["current_method"]],
        axis=1,
    )

    alternatives = prediction_table[PAYMENT_METHODS].copy()

    for method in PAYMENT_METHODS:
        alternatives[method] = alternatives[method].where(
            prediction_table["current_method"] != method
        )

    prediction_table["best_alternative_method"] = alternatives.idxmax(axis=1)

    prediction_table["predicted_improvement"] = (
        prediction_table["best_alternative_method"].map(
            prediction_table.set_index("transaction_id")[
                PAYMENT_METHODS
            ].idxmax(axis=1)
        )
        if False
        else prediction_table.apply(
            lambda row: row[row["best_alternative_method"]]
            - row["current_success_probability"],
            axis=1,
        )
    )

    prediction_table["recommended"] = (
        prediction_table["predicted_improvement"] >= ROUTING_THRESHOLD
    )

    return prediction_table[
        [
            "transaction_id",
            "current_method",
            "best_alternative_method",
            "predicted_improvement",
            "recommended",
        ]
    ]


def main() -> None:
    print("Generating fresh synthetic transactions...")

    contexts = generate_transactions(
        N_TRANSACTIONS,
        seed=CONTEXT_SEED,
    )

    # Add simulated outcomes without changing the existing test data.
    fresh_data = simulate_payment_outcomes(
        contexts,
        seed=OUTCOME_SEED,
    )

    print(f"Fresh transactions: {len(fresh_data):,}")
    print(f"Actual simulated failure rate: {(1 - fresh_data['payment_success'].mean()):.4%}")

    # Calculate simulator ground-truth risk for every payment method.
    risk_frames = []

    for method in PAYMENT_METHODS:
        candidates = fresh_data.copy()
        candidates["payment_method"] = method
        candidates["candidate_method"] = method
        risk_frames.append(candidates)

    all_risks = pd.concat(risk_frames, ignore_index=True)

    all_risks["true_failure_risk"] = calculate_failure_probability(
        all_risks
    )

    risk_table = all_risks.pivot(
        index="transaction_id",
        columns="candidate_method",
        values="true_failure_risk",
    ).reset_index()

    risk_table = risk_table.merge(
        fresh_data[["transaction_id", "payment_method"]].rename(
            columns={"payment_method": "current_method"}
        ),
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    risk_table["current_failure_risk"] = risk_table.apply(
        lambda row: row[row["current_method"]],
        axis=1,
    )

    baseline_risk = risk_table["current_failure_risk"].mean()

    print("\nFresh-Data Routing Robustness")
    print("=" * 55)
    print(f"Current expected failure risk: {baseline_risk:.4%}")

    for name, model_path in MODELS.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model artifact: {model_path}"
            )

        model = joblib.load(model_path)
        decisions = predict_best_alternatives(fresh_data, model)

        evaluation = decisions.merge(
            risk_table,
            on=["transaction_id", "current_method"],
            how="left",
            validate="one_to_one",
        )

        evaluation["selected_failure_risk"] = evaluation.apply(
            lambda row: row[row["best_alternative_method"]],
            axis=1,
        )

        actual_risk_improvement = (
            evaluation["current_failure_risk"]
            - evaluation["selected_failure_risk"]
        )

        recommended = evaluation["recommended"]

        post_routing_risk = evaluation["current_failure_risk"].where(
            ~recommended,
            evaluation["selected_failure_risk"],
        ).mean()

        recommendation_count = int(recommended.sum())

        genuinely_better = (
            recommended & (actual_risk_improvement > 0)
        )

        print(f"\n{name.title()} model")
        print("-" * 40)
        print(f"Recommendations: {recommendation_count:,} ({recommended.mean():.2%})")
        print(f"Expected failure risk after routing: {post_routing_risk:.4%}")
        print(f"Expected risk reduction: {baseline_risk - post_routing_risk:.4%}")

        if recommendation_count:
            print(
                "Recommendations with lower true risk: "
                f"{genuinely_better.sum():,} "
                f"({genuinely_better.sum() / recommendation_count:.2%})"
            )
            print(
                "Mean true risk reduction among recommendations: "
                f"{actual_risk_improvement[recommended].mean():.4%}"
            )


if __name__ == "__main__":
    main()
