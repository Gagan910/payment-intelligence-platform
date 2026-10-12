
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from evaluate_feature_ablation import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
)
from train_logistic_regression import load_data


OUTPUT_PATH = Path("data/processed/evaluation/logistic_feature_comparison.csv")


def main() -> None:
    X_train, y_train, X_validation, y_validation = load_data()

    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    interaction_features = [
        "merchant_payment_method",
        "device_payment_method",
        "network_payment_method",
    ]

    experiments = {
        "all_features": all_features,
        "without_interactions": [
            column
            for column in all_features
            if column not in interaction_features
        ],
    }

    results = []

    for name, columns in experiments.items():
        print(f"Training {name}...")

        model = build_pipeline(columns)
        model.fit(X_train[columns], y_train)

        classes = list(model.classes_)
        failure_index = classes.index(0)

        failure_probabilities = model.predict_proba(
            X_validation[columns]
        )[:, failure_index]

        failure_labels = (y_validation == 0).astype(int)

        results.append(
            {
                "experiment": name,
                "feature_count": len(columns),
                "failure_roc_auc": roc_auc_score(
                    failure_labels,
                    failure_probabilities,
                ),
                "failure_brier_score": brier_score_loss(
                    failure_labels,
                    failure_probabilities,
                ),
            }
        )

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nValidation Feature Comparison")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
