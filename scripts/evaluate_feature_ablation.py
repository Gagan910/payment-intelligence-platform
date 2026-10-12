from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_logistic_regression import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_data,
)


OUTPUT_PATH = Path("data/processed/evaluation/feature_ablation.csv")


def build_pipeline(columns: list[str]) -> Pipeline:
    categorical = [c for c in CATEGORICAL_FEATURES if c in columns]
    numeric = [c for c in NUMERIC_FEATURES if c in columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def evaluate(name, columns, X_train, y_train, X_validation, y_validation):
    model = build_pipeline(columns)
    model.fit(X_train[columns], y_train)

    failure_index = list(model.classes_).index(0)
    p_failure = model.predict_proba(X_validation[columns])[:, failure_index]
    y_failure = (y_validation == 0).astype(int)

    return {
        "experiment": name,
        "features": len(columns),
        "failure_roc_auc": roc_auc_score(y_failure, p_failure),
        "failure_brier_score": brier_score_loss(y_failure, p_failure),
    }


def main():
    X_train, y_train, X_validation, y_validation = load_data()

    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    history = ["user_method_success_rate", "merchant_method_success_rate"]
    interactions = [
        "merchant_payment_method",
        "device_payment_method",
        "network_payment_method",
    ]

    experiments = [
        ("baseline_all_features", all_features),
        ("without_history_features", [c for c in all_features if c not in history]),
        (
            "without_interaction_features",
            [c for c in all_features if c not in interactions],
        ),
        (
            "without_history_or_interactions",
            [c for c in all_features if c not in history + interactions],
        ),
    ]

    results = [
        evaluate(name, columns, X_train, y_train, X_validation, y_validation)
        for name, columns in experiments
    ]

    output = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\nFailure-focused feature ablation")
    print(output.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
