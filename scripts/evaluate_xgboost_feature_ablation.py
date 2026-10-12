from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from train_xgboost import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_data,
)


OUTPUT_PATH = Path("data/processed/evaluation/xgboost_feature_ablation.csv")

INTERACTION_FEATURES = [
    "merchant_payment_method",
    "device_payment_method",
    "network_payment_method",
]


def build_pipeline(columns: list[str]) -> Pipeline:
    categorical = [c for c in CATEGORICAL_FEATURES if c in columns]
    numeric = [c for c in NUMERIC_FEATURES if c in columns]

    transformers = []

    if numeric:
        transformers.append(("numeric", "passthrough", numeric))

    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
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
        "feature_count": len(columns),
        "failure_roc_auc": roc_auc_score(y_failure, p_failure),
        "failure_brier_score": brier_score_loss(y_failure, p_failure),
    }


def main():
    X_train, y_train, X_validation, y_validation = load_data()

    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    all_features += INTERACTION_FEATURES

    missing = sorted(set(all_features) - set(X_train.columns))
    if missing:
        raise ValueError(f"Expected features missing from training data: {missing}")

    experiments = [
        ("all_features", all_features),
        (
            "without_interactions",
            [c for c in all_features if c not in INTERACTION_FEATURES],
        ),
    ]

    results = [
        evaluate(name, columns, X_train, y_train, X_validation, y_validation)
        for name, columns in experiments
    ]

    output = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\nXGBoost failure-focused feature ablation")
    print(output.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
