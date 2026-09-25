from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


FEATURE_DIR = Path("data/processed/features")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("data/processed/evaluation")

VALIDATION_FEATURES = FEATURE_DIR / "validation_features.parquet"
VALIDATION_TARGET = FEATURE_DIR / "validation_target.parquet"


MODELS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression_baseline.joblib",
    "Random Forest": MODEL_DIR / "random_forest_baseline.joblib",
    "XGBoost": MODEL_DIR / "xgboost_baseline.joblib",
}


def evaluate_model(model, X_validation, y_validation):
    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y_validation, probabilities),
        "pr_auc": average_precision_score(y_validation, probabilities),
        "precision": precision_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "brier_score": brier_score_loss(
            y_validation,
            probabilities,
        ),
    }


def main() -> None:
    print("Loading validation dataset...")

    X_validation = pd.read_parquet(VALIDATION_FEATURES)
    y_validation = pd.read_parquet(VALIDATION_TARGET)["payment_success"]

    print(f"Validation rows: {len(X_validation):,}")

    results = []

    for model_name, model_path in MODELS.items():
        print(f"\nEvaluating {model_name}...")

        model = joblib.load(model_path)

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        results.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    comparison = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "model_comparison_validation.csv"
    comparison.to_csv(output_path, index=False)

    print("\nModel Comparison")
    print("=" * 80)

    print(
        comparison.to_string(
            index=False,
            formatters={
                "roc_auc": "{:.4f}".format,
                "pr_auc": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "brier_score": "{:.4f}".format,
            },
        )
    )

    print("\nComparison saved to:")
    print(output_path)

    print("\nModel comparison completed.")


if __name__ == "__main__":
    main()