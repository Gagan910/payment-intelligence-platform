from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


FEATURE_DIR = Path("data/processed/features")
MODEL_DIR = Path("models")

TRAIN_FEATURES = FEATURE_DIR / "train_features.parquet"
TRAIN_TARGET = FEATURE_DIR / "train_target.parquet"

VALIDATION_FEATURES = FEATURE_DIR / "validation_features.parquet"
VALIDATION_TARGET = FEATURE_DIR / "validation_target.parquet"

MODEL_PATH = MODEL_DIR / "xgboost_baseline.joblib"


CATEGORICAL_FEATURES = [
    "merchant_category",
    "payment_method",
    "user_segment",
    "device_type",
    "network_quality",
]

NUMERIC_FEATURES = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "retry_count",
    "transaction_velocity",
    "user_method_success_rate",
    "merchant_method_success_rate",
    "log_amount",
    "log_transaction_velocity",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]


def load_data():
    X_train = pd.read_parquet(TRAIN_FEATURES)
    y_train = pd.read_parquet(TRAIN_TARGET)["payment_success"]

    X_validation = pd.read_parquet(VALIDATION_FEATURES)
    y_validation = pd.read_parquet(VALIDATION_TARGET)["payment_success"]

    return X_train, y_train, X_validation, y_validation


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = XGBClassifier(
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
            ("model", model),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> None:
    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_validation, probabilities)
    pr_auc = average_precision_score(y_validation, probabilities)
    precision = precision_score(y_validation, predictions, zero_division=0)
    recall = recall_score(y_validation, predictions, zero_division=0)
    f1 = f1_score(y_validation, predictions, zero_division=0)
    brier = brier_score_loss(y_validation, probabilities)

    matrix = confusion_matrix(y_validation, predictions)

    print("\nValidation Results")
    print("=" * 50)
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"Brier Score: {brier:.4f}")

    print("\nConfusion Matrix")
    print("[[TN, FP],")
    print(" [FN, TP]]")
    print(matrix)

    print("\nValidation rows:", f"{len(y_validation):,}")
    print("Predicted positive rate:", f"{predictions.mean():.4f}")
    print("Actual positive rate:", f"{y_validation.mean():.4f}")


def main() -> None:
    print("Loading model-ready datasets...")

    X_train, y_train, X_validation, y_validation = load_data()

    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Features:        {X_train.shape[1]}")

    print("\nBuilding XGBoost pipeline...")

    pipeline = build_pipeline()

    print("Training XGBoost...")
    print("This may take longer than the previous models.")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    evaluate_model(
        pipeline,
        X_validation,
        y_validation,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print("\nModel saved successfully.")
    print(f"Model path: {MODEL_PATH}")


if __name__ == "__main__":
    main()