from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


FEATURE_DIR = Path("data/processed/features")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("data/processed/evaluation")

TRAIN_FEATURES = FEATURE_DIR / "train_features.parquet"
TRAIN_TARGET = FEATURE_DIR / "train_target.parquet"

VALIDATION_FEATURES = FEATURE_DIR / "validation_features.parquet"
VALIDATION_TARGET = FEATURE_DIR / "validation_target.parquet"

TEST_FEATURES = FEATURE_DIR / "test_features.parquet"
TEST_TARGET = FEATURE_DIR / "test_target.parquet"

BASE_MODEL_PATH = MODEL_DIR / "logistic_regression_baseline.joblib"
CALIBRATED_MODEL_PATH = MODEL_DIR / "logistic_regression_calibrated.joblib"


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


def build_base_model() -> Pipeline:
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

    model = LogisticRegression(
        max_iter=5000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    print("Loading datasets...")

    X_train = pd.read_parquet(TRAIN_FEATURES)
    y_train = pd.read_parquet(TRAIN_TARGET)["payment_success"]

    X_validation = pd.read_parquet(VALIDATION_FEATURES)
    y_validation = pd.read_parquet(VALIDATION_TARGET)["payment_success"]

    X_test = pd.read_parquet(TEST_FEATURES)
    y_test = pd.read_parquet(TEST_TARGET)["payment_success"]

    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Test rows:       {len(X_test):,}")

    # ---------------------------------------------------------
    # 1. Fit the base Logistic Regression model on TRAIN only.
    # ---------------------------------------------------------
    print("\nFitting base Logistic Regression on training data...")

    base_model = build_base_model()
    base_model.fit(X_train, y_train)

    validation_probabilities = base_model.predict_proba(
        X_validation
    )[:, 1]

    test_raw_probabilities = base_model.predict_proba(
        X_test
    )[:, 1]

    validation_raw_brier = brier_score_loss(
        y_validation,
        validation_probabilities,
    )

    test_raw_brier = brier_score_loss(
        y_test,
        test_raw_probabilities,
    )

    print("\nRaw Logistic Regression Brier Scores")
    print("=" * 50)
    print(f"Validation: {validation_raw_brier:.4f}")
    print(f"Test:       {test_raw_brier:.4f}")

    # ---------------------------------------------------------
    # 2. Fit sigmoid calibration on VALIDATION probabilities.
    #
    # Logistic regression on log-odds is Platt-style sigmoid
    # calibration.
    # ---------------------------------------------------------
    print("\nFitting sigmoid calibration on validation predictions...")

    validation_probabilities_clipped = np.clip(
        validation_probabilities,
        1e-6,
        1 - 1e-6,
    )

    validation_log_odds = np.log(
        validation_probabilities_clipped
        / (1 - validation_probabilities_clipped)
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        random_state=42,
    )

    calibrator.fit(
        validation_log_odds,
        y_validation,
    )

    # ---------------------------------------------------------
    # 3. Apply the fitted calibrator to TEST predictions.
    # ---------------------------------------------------------
    test_probabilities_clipped = np.clip(
        test_raw_probabilities,
        1e-6,
        1 - 1e-6,
    )

    test_log_odds = np.log(
        test_probabilities_clipped
        / (1 - test_probabilities_clipped)
    ).reshape(-1, 1)

    calibrated_test_probabilities = calibrator.predict_proba(
        test_log_odds
    )[:, 1]

    test_calibrated_brier = brier_score_loss(
        y_test,
        calibrated_test_probabilities,
    )

    print("\nFinal Test Calibration Results")
    print("=" * 50)
    print(f"Raw Brier Score:        {test_raw_brier:.4f}")
    print(f"Calibrated Brier Score: {test_calibrated_brier:.4f}")

    # ---------------------------------------------------------
    # 4. Calibration curves on TEST.
    # ---------------------------------------------------------
    raw_fraction, raw_mean = calibration_curve(
        y_test,
        test_raw_probabilities,
        n_bins=10,
        strategy="uniform",
    )

    calibrated_fraction, calibrated_mean = calibration_curve(
        y_test,
        calibrated_test_probabilities,
        n_bins=10,
        strategy="uniform",
    )

    calibration_data = pd.DataFrame(
        {
            "raw_mean_predicted_probability": pd.Series(raw_mean),
            "raw_actual_success_rate": pd.Series(raw_fraction),
            "calibrated_mean_predicted_probability": pd.Series(
                calibrated_mean
            ),
            "calibrated_actual_success_rate": pd.Series(
                calibrated_fraction
            ),
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "logistic_regression_calibration_test.csv"
    calibration_data.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 6))

    plt.plot(
        raw_mean,
        raw_fraction,
        marker="o",
        label="Raw Logistic Regression",
    )

    plt.plot(
        calibrated_mean,
        calibrated_fraction,
        marker="o",
        label="Sigmoid Calibrated",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Actual Success Rate")
    plt.title("Logistic Regression Calibration on Test Set")
    plt.legend()
    plt.grid(alpha=0.3)

    plot_path = OUTPUT_DIR / "logistic_regression_calibration_test.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nTest Calibration Data")
    print("=" * 50)
    print(calibration_data.to_string(index=False))

    print("\nArtifacts saved:")
    print(f"  Calibration CSV:  {csv_path}")
    print(f"  Calibration plot: {plot_path}")

    # Save the components required to reproduce the calibrated model.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    calibration_artifact = {
        "base_model": base_model,
        "calibrator": calibrator,
    }

    joblib.dump(
        calibration_artifact,
        CALIBRATED_MODEL_PATH,
    )

    print("\nCalibrated model artifact saved:")
    print(f"  {CALIBRATED_MODEL_PATH}")

    print("\nCalibration experiment completed.")


if __name__ == "__main__":
    main()
    