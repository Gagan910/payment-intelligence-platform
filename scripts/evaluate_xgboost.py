from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


FEATURE_DIR = Path("data/processed/features")
MODEL_PATH = Path("models/xgboost_baseline.joblib")
OUTPUT_DIR = Path("data/processed/evaluation")

VALIDATION_FEATURES = FEATURE_DIR / "validation_features.parquet"
VALIDATION_TARGET = FEATURE_DIR / "validation_target.parquet"


def main() -> None:
    print("Loading XGBoost baseline...")

    model = joblib.load(MODEL_PATH)

    X_validation = pd.read_parquet(VALIDATION_FEATURES)
    y_validation = pd.read_parquet(VALIDATION_TARGET)["payment_success"]

    print(f"Validation rows: {len(X_validation):,}")

    print("\nGenerating probability predictions...")

    probabilities = model.predict_proba(X_validation)[:, 1]

    roc_auc = roc_auc_score(y_validation, probabilities)
    pr_auc = average_precision_score(y_validation, probabilities)
    brier = brier_score_loss(y_validation, probabilities)

    print("\nProbability Metrics")
    print("=" * 50)
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")

    print("\nProbability Summary")
    print("=" * 50)
    print(f"Minimum:     {probabilities.min():.4f}")
    print(f"Maximum:     {probabilities.max():.4f}")
    print(f"Mean:        {probabilities.mean():.4f}")
    print(f"Median:      {np.median(probabilities):.4f}")

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_validation,
        probabilities,
        n_bins=10,
        strategy="uniform",
    )

    calibration_data = pd.DataFrame(
        {
            "mean_predicted_probability": mean_predicted_value,
            "actual_success_rate": fraction_of_positives,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calibration_csv = OUTPUT_DIR / "xgboost_calibration.csv"
    calibration_data.to_csv(calibration_csv, index=False)

    plt.figure(figsize=(8, 6))

    plt.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        label="XGBoost",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Actual Success Rate")
    plt.title("XGBoost Calibration Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plot_path = OUTPUT_DIR / "xgboost_calibration.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nCalibration Data")
    print("=" * 50)
    print(calibration_data.to_string(index=False))

    print("\nEvaluation artifacts saved:")
    print(f"  Calibration CSV:  {calibration_csv}")
    print(f"  Calibration plot: {plot_path}")

    print("\nXGBoost evaluation completed.")


if __name__ == "__main__":
    main()