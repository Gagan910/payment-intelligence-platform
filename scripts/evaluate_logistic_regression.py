from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


FEATURE_DIR = Path("data/processed/features")
MODEL_PATH = Path("models/logistic_regression_baseline.joblib")
OUTPUT_DIR = Path("data/processed/evaluation")

TEST_FEATURES = FEATURE_DIR / "test_features.parquet"
TEST_TARGET = FEATURE_DIR / "test_target.parquet"


def main() -> None:
    print("Loading Logistic Regression baseline...")

    model = joblib.load(MODEL_PATH)

    X_test = pd.read_parquet(TEST_FEATURES)
    y_test = pd.read_parquet(TEST_TARGET)["payment_success"]

    print(f"Test rows: {len(X_test):,}")

    print("\nGenerating probability predictions...")

    probabilities = model.predict_proba(X_test)[:, 1]

    
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)
    brier = brier_score_loss(y_test, probabilities)

    failure_labels = 1 - y_test
    failure_probabilities = 1 - probabilities

    failure_pr_auc = average_precision_score(
        failure_labels,
        failure_probabilities,
    )

    failure_rate = failure_labels.mean()

    
    print("\nFailure Threshold Analysis")
    print("=" * 85)

    threshold_results = []

    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]:
        failure_predictions = (
            failure_probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            failure_labels,
            failure_predictions,
            labels=[0, 1],
        ).ravel()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "failures_caught": tp,
                "failures_missed": fn,
                "successful_payments_flagged": fp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    print(
        pd.DataFrame(threshold_results).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


    print("\nFailure Classification Metrics (threshold = 0.50)")
    print("=" * 50)

    print(
        classification_report(
            failure_labels,
            failure_predictions,
            target_names=["Success", "Failure"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            failure_labels,
            failure_predictions,
            labels=[0, 1],
        )
    )


    print("\nProbability Metrics")
    print("=" * 50)
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Failure PR-AUC: {failure_pr_auc:.4f}")
    print(f"Actual Failure Rate: {failure_rate:.4f}")

    print("\nProbability Summary")
    print("=" * 50)
    print(f"Minimum:     {probabilities.min():.4f}")
    print(f"Maximum:     {probabilities.max():.4f}")
    print(f"Mean:        {probabilities.mean():.4f}")
    print(f"Median:      {np.median(probabilities):.4f}")

    # Calculate calibration curve.
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test,
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

    calibration_csv = OUTPUT_DIR / "logistic_regression_calibration.csv"
    calibration_data.to_csv(calibration_csv, index=False)

    # Create calibration plot.
    plt.figure(figsize=(8, 6))

    plt.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        label="Logistic Regression",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Actual Success Rate")
    plt.title("Logistic Regression Calibration Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plot_path = OUTPUT_DIR / "logistic_regression_calibration.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nCalibration Data")
    print("=" * 50)
    print(calibration_data.to_string(index=False))

    print("\nEvaluation artifacts saved:")
    print(f"  Calibration CSV:   {calibration_csv}")
    print(f"  Calibration plot:  {plot_path}")

    print("\nLogistic Regression evaluation completed.")


if __name__ == "__main__":
    main()