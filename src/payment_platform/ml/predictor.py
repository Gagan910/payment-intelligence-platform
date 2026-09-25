from pathlib import Path

import joblib
import pandas as pd

from payment_platform.data.features import get_model_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_baseline.joblib"
MODEL_VERSION = "logistic_regression_baseline_v1"


class PaymentPredictor:
    """Load and serve the trained payment failure model."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        model_version: str = MODEL_VERSION,
    ) -> None:
        self.model_path = model_path
        self.model_version = model_version
        self.model = joblib.load(self.model_path)

    def predict_failure_probability(
        self,
        transaction: pd.DataFrame,
    ) -> float:
        """Return predicted payment failure probability."""
        model_features = get_model_features(transaction)

        success_probability = float(
            self.model.predict_proba(model_features)[0, 1]
        )

        return 1.0 - success_probability