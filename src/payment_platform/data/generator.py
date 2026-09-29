from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import pandas as pd


PAYMENT_METHODS: Final[tuple[str, ...]] = (
    "upi",
    "credit_card",
    "debit_card",
    "net_banking",
    "wallet",
)

MERCHANT_CATEGORIES: Final[tuple[str, ...]] = (
    "electronics",
    "grocery",
    "fashion",
    "travel",
    "food",
    "entertainment",
    "healthcare",
    "education",
)

USER_SEGMENTS: Final[tuple[str, ...]] = (
    "new",
    "regular",
    "frequent",
)

DEVICE_TYPES: Final[tuple[str, ...]] = (
    "mobile",
    "desktop",
    "tablet",
)

NETWORK_QUALITY: Final[tuple[str, ...]] = (
    "poor",
    "average",
    "good",
    "excellent",
)


METHOD_FAILURE_LOG_ODDS: Final[dict[str, float]] = {
    "upi": -2.20,
    "credit_card": -2.00,
    "debit_card": -1.85,
    "net_banking": -1.65,
    "wallet": -1.95,
}


NETWORK_FAILURE_EFFECT: Final[dict[str, float]] = {
    "poor": 0.90,
    "average": 0.30,
    "good": 0.00,
    "excellent": -0.20,
}


MERCHANT_METHOD_INTERACTION: Final[dict[str, dict[str, float]]] = {
    "electronics": {"credit_card": -0.35},
    "travel": {"credit_card": -0.40},
    "food": {"upi": -0.10},
    "grocery": {"upi": -0.10},
    "entertainment": {"wallet": -0.35},
    "healthcare": {"credit_card": -0.30},
    "education": {"net_banking": -0.60},
    "fashion": {"wallet": -0.35},
}


DEVICE_METHOD_INTERACTION: Final[dict[str, dict[str, float]]] = {
    "mobile": {"upi": -0.25},
    "desktop": {"credit_card": -0.10},
    "tablet": {"wallet": -0.10},
}


@dataclass(frozen=True)
class TransactionContext:
    """Features available before the current payment attempt."""

    transaction_id: str
    timestamp: str
    user_id: str
    merchant_id: str
    amount: float
    merchant_category: str
    payment_method: str
    user_segment: str
    device_type: str
    network_quality: str
    hour_of_day: int
    day_of_week: int
    retry_count: int
    transaction_velocity: int
    user_method_success_rate: float
    merchant_method_success_rate: float


def _generate_user_method_success_rate(
    rng: np.random.Generator,
    user_segment: str,
) -> float:
    """Generate a historical user-method success rate."""

    segment_parameters = {
        "new": (8.5, 1.5),
        "regular": (16.0, 2.0),
        "frequent": (24.0, 1.8),
    }

    alpha, beta = segment_parameters[user_segment]
    return float(rng.beta(alpha, beta))


def _generate_merchant_method_success_rate(
    rng: np.random.Generator,
    merchant_category: str,
) -> float:
    """Generate a historical merchant-method success rate."""

    category_parameters = {
        "electronics": (18.0, 2.5),
        "grocery": (22.0, 2.0),
        "fashion": (20.0, 2.5),
        "travel": (16.0, 3.0),
        "food": (24.0, 2.0),
        "entertainment": (21.0, 2.5),
        "healthcare": (19.0, 2.5),
        "education": (23.0, 2.0),
    }

    alpha, beta = category_parameters[merchant_category]
    return float(rng.beta(alpha, beta))


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    """Convert a log-odds score into a probability."""

    return 1.0 / (1.0 + np.exp(-value))


def calculate_failure_probability(
    transactions: pd.DataFrame,
) -> pd.Series:
    """
    Calculate the synthetic ground-truth probability of payment failure.

    This represents the simulated payment environment, not an ML model.
    """

    required_columns = {
        "amount",
        "merchant_category",
        "payment_method",
        "device_type",
        "network_quality",
        "retry_count",
        "transaction_velocity",
        "user_method_success_rate",
        "merchant_method_success_rate",
        "hour_of_day",
    }

    missing_columns = required_columns - set(transactions.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    score = np.array(
        [
            METHOD_FAILURE_LOG_ODDS[method]
            for method in transactions["payment_method"]
        ],
        dtype=float,
    )

    # Larger transactions carry slightly higher synthetic risk.
    amount_effect = (
        np.log1p(transactions["amount"].to_numpy()) - 6.0
    ) * 0.18

    # Poor network conditions increase failure probability.
    network_effect = np.array(
        [
            NETWORK_FAILURE_EFFECT[quality]
            for quality in transactions["network_quality"]
        ],
        dtype=float,
    )

    # Merchant category can favor specific payment methods.
    merchant_method_effect = np.array(
        [
            MERCHANT_METHOD_INTERACTION.get(
                category,
                {},
            ).get(method, 0.0)
            for category, method in zip(
                transactions["merchant_category"],
                transactions["payment_method"],
            )
        ],
        dtype=float,
    )

    # Device type can favor specific payment methods.
    device_method_effect = np.array(
        [
            DEVICE_METHOD_INTERACTION.get(
                device,
                {},
            ).get(method, 0.0)
            for device, method in zip(
                transactions["device_type"],
                transactions["payment_method"],
            )
        ],
        dtype=float,
    )

    # Retry activity and high transaction velocity increase risk.
    retry_effect = transactions["retry_count"].to_numpy() * 0.35

    velocity_effect = (
        np.maximum(
            transactions["transaction_velocity"].to_numpy() - 3,
            0,
        )
        * 0.12
    )

    # Lower historical success rates increase current risk.
    user_history_effect = (
        0.90
        * (
            1.0
            - transactions["user_method_success_rate"].to_numpy()
        )
    )

    merchant_history_effect = (
        1.10
        * (
            1.0
            - transactions["merchant_method_success_rate"].to_numpy()
        )
    )

    # Mild temporal effect for late-night transactions.
    late_night_effect = np.where(
        (transactions["hour_of_day"].to_numpy() <= 5)
        | (transactions["hour_of_day"].to_numpy() >= 23),
        0.20,
        0.0,
    )

    log_odds = (
        score
        + amount_effect
        + network_effect
        + merchant_method_effect
        + device_method_effect
        + retry_effect
        + velocity_effect
        + user_history_effect
        + merchant_history_effect
        + late_night_effect
    )

    probabilities = _sigmoid(log_odds)

    return pd.Series(
        np.clip(probabilities, 0.001, 0.999),
        index=transactions.index,
        name="failure_probability",
    )


def simulate_payment_outcomes(
    transactions: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate payment outcomes from the synthetic ground-truth
    failure probabilities.

    Returns a copy of the input DataFrame with:
        failure_probability
        payment_success
    """

    rng = np.random.default_rng(seed)

    result = transactions.copy()

    failure_probability = calculate_failure_probability(result)

    random_draw = rng.random(len(result))

    payment_failure = (
        random_draw < failure_probability.to_numpy()
    )

    result["failure_probability"] = np.round(
        failure_probability,
        6,
    )

    result["payment_success"] = (
        ~payment_failure
    ).astype(int)

    return result


def generate_transactions(
    n_transactions: int,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic transaction contexts.

    This function generates only information that would be available
    before the current payment attempt. The payment outcome is not
    generated here.
    """

    if n_transactions <= 0:
        raise ValueError(
            "n_transactions must be greater than zero."
        )

    rng = np.random.default_rng(seed)

    user_count = max(
        1_000,
        n_transactions // 5,
    )

    merchant_count = max(
        100,
        n_transactions // 50,
    )

    user_ids = np.array(
        [
            f"user_{index:06d}"
            for index in range(1, user_count + 1)
        ]
    )

    merchant_ids = np.array(
        [
            f"merchant_{index:05d}"
            for index in range(1, merchant_count + 1)
        ]
    )

    selected_users = rng.choice(
        user_ids,
        size=n_transactions,
    )

    selected_merchants = rng.choice(
        merchant_ids,
        size=n_transactions,
    )

    merchant_categories = rng.choice(
        MERCHANT_CATEGORIES,
        size=n_transactions,
    )

    user_segments = rng.choice(
        USER_SEGMENTS,
        size=n_transactions,
        p=[0.20, 0.55, 0.25],
    )

    payment_methods = rng.choice(
        PAYMENT_METHODS,
        size=n_transactions,
        p=[0.40, 0.20, 0.18, 0.10, 0.12],
    )

    device_types = rng.choice(
        DEVICE_TYPES,
        size=n_transactions,
        p=[0.70, 0.25, 0.05],
    )

    network_quality = rng.choice(
        NETWORK_QUALITY,
        size=n_transactions,
        p=[0.08, 0.22, 0.50, 0.20],
    )

    # Log-normal distribution produces mostly small/medium payments
    # with a realistic long tail of larger transactions.
    amounts = np.round(
        rng.lognormal(
            mean=6.2,
            sigma=0.85,
            size=n_transactions,
        ),
        2,
    )

    amounts = np.clip(
        amounts,
        50.0,
        100_000.0,
    )

    timestamps = pd.to_datetime(
        rng.integers(
            pd.Timestamp("2024-01-01").value,
            pd.Timestamp("2025-12-31 23:59:59").value,
            size=n_transactions,
        )
    )

    hours = timestamps.hour
    days = timestamps.dayofweek

    retry_counts = rng.choice(
        [0, 1, 2],
        size=n_transactions,
        p=[0.88, 0.10, 0.02],
    )

    transaction_velocity = rng.poisson(
        lam=2.5,
        size=n_transactions,
    )

    user_success_rates = np.array(
        [
            _generate_user_method_success_rate(
                rng,
                segment,
            )
            for segment in user_segments
        ]
    )

    merchant_success_rates = np.array(
        [
            _generate_merchant_method_success_rate(
                rng,
                category,
            )
            for category in merchant_categories
        ]
    )

    transactions = pd.DataFrame(
        {
            "transaction_id": [
                f"txn_{index:08d}"
                for index in range(1, n_transactions + 1)
            ],
            "timestamp": timestamps,
            "user_id": selected_users,
            "merchant_id": selected_merchants,
            "amount": amounts,
            "merchant_category": merchant_categories,
            "payment_method": payment_methods,
            "user_segment": user_segments,
            "device_type": device_types,
            "network_quality": network_quality,
            "hour_of_day": hours,
            "day_of_week": days,
            "retry_count": retry_counts,
            "transaction_velocity": transaction_velocity,
            "user_method_success_rate": np.round(
                user_success_rates,
                4,
            ),
            "merchant_method_success_rate": np.round(
                merchant_success_rates,
                4,
            ),
        }
    )

    return transactions


def generate_transaction_contexts(
    n_transactions: int,
    *,
    seed: int = 42,
) -> list[TransactionContext]:
    """Generate transaction contexts as typed Python objects."""

    dataframe = generate_transactions(
        n_transactions=n_transactions,
        seed=seed,
    )

    return [
        TransactionContext(**row)
        for row in dataframe.to_dict(
            orient="records"
        )
    ]


def contexts_to_dataframe(
    contexts: list[TransactionContext],
) -> pd.DataFrame:
    """Convert transaction contexts to a DataFrame."""

    return pd.DataFrame(
        [
            asdict(context)
            for context in contexts
        ]
    )