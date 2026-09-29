import numpy as np
import pandas as pd

from payment_platform.data.generator import (
    METHOD_FAILURE_LOG_ODDS,
    NETWORK_FAILURE_EFFECT,
    generate_transactions,
)


METHODS = [
    "upi",
    "credit_card",
    "debit_card",
    "net_banking",
    "wallet",
]


NETWORK_METHOD_INTERACTION = {
    "upi": {
        "poor": 0.30,
        "average": 0.10,
        "good": 0.00,
        "excellent": -0.05,
    },
    "credit_card": {
        "poor": -0.10,
        "average": -0.05,
        "good": 0.00,
        "excellent": 0.00,
    },
    "debit_card": {
        "poor": 0.05,
        "average": 0.00,
        "good": 0.00,
        "excellent": 0.00,
    },
    "net_banking": {
        "poor": 0.10,
        "average": 0.05,
        "good": 0.00,
        "excellent": 0.00,
    },
    "wallet": {
        "poor": 0.05,
        "average": 0.00,
        "good": 0.00,
        "excellent": 0.00,
    },
}


MERCHANT_METHOD_INTERACTION = {
    "electronics": {
        "credit_card": -0.35,
    },
    "travel": {
        "credit_card": -0.40,
    },
    "food": {
        "upi": -0.10,
    },
    "grocery": {
        "upi": -0.10,
    },
    "entertainment": {
        "wallet": -0.35,
    },
    "healthcare": {
        "credit_card": -0.30,
    },
    "education": {
        "net_banking": -0.60,
    },
    "fashion": {
        "wallet": -0.35,
    },
}


DEVICE_METHOD_INTERACTION = {
    "mobile": {
        "upi": -0.10,
    },
    "desktop": {
        "credit_card": -0.10,
    },
    "tablet": {
        "wallet": -0.10,
    },
}


def get_network_interaction(
    df: pd.DataFrame,
    payment_method: str,
) -> np.ndarray:
    return np.array(
        [
            NETWORK_METHOD_INTERACTION[payment_method][quality]
            for quality in df["network_quality"]
        ]
    )


def get_merchant_interaction(
    df: pd.DataFrame,
    payment_method: str,
) -> np.ndarray:
    return np.array(
        [
            MERCHANT_METHOD_INTERACTION
            .get(category, {})
            .get(payment_method, 0.0)
            for category in df["merchant_category"]
        ]
    )


def get_device_interaction(
    df: pd.DataFrame,
    payment_method: str,
) -> np.ndarray:
    return np.array(
        [
            DEVICE_METHOD_INTERACTION
            .get(device, {})
            .get(payment_method, 0.0)
            for device in df["device_type"]
        ]
    )


def calculate_candidate_failure_probability(
    df: pd.DataFrame,
    payment_method: str,
) -> np.ndarray:
    method_effect = np.full(
        len(df),
        METHOD_FAILURE_LOG_ODDS[payment_method],
    )

    network_effect = np.array(
        [
            NETWORK_FAILURE_EFFECT[quality]
            for quality in df["network_quality"]
        ]
    )

    amount_effect = (
        np.log1p(df["amount"].to_numpy()) - 6.0
    ) * 0.18

    retry_effect = (
        df["retry_count"].to_numpy() * 0.35
    )

    velocity_effect = (
        np.maximum(
            df["transaction_velocity"].to_numpy() - 3,
            0,
        )
        * 0.12
    )

    user_history_effect = (
        0.90
        * (1.0 - df["user_method_success_rate"].to_numpy())
    )

    merchant_history_effect = (
        1.10
        * (1.0 - df["merchant_method_success_rate"].to_numpy())
    )

    late_night_effect = np.where(
        (df["hour_of_day"].to_numpy() <= 5)
        | (df["hour_of_day"].to_numpy() >= 23),
        0.20,
        0.0,
    )

    network_interaction = get_network_interaction(
        df,
        payment_method,
    )

    merchant_interaction = get_merchant_interaction(
        df,
        payment_method,
    )

    device_interaction = get_device_interaction(
        df,
        payment_method,
    )

    log_odds = (
        method_effect
        + network_effect
        + amount_effect
        + retry_effect
        + velocity_effect
        + user_history_effect
        + merchant_history_effect
        + late_night_effect
        + network_interaction
        + merchant_interaction
        + device_interaction
    )

    return np.clip(
        1.0 / (1.0 + np.exp(-log_odds)),
        0.001,
        0.999,
    )


def main() -> None:
    df = generate_transactions(
        n_transactions=100_000,
        seed=42,
    )

    probabilities = pd.DataFrame(
        {
            method: calculate_candidate_failure_probability(
                df,
                method,
            )
            for method in METHODS
        }
    )

    best_methods = probabilities.idxmin(axis=1)

    print("Overall expected failure rate:")
    print(
        f"{probabilities.mean(axis=1).mean() * 100:.2f}%"
    )

    print("\nExpected failure rate by method:")
    print(
        probabilities.mean()
        .mul(100)
        .round(2)
        .sort_values()
        .to_string()
    )

    print("\nBest ground-truth method distribution:")
    print(
        best_methods
        .value_counts()
        .to_string()
    )

    print("\nBest-method percentage:")
    print(
        best_methods
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_string()
    )

    print("\nBest method by network:")
    network_results = {}

    for network in df["network_quality"].unique():
        mask = df["network_quality"] == network
        network_results[network] = (
            probabilities.loc[mask].mean().idxmin()
        )

    print(
        pd.Series(network_results)
        .sort_index()
        .to_string()
    )

    print("\nBest method by merchant category:")
    merchant_results = {}

    for category in df["merchant_category"].unique():
        mask = df["merchant_category"] == category
        merchant_results[category] = (
            probabilities.loc[mask].mean().idxmin()
        )

    print(
        pd.Series(merchant_results)
        .sort_index()
        .to_string()
    )

    print("\nBest method by device:")
    device_results = {}

    for device in df["device_type"].unique():
        mask = df["device_type"] == device
        device_results[device] = (
            probabilities.loc[mask].mean().idxmin()
        )

    print(
        pd.Series(device_results)
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()