from __future__ import annotations

from typing import Sequence

import pandas as pd

from payment_platform.data.features import get_model_features


DEFAULT_PAYMENT_METHODS = (
    "upi",
    "credit_card",
    "debit_card",
    "net_banking",
    "wallet",
)


def build_counterfactual_transactions(
    transaction: pd.Series,
    payment_methods: Sequence[str] = DEFAULT_PAYMENT_METHODS,
) -> pd.DataFrame:
    """
    Build counterfactual versions of one transaction.

    Every transaction attribute remains unchanged except payment_method.
    """
    if not payment_methods:
        raise ValueError("payment_methods cannot be empty.")

    if "payment_method" not in transaction.index:
        raise ValueError("Transaction must contain 'payment_method'.")

    counterfactuals = pd.DataFrame(
        [transaction.to_dict()] * len(payment_methods)
    )

    counterfactuals["payment_method"] = list(payment_methods)

    return counterfactuals.reset_index(drop=True)


def predict_counterfactual_failure_probabilities(
    model,
    transaction: pd.Series,
    payment_methods: Sequence[str] = DEFAULT_PAYMENT_METHODS,
) -> dict[str, float]:
    """
    Predict failure probability for each candidate payment method.

    The trained model predicts payment success probability. This function
    converts that output to failure probability before returning it.
    """
    counterfactuals = build_counterfactual_transactions(
        transaction=transaction,
        payment_methods=payment_methods,
    )

    model_features = get_model_features(counterfactuals)

    success_probabilities = model.predict_proba(model_features)[:, 1]

    failure_probabilities = 1.0 - success_probabilities

    return {
        method: float(probability)
        for method, probability in zip(
            payment_methods,
            failure_probabilities,
        )
    }