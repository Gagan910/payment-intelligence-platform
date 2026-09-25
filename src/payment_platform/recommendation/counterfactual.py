from __future__ import annotations

from typing import Sequence

import pandas as pd


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

    Parameters
    ----------
    transaction:
        Original transaction context.

    payment_methods:
        Payment methods to evaluate.

    Returns
    -------
    pd.DataFrame
        One row per candidate payment method.
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

    The supplied model must expose ``predict_proba``.

    Returns
    -------
    dict[str, float]
        Mapping of payment method to predicted failure probability.
    """
    counterfactuals = build_counterfactual_transactions(
        transaction=transaction,
        payment_methods=payment_methods,
    )

    probabilities = model.predict_proba(counterfactuals)[:, 1]

    return {
        method: float(probability)
        for method, probability in zip(payment_methods, probabilities)
    }