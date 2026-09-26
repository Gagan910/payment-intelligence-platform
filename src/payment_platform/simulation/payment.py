from __future__ import annotations

import numpy as np
import pandas as pd

from payment_platform.data.generator import calculate_failure_probability


def simulate_payment_outcome(transaction: pd.Series) -> tuple[float, str]:
    """
    Simulate the ground-truth payment outcome for one transaction.

    Returns:
        failure_probability: Synthetic environment failure probability.
        outcome: "success" or "failure".

    The ground-truth simulation is independent of the ML prediction.
    """

    transaction_frame = pd.DataFrame([transaction.to_dict()])

    failure_probability = float(
        calculate_failure_probability(transaction_frame).iloc[0]
    )

    random_draw = float(np.random.default_rng().random())

    outcome = (
        "failure"
        if random_draw < failure_probability
        else "success"
    )

    return failure_probability, outcome