from math import exp


METHOD_FAILURE_LOG_ODDS = {
    "upi": -2.20,
    "credit_card": -2.00,
    "debit_card": -1.85,
    "net_banking": -1.65,
    "wallet": -1.95,
}

NETWORK_FAILURE_EFFECT = {
    "poor": 0.90,
    "average": 0.30,
    "good": 0.00,
    "excellent": -0.20,
}

MERCHANT_METHOD_INTERACTION = {
    "electronics": {"credit_card": -0.35},
    "travel": {"credit_card": -0.40},
    "food": {"upi": -0.10},
    "grocery": {"upi": -0.10},
    "entertainment": {"wallet": -0.35},
    "healthcare": {"credit_card": -0.30},
    "education": {"net_banking": -0.60},
    "fashion": {"wallet": -0.35},
}

DEVICE_METHOD_INTERACTION = {
    "mobile": {"upi": -0.25},
    "desktop": {"credit_card": -0.10},
    "tablet": {"wallet": -0.10},
}


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def calculate_failure_probability(
    *,
    payment_method: str,
    merchant_category: str,
    network_quality: str,
    device_type: str,
) -> float:
    score = METHOD_FAILURE_LOG_ODDS[payment_method]

    score += NETWORK_FAILURE_EFFECT[network_quality]

    score += MERCHANT_METHOD_INTERACTION.get(
        merchant_category, {}
    ).get(payment_method, 0.0)

    score += DEVICE_METHOD_INTERACTION.get(
        device_type, {}
    ).get(payment_method, 0.0)

    return sigmoid(score)


def print_scenario(
    name: str,
    *,
    merchant_category: str,
    network_quality: str,
    device_type: str,
) -> None:
    probabilities = {
        method: calculate_failure_probability(
            payment_method=method,
            merchant_category=merchant_category,
            network_quality=network_quality,
            device_type=device_type,
        )
        for method in METHOD_FAILURE_LOG_ODDS
    }

    best_method = min(probabilities, key=probabilities.get)

    print(f"\n{name}")
    print(
        f"Context: merchant={merchant_category}, "
        f"network={network_quality}, device={device_type}"
    )

    for method, probability in sorted(probabilities.items(), key=lambda x: x[1]):
        print(f"  {method:<12} failure={probability:.4f}")

    print(f"  BEST METHOD: {best_method}")


scenarios = [
    (
        "Electronics + Desktop + Good Network",
        "electronics",
        "good",
        "desktop",
    ),
    (
        "Travel + Desktop + Good Network",
        "travel",
        "good",
        "desktop",
    ),
    (
        "Entertainment + Tablet + Good Network",
        "entertainment",
        "good",
        "tablet",
    ),
    (
        "Education + Desktop + Good Network",
        "education",
        "good",
        "desktop",
    ),
    (
        "Food + Mobile + Good Network",
        "food",
        "good",
        "mobile",
    ),
    (
        "Grocery + Mobile + Good Network",
        "grocery",
        "good",
        "mobile",
    ),
    (
        "Healthcare + Desktop + Good Network",
        "healthcare",
        "good",
        "desktop",
    ),
    (
        "Fashion + Tablet + Good Network",
        "fashion",
        "good",
        "tablet",
    ),
    (
        "Poor Network + Desktop",
        "electronics",
        "poor",
        "desktop",
    ),
    (
        "Excellent Network + Mobile",
        "electronics",
        "excellent",
        "mobile",
    ),
]


for scenario in scenarios:
    print_scenario(
        scenario[0],
        merchant_category=scenario[1],
        network_quality=scenario[2],
        device_type=scenario[3],
    )