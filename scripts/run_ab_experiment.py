from __future__ import annotations

import argparse
import random
import uuid
from datetime import datetime
from typing import Any

from fastapi.testclient import TestClient

from payment_platform.api.app import app
from payment_platform.experiments.config import EXPERIMENT_ID


client = TestClient(app)


MERCHANT_CATEGORIES = (
    "electronics",
    "grocery",
    "fashion",
    "travel",
    "food",
    "entertainment",
    "healthcare",
    "education",
)

USER_SEGMENTS = (
    "new",
    "regular",
    "frequent",
)

DEVICE_TYPES = (
    "mobile",
    "desktop",
    "tablet",
)

NETWORK_QUALITY = (
    "poor",
    "average",
    "good",
    "excellent",
)

PAYMENT_METHOD = "debit_card"

RANDOM_SEED = 42


def create_parent_records(
    *,
    user_id: str,
    merchant_id: str,
    user_segment: str,
    merchant_category: str,
) -> None:
    from payment_platform.db.connection import get_connection

    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (
                    user_id,
                    user_segment,
                    preferred_payment_method
                )
                VALUES (%s, %s, %s)
                """,
                (
                    user_id,
                    user_segment,
                    PAYMENT_METHOD,
                ),
            )

            cursor.execute(
                """
                INSERT INTO merchants (
                    merchant_id,
                    merchant_category,
                    merchant_risk_score
                )
                VALUES (%s, %s, %s)
                """,
                (
                    merchant_id,
                    merchant_category,
                    0.20,
                ),
            )

        conn.commit()

    finally:
        conn.close()


def build_transaction_context(
    rng: random.Random,
) -> dict[str, Any]:
    return {
        "device_type": rng.choice(DEVICE_TYPES),
        "network_quality": rng.choice(NETWORK_QUALITY),
        "retry_count": rng.randint(0, 2),
        "transaction_velocity": rng.randint(1, 8),
        "user_method_success_rate": round(
            rng.uniform(0.60, 0.98),
            4,
        ),
        "merchant_method_success_rate": round(
            rng.uniform(0.60, 0.98),
            4,
        ),
    }


def run_single_transaction(
    *,
    run_id: str,
    index: int,
    rng: random.Random,
) -> dict[str, Any]:
    transaction_id = (
        f"{run_id}_{uuid.uuid4().hex}"
    )

    user_id = (
        f"ab_user_{uuid.uuid4().hex}"
    )

    merchant_id = (
        f"ab_merchant_{uuid.uuid4().hex}"
    )

    merchant_category = rng.choice(
        MERCHANT_CATEGORIES
    )

    user_segment = rng.choice(
        USER_SEGMENTS
    )

    amount = round(
        rng.uniform(100, 5000),
        2,
    )

    context = build_transaction_context(rng)

    create_parent_records(
        user_id=user_id,
        merchant_id=merchant_id,
        user_segment=user_segment,
        merchant_category=merchant_category,
    )

    transaction_request = {
        "transaction_id": transaction_id,
        "user_id": user_id,
        "merchant_id": merchant_id,
        "amount": amount,
        "currency": "INR",
        "selected_payment_method": PAYMENT_METHOD,
        "status": "initiated",
        "context": context,
    }

    transaction_response = client.post(
        "/transactions",
        json=transaction_request,
    )

    if transaction_response.status_code != 200:
        raise RuntimeError(
            "Transaction creation failed: "
            f"{transaction_response.status_code} "
            f"{transaction_response.text}"
        )

    transaction_data = transaction_response.json()
    variant = transaction_data["experiment_variant"]

    hour_of_day = rng.randint(0, 23)
    day_of_week = index % 7

    recommendation_request = {
        "transaction_id": transaction_id,
        "amount": amount,
        "merchant_category": merchant_category,
        "payment_method": PAYMENT_METHOD,
        "user_segment": user_segment,
        "device_type": context["device_type"],
        "network_quality": context["network_quality"],
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "retry_count": context["retry_count"],
        "transaction_velocity": context[
            "transaction_velocity"
        ],
        "user_method_success_rate": context[
            "user_method_success_rate"
        ],
        "merchant_method_success_rate": context[
            "merchant_method_success_rate"
        ],
    }

    recommendation_response = client.post(
        "/recommend",
        json=recommendation_request,
    )

    if recommendation_response.status_code != 200:
        raise RuntimeError(
            "Recommendation failed: "
            f"{recommendation_response.status_code} "
            f"{recommendation_response.text}"
        )

    recommendation_data = recommendation_response.json()
    recommendation_id = recommendation_data[
        "recommendation_id"
    ]

    # This runner measures the primary payment-success outcome.
    # It does not measure natural user recommendation acceptance.
    accepted = True

    decision_response = client.post(
        "/recommend/decision",
        json={
            "recommendation_id": recommendation_id,
            "accepted": accepted,
        },
    )

    if decision_response.status_code != 200:
        raise RuntimeError(
            "Recommendation decision failed: "
            f"{decision_response.status_code} "
            f"{decision_response.text}"
        )

    decision_data = decision_response.json()

    selected_payment_method = decision_data[
        "selected_payment_method"
    ]

    payment_attempt_response = client.post(
        "/payment-attempts",
        json={
            "transaction_id": transaction_id,
            "payment_method": selected_payment_method,
            "attempt_number": 1,
        },
    )

    if payment_attempt_response.status_code != 200:
        raise RuntimeError(
            "Payment attempt failed: "
            f"{payment_attempt_response.status_code} "
            f"{payment_attempt_response.text}"
        )

    payment_data = payment_attempt_response.json()

    return {
        "transaction_id": transaction_id,
        "variant": variant,
        "recommendation_action": recommendation_data[
            "recommendation_action"
        ],
        "recommended_method": recommendation_data[
            "recommended_method"
        ],
        "accepted": accepted,
        "selected_payment_method": selected_payment_method,
        "outcome": payment_data["outcome"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run synthetic traffic through the payment "
            "routing A/B experiment."
        )
    )

    parser.add_argument(
        "--transactions",
        type=int,
        default=1800,
        help="Number of synthetic transactions to execute.",
    )

    args = parser.parse_args()

    if args.transactions <= 0:
        raise ValueError(
            "--transactions must be greater than zero"
        )

    rng = random.Random(RANDOM_SEED)

    run_id = (
        f"ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{uuid.uuid4().hex[:8]}"
    )

    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Run ID: {run_id}")
    print(f"Transactions: {args.transactions}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    results: list[dict[str, Any]] = []

    for index in range(args.transactions):
        result = run_single_transaction(
            run_id=run_id,
            index=index,
            rng=rng,
        )

        results.append(result)

        print(
            f"{index + 1}/{args.transactions} "
            f"| variant={result['variant']} "
            f"| action={result['recommendation_action']} "
            f"| selected={result['selected_payment_method']} "
            f"| outcome={result['outcome']}"
        )

    print()
    print("Experiment execution complete.")
    print(f"Run ID: {run_id}")
    print()

    for variant in ("control", "treatment"):
        variant_results = [
            result
            for result in results
            if result["variant"] == variant
        ]

        if not variant_results:
            print(f"{variant}: no observations")
            continue

        successes = sum(
            result["outcome"] == "success"
            for result in variant_results
        )

        success_rate = (
            successes / len(variant_results)
        )

        print(
            f"{variant}: "
            f"n={len(variant_results)}, "
            f"successes={successes}, "
            f"success_rate={success_rate:.4f}"
        )


if __name__ == "__main__":
    main()