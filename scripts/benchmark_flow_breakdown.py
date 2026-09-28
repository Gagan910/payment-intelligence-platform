from __future__ import annotations

import statistics
import time

from fastapi.testclient import TestClient

from payment_platform.api.app import app
from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.transaction_context import (
    create_transaction_context,
)


WARMUP_REQUESTS = 10
MEASURED_REQUESTS = 100

TRANSACTION_ID = "benchmark_flow_breakdown"
USER_ID = "benchmark_flow_breakdown_user"
MERCHANT_ID = "benchmark_flow_breakdown_merchant"


PREDICTION_REQUEST = {
    "transaction_id": TRANSACTION_ID,
    "amount": 500,
    "merchant_category": "electronics",
    "payment_method": "debit_card",
    "user_segment": "regular",
    "device_type": "mobile",
    "network_quality": "poor",
    "hour_of_day": 0,
    "day_of_week": 2,
    "retry_count": 0,
    "transaction_velocity": 2,
    "user_method_success_rate": 0.90,
    "merchant_method_success_rate": 0.92,
}


def percentile(values: list[float], percentile_value: float) -> float:
    sorted_values = sorted(values)

    index = (len(sorted_values) - 1) * (percentile_value / 100)

    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)

    weight = index - lower

    return (
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * weight
    )


def cleanup_test_data() -> None:
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM payment_attempts
                WHERE transaction_id = %s
                """,
                (TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM recommendations
                WHERE transaction_id = %s
                """,
                (TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM predictions
                WHERE transaction_id = %s
                """,
                (TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM transaction_context
                WHERE transaction_id = %s
                """,
                (TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM transactions
                WHERE transaction_id = %s
                """,
                (TRANSACTION_ID,),
            )

            cursor.execute(
                """
                DELETE FROM users
                WHERE user_id = %s
                """,
                (USER_ID,),
            )

            cursor.execute(
                """
                DELETE FROM merchants
                WHERE merchant_id = %s
                """,
                (MERCHANT_ID,),
            )

        conn.commit()

    finally:
        conn.close()


def create_parent_records() -> None:
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
                    USER_ID,
                    "regular",
                    "debit_card",
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
                    MERCHANT_ID,
                    "electronics",
                    0.20,
                ),
            )

        conn.commit()

    finally:
        conn.close()


def create_benchmark_transaction(client: TestClient) -> None:
    response = client.post(
        "/transactions",
        json={
            "transaction_id": TRANSACTION_ID,
            "user_id": USER_ID,
            "merchant_id": MERCHANT_ID,
            "amount": 500,
            "currency": "INR",
            "selected_payment_method": "debit_card",
            "experiment_variant": "control",
            "status": "initiated",
        },
    )

    response.raise_for_status()

    conn = get_connection()

    try:
        create_transaction_context(
            conn,
            transaction_id=TRANSACTION_ID,
            device_type="mobile",
            network_quality="poor",
            retry_count=0,
            transaction_velocity=2,
            user_method_success_rate=0.90,
            merchant_method_success_rate=0.92,
        )
    finally:
        conn.close()


def run_flow_with_breakdown(
    client: TestClient,
) -> dict[str, float]:
    timings: dict[str, float] = {}

    # ---------------------------------------------------------
    # 1. Prediction
    # ---------------------------------------------------------
    start = time.perf_counter()

    prediction_response = client.post(
        "/predict",
        json=PREDICTION_REQUEST,
    )

    timings["prediction"] = (
        time.perf_counter() - start
    ) * 1000.0

    prediction_response.raise_for_status()

    # ---------------------------------------------------------
    # 2. Recommendation
    # ---------------------------------------------------------
    start = time.perf_counter()

    recommendation_response = client.post(
        "/recommend",
        json=PREDICTION_REQUEST,
    )

    timings["recommendation"] = (
        time.perf_counter() - start
    ) * 1000.0

    recommendation_response.raise_for_status()

    recommendation_data = recommendation_response.json()

    recommendation_id = recommendation_data["recommendation_id"]

    # ---------------------------------------------------------
    # 3. Recommendation decision
    # ---------------------------------------------------------
    start = time.perf_counter()

    decision_response = client.post(
        "/recommend/decision",
        json={
            "recommendation_id": recommendation_id,
            "accepted": True,
        },
    )

    timings["recommendation_decision"] = (
        time.perf_counter() - start
    ) * 1000.0

    decision_response.raise_for_status()

    decision_data = decision_response.json()

    selected_payment_method = decision_data[
        "selected_payment_method"
    ]

    # ---------------------------------------------------------
    # 4. Payment attempt
    # ---------------------------------------------------------
    start = time.perf_counter()

    payment_attempt_response = client.post(
        "/payment-attempts",
        json={
            "transaction_id": TRANSACTION_ID,
            "payment_method": selected_payment_method,
            "attempt_number": 1,
        },
    )

    timings["payment_attempt"] = (
        time.perf_counter() - start
    ) * 1000.0

    payment_attempt_response.raise_for_status()

    timings["total"] = sum(timings.values())

    return timings


def print_statistics(
    name: str,
    values: list[float],
) -> None:
    print(
        f"{name:<24}"
        f"Mean: {statistics.mean(values):>9.3f} ms   "
        f"P50: {percentile(values, 50):>9.3f} ms   "
        f"P95: {percentile(values, 95):>9.3f} ms   "
        f"P99: {percentile(values, 99):>9.3f} ms"
    )


def main() -> None:
    cleanup_test_data()

    client = TestClient(app)

    create_parent_records()
    create_benchmark_transaction(client)

    try:
        # -----------------------------------------------------
        # Warmup
        # -----------------------------------------------------
        for _ in range(WARMUP_REQUESTS):
            run_flow_with_breakdown(client)

        measurements = {
            "prediction": [],
            "recommendation": [],
            "recommendation_decision": [],
            "payment_attempt": [],
            "total": [],
        }

        # -----------------------------------------------------
        # Measurement
        # -----------------------------------------------------
        for _ in range(MEASURED_REQUESTS):
            timings = run_flow_with_breakdown(client)

            for stage, latency in timings.items():
                measurements[stage].append(latency)

        print()
        print("End-to-end payment flow latency breakdown")
        print("-----------------------------------------")
        print(f"Warmup flows:       {WARMUP_REQUESTS}")
        print(f"Measured flows:     {MEASURED_REQUESTS}")
        print()

        print_statistics(
            "Prediction",
            measurements["prediction"],
        )

        print_statistics(
            "Recommendation",
            measurements["recommendation"],
        )

        print_statistics(
            "Recommendation decision",
            measurements["recommendation_decision"],
        )

        print_statistics(
            "Payment attempt",
            measurements["payment_attempt"],
        )

        print_statistics(
            "TOTAL FLOW",
            measurements["total"],
        )

        print()

        mean_total = statistics.mean(measurements["total"])

        mean_components = sum(
            statistics.mean(measurements[stage])
            for stage in (
                "prediction",
                "recommendation",
                "recommendation_decision",
                "payment_attempt",
            )
        )

        print("Mean latency composition")
        print("------------------------")

        for stage in (
            "prediction",
            "recommendation",
            "recommendation_decision",
            "payment_attempt",
        ):
            mean_stage = statistics.mean(measurements[stage])
            percentage = (mean_stage / mean_total) * 100

            print(
                f"{stage:<24}"
                f"{mean_stage:>9.3f} ms   "
                f"{percentage:>6.2f}%"
            )

        print(
            f"{'Component sum':<24}"
            f"{mean_components:>9.3f} ms"
        )

        print(
            f"{'Measured total':<24}"
            f"{mean_total:>9.3f} ms"
        )

    finally:
        cleanup_test_data()


if __name__ == "__main__":
    main()