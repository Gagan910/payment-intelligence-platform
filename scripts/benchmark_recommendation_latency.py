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

TRANSACTION_ID = "benchmark_recommendation_latency"
USER_ID = "benchmark_recommendation_user"
MERCHANT_ID = "benchmark_recommendation_merchant"


REQUEST_PAYLOAD = {
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
                DELETE FROM recommendations
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


def main() -> None:
    cleanup_test_data()

    client = TestClient(app)

    create_parent_records()
    create_benchmark_transaction(client)

    try:
        # Warmup
        for _ in range(WARMUP_REQUESTS):
            response = client.post(
                "/recommend",
                json=REQUEST_PAYLOAD,
            )
            response.raise_for_status()

        # Measurement
        latencies_ms: list[float] = []

        for _ in range(MEASURED_REQUESTS):
            start = time.perf_counter()

            response = client.post(
                "/recommend",
                json=REQUEST_PAYLOAD,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            response.raise_for_status()

            latencies_ms.append(elapsed_ms)

        print()
        print("Recommendation API latency benchmark")
        print("-------------------------------------")
        print(f"Warmup requests:   {WARMUP_REQUESTS}")
        print(f"Measured requests: {MEASURED_REQUESTS}")
        print(f"Mean:   {statistics.mean(latencies_ms):.3f} ms")
        print(f"P50:    {percentile(latencies_ms, 50):.3f} ms")
        print(f"P95:    {percentile(latencies_ms, 95):.3f} ms")
        print(f"P99:    {percentile(latencies_ms, 99):.3f} ms")
        print(f"Min:    {min(latencies_ms):.3f} ms")
        print(f"Max:    {max(latencies_ms):.3f} ms")

    finally:
        cleanup_test_data()


if __name__ == "__main__":
    main()