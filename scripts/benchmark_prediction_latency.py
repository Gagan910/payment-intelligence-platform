from __future__ import annotations

import statistics
import time

from fastapi.testclient import TestClient

from payment_platform.api.app import app
from payment_platform.db.connection import get_connection


client = TestClient(app)

TRANSACTION_ID = "txn_prediction_benchmark"


REQUEST = {
    "transaction_id": TRANSACTION_ID,
    "amount": 500,
    "merchant_category": "electronics",
    "payment_method": "debit_card",
    "user_segment": "regular",
    "device_type": "mobile",
    "network_quality": "good",
    "hour_of_day": 14,
    "day_of_week": 2,
    "retry_count": 0,
    "transaction_velocity": 2,
    "user_method_success_rate": 0.90,
    "merchant_method_success_rate": 0.92,
}


def create_benchmark_transaction() -> None:
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
                ON CONFLICT (user_id) DO NOTHING
                """,
                (
                    "user_prediction_benchmark",
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
                ON CONFLICT (merchant_id) DO NOTHING
                """,
                (
                    "merchant_prediction_benchmark",
                    "electronics",
                    0.20,
                ),
            )

            cursor.execute(
                """
                INSERT INTO transactions (
                    transaction_id,
                    user_id,
                    merchant_id,
                    amount,
                    currency,
                    timestamp,
                    selected_payment_method,
                    experiment_variant,
                    status
                )
                VALUES (
                    %s, %s, %s, %s, %s, NOW(), %s, %s, %s
                )
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                (
                    TRANSACTION_ID,
                    "user_prediction_benchmark",
                    "merchant_prediction_benchmark",
                    500,
                    "INR",
                    "debit_card",
                    None,
                    "initiated",
                ),
            )

        conn.commit()

    finally:
        conn.close()


def cleanup_benchmark_transaction() -> None:
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM predictions
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
                ("user_prediction_benchmark",),
            )

            cursor.execute(
                """
                DELETE FROM merchants
                WHERE merchant_id = %s
                """,
                ("merchant_prediction_benchmark",),
            )

        conn.commit()

    finally:
        conn.close()


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)

    if not ordered:
        raise ValueError("Cannot calculate percentile of empty data.")

    index = (len(ordered) - 1) * percentile_value
    lower = int(index)
    upper = min(lower + 1, len(ordered))
    fraction = index - lower

    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def main() -> None:
    warmup_requests = 10
    measured_requests = 100

    cleanup_benchmark_transaction()
    create_benchmark_transaction()

    try:
        for _ in range(warmup_requests):
            response = client.post(
                "/predict",
                json=REQUEST,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Warm-up request failed: "
                    f"{response.status_code} {response.text}"
                )

        latencies_ms: list[float] = []

        for _ in range(measured_requests):
            started_at = time.perf_counter()

            response = client.post(
                "/predict",
                json=REQUEST,
            )

            completed_at = time.perf_counter()

            if response.status_code != 200:
                raise RuntimeError(
                    f"Benchmark request failed: "
                    f"{response.status_code} {response.text}"
                )

            latency_ms = (
                completed_at - started_at
            ) * 1000.0

            latencies_ms.append(latency_ms)

        print(f"Requests: {len(latencies_ms)}")
        print(f"Mean latency: {statistics.mean(latencies_ms):.3f} ms")
        print(
            f"Median latency: "
            f"{statistics.median(latencies_ms):.3f} ms"
        )
        print(
            f"P50 latency: "
            f"{percentile(latencies_ms, 0.50):.3f} ms"
        )
        print(
            f"P95 latency: "
            f"{percentile(latencies_ms, 0.95):.3f} ms"
        )
        print(
            f"P99 latency: "
            f"{percentile(latencies_ms, 0.99):.3f} ms"
        )
        print(
            f"Minimum latency: "
            f"{min(latencies_ms):.3f} ms"
        )
        print(
            f"Maximum latency: "
            f"{max(latencies_ms):.3f} ms"
        )

    finally:
        cleanup_benchmark_transaction()


if __name__ == "__main__":
    main()