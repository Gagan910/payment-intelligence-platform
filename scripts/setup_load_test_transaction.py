from __future__ import annotations

from fastapi.testclient import TestClient

from payment_platform.api.app import app
from payment_platform.db.connection import get_connection
from payment_platform.db.repositories.transaction_context import (
    create_transaction_context,
)


LOAD_TEST_COUNT = 50
TRANSACTION_PREFIX = "load_test_prediction_"
USER_PREFIX = "load_test_user_"
MERCHANT_PREFIX = "load_test_merchant_"


def cleanup() -> None:
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            for index in range(LOAD_TEST_COUNT):
                transaction_id = f"{TRANSACTION_PREFIX}{index}"
                user_id = f"{USER_PREFIX}{index}"
                merchant_id = f"{MERCHANT_PREFIX}{index}"

                cursor.execute(
                    """
                    DELETE FROM predictions
                    WHERE transaction_id = %s
                    """,
                    (transaction_id,),
                )

                cursor.execute(
                    """
                    DELETE FROM transaction_context
                    WHERE transaction_id = %s
                    """,
                    (transaction_id,),
                )

                cursor.execute(
                    """
                    DELETE FROM transactions
                    WHERE transaction_id = %s
                    """,
                    (transaction_id,),
                )

                cursor.execute(
                    """
                    DELETE FROM users
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )

                cursor.execute(
                    """
                    DELETE FROM merchants
                    WHERE merchant_id = %s
                    """,
                    (merchant_id,),
                )

        conn.commit()

    finally:
        conn.close()


def create_parent_records() -> None:
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            for index in range(LOAD_TEST_COUNT):
                user_id = f"{USER_PREFIX}{index}"
                merchant_id = f"{MERCHANT_PREFIX}{index}"

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
                        merchant_id,
                        "electronics",
                        0.20,
                    ),
                )

        conn.commit()

    finally:
        conn.close()


def create_transactions() -> None:
    client = TestClient(app)

    for index in range(LOAD_TEST_COUNT):
        transaction_id = f"{TRANSACTION_PREFIX}{index}"
        user_id = f"{USER_PREFIX}{index}"
        merchant_id = f"{MERCHANT_PREFIX}{index}"

        response = client.post(
            "/transactions",
            json={
                "transaction_id": transaction_id,
                "user_id": user_id,
                "merchant_id": merchant_id,
                "amount": 500,
                "currency": "INR",
                "selected_payment_method": "debit_card",
                "experiment_variant": "control",
                "status": "initiated",
            },
        )

        response.raise_for_status()


def create_transaction_contexts() -> None:
    conn = get_connection()

    try:
        for index in range(LOAD_TEST_COUNT):
            transaction_id = f"{TRANSACTION_PREFIX}{index}"

            create_transaction_context(
                conn,
                transaction_id=transaction_id,
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
    cleanup()
    create_parent_records()
    create_transactions()
    create_transaction_contexts()

    print("Load-test transactions created successfully.")
    print(f"Transaction count: {LOAD_TEST_COUNT}")
    print(
        f"Transaction IDs: "
        f"{TRANSACTION_PREFIX}0 -> "
        f"{TRANSACTION_PREFIX}{LOAD_TEST_COUNT - 1}"
    )


if __name__ == "__main__":
    main()