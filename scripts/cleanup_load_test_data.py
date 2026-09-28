from __future__ import annotations

from payment_platform.db.connection import get_connection


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
                    "DELETE FROM predictions WHERE transaction_id = %s",
                    (transaction_id,),
                )

                cursor.execute(
                    "DELETE FROM payment_attempts WHERE transaction_id = %s",
                    (transaction_id,),
                )

                cursor.execute(
                    "DELETE FROM recommendations WHERE transaction_id = %s",
                    (transaction_id,),
                )

                cursor.execute(
                    "DELETE FROM transaction_context WHERE transaction_id = %s",
                    (transaction_id,),
                )

                cursor.execute(
                    "DELETE FROM transactions WHERE transaction_id = %s",
                    (transaction_id,),
                )

                cursor.execute(
                    "DELETE FROM users WHERE user_id = %s",
                    (user_id,),
                )

                cursor.execute(
                    "DELETE FROM merchants WHERE merchant_id = %s",
                    (merchant_id,),
                )

        conn.commit()

    finally:
        conn.close()


def main() -> None:
    cleanup()
    print("Load-test data cleaned successfully.")
    print(f"Removed load-test records for {LOAD_TEST_COUNT} transaction slots.")


if __name__ == "__main__":
    main()