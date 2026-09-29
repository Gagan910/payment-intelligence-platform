from __future__ import annotations

import psycopg


def create_transaction_context(
    conn: psycopg.Connection,
    *,
    transaction_id: str,
    device_type: str,
    network_quality: str,
    retry_count: int,
    transaction_velocity: int,
    user_method_success_rate: float,
    merchant_method_success_rate: float,
    commit: bool = True,
) -> None:
    """Insert the pre-attempt context for a transaction."""
    query = """
        INSERT INTO transaction_context (
            transaction_id,
            device_type,
            network_quality,
            retry_count,
            transaction_velocity,
            user_method_success_rate,
            merchant_method_success_rate
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s
        )
    """

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                transaction_id,
                device_type,
                network_quality,
                retry_count,
                transaction_velocity,
                user_method_success_rate,
                merchant_method_success_rate,
            ),
        )

    if commit:
        conn.commit()


def get_transaction_context(
    conn: psycopg.Connection,
    *,
    transaction_id: str,
) -> dict | None:
    """Retrieve the pre-attempt context for a transaction."""
    query = """
        SELECT
            transaction_id,
            device_type,
            network_quality,
            retry_count,
            transaction_velocity,
            user_method_success_rate,
            merchant_method_success_rate
        FROM transaction_context
        WHERE transaction_id = %s
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (transaction_id,))
        row = cursor.fetchone()

    if row is None:
        return None

    return {
        "transaction_id": row[0],
        "device_type": row[1],
        "network_quality": row[2],
        "retry_count": row[3],
        "transaction_velocity": row[4],
        "user_method_success_rate": float(row[5]),
        "merchant_method_success_rate": float(row[6]),
    }