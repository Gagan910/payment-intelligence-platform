from __future__ import annotations

from datetime import datetime

import psycopg


def create_transaction(
    conn: psycopg.Connection,
    *,
    transaction_id: str,
    user_id: str,
    merchant_id: str,
    amount: float,
    currency: str,
    timestamp: datetime,
    selected_payment_method: str,
    experiment_variant: str | None,
    status: str,
) -> None:
    """Insert a transaction into the database."""

    query = """
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
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (
                transaction_id,
                user_id,
                merchant_id,
                amount,
                currency,
                timestamp,
                selected_payment_method,
                experiment_variant,
                status,
            ),
        )

    conn.commit()


def get_transaction(
    conn: psycopg.Connection,
    *,
    transaction_id: str,
) -> dict | None:
    """Retrieve a transaction by its transaction ID."""

    query = """
        SELECT
            transaction_id,
            user_id,
            merchant_id,
            amount,
            currency,
            timestamp,
            selected_payment_method,
            experiment_variant,
            status
        FROM transactions
        WHERE transaction_id = %s
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (transaction_id,))
        row = cursor.fetchone()

    if row is None:
        return None

    return {
        "transaction_id": row[0],
        "user_id": row[1],
        "merchant_id": row[2],
        "amount": float(row[3]),
        "currency": row[4],
        "timestamp": row[5],
        "selected_payment_method": row[6],
        "experiment_variant": row[7],
        "status": row[8],
    }


def update_transaction_status(
    conn: psycopg.Connection,
    *,
    transaction_id: str,
    status: str,
) -> None:
    """Update the lifecycle status of a transaction."""

    query = """
        UPDATE transactions
        SET status = %s
        WHERE transaction_id = %s
    """

    with conn.cursor() as cursor:
        cursor.execute(
            query,
            (status, transaction_id),
        )

    conn.commit()


def get_transaction_context(
    conn: psycopg.Connection,
    *,
    transaction_id: str,
    payment_method: str | None = None,
) -> dict | None:
    """Retrieve the complete pre-attempt context for a transaction."""

    query = """
        SELECT
            t.transaction_id,
            t.user_id,
            t.merchant_id,
            t.amount,
            t.timestamp,
            m.merchant_category,
            u.user_segment,
            tc.device_type,
            tc.network_quality,
            tc.retry_count,
            tc.transaction_velocity,
            tc.user_method_success_rate,
            tc.merchant_method_success_rate
        FROM transactions AS t
        INNER JOIN users AS u
            ON u.user_id = t.user_id
        INNER JOIN merchants AS m
            ON m.merchant_id = t.merchant_id
        INNER JOIN transaction_context AS tc
            ON tc.transaction_id = t.transaction_id
        WHERE t.transaction_id = %s
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (transaction_id,))
        row = cursor.fetchone()

    if row is None:
        return None

    resolved_payment_method = (
        payment_method
        if payment_method is not None
        else get_transaction_payment_method(
            conn,
            transaction_id=transaction_id,
        )
    )

    timestamp = row[4]

    return {
        "transaction_id": row[0],
        "user_id": row[1],
        "merchant_id": row[2],
        "amount": float(row[3]),
        "merchant_category": row[5],
        "user_segment": row[6],
        "payment_method": resolved_payment_method,
        "device_type": row[7],
        "network_quality": row[8],
        "hour_of_day": timestamp.hour,
        "day_of_week": timestamp.weekday(),
        "retry_count": row[9],
        "transaction_velocity": row[10],
        "user_method_success_rate": float(row[11]),
        "merchant_method_success_rate": float(row[12]),
    }


def get_transaction_payment_method(
    conn: psycopg.Connection,
    *,
    transaction_id: str,
) -> str | None:
    """Retrieve the currently selected payment method for a transaction."""

    query = """
        SELECT selected_payment_method
        FROM transactions
        WHERE transaction_id = %s
    """

    with conn.cursor() as cursor:
        cursor.execute(query, (transaction_id,))
        row = cursor.fetchone()

    if row is None:
        return None

    return row[0]