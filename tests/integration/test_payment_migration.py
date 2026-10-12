
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from payment_platform.db.connection import get_connection


MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "db"
    / "migrations"
    / "001_add_payment_attempt_uniqueness.sql"
)


@pytest.fixture
def migration_schema():
    """Create an isolated PostgreSQL schema for one migration test."""
    schema_name = f"migration_test_{uuid4().hex}"

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                psycopg.sql.SQL("CREATE SCHEMA {}").format(
                    psycopg.sql.Identifier(schema_name)
                )
            )
            cursor.execute(
                psycopg.sql.SQL(
                    """
                    CREATE TABLE {}.transactions (
                        transaction_id VARCHAR(100) PRIMARY KEY
                    )
                    """
                ).format(psycopg.sql.Identifier(schema_name))
            )
            cursor.execute(
                psycopg.sql.SQL(
                    """
                    CREATE TABLE {}.payment_attempts (
                        attempt_id BIGSERIAL PRIMARY KEY,
                        transaction_id VARCHAR(100) NOT NULL
                            REFERENCES {}.transactions(transaction_id),
                        payment_method VARCHAR(50) NOT NULL,
                        attempt_number INTEGER NOT NULL
                            CHECK (attempt_number > 0),
                        started_at TIMESTAMPTZ NOT NULL,
                        completed_at TIMESTAMPTZ,
                        outcome VARCHAR(50) NOT NULL
                    )
                    """
                ).format(
                    psycopg.sql.Identifier(schema_name),
                    psycopg.sql.Identifier(schema_name),
                )
            )
        conn.commit()
        yield schema_name
    finally:
        conn.rollback()
        with conn.cursor() as cursor:
            cursor.execute(
                psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(schema_name)
                )
            )
        conn.commit()
        conn.close()



def apply_migration(conn, schema_name: str) -> None:
    """Execute the migration against the isolated schema."""
    migration_sql = MIGRATION_PATH.read_text(encoding="utf-8")

    # Rewrite the regclass reference first, before replacing table references.
    migration_sql = migration_sql.replace(
        "'public.payment_attempts'::regclass",
        f"'{schema_name}.payment_attempts'::regclass",
    )

    migration_sql = migration_sql.replace(
        "public.payment_attempts",
        f'"{schema_name}".payment_attempts',
    )

    with conn.cursor() as cursor:
        cursor.execute(migration_sql)

    conn.commit()


def constraint_exists(conn, schema_name: str) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_constraint AS c
                JOIN pg_class AS t ON t.oid = c.conrelid
                JOIN pg_namespace AS n ON n.oid = t.relnamespace
                WHERE n.nspname = %s
                  AND t.relname = 'payment_attempts'
                  AND c.conname =
                      'uq_payment_attempt_transaction_number'
            )
            """,
            (schema_name,),
        )
        return cursor.fetchone()[0]


def test_migration_adds_unique_constraint(migration_schema):
    conn = get_connection()
    try:
        apply_migration(conn, migration_schema)
        assert constraint_exists(conn, migration_schema)
    finally:
        conn.close()


def test_migration_is_repeatable(migration_schema):
    conn = get_connection()
    try:
        apply_migration(conn, migration_schema)
        apply_migration(conn, migration_schema)
        assert constraint_exists(conn, migration_schema)
    finally:
        conn.close()


def test_migration_rejects_duplicates_without_deleting_rows(
    migration_schema,
):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                psycopg.sql.SQL(
                    """
                    INSERT INTO {}.transactions (transaction_id)
                    VALUES ('duplicate_test')
                    """
                ).format(psycopg.sql.Identifier(migration_schema))
            )
            cursor.execute(
                psycopg.sql.SQL(
                    """
                    INSERT INTO {}.payment_attempts (
                        transaction_id,
                        payment_method,
                        attempt_number,
                        started_at,
                        outcome
                    )
                    VALUES
                        ('duplicate_test', 'upi', 1, CURRENT_TIMESTAMP, 'failure'),
                        ('duplicate_test', 'upi', 1, CURRENT_TIMESTAMP, 'failure')
                    """
                ).format(psycopg.sql.Identifier(migration_schema))
            )
        conn.commit()

        with pytest.raises(psycopg.Error, match="Duplicate transaction_id"):
            apply_migration(conn, migration_schema)

        conn.rollback()

        with conn.cursor() as cursor:
            cursor.execute(
                psycopg.sql.SQL(
                    """
                    SELECT COUNT(*)
                    FROM {}.payment_attempts
                    WHERE transaction_id = 'duplicate_test'
                    """
                ).format(psycopg.sql.Identifier(migration_schema))
            )
            assert cursor.fetchone()[0] == 2

        assert not constraint_exists(conn, migration_schema)
    finally:
        conn.close()


def test_migration_rejects_incompatible_constraint(migration_schema):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                psycopg.sql.SQL(
                    """
                    ALTER TABLE {}.payment_attempts
                    ADD CONSTRAINT uq_payment_attempt_transaction_number
                    UNIQUE (attempt_id)
                    """
                ).format(psycopg.sql.Identifier(migration_schema))
            )
        conn.commit()

        with pytest.raises(
            psycopg.Error,
            match="Constraint exists with unexpected definition",
        ):
            apply_migration(conn, migration_schema)

        conn.rollback()
    finally:
        conn.close()
