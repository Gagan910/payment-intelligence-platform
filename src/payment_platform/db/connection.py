from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATABASE_URL = (
    "postgresql://postgres:postgres@127.0.0.1:5432/payment_intelligence"
)


def get_database_url() -> str:
    """Return the PostgreSQL connection URL from the environment."""
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_connection() -> psycopg.Connection:
    """Create and return a PostgreSQL database connection."""
    return psycopg.connect(get_database_url())