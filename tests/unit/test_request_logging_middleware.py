import logging
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from payment_platform.api.app import application
from payment_platform.api.middleware import RequestLoggingMiddleware

client = TestClient(application)

def test_response_contains_valid_request_id() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None

    # Validate that the identifier is a UUID.
    UUID(request_id)


def test_each_request_receives_a_unique_request_id() -> None:
    first_response = client.get("/health")
    second_response = client.get("/health")

    first_request_id = first_response.headers["X-Request-ID"]
    second_request_id = second_response.headers["X-Request-ID"]

    assert first_request_id != second_request_id


def test_request_completion_is_logged(caplog) -> None:
    with caplog.at_level(
        logging.INFO,
        logger="payment_platform.api",
    ):
        response = client.get("/health")

    assert response.status_code == 200

    records = [
        record
        for record in caplog.records
        if record.name == "payment_platform.api"
        and record.getMessage() == "request_completed"
    ]
    assert records

    record = records[-1]
    assert record.request_id == response.headers["X-Request-ID"]
    assert record.method == "GET"
    assert record.path == "/health"
    assert record.status_code == 200
    assert record.duration_ms >= 0

def test_server_error_response_contains_request_id() -> None:

    test_app = FastAPI()
    test_app.add_middleware(RequestLoggingMiddleware)

    @test_app.get("/test-error")
    def raise_server_error() -> None:
        raise HTTPException(status_code=500, detail="Test error")

    error_client = TestClient(test_app, raise_server_exceptions=False)
    response = error_client.get("/test-error")

    assert response.status_code == 500
    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    UUID(request_id)

def test_unhandled_exception_is_logged_with_request_id(
    caplog,
) -> None:

    test_app = FastAPI()
    test_app.add_middleware(RequestLoggingMiddleware)

    @test_app.get("/test-unhandled-error")
    def raise_unhandled_error() -> None:
        raise RuntimeError("Simulated internal failure")

    error_client = TestClient(
        test_app,
        raise_server_exceptions=False,
    )

    with caplog.at_level(
        logging.ERROR,
        logger="payment_platform.api",
    ):
        response = error_client.get("/test-unhandled-error")

    assert response.status_code == 500

    records = [
        record
        for record in caplog.records
        if record.name == "payment_platform.api"
        and record.getMessage() == "request_failed"
    ]
    assert records

    record = records[-1]
    assert record.request_id
    UUID(record.request_id)
    assert record.method == "GET"
    assert record.path == "/test-unhandled-error"
    assert record.duration_ms >= 0
