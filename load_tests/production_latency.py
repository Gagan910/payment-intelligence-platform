
from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://payment-intelligence-platform.onrender.com"
HEALTH_URL = f"{BASE_URL}/health"
SAMPLES = 30
TIMEOUT_SECONDS = 60
OUTPUT_DIR = Path("load_tests/results")


def request_latency(url: str) -> tuple[float, int]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "payment-platform-latency-benchmark/1.0"},
    )
    started = time.perf_counter()

    try:
        with urllib.request.urlopen(
            request, timeout=TIMEOUT_SECONDS
        ) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        exc.read()
        status = exc.code
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return elapsed_ms, 0

    elapsed_ms = (time.perf_counter() - started) * 1000
    return elapsed_ms, status


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percent / 100)
    return ordered[index]


def main() -> None:
    print(f"Target: {HEALTH_URL}")
    print("Sending one warm-up request...")

    warmup_ms, warmup_status = request_latency(HEALTH_URL)
    print(f"Warm-up: HTTP {warmup_status}, {warmup_ms:.2f} ms")

    measurements: list[float] = []
    statuses: list[int] = []

    print(f"\nCollecting {SAMPLES} sequential requests...")

    for index in range(SAMPLES):
        elapsed_ms, status = request_latency(HEALTH_URL)
        measurements.append(elapsed_ms)
        statuses.append(status)
        print(
            f"{index + 1:02d}/{SAMPLES} "
            f"HTTP {status} | {elapsed_ms:.2f} ms"
        )

    successful = [
        latency
        for latency, status in zip(measurements, statuses)
        if 200 <= status < 300
    ]

    latency_summary = None

    if successful:
        latency_summary = {
            "mean": round(statistics.mean(successful), 2),
            "median": round(statistics.median(successful), 2),
            "p50": round(percentile(successful, 50), 2),
            "p95": round(percentile(successful, 95), 2),
            "p99": round(percentile(successful, 99), 2),
            "max": round(max(successful), 2),
        }

    results = {
        "target": HEALTH_URL,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "/health",
        "warmup_status": warmup_status,
        "warmup_latency_ms": round(warmup_ms, 2),
        "sample_count": SAMPLES,
        "successful_requests": len(successful),
        "failed_requests": SAMPLES - len(successful),
        "http_statuses": statuses,
        "latency_ms_successful_requests_only": latency_summary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "production_health_latency.json"
    output_file.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n--- Results ---")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
