# Production Monitoring and Latency Benchmarks

## Health endpoint

Target: `https://payment-intelligence-platform.onrender.com/health`

The health endpoint supports both GET and HEAD requests. UptimeRobot monitors this endpoint at a five-minute interval.

## Historical health benchmark

File: `results/production_health_latency.json`

The saved benchmark contains 30 successful HTTP 200 responses, with a mean latency of 471.67 ms and p95 latency of 508.67 ms.

These measurements were collected before the latest health endpoint compatibility fix. They are historical observations, not a service-level objective (SLO) or availability guarantee.

## Health benchmark script

Script: `production_latency.py`

The current script calculates latency statistics from successful HTTP 2xx requests only. Its output uses the key `latency_ms_successful_requests_only`.

The historical JSON uses the older key `latency_ms`. This difference is intentional: the historical measurement is preserved without rewriting its original schema.

## Prediction benchmark

File: `results/production_prediction_latency.json`

The recorded sample contains five successful production prediction requests.

- Mean latency: 773.88 ms
- Median latency: 482.83 ms
- Minimum latency: 395.93 ms
- Maximum latency: 1,621.76 ms

This is a small initial sample, not an SLA or a statistically reliable percentile baseline. Each successful prediction request inserts a prediction record, so avoid unnecessary production benchmark runs.

## Interpretation and limitations

Client-observed latency includes client, network, hosting, API, and database-related overhead. These measurements do not isolate model inference latency.

A successful health check confirms endpoint responsiveness, not the health of every prediction, recommendation, or database operation.

Production latency and availability should be evaluated using repeated observations over time, with sample sizes and failure rates reported explicitly.

