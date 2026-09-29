from __future__ import annotations

import argparse
import math
from statistics import mean
from typing import Any

from payment_platform.db.connection import get_connection


EXPERIMENT_ID = "payment_routing_v1"

ALPHA = 0.05
Z_95 = 1.96


def proportion_confidence_interval(
    successes: int,
    total: int,
    z: float = Z_95,
) -> tuple[float, float]:
    """Calculate a Wilson confidence interval for one proportion."""

    if total <= 0:
        raise ValueError("total must be greater than zero")

    proportion = successes / total

    denominator = 1 + (z**2 / total)

    center = (
        proportion
        + (z**2 / (2 * total))
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (
                proportion * (1 - proportion)
                / total
            )
            + (z**2 / (4 * total**2))
        )
        / denominator
    )

    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
    )


def normal_cdf(value: float) -> float:
    """Standard normal cumulative distribution function."""

    return 0.5 * (
        1.0 + math.erf(
            value / math.sqrt(2.0)
        )
    )


def two_proportion_z_test(
    *,
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
) -> tuple[float, float]:
    """
    Two-sided two-proportion z-test.

    H0: treatment success rate == control success rate
    H1: treatment success rate != control success rate
    """

    if control_total <= 0 or treatment_total <= 0:
        raise ValueError(
            "Both variants must contain observations."
        )

    control_rate = (
        control_successes / control_total
    )

    treatment_rate = (
        treatment_successes / treatment_total
    )

    pooled_successes = (
        control_successes
        + treatment_successes
    )

    pooled_total = (
        control_total
        + treatment_total
    )

    pooled_rate = (
        pooled_successes / pooled_total
    )

    standard_error = math.sqrt(
        pooled_rate
        * (1 - pooled_rate)
        * (
            (1 / control_total)
            + (1 / treatment_total)
        )
    )

    if standard_error == 0:
        return 0.0, 1.0

    z_statistic = (
        treatment_rate - control_rate
    ) / standard_error

    p_value = 2.0 * (
        1.0 - normal_cdf(abs(z_statistic))
    )

    return z_statistic, p_value


def difference_confidence_interval(
    *,
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    z: float = Z_95,
) -> tuple[float, float]:
    """Calculate a 95% normal-approximation CI for rate difference."""

    control_rate = (
        control_successes / control_total
    )

    treatment_rate = (
        treatment_successes / treatment_total
    )

    difference = (
        treatment_rate - control_rate
    )

    standard_error = math.sqrt(
        (
            control_rate
            * (1 - control_rate)
            / control_total
        )
        + (
            treatment_rate
            * (1 - treatment_rate)
            / treatment_total
        )
    )

    margin = z * standard_error

    return (
        difference - margin,
        difference + margin,
    )


def percentile(
    values: list[float],
    percentile_value: float,
) -> float | None:
    """Calculate a linearly interpolated percentile."""

    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        (len(ordered) - 1)
        * percentile_value
    )

    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]

    weight = position - lower_index

    return (
        lower_value
        + weight * (upper_value - lower_value)
    )


def load_results(
    run_id: str,
) -> list[dict[str, Any]]:
    conn = get_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    t.transaction_id,
                    t.experiment_variant,
                    t.selected_payment_method,
                    r.original_method,
                    r.recommended_method,
                    r.accepted,
                    pa.payment_method,
                    pa.attempt_number,
                    pa.outcome,
                    pa.started_at,
                    pa.completed_at
                FROM transactions t
                LEFT JOIN recommendations r
                    ON r.transaction_id = t.transaction_id
                LEFT JOIN payment_attempts pa
                    ON pa.transaction_id = t.transaction_id
                WHERE
                    LEFT(t.transaction_id, %s) = %s
                ORDER BY t.transaction_id
                """,
                (
                    len(run_id),
                    run_id,
                ),
            )

            rows = cursor.fetchall()

    finally:
        conn.close()

    results: list[dict[str, Any]] = []

    for row in rows:
        (
            transaction_id,
            variant,
            initial_method,
            original_method,
            recommended_method,
            accepted,
            payment_method,
            attempt_number,
            outcome,
            started_at,
            completed_at,
        ) = row

        latency_ms = None

        if (
            started_at is not None
            and completed_at is not None
        ):
            latency_ms = (
                completed_at - started_at
            ).total_seconds() * 1000.0

        results.append(
            {
                "transaction_id": transaction_id,
                "variant": variant,
                "initial_method": initial_method,
                "original_method": original_method,
                "recommended_method": recommended_method,
                "accepted": accepted,
                "payment_method": payment_method,
                "attempt_number": attempt_number,
                "outcome": outcome,
                "latency_ms": latency_ms,
            }
        )

    return results


def calculate_variant_metrics(
    results: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    observations = [
        result
        for result in results
        if result["variant"] == variant
    ]

    total = len(observations)

    if total == 0:
        return {
            "variant": variant,
            "n": 0,
            "payment_observations": 0,
            "successes": 0,
            "failures": 0,
            "success_rate": None,
            "failure_rate": None,
            "ci_low": None,
            "ci_high": None,
            "recommendations": 0,
            "switches": 0,
            "keep_current": 0,
            "accepted": 0,
            "acceptance_rate": None,
            "recommendation_rate": None,
            "switch_rate": None,
            "completion_rate": None,
            "latency_mean_ms": None,
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "latency_p99_ms": None,
        }

    payment_observations = [
        result
        for result in observations
        if result["outcome"] in {
            "success",
            "failure",
        }
    ]

    successes = sum(
        result["outcome"] == "success"
        for result in payment_observations
    )

    failures = sum(
        result["outcome"] == "failure"
        for result in payment_observations
    )

    payment_total = len(payment_observations)

    if payment_total > 0:
        success_rate = successes / payment_total
        failure_rate = failures / payment_total

        ci_low, ci_high = proportion_confidence_interval(
            successes=successes,
            total=payment_total,
        )
    else:
        success_rate = None
        failure_rate = None
        ci_low = None
        ci_high = None

    recommendation_observations = [
        result
        for result in observations
        if result["original_method"] is not None
    ]

    recommendations = sum(
        result["recommended_method"] is not None
        for result in recommendation_observations
    )

    switches = sum(
        result["recommended_method"] is not None
        and result["recommended_method"]
        != result["original_method"]
        for result in recommendation_observations
    )

    keep_current = sum(
        result["recommended_method"] is None
        or result["recommended_method"]
        == result["original_method"]
        for result in recommendation_observations
    )

    accepted = sum(
        result["accepted"] is True
        and result["recommended_method"] is not None
        and result["recommended_method"]
        != result["original_method"]
        for result in recommendation_observations
    )

    recommendation_count = len(
        recommendation_observations
    )

    recommendation_rate = (
        recommendations / total
        if total > 0
        else None
    )

    switch_rate = (
        switches / total
        if total > 0
        else None
    )

    acceptance_rate = (
        accepted / switches
        if switches > 0
        else None
    )

    completion_rate = (
        payment_total / total
        if total > 0
        else None
    )

    latency_values = [
        result["latency_ms"]
        for result in payment_observations
        if result["latency_ms"] is not None
    ]

    return {
        "variant": variant,
        "n": total,
        "payment_observations": payment_total,
        "successes": successes,
        "failures": failures,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "recommendations": recommendations,
        "switches": switches,
        "keep_current": keep_current,
        "accepted": accepted,
        "acceptance_rate": acceptance_rate,
        "recommendation_rate": recommendation_rate,
        "switch_rate": switch_rate,
        "completion_rate": completion_rate,
        "latency_mean_ms": (
            mean(latency_values)
            if latency_values
            else None
        ),
        "latency_p50_ms": percentile(
            latency_values,
            0.50,
        ),
        "latency_p95_ms": percentile(
            latency_values,
            0.95,
        ),
        "latency_p99_ms": percentile(
            latency_values,
            0.99,
        ),
    }


def print_variant_metrics(
    metrics: dict[str, Any],
) -> None:
    print(f"{metrics['variant']}:")

    print(f"  n = {metrics['n']}")
    print(
        f"  payment_observations = "
        f"{metrics['payment_observations']}"
    )
    print(f"  successes = {metrics['successes']}")
    print(f"  failures = {metrics['failures']}")

    if metrics["success_rate"] is not None:
        print(
            f"  success_rate = "
            f"{metrics['success_rate']:.4f}"
        )
        print(
            f"  failure_rate = "
            f"{metrics['failure_rate']:.4f}"
        )
        print(
            f"  95% CI = "
            f"[{metrics['ci_low']:.4f}, "
            f"{metrics['ci_high']:.4f}]"
        )
    else:
        print("  success_rate = N/A")
        print("  failure_rate = N/A")

    print(
        f"  recommendation_events = "
        f"{metrics['recommendations']}"
    )

    print(
        f"  switches = "
        f"{metrics['switches']}"
    )

    print(
        f"  keep_current = "
        f"{metrics['keep_current']}"
    )

    if metrics["recommendation_rate"] is not None:
        print(
            f"  recommendation_rate = "
            f"{metrics['recommendation_rate']:.4f}"
        )
        print(
            f"  switch_rate = "
            f"{metrics['switch_rate']:.4f}"
        )
    else:
        print("  recommendation_rate = N/A")
        print("  switch_rate = N/A")

    print(
        f"  accepted_switches = "
        f"{metrics['accepted']}"
    )

    if metrics["acceptance_rate"] is not None:
        print(
            f"  acceptance_rate = "
            f"{metrics['acceptance_rate']:.4f}"
        )
    else:
        print("  acceptance_rate = N/A")

    if metrics["completion_rate"] is not None:
        print(
            f"  payment_completion_rate = "
            f"{metrics['completion_rate']:.4f}"
        )
    else:
        print("  payment_completion_rate = N/A")

    if metrics["latency_mean_ms"] is not None:
        print(
            f"  payment_latency_mean_ms = "
            f"{metrics['latency_mean_ms']:.3f}"
        )
        print(
            f"  payment_latency_p50_ms = "
            f"{metrics['latency_p50_ms']:.3f}"
        )
        print(
            f"  payment_latency_p95_ms = "
            f"{metrics['latency_p95_ms']:.3f}"
        )
        print(
            f"  payment_latency_p99_ms = "
            f"{metrics['latency_p99_ms']:.3f}"
        )
    else:
        print("  payment_latency = N/A")

    print()


def print_routing_distribution(
    results: list[dict[str, Any]],
) -> None:
    print("Treatment routing distribution:")

    treatment_results = [
        result
        for result in results
        if result["variant"] == "treatment"
    ]

    switches = [
        result
        for result in treatment_results
        if result["recommended_method"] is not None
        and result["recommended_method"]
        != result["original_method"]
    ]

    if not switches:
        print("  No treatment switches recorded.")
        print()
        return

    counts: dict[tuple[str, str], int] = {}

    for result in switches:
        key = (
            result["original_method"],
            result["recommended_method"],
        )

        counts[key] = counts.get(key, 0) + 1

    for (
        original_method,
        recommended_method,
    ), count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(
            f"  {original_method} -> "
            f"{recommended_method}: {count}"
        )

    print()


def print_unavailable_metrics() -> None:
    print("Unavailable experiment metrics:")

    print(
        "  abandonment_rate = N/A "
        "(no abandonment event is currently recorded)"
    )

    print(
        "  natural_user_acceptance_rate = N/A "
        "(runner currently forces accepted=True)"
    )

    print(
        "  time_to_success = N/A "
        "(not defined as an experiment metric)"
    )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate payment success, routing, "
            "latency, and guardrail metrics for "
            "a specific A/B experiment run."
        )
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Experiment run identifier.",
    )

    args = parser.parse_args()

    results = load_results(args.run_id)

    if not results:
        print(
            f"No experiment results found "
            f"for run_id={args.run_id}"
        )
        return

    control = calculate_variant_metrics(
        results,
        "control",
    )

    treatment = calculate_variant_metrics(
        results,
        "treatment",
    )

    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Run: {args.run_id}")
    print(f"Alpha: {ALPHA}")
    print()

    print_variant_metrics(control)
    print_variant_metrics(treatment)

    print_routing_distribution(results)

    print_unavailable_metrics()

    if (
        control["success_rate"] is None
        or treatment["success_rate"] is None
    ):
        print(
            "Statistical comparison skipped: "
            "both variants require payment observations."
        )
        return

    absolute_difference = (
        treatment["success_rate"]
        - control["success_rate"]
    )

    if control["success_rate"] > 0:
        relative_improvement = (
            absolute_difference
            / control["success_rate"]
        )
    else:
        relative_improvement = None

    ci_low, ci_high = difference_confidence_interval(
        control_successes=control["successes"],
        control_total=control["payment_observations"],
        treatment_successes=treatment["successes"],
        treatment_total=treatment["payment_observations"],
    )

    z_statistic, p_value = two_proportion_z_test(
        control_successes=control["successes"],
        control_total=control["payment_observations"],
        treatment_successes=treatment["successes"],
        treatment_total=treatment["payment_observations"],
    )

    statistically_significant = (
        p_value < ALPHA
    )

    print("Treatment vs control:")
    print(
        f"  absolute difference = "
        f"{absolute_difference:.4f}"
    )

    print(
        "  difference in percentage points = "
        f"{absolute_difference * 100:.2f}"
    )

    if relative_improvement is not None:
        print(
            f"  relative improvement = "
            f"{relative_improvement * 100:.2f}%"
        )
    else:
        print(
            "  relative improvement = N/A"
        )

    print(
        f"  95% CI for difference = "
        f"[{ci_low:.4f}, {ci_high:.4f}]"
    )

    print(
        f"  z-statistic = "
        f"{z_statistic:.4f}"
    )

    print(
        f"  p-value = "
        f"{p_value:.6f}"
    )

    print(
        f"  statistically significant at "
        f"alpha={ALPHA}: "
        f"{statistically_significant}"
    )

    print()
    print("Evaluation notes:")

    if treatment["switches"] > 0:
        print(
            "  Treatment includes payment-method "
            "switches."
        )
    else:
        print(
            "  Treatment produced no payment-method "
            "switches."
        )

    if (
        treatment["switches"] > 0
        and treatment["accepted"]
        == treatment["switches"]
    ):
        print(
            "  WARNING: All treatment switches were "
            "marked accepted."
        )
        print(
            "  Acceptance rate should not be interpreted "
            "as natural user behavior for this run."
        )

    print(
        "  Results represent the synthetic experiment "
        "environment and should not be treated as "
        "real-world payment performance."
    )


if __name__ == "__main__":
    main()