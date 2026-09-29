from __future__ import annotations

import argparse
import math
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
                    ee.event_type,
                    ee.metadata
                FROM transactions t
                JOIN experiment_events ee
                    ON ee.transaction_id = t.transaction_id
                WHERE
                    ee.experiment_id = %s
                    AND ee.event_type = 'payment_completed'
                    AND LEFT(t.transaction_id, %s) = %s
                ORDER BY ee.timestamp
                """,
                (
                    EXPERIMENT_ID,
                    len(run_id),
                    run_id,
                ),
            )

            rows = cursor.fetchall()

    finally:
        conn.close()

    results: list[dict[str, Any]] = []

    for row in rows:
        transaction_id = row[0]
        variant = row[1]
        event_type = row[2]
        metadata = row[3] or {}

        results.append(
            {
                "transaction_id": transaction_id,
                "variant": variant,
                "event_type": event_type,
                "outcome": metadata.get("outcome"),
                "payment_method": metadata.get(
                    "payment_method"
                ),
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
            "successes": 0,
            "failures": 0,
            "success_rate": None,
            "failure_rate": None,
            "ci_low": None,
            "ci_high": None,
        }

    successes = sum(
        result["outcome"] == "success"
        for result in observations
    )

    failures = sum(
        result["outcome"] == "failure"
        for result in observations
    )

    success_rate = successes / total
    failure_rate = failures / total

    ci_low, ci_high = proportion_confidence_interval(
        successes=successes,
        total=total,
    )

    return {
        "variant": variant,
        "n": total,
        "successes": successes,
        "failures": failures,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate payment success outcomes for "
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
            f"No payment results found for run_id={args.run_id}"
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

    for metrics in (control, treatment):
        print(f"{metrics['variant']}:")
        print(f"  n = {metrics['n']}")
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

        print()

    if (
        control["success_rate"] is None
        or treatment["success_rate"] is None
    ):
        print(
            "Statistical comparison skipped: "
            "both variants require observations."
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
        control_total=control["n"],
        treatment_successes=treatment["successes"],
        treatment_total=treatment["n"],
    )

    z_statistic, p_value = two_proportion_z_test(
        control_successes=control["successes"],
        control_total=control["n"],
        treatment_successes=treatment["successes"],
        treatment_total=treatment["n"],
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


if __name__ == "__main__":
    main()