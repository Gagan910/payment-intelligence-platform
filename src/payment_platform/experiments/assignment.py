from __future__ import annotations

import hashlib


CONTROL_VARIANT = "control"
TREATMENT_VARIANT = "treatment"

VALID_VARIANTS = frozenset(
    {
        CONTROL_VARIANT,
        TREATMENT_VARIANT,
    }
)


def assign_variant(
    *,
    experiment_id: str,
    subject_id: str,
    treatment_percentage: int = 50,
) -> str:
    """
    Deterministically assign a subject to an A/B experiment variant.

    The same experiment_id + subject_id always produces the same variant.
    """

    if not experiment_id:
        raise ValueError("experiment_id must not be empty")

    if not subject_id:
        raise ValueError("subject_id must not be empty")

    if not 0 <= treatment_percentage <= 100:
        raise ValueError("treatment_percentage must be between 0 and 100")

    assignment_key = f"{experiment_id}:{subject_id}"

    digest = hashlib.sha256(
        assignment_key.encode("utf-8")
    ).hexdigest()

    bucket = int(digest[:8], 16) % 100

    if bucket < treatment_percentage:
        return TREATMENT_VARIANT

    return CONTROL_VARIANT