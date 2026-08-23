"""Small reporting helpers for independent ROM validation sets."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def empirical_coverage(errors: Iterable[float], threshold: float) -> dict[str, float | int]:
    """Summarize coverage, counting non-finite reduced results as failures."""
    values = np.asarray(list(errors), dtype=np.float64)
    threshold = float(threshold)
    if values.size == 0:
        raise ValueError("At least one validation error is required.")
    finite = np.isfinite(values)
    if np.any(values[finite] < 0.0):
        raise ValueError("Finite validation errors must be nonnegative.")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("The reporting threshold must be finite and positive.")

    passed = int(np.count_nonzero(finite & (values <= threshold)))
    count = int(values.size)
    finite_count = int(np.count_nonzero(finite))
    fraction = passed / count
    observed_max = float(np.max(values[finite])) if finite_count else float("nan")
    return {
        "target_error": threshold,
        "target_error_percent": 100.0 * threshold,
        "observed_count": count,
        "finite_error_count": finite_count,
        "numerical_failure_count": count - finite_count,
        "below_target_count": passed,
        "above_target_count": count - passed,
        "below_target_fraction": fraction,
        "below_target_percent": 100.0 * fraction,
        "observed_error_max": observed_max,
        "observed_error_max_percent": 100.0 * observed_max,
    }
