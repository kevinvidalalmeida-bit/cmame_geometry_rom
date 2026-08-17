"""Small reporting helpers for independent ROM validation sets."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def empirical_coverage(errors: Iterable[float], threshold: float) -> dict[str, float | int]:
    """Summarize how many observed relative errors meet a fixed threshold."""
    values = np.asarray(list(errors), dtype=np.float64)
    threshold = float(threshold)
    if values.size == 0:
        raise ValueError("At least one validation error is required.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Validation errors must be finite and nonnegative.")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("The reporting threshold must be finite and positive.")

    passed = int(np.count_nonzero(values <= threshold))
    count = int(values.size)
    fraction = passed / count
    return {
        "target_error": threshold,
        "target_error_percent": 100.0 * threshold,
        "observed_count": count,
        "below_target_count": passed,
        "above_target_count": count - passed,
        "below_target_fraction": fraction,
        "below_target_percent": 100.0 * fraction,
        "observed_error_max": float(np.max(values)),
        "observed_error_max_percent": 100.0 * float(np.max(values)),
    }
