"""Constitutive nearest-neighbor selection for FFT snapshot initialization."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigvalsh


def nearest_constitutive_snapshot(current_phases, previous_phases):
    """Return the closest stored phase pair in a scale-invariant SPD metric.

    Common rotations and common stiffness scaling do not affect the generalized
    eigenvalue interval.  A tie selects the latest snapshot, whose physical
    field is already resident in the reusable work buffer.
    """
    distances = []
    for phases in previous_phases:
        eigenvalues = np.concatenate([
            eigvalsh(new, old, check_finite=True)
            for new, old in zip(current_phases, phases)
        ])
        alpha, beta = eigenvalues.min(), eigenvalues.max()
        if alpha <= 0 or not np.isfinite(beta):
            raise ValueError("Constitutive nearest-neighbor search requires SPD phases.")
        distances.append(float((beta - alpha) / (beta + alpha)))
    if not distances:
        return None, None
    best = len(distances) - 1 - int(np.argmin(distances[::-1]))
    return best, distances[best]
