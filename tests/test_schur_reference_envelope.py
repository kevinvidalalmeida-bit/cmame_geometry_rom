from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constitutive_transfer.schur_estimator import optimize_convex_reference_envelope


def test_convex_reference_envelope_is_valid_and_sharper() -> None:
    error = np.diag([1.0, 1.0, 0.2, 0.2, 0.2, 0.2])
    first = np.diag([1.0, 4.0, 0.4, 0.4, 0.4, 0.4])
    second = np.diag([4.0, 1.0, 0.4, 0.4, 0.4, 0.4])

    selection = optimize_convex_reference_envelope(
        np.stack([first, second]), relative_tolerance=1.0e-14
    )

    assert np.isclose(selection.weights.sum(), 1.0)
    assert np.all(selection.weights >= 0.0)
    assert selection.objective < selection.best_single_objective
    assert np.linalg.eigvalsh(selection.upper_bound_matrix - error).min() >= -1.0e-14
    assert selection.relative_duality_gap <= 1.0e-14


def test_convex_reference_envelope_keeps_best_vertex_when_optimal() -> None:
    best = np.eye(6)
    worse = 2.0 * np.eye(6)

    selection = optimize_convex_reference_envelope(np.stack([best, worse]))

    np.testing.assert_allclose(selection.upper_bound_matrix, best, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(selection.weights, [1.0, 0.0], rtol=0.0, atol=0.0)
