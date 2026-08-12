from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import inverse_identification_utils as inverse


def test_noise_matrix_has_requested_symmetric_frobenius_norm() -> None:
    target = np.diag([12.0, 10.0, 9.0, 5.0, 4.0, 3.0])
    perturbation = inverse._noise_matrix(target, 0.005, np.random.default_rng(17))

    np.testing.assert_allclose(perturbation, perturbation.T, rtol=0.0, atol=0.0)
    assert np.isclose(np.linalg.norm(perturbation) / np.linalg.norm(target), 0.005)


def test_weighted_upper_triangle_preserves_symmetric_frobenius_norm() -> None:
    rng = np.random.default_rng(19)
    matrix = rng.normal(size=(6, 6))
    matrix = 0.5 * (matrix + matrix.T)

    vector = inverse._weighted_symmetric_vector(matrix)

    assert np.isclose(np.linalg.norm(vector), np.linalg.norm(matrix))
