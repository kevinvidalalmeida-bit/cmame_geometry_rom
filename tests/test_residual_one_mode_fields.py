import numpy as np

from scripts.residual_one_mode_fields import dominant_residual_stress, residual_stresses


class DiagonalAffine:
    def __init__(self, matrices):
        self.matrices = matrices

    def apply_all(self, values):
        return np.stack([np.einsum("ab,lbn->lan", matrix, values) for matrix in self.matrices])


def test_field_residual_is_exact_affine_macro_minus_ritz_stress():
    rng = np.random.default_rng(7)
    q, r, n = 3, 2, 5
    matrices = rng.normal(size=(q, 6, 6))
    V = rng.normal(size=(r, 6, n))
    gamma = rng.normal(size=q)
    Y = rng.normal(size=(r, 6))
    residual = residual_stresses(DiagonalAffine(matrices), V, gamma, Y)
    K = np.einsum("q,qab->ab", gamma, matrices)
    expected = -np.einsum("ab,lbn->lan", K, np.eye(6)[:, :, None] + np.einsum("rl,rbn->lbn", Y, V))
    assert np.allclose(residual, expected)
    direction = rng.normal(size=6)
    assert np.allclose(dominant_residual_stress(residual, direction), np.einsum("l,lcn->cn", direction, expected))
