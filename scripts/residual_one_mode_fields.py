"""Field-level affine residual assembly for one-mode greedy enrichment."""
from __future__ import annotations

import numpy as np


def residual_stresses(
    affine_action: object,
    basis: np.ndarray,
    coefficients: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Return unprojected stresses whose projection is ``B - K VY``.

    ``basis`` is stored as ``(r,6,nvox)`` in the phase/orientation order used
    by the affine compiler and ``amplitudes`` is ``(r,6)``.  The six unit
    macroscopic strains are included explicitly, so no FOM solve is involved.
    The returned field has shape ``(6,6,nvox)``: first axis is macro load and
    second axis is Mandel stress component.
    """
    V = np.asarray(basis)
    gamma = np.asarray(coefficients, dtype=float)
    Y = np.asarray(amplitudes, dtype=float)
    if V.ndim != 3 or V.shape[1] != 6 or Y.shape != (V.shape[0], 6):
        raise ValueError("basis must be (r,6,nvox) and amplitudes (r,6).")
    if gamma.ndim != 1:
        raise ValueError("coefficients must be one-dimensional.")
    total = np.einsum("rl,rcn->lcn", Y, V, optimize=True)
    total += np.eye(6, dtype=total.dtype)[:, :, None]
    all_stresses = affine_action.apply_all(total)
    if all_stresses.shape[:2] != (len(gamma), 6):
        raise ValueError("affine action does not match the coefficient count.")
    # Corrector equilibrium is G A(E+x)=0, hence the residual is -G stress.
    return -np.einsum("q,qlcn->lcn", gamma, all_stresses, optimize=True)


def dominant_residual_stress(residual_stress: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Collapse six residual loads to the selected unit macro direction."""
    stress = np.asarray(residual_stress)
    vector = np.asarray(direction, dtype=float)
    if stress.ndim != 3 or stress.shape[0:2] != (6, 6) or vector.shape != (6,):
        raise ValueError("expected residual_stress=(6,6,nvox), direction=(6,).")
    return np.einsum("l,lcn->cn", vector, stress, optimize=True)
