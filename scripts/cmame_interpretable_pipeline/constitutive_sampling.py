"""Material-only covering designs on the cone of two-phase elastic tensors."""

import numpy as np


def phase_spectral_distances(phases, anchor):
    """Projective distance log(beta/alpha) from two generalized SPD spectra.

    alpha and beta bound both phases together; independent phase scalings
    must not be removed, because their ratio affects homogenized fields.
    """
    phases, anchor = np.asarray(phases, dtype=float), np.asarray(anchor, dtype=float)
    inverse = np.linalg.inv(np.linalg.cholesky(anchor))
    whitened = inverse @ phases @ np.swapaxes(inverse, -1, -2)
    eigenvalues = np.linalg.eigvalsh((whitened+np.swapaxes(whitened, -1, -2))/2)
    alpha, beta = eigenvalues.min(axis=(-1,-2)), eigenvalues.max(axis=(-1,-2))
    if not np.all(np.isfinite(eigenvalues)) or np.any(alpha <= 0):
        raise ValueError('Constitutive covering requires finite SPD phases.')
    return np.maximum(0., np.log(beta)-np.log(alpha))


def constitutive_maximin_indices(phases, count):
    """Farthest-point design; no geometry or response values enter selection."""
    phases = np.asarray(phases, dtype=float)
    if phases.ndim != 4 or phases.shape[1:] != (2,6,6):
        raise ValueError('Expected an array of shape (materials, 2, 6, 6).')
    if not 1 <= count <= len(phases):
        raise ValueError('Requested covering count is outside the candidate pool.')
    # Normalize both phases by the SAME scalar before constructing the center.
    scale = np.linalg.norm(phases[:,0], axis=(-1,-2))
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError('Invalid phase scale.')
    normalized = phases / scale[:,None,None,None]
    center = normalized.mean(axis=0)
    selected = [int(np.argmin(phase_spectral_distances(normalized, center)))]
    minimum = phase_spectral_distances(normalized, normalized[selected[0]])
    radii = [float(minimum.max())]
    minimum[selected] = -np.inf
    while len(selected) < count:
        next_index = int(np.argmax(minimum))
        selected.append(next_index)
        distance = phase_spectral_distances(normalized, normalized[next_index])
        minimum = np.minimum(minimum, distance)
        minimum[selected] = -np.inf
        radii.append(float(max(0., minimum.max())))
    return np.asarray(selected, dtype=int), np.asarray(radii)
