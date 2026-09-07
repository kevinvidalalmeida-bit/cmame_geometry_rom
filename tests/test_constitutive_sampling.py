"""Spectral covering checks using independent generalized eigenproblems."""

import numpy as np
from scipy.linalg import eigvalsh

from cmame_interpretable_pipeline.constitutive_sampling import (
    constitutive_maximin_indices, phase_spectral_distances,
)


def phases():
    a = np.random.default_rng(916).normal(size=(25,2,6,6))
    return a @ np.swapaxes(a,-1,-2) + np.eye(6)


def test_distance_matches_generalized_eigenvalues_and_preserves_phase_ratio():
    values = phases()
    observed = phase_spectral_distances(values, values[0])
    expected = []
    for candidate in values:
        spectrum = np.concatenate([eigvalsh(new, old) for new,old in zip(candidate,values[0])])
        expected.append(np.log(spectrum.max()/spectrum.min()))
    np.testing.assert_allclose(observed, expected, atol=1e-13)
    scaled = values[0].copy()
    scaled[1] *= 3
    np.testing.assert_allclose(phase_spectral_distances(scaled[None], values[0]), np.log(3), atol=1e-13)


def test_covering_is_invariant_to_material_scale_and_common_rotations():
    values = phases()
    indices, radii = constitutive_maximin_indices(values, 10)
    rotation,_ = np.linalg.qr(np.random.default_rng(104).normal(size=(6,6)))
    scaled = (rotation @ values @ rotation.T)*np.geomspace(1e-3,1e3,len(values))[:,None,None,None]
    other, other_radii = constitutive_maximin_indices(scaled, 10)
    np.testing.assert_array_equal(other, indices)
    np.testing.assert_allclose(other_radii, radii, atol=1e-12)
    assert len(set(indices)) == len(indices)
    assert np.all(np.diff(radii) <= 1e-13)


def test_reported_cover_radius_bounds_every_candidate_distance():
    values = phases()
    indices, radii = constitutive_maximin_indices(values, 9)
    all_distances = np.stack([phase_spectral_distances(values, values[i]) for i in indices])
    np.testing.assert_allclose(all_distances.min(axis=0).max(), radii[-1], atol=1e-13)
