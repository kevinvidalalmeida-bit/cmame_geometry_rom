"""Exactness tests for the two-kernel offline-online Schur estimator."""

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constitutive_transfer.schur_estimator import (
    TwoKernelEstimator,
    compile_two_kernel_contractions,
    hierarchical_matrix_upper_bound,
    optimize_isotropic_reference,
    optimize_isotropic_reference_batch,
    traction_kernel_features,
)
import schur_energy_indicators as direct


def test_traction_two_kernel_identity_matches_direct_fourier():
    rng = np.random.default_rng(20260811)
    shape = (5, 4, 3)
    nvox = int(np.prod(shape))
    stresses = rng.normal(size=(4, 6, nvox))
    nvec, nonzero = direct._frequency_unit_vectors(shape)
    residual = direct._project_compatible_fourier(
        stresses, shape=shape, nvec=nvec, nonzero=nonzero
    )
    correction = direct._reference_inverse_fourier(
        residual,
        nvec=nvec,
        nonzero=nonzero,
        lam0=1.7,
        mu0=2.3,
    )
    direct_energy = direct._energy_matrix_from_fourier(
        direct._tensor_loads_to_mandel_flat(residual),
        direct._tensor_loads_to_mandel_flat(correction),
        nvox=nvox,
    )

    transverse, longitudinal = traction_kernel_features(stresses, shape)
    dense_energy = transverse @ np.conjugate(transverse).T / (2.3 * nvox**2)
    dense_energy += longitudinal @ np.conjugate(longitudinal).T / (
        (1.7 + 2.0 * 2.3) * nvox**2
    )
    dense_energy = 0.5 * (dense_energy.real + dense_energy.real.T)
    np.testing.assert_allclose(dense_energy, direct_energy, rtol=2.0e-13, atol=2.0e-13)


def test_real_fourier_half_spectrum_matches_full_spectrum_on_odd_grid():
    rng = np.random.default_rng(20260820)
    shape = (5, 5, 5)
    stresses = rng.normal(size=(3, 6, int(np.prod(shape))))
    transverse_full, longitudinal_full = traction_kernel_features(
        stresses, shape, half_spectrum=False
    )
    transverse_half, longitudinal_half = traction_kernel_features(
        stresses, shape, half_spectrum=True
    )
    full_energy = transverse_full @ np.conjugate(transverse_full).T
    full_energy += longitudinal_full @ np.conjugate(longitudinal_full).T
    half_energy = transverse_half @ np.conjugate(transverse_half).T
    half_energy += longitudinal_half @ np.conjugate(longitudinal_half).T
    full_energy = 0.5 * (full_energy.real + full_energy.real.T)
    half_energy = 0.5 * (half_energy.real + half_energy.real.T)
    np.testing.assert_allclose(half_energy, full_energy, rtol=2.0e-13, atol=2.0e-13)


def test_cupy_features_match_scipy_in_float64_when_available():
    pytest.importorskip("cupy")
    rng = np.random.default_rng(20260822)
    shape = (5, 5, 5)
    stresses = rng.normal(size=(2, 6, int(np.prod(shape))))
    cpu = traction_kernel_features(
        stresses, shape, half_spectrum=True, fft_workers=1, fft_backend="scipy"
    )
    gpu = traction_kernel_features(
        stresses, shape, half_spectrum=True, fft_backend="cupy"
    )
    for cpu_values, gpu_values in zip(cpu, gpu, strict=True):
        np.testing.assert_allclose(gpu_values, cpu_values, rtol=2.0e-12, atol=2.0e-12)


def test_compiled_contractions_match_direct_fields(tmp_path: Path):
    rng = np.random.default_rng(17)
    shape = (4, 3, 2)
    nvox = int(np.prod(shape))
    rank = 3
    q_count = 2
    basis = rng.normal(size=(rank, 6, nvox))
    basis -= basis.mean(axis=2, keepdims=True)

    atoms = rng.normal(size=(q_count, nvox, 6, 6))
    atoms = 0.5 * (atoms + np.swapaxes(atoms, -1, -2))

    def affine_stress(q: int, strains: np.ndarray) -> np.ndarray:
        return np.einsum("nab,lbn->lan", atoms[q], strains, optimize=True)

    artifact = tmp_path / "online_estimator.npz"
    compile_two_kernel_contractions(
        output_path=artifact,
        shape=shape,
        basis_fields=basis,
        coefficient_names=("q0", "q1"),
        affine_stress_batch=affine_stress,
        atom_batch_size=2,
        feature_block=11,
    )
    estimator = TwoKernelEstimator.load(artifact)
    coefficients = np.array((1.2, -0.35))
    amplitudes = rng.normal(size=(rank, 6))
    dense = estimator.energy_matrix(
        coefficients, amplitudes, lambda0=0.8, mu0=1.4
    )

    strains = np.einsum("rl,rcn->lcn", amplitudes, basis, optimize=True)
    for load in range(6):
        strains[load, load] += 1.0
    stress = sum(coefficients[q] * affine_stress(q, strains) for q in range(q_count))
    transverse, longitudinal = traction_kernel_features(stress, shape)
    expected = transverse @ np.conjugate(transverse).T / (1.4 * nvox**2)
    expected += longitudinal @ np.conjugate(longitudinal).T / (
        (0.8 + 2.0 * 1.4) * nvox**2
    )
    expected = 0.5 * (expected.real + expected.real.T)
    relative_error = np.linalg.norm(dense - expected) / np.linalg.norm(expected)
    assert relative_error < 1.0e-12


def test_reference_search_returns_conservative_generalized_beta():
    rng = np.random.default_rng(9)
    phases = []
    for shift in (2.0, 5.0):
        factor = rng.normal(size=(6, 6))
        phases.append(factor.T @ factor + shift * np.eye(6))
    raw = rng.normal(size=(2, 6, 8))
    kernels = np.einsum("tli,tmi->tlm", raw, raw, optimize=True)
    selected = optimize_isotropic_reference(kernels, phases, beta_margin=1.0e-10)
    assert selected.beta_safe < selected.beta
    assert selected.beta_safe > 0.0
    assert selected.upper_bound_matrix.shape == (6, 6)
    assert 0 <= selected.index < 129


def test_batched_reference_search_matches_scalar_search():
    rng = np.random.default_rng(19)
    phases = []
    for shift in (2.0, 5.0):
        factor = rng.normal(size=(6, 6))
        phases.append(factor.T @ factor + shift * np.eye(6))
    kernels = rng.normal(size=(4, 2, 6, 6))
    kernels = 0.5 * (kernels + np.swapaxes(kernels, -1, -2))
    batch = optimize_isotropic_reference_batch(
        kernels,
        tuple(np.broadcast_to(value, (len(kernels), 6, 6)) for value in phases),
        beta_margin=1.0e-10,
    )
    for candidate in range(len(kernels)):
        scalar = optimize_isotropic_reference(
            kernels[candidate], phases, beta_margin=1.0e-10
        )
        assert int(batch["index"][candidate]) == scalar.index
        np.testing.assert_allclose(
            batch["upper_bound_matrices"][candidate],
            scalar.upper_bound_matrix,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            batch["beta_safe"][candidate], scalar.beta_safe, rtol=2.0e-13, atol=2.0e-13
        )


def test_cross_kernel_diagonal_and_reverse_identity():
    rng = np.random.default_rng(20260812)
    q_count, rank, feature_count = 5, 4, 19
    features = rng.normal(size=(2, q_count, 6 + rank, feature_count))
    bb = np.einsum("tpaf,tqbf->tpqab", features[:, :, :6], features[:, :, :6])
    bk = np.einsum("tpaf,tqrf->tpqar", features[:, :, :6], features[:, :, 6:])
    kk = np.einsum("tpaf,tqrf->tpqar", features[:, :, 6:], features[:, :, 6:])
    estimator = TwoKernelEstimator(bb, bk, kk)
    coefficients = rng.normal(size=(3, q_count))
    amplitudes = rng.normal(size=(3, rank, 6))

    cross = estimator.kernel_cross_matrices_batch(
        coefficients, amplitudes, coefficients, amplitudes
    )
    diagonal = cross[np.arange(3), np.arange(3)]
    direct = estimator.kernel_energy_matrices_batch(coefficients, amplitudes)
    np.testing.assert_allclose(diagonal, direct, rtol=1.0e-11, atol=1.0e-11)

    left_coefficients = coefficients[:2]
    left_amplitudes = amplitudes[:2]
    right_coefficients = coefficients[1:]
    right_amplitudes = amplitudes[1:]
    forward = estimator.cross_energy_matrices_batch(
        left_coefficients,
        left_amplitudes,
        right_coefficients,
        right_amplitudes,
        lambda0=1.3,
        mu0=0.9,
    )
    reverse = estimator.cross_energy_matrices_batch(
        right_coefficients,
        right_amplitudes,
        left_coefficients,
        left_amplitudes,
        lambda0=1.3,
        mu0=0.9,
    )
    np.testing.assert_allclose(
        forward,
        np.swapaxes(reverse, 0, 1).swapaxes(-1, -2),
        rtol=1.0e-11,
        atol=1.0e-11,
    )

    factor, metadata = estimator.reference_gram_factor(
        lambda0=1.3,
        mu0=0.9,
        relative_eigenvalue_tolerance=1.0e-13,
    )
    left_embedding = estimator.residual_embeddings_batch(
        left_coefficients, left_amplitudes, factor
    )
    right_embedding = estimator.residual_embeddings_batch(
        right_coefficients, right_amplitudes, factor
    )
    factorized = np.einsum(
        "nkl,mkr->nmlr", left_embedding, right_embedding, optimize=True
    )
    np.testing.assert_allclose(factorized, forward, rtol=2.0e-11, atol=2.0e-11)
    assert metadata["embedding_rank"] <= metadata["atom_dimension"]
    assert metadata["retained_trace_fraction"] > 1.0 - 1.0e-12


def test_factorized_global_gain_matches_explicit_pairs():
    rng = np.random.default_rng(20260814)
    probe = rng.normal(size=(7, 9, 6))
    candidates = rng.normal(size=(4, 9, 6))
    weights = rng.random(7)
    weights /= weights.sum()
    covariance = np.einsum(
        "n,nkl,nml->km", weights, probe, probe, optimize=True
    )
    factorized = []
    explicit = []
    for candidate in candidates:
        self_gram = candidate.T @ candidate
        inverse = np.linalg.pinv(self_gram, rcond=1.0e-12)
        compressed = candidate.T @ covariance @ candidate
        factorized.append(float(np.trace(inverse @ compressed)))
        cross = np.einsum("nkl,kr->nlr", probe, candidate, optimize=True)
        gains = np.einsum(
            "nlr,rs,nls->n", cross, inverse, cross, optimize=True
        )
        explicit.append(float(np.sum(weights * gains)))
    np.testing.assert_allclose(factorized, explicit, rtol=2.0e-13, atol=2.0e-13)


def test_estimator_global_influence_matches_explicit_cross_material_sum():
    rng = np.random.default_rng(20260815)
    q_count, rank, feature_count = 4, 5, 23
    features = rng.normal(size=(2, q_count, 6 + rank, feature_count))
    bb = np.einsum("tpaf,tqbf->tpqab", features[:, :, :6], features[:, :, :6])
    bk = np.einsum("tpaf,tqrf->tpqar", features[:, :, :6], features[:, :, 6:])
    kk = np.einsum("tpaf,tqrf->tpqar", features[:, :, 6:], features[:, :, 6:])
    estimator = TwoKernelEstimator(bb, bk, kk)
    probe_coefficients = rng.normal(size=(8, q_count))
    probe_amplitudes = rng.normal(size=(8, rank, 6))
    candidate_coefficients = rng.normal(size=(5, q_count))
    candidate_amplitudes = rng.normal(size=(5, rank, 6))
    weights = rng.random(8)
    weights /= weights.sum()
    lambda0, mu0 = 1.4, 0.8

    scores, metadata = estimator.global_influence_scores_batch(
        probe_coefficients,
        probe_amplitudes,
        candidate_coefficients,
        candidate_amplitudes,
        weights,
        lambda0=lambda0,
        mu0=mu0,
        max_embedding_rank=None,
    )
    cross = estimator.cross_energy_matrices_batch(
        probe_coefficients,
        probe_amplitudes,
        candidate_coefficients,
        candidate_amplitudes,
        lambda0=lambda0,
        mu0=mu0,
    )
    candidate_self = estimator.cross_energy_matrices_batch(
        candidate_coefficients,
        candidate_amplitudes,
        candidate_coefficients,
        candidate_amplitudes,
        lambda0=lambda0,
        mu0=mu0,
    )
    explicit = []
    for candidate in range(len(candidate_coefficients)):
        inverse = np.linalg.pinv(candidate_self[candidate, candidate], rcond=1.0e-12)
        explained = np.einsum(
            "nab,bc,ncd->nad",
            cross[:, candidate],
            inverse,
            np.swapaxes(cross[:, candidate], -1, -2),
            optimize=True,
        )
        explicit.append(float(np.sum(weights * np.trace(explained, axis1=1, axis2=2))))
    np.testing.assert_allclose(scores, explicit, rtol=2.0e-11, atol=2.0e-11)
    assert metadata["pair_count_avoided"] == 40




def test_hierarchical_bound_adds_exact_reduced_drop_and_tail_bound():
    truth = np.eye(6)
    tail_error = np.diag(np.linspace(0.01, 0.06, 6))
    reduced_drop = np.diag(np.linspace(0.2, 0.7, 6))
    enriched = truth + tail_error
    coarse = enriched + reduced_drop
    tail_upper = tail_error + 0.005 * np.eye(6)

    upper = hierarchical_matrix_upper_bound(coarse, enriched, tail_upper)

    np.testing.assert_allclose(upper, reduced_drop + tail_upper)
    coarse_error = coarse - truth
    assert np.min(np.linalg.eigvalsh(upper - coarse_error)) >= 0.0


def test_hierarchical_bound_rejects_non_nested_outputs():
    coarse = np.eye(6)
    enriched = np.eye(6) + 0.1 * np.eye(6)
    with np.testing.assert_raises_regex(ValueError, "nested Ritz hierarchy"):
        hierarchical_matrix_upper_bound(coarse, enriched, np.eye(6))
