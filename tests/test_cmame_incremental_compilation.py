"""Equivalence checks for block basis and incremental Ritz compilation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cmame_campaign_common as common
import schur_estimator_compiler as compile_estimator
import schur_energy_indicators as qoi
import rom_reduced_operator as reduced


def _scalar_cgs2(
    basis: list[np.ndarray], fields: list[np.ndarray], tolerance: float
) -> list[np.ndarray]:
    appended: list[np.ndarray] = []
    for field in fields:
        vector = np.asarray(field, dtype=np.float64).copy()
        for old in basis:
            vector -= np.mean(old * vector) * old
        for old in appended:
            vector -= np.mean(old * vector) * old
        for old in basis:
            vector -= np.mean(old * vector) * old
        for old in appended:
            vector -= np.mean(old * vector) * old
        norm = np.sqrt(max(float(np.mean(vector * vector)), 0.0))
        if norm > tolerance:
            appended.append(vector / norm)
    basis.extend(appended)
    return appended


def _orthonormal_fields(rng: np.random.Generator, count: int, shape: tuple[int, ...]):
    dimension = int(np.prod(shape))
    raw = rng.normal(size=(dimension, count))
    q, _ = np.linalg.qr(raw)
    return [(np.sqrt(dimension) * q[:, index]).reshape(shape) for index in range(count)]


def test_block_cgs2_preserves_scalar_cgs2_subspace():
    rng = np.random.default_rng(20260816)
    shape = (6, 4, 3, 2)
    initial = _orthonormal_fields(rng, 9, shape)
    incoming = [rng.normal(size=shape) for _ in range(6)]
    scalar_basis = [value.copy() for value in initial]
    block_basis = [value.copy() for value in initial]

    scalar = _scalar_cgs2(scalar_basis, incoming, 1.0e-11)
    block = common._append_orthonormal(
        block_basis, incoming, tolerance=1.0e-11, basis_block_size=4
    )
    assert len(block) == len(scalar) == 6
    scalar_matrix = np.stack([value.reshape(-1) for value in scalar])
    block_matrix = np.stack([value.reshape(-1) for value in block])
    overlap = scalar_matrix @ block_matrix.T / float(np.prod(shape))
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    np.testing.assert_allclose(singular_values, 1.0, rtol=2.0e-12, atol=2.0e-12)
    gram = block_matrix @ block_matrix.T / float(np.prod(shape))
    np.testing.assert_allclose(gram, np.eye(6), rtol=2.0e-12, atol=2.0e-12)


def test_incremental_ritz_operators_match_full_assembly():
    rng = np.random.default_rng(20260817)
    shape = (5, 4, 3)
    phase = np.zeros(shape, dtype=np.uint8)
    phase[2:] = 1
    ori = np.zeros(shape + (3,), dtype=np.float64)
    ori[..., 0] = 1.0
    fields = _orthonormal_fields(rng, 8, (6,) + shape)
    old_basis, new_basis = fields[:5], fields[5:]
    K0, B0, D0, _ = reduced._assemble_reduced_operators(
        phase=phase, ori=ori, basis=old_basis
    )
    matrix_idx, fiber_groups = qoi._geometry_groups(phase, ori)
    affine = compile_estimator._affine_stress_factory(matrix_idx, fiber_groups)
    Ki, Bi, Di, metadata = reduced._extend_reduced_operators(
        existing={"Kq": K0, "Bq": B0, "Dq": D0},
        old_basis=old_basis,
        new_basis=new_basis,
        affine_stress_batch=affine,
        basis_block_size=3,
    )
    Kf, Bf, Df, _ = reduced._assemble_reduced_operators(
        phase=phase, ori=ori, basis=fields
    )
    np.testing.assert_allclose(Ki, Kf, rtol=3.0e-13, atol=3.0e-13)
    np.testing.assert_allclose(Bi, Bf, rtol=3.0e-13, atol=3.0e-13)
    np.testing.assert_allclose(Di, Df, rtol=3.0e-13, atol=3.0e-13)
    assert metadata["assembly_mode"] == "incremental"


def test_incremental_batch_cholesky_matches_dense_batched_solves():
    rng = np.random.default_rng(20260819)
    candidates, coefficients, final_rank = 19, 4, 18
    coefficient_values = 0.1 + rng.random((candidates, coefficients))
    raw = rng.normal(size=(coefficients, final_rank, final_rank))
    Kq = np.stack(
        [factor.T @ factor + (index + 1.0) * np.eye(final_rank) for index, factor in enumerate(raw)]
    )
    Bq = rng.normal(size=(coefficients, final_rank, 6))
    Dq = rng.normal(size=(coefficients, 6, 6))
    Dq = 0.5 * (Dq + np.swapaxes(Dq, -1, -2))

    evaluator = reduced.IncrementalAffineBatchEvaluator(
        coefficient_values, Kq[:, :6, :6], Bq[:, :6], Dq
    )
    for rank in (12, final_rank):
        metadata = evaluator.extend(Kq[:, :rank, :rank], Bq[:, :rank], Dq)
        cached_C, cached_amplitudes = evaluator.evaluate()
        dense_C, dense_amplitudes, _ = reduced._rom_ceff_batch(
            coefficient_values, Kq[:, :rank, :rank], Bq[:, :rank], Dq
        )
        np.testing.assert_allclose(cached_C, dense_C, rtol=2.0e-12, atol=2.0e-12)
        np.testing.assert_allclose(
            cached_amplitudes, dense_amplitudes, rtol=2.0e-12, atol=2.0e-12
        )
        assert metadata["update_mode"] == "block_cholesky"
