"""Equivalence checks for block basis and incremental Ritz compilation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cmame_campaign_common as common
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


def _ill_conditioned_raw_ritz_case(seed: int):
    rng = np.random.default_rng(seed)
    shape = (4, 4, 4)
    rank = 60
    dimension = 6 * int(np.prod(shape))
    q, _ = np.linalg.qr(rng.standard_normal((dimension, rank)))
    orthonormal = (np.sqrt(dimension) * q.T).reshape((rank, 6) + shape)
    left, _ = np.linalg.qr(rng.standard_normal((rank, rank)))
    right, _ = np.linalg.qr(rng.standard_normal((rank, rank)))
    mixing = left @ np.diag(np.geomspace(1.0, 2.5e-4, rank)) @ right.T
    raw = (mixing @ orthonormal.reshape(rank, -1)).reshape(
        orthonormal.shape
    ).astype(np.float32)
    phase = np.zeros(shape, dtype=np.uint8)
    phase.reshape(-1)[::3] = 1
    ori = np.zeros(shape + (3,), dtype=np.float64)
    ori[..., 0] = 1.0
    coefficients = reduced._material_coefficients(
        {
            "Em": 3.5,
            "nu_m": 0.35,
            "Ef_L": 120.0,
            "Ef_T": 14.0,
            "G_LT": 5.5,
            "nu_LT": 0.22,
            "nu_TT": 0.32,
        }
    )
    return raw, phase, ori, coefficients


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


def test_contiguous_float32_basis_preserves_full_rank_subspace():
    rng = np.random.default_rng(20260830)
    shape = (6, 5, 4, 3)
    incoming = [rng.normal(size=shape) for _ in range(12)]
    reference: list[np.ndarray] = []
    common._append_orthonormal(reference, incoming[:6], tolerance=1.0e-12)
    common._append_orthonormal(reference, incoming[6:], tolerance=1.0e-12)

    contiguous = common.ContiguousBasis(12, shape, dtype=np.float32)
    contiguous.append(incoming[:6], tolerance=1.0e-12)
    contiguous.append(incoming[6:], tolerance=1.0e-12)

    assert len(contiguous) == 12
    reference_matrix = np.stack(reference).reshape(12, -1)
    overlap = reference_matrix @ contiguous.active_flat.astype(np.float64).T / np.prod(shape)
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    np.testing.assert_allclose(singular_values, 1.0, rtol=3.0e-6, atol=3.0e-6)
    gram = contiguous.active_flat @ contiguous.active_flat.T / np.float32(np.prod(shape))
    np.testing.assert_allclose(gram, np.eye(12), rtol=3.0e-6, atol=3.0e-6)


def test_contiguous_basis_accepts_preordered_blocks():
    rng = np.random.default_rng(20260902)
    shape = (6, 5, 4, 3)
    incoming = rng.standard_normal((9,) + shape).astype(np.float32)
    regular = common.ContiguousBasis(9, shape, dtype=np.float32)
    preordered = common.ContiguousBasis(9, shape, dtype=np.float32)

    regular.append(incoming, tolerance=1.0e-12)
    preordered.append_preordered(incoming.copy(order="C"), tolerance=1.0e-12)

    overlap = regular.active_flat @ preordered.active_flat.T / np.float32(np.prod(shape))
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    np.testing.assert_allclose(singular_values, 1.0, rtol=3.0e-6, atol=3.0e-6)
    assert preordered.last_projection_backend == "initial_block_no_projection"


def test_contiguous_basis_raw_preordered_preserves_full_rank_fields():
    rng = np.random.default_rng(20260910)
    shape = (6, 5, 4, 3)
    first = rng.standard_normal((4,) + shape).astype(np.float32)
    second = rng.standard_normal((3,) + shape).astype(np.float32)
    basis = common.ContiguousBasis(7, shape, dtype=np.float32)

    appended_first = basis.append_raw_preordered(first)
    appended_second = basis.append_raw_preordered(second)

    np.testing.assert_array_equal(appended_first, first)
    np.testing.assert_array_equal(appended_second, second)
    np.testing.assert_array_equal(basis.active_fields, np.concatenate((first, second)))
    assert len(basis) == 7
    assert basis.last_projection_backend == "raw_full_rank_no_projection"


def test_raw_full_rank_ritz_is_invariant_to_basis_coordinates():
    rng = np.random.default_rng(20260911)
    shape = (5, 4, 3)
    phase = np.zeros(shape, dtype=np.uint8)
    phase.reshape(-1)[::3] = 1
    ori = np.zeros(shape + (3,), dtype=np.float64)
    ori[..., 0] = 1.0
    orthonormal = np.stack(_orthonormal_fields(rng, 8, (6,) + shape))
    mixing = np.eye(8) + 0.08 * rng.standard_normal((8, 8))
    raw = (mixing @ orthonormal.reshape(8, -1)).reshape(orthonormal.shape)

    K_pod, B_pod, D_pod, _ = reduced._assemble_reduced_operators(
        phase=phase, ori=ori, basis=orthonormal
    )
    K_raw, B_raw, D_raw, metadata = reduced._assemble_reduced_operators(
        phase=phase, ori=ori, basis=raw
    )
    material = {
        "Em": 3.5,
        "nu_m": 0.35,
        "Ef_L": 120.0,
        "Ef_T": 14.0,
        "G_LT": 5.5,
        "nu_LT": 0.22,
        "nu_TT": 0.32,
    }
    coefficients = reduced._material_coefficients(material)
    C_pod, _, _ = reduced._rom_ceff(coefficients, K_pod, B_pod, D_pod)
    C_raw, _, _ = reduced._rom_ceff(coefficients, K_raw, B_raw, D_raw)

    np.testing.assert_allclose(C_raw, C_pod, rtol=2.0e-12, atol=2.0e-12)
    assert metadata["gram_condition"] > 1.0


def test_raw_ritz_rank_reveal_discards_only_dependent_directions():
    rng = np.random.default_rng(20260912)
    shape = (5, 4, 3)
    phase = np.zeros(shape, dtype=np.uint8)
    phase.reshape(-1)[::3] = 1
    ori = np.zeros(shape + (3,), dtype=np.float64)
    ori[..., 0] = 1.0
    independent = np.stack(_orthonormal_fields(rng, 5, (6,) + shape))
    dependent = np.concatenate((independent, independent[[2]]), axis=0)

    Kq, Bq, _, metadata = reduced._assemble_reduced_operators(
        phase=phase,
        ori=ori,
        basis=dependent,
        gram_rank_reveal=True,
    )

    assert Kq.shape[1:] == (5, 5)
    assert Bq.shape[1] == 5
    assert metadata["effective_rank"] == 5
    assert metadata["discarded_rank"] == 1
    assert metadata["gram_transform_mode"] == "eigh_rank_reveal"


def test_ill_conditioned_float32_raw_ritz_keeps_physical_stiffness_spd():
    raw, phase, ori, coefficients = _ill_conditioned_raw_ritz_case(20260914)
    rank = len(raw)

    Kq, _, _, metadata = reduced._assemble_reduced_operators(
        phase=phase,
        ori=ori,
        basis=raw,
        gram_rank_reveal=True,
    )
    stiffness = np.einsum("q,qij->ij", coefficients, Kq, optimize=True)
    stiffness = 0.5 * (stiffness + stiffness.T)

    assert metadata["gram_condition"] > 1.0e7
    assert metadata["effective_rank"] == rank
    assert metadata["contraction_compute_dtype"] == "float64"
    assert metadata["gram_product_dtype"] == "float64"
    assert np.linalg.eigvalsh(stiffness)[0] > 0.0


def test_float32_ritz_rank_reveal_drops_unresolved_coordinates_and_keeps_spd():
    raw, phase, ori, coefficients = _ill_conditioned_raw_ritz_case(20260915)
    rank = len(raw)

    Kq, _, _, metadata = reduced._assemble_reduced_operators(
        phase=phase,
        ori=ori,
        basis=raw,
        gram_rank_reveal=True,
        gram_rank_rtol=1.0e-6,
        contraction_compute_dtype="float32",
    )
    stiffness = np.einsum("q,qij->ij", coefficients, Kq, optimize=True)
    stiffness = 0.5 * (stiffness + stiffness.T)

    assert metadata["effective_rank"] < rank
    assert metadata["discarded_rank"] == rank - metadata["effective_rank"]
    assert metadata["gram_transform_mode"] == "eigh_rank_reveal"
    assert metadata["contraction_compute_dtype"] == "float32"
    assert np.linalg.eigvalsh(stiffness)[0] > 0.0


def test_raw_ritz_incremental_extension_matches_full_assembly():
    rng = np.random.default_rng(20260913)
    shape = (5, 4, 3)
    phase = np.zeros(shape, dtype=np.uint8)
    phase.reshape(-1)[::3] = 1
    ori = np.zeros(shape + (3,), dtype=np.float64)
    ori[..., 0] = 1.0
    orthonormal = np.stack(_orthonormal_fields(rng, 9, (6,) + shape))
    mixing = np.eye(9) + 0.05 * rng.standard_normal((9, 9))
    raw = (mixing @ orthonormal.reshape(9, -1)).reshape(orthonormal.shape)
    order = reduced.phase_orientation_voxel_order(phase, ori)
    ordered_phase = phase.reshape(-1)[order]
    ordered_ori = ori.reshape(-1, 3)[order]
    raw = np.take(raw.reshape(9, 6, -1), order, axis=2)
    split = 5
    affine = reduced.affine_stress_batch_factory(ordered_phase, ordered_ori)

    K0, B0, D0, first = reduced._assemble_reduced_operators(
        phase=ordered_phase,
        ori=ordered_ori,
        basis=raw[:split],
        affine_stress_batch=affine,
        gram_rank_reveal=True,
    )
    existing = {
        "Kq": K0,
        "Bq": B0,
        "Dq": D0,
        "raw_Kq": first["raw_Kq"],
        "raw_Bq": first["raw_Bq"],
        "G": first["G"],
        "invR": first["invR"],
    }
    Ki, Bi, Di, _ = reduced._extend_reduced_operators(
        existing=existing,
        old_basis=raw[:split],
        new_basis=raw[split:],
        affine_stress_batch=affine,
        gram_rank_reveal=True,
    )
    Kf, Bf, Df, _ = reduced._assemble_reduced_operators(
        phase=ordered_phase,
        ori=ordered_ori,
        basis=raw,
        affine_stress_batch=affine,
        gram_rank_reveal=True,
    )

    np.testing.assert_allclose(Ki, Kf, rtol=3.0e-12, atol=3.0e-12)
    np.testing.assert_allclose(Bi, Bf, rtol=3.0e-12, atol=3.0e-12)
    np.testing.assert_allclose(Di, Df, rtol=3.0e-13, atol=3.0e-13)


def test_contiguous_basis_blocked_projection_preserves_subspace():
    rng = np.random.default_rng(20260904)
    shape = (6, 5, 4, 3)
    initial = rng.standard_normal((6,) + shape).astype(np.float32)
    incoming = rng.standard_normal((13,) + shape).astype(np.float32)
    unblocked = common.ContiguousBasis(
        19, shape, dtype=np.float32, projection_row_block_size=len(incoming)
    )
    blocked = common.ContiguousBasis(
        19, shape, dtype=np.float32, projection_row_block_size=3
    )

    unblocked.append_preordered(initial.copy(order="C"), tolerance=1.0e-12)
    blocked.append_preordered(initial.copy(order="C"), tolerance=1.0e-12)
    unblocked.append_preordered(incoming.copy(order="C"), tolerance=1.0e-12)
    blocked.append_preordered(incoming.copy(order="C"), tolerance=1.0e-12)

    overlap = unblocked.active_flat @ blocked.active_flat.T / np.float32(np.prod(shape))
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    np.testing.assert_allclose(singular_values, 1.0, rtol=5.0e-6, atol=5.0e-6)
    assert blocked.last_projection_backend == "scipy_blas_gemm_in_place"


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
    affine = reduced.affine_stress_batch_factory(phase, ori)
    Ki, Bi, Di, metadata = reduced._extend_reduced_operators(
        existing={"Kq": K0, "Bq": B0, "Dq": D0},
        old_basis=old_basis,
        new_basis=new_basis,
        affine_stress_batch=affine,
    )
    Kf, Bf, Df, _ = reduced._assemble_reduced_operators(
        phase=phase, ori=ori, basis=fields
    )
    np.testing.assert_allclose(Ki, Kf, rtol=3.0e-13, atol=3.0e-13)
    np.testing.assert_allclose(Bi, Bf, rtol=3.0e-13, atol=3.0e-13)
    np.testing.assert_allclose(Di, Df, rtol=3.0e-13, atol=3.0e-13)
    assert metadata["assembly_mode"] == "incremental"


def test_vectorized_affine_assembly_matches_direct_contractions():
    rng = np.random.default_rng(20260827)
    shape = (4, 3, 2)
    phase = np.zeros(shape, dtype=np.uint8)
    phase[1::2] = 1
    ori = np.zeros(shape + (3,), dtype=np.float64)
    orientations = np.eye(3)
    fiber_indices = np.flatnonzero(phase.reshape(-1) != 0)
    ori.reshape(-1, 3)[fiber_indices] = orientations[
        np.arange(len(fiber_indices)) % len(orientations)
    ]
    basis = _orthonormal_fields(rng, 7, (6,) + shape)
    values = np.stack(basis).reshape(len(basis), 6, -1)
    affine = reduced.affine_stress_batch_factory(phase, ori)

    Kq, Bq, Dq, metadata = reduced._assemble_reduced_operators(
        phase=phase,
        ori=ori,
        basis=basis,
        affine_stress_batch=affine,
    )
    expected_K = []
    expected_B = []
    for q in range(len(reduced.COEFF_NAMES)):
        stress = affine(q, values)
        expected_K.append(
            np.einsum("ian,jan->ij", values, stress, optimize=True) / phase.size
        )
        expected_B.append(np.mean(stress, axis=2))
    np.testing.assert_allclose(Kq, np.stack(expected_K), rtol=3.0e-13, atol=3.0e-13)
    np.testing.assert_allclose(Bq, np.stack(expected_B), rtol=3.0e-13, atol=3.0e-13)
    np.testing.assert_allclose(
        Dq,
        affine.averaged_stiffness,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    assert metadata["assembly_mode"] == "batched_affine_cpu"

    stresses = affine.apply_all(values)
    np.testing.assert_allclose(
        stresses,
        np.stack([affine(q, values) for q in range(len(reduced.COEFF_NAMES))]),
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    selected = np.array([0, 2, 5])
    np.testing.assert_allclose(
        affine.apply_indices(selected, values),
        np.stack([affine(int(q), values) for q in selected]),
        rtol=3.0e-13,
        atol=3.0e-13,
    )


def test_gpu_supported_affine_chunks_match_cpu():
    try:
        import cupy as cp

        cp.cuda.Device().compute_capability
    except Exception:
        pytest.skip("CUDA/CuPy is unavailable")
    rng = np.random.default_rng(20260914)
    shape = (5, 4, 3)
    phase = np.zeros(shape, dtype=np.uint8)
    phase.reshape(-1)[::3] = 1
    ori = np.zeros(shape + (3,), dtype=np.float32)
    fiber = np.flatnonzero(phase.reshape(-1))
    ori.reshape(-1, 3)[fiber, np.arange(len(fiber)) % 3] = 1.0
    order = reduced.phase_orientation_voxel_order(phase, ori)
    ordered_phase = phase.reshape(-1)[order]
    ordered_ori = ori.reshape(-1, 3)[order]
    affine = reduced.affine_stress_batch_factory(ordered_phase, ordered_ori)

    for support, indices, selector in affine.support_blocks:
        count = int(selector.stop) - int(selector.start)
        values = rng.standard_normal((7, 6, count)).astype(np.float32)
        expected = affine.apply_supported_chunk(indices, values, support, 0)
        actual = cp.asnumpy(
            affine.apply_supported_chunk_gpu(
                indices,
                cp.asarray(values.reshape(len(values), -1)),
                support,
                0,
            )
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)


def test_gpu_affine_failure_falls_back_to_cpu_without_changing_operators():
    try:
        import cupy as cp

        cp.cuda.Device().compute_capability
    except Exception:
        pytest.skip("CUDA/CuPy is unavailable")
    rng = np.random.default_rng(20260915)
    shape = (5, 4, 3)
    phase = np.zeros(shape, dtype=np.uint8)
    phase.reshape(-1)[::3] = 1
    ori = np.zeros(shape + (3,), dtype=np.float32)
    ori[..., 0] = 1.0
    order = reduced.phase_orientation_voxel_order(phase, ori)
    ordered_phase = phase.reshape(-1)[order]
    ordered_ori = ori.reshape(-1, 3)[order]
    basis = np.take(
        np.stack(_orthonormal_fields(rng, 7, (6,) + shape))
        .astype(np.float32)
        .reshape(7, 6, -1),
        order,
        axis=2,
    )
    cpu_affine = reduced.affine_stress_batch_factory(ordered_phase, ordered_ori)
    cpu_affine.apply_supported_chunk_gpu = None
    expected = reduced._assemble_reduced_operators(
        phase=ordered_phase,
        ori=ordered_ori,
        basis=basis,
        affine_stress_batch=cpu_affine,
    )[:3]

    fallback_affine = reduced.affine_stress_batch_factory(ordered_phase, ordered_ori)

    def fail_gpu(*_args, **_kwargs):
        raise RuntimeError("intentional GPU test failure")

    fallback_affine.apply_supported_chunk_gpu = fail_gpu
    *actual, metadata = reduced._assemble_reduced_operators(
        phase=ordered_phase,
        ori=ordered_ori,
        basis=basis,
        affine_stress_batch=fallback_affine,
    )

    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=2.0e-6, atol=2.0e-6)
    assert metadata["affine_stress_backend"] == "cpu"
    assert metadata["cpu_affine_chunks"] > 0
    assert "intentional GPU test failure" in metadata["gpu_affine_fallback"]


def test_affine_coefficient_blocks_preserve_reduced_operators():
    rng = np.random.default_rng(20260903)
    shape = (5, 4, 3)
    phase = np.zeros(shape, dtype=np.uint8)
    phase.reshape(-1)[::3] = 1
    ori = np.zeros(shape + (3,), dtype=np.float32)
    fiber = np.flatnonzero(phase.reshape(-1))
    ori.reshape(-1, 3)[fiber, np.arange(len(fiber)) % 3] = 1.0
    basis = np.stack(_orthonormal_fields(rng, 8, (6,) + shape)).astype(np.float32)

    expected = reduced._assemble_reduced_operators(
        phase=phase,
        ori=ori,
        basis=basis,
    )[:3]
    blocked = reduced._assemble_reduced_operators(
        phase=phase,
        ori=ori,
        basis=basis,
        affine_q_block_size=2,
    )[:3]
    for actual, reference in zip(blocked, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2.0e-6, atol=2.0e-6)


def test_float32_incremental_ritz_matches_float64_assembly():
    rng = np.random.default_rng(20260831)
    shape = (5, 4, 3)
    phase = np.zeros(shape, dtype=np.uint8)
    phase[1::2] = 1
    ori = np.zeros(shape + (3,), dtype=np.float32)
    ori[..., 0] = 1.0
    basis64 = np.stack(_orthonormal_fields(rng, 9, (6,) + shape))
    basis32 = basis64.astype(np.float32)
    split = 5

    K0, B0, D0, _ = reduced._assemble_reduced_operators(
        phase=phase,
        ori=ori,
        basis=basis32[:split],
    )
    affine = reduced.affine_stress_batch_factory(phase, ori)
    Ki, Bi, Di, metadata = reduced._extend_reduced_operators(
        existing={"Kq": K0, "Bq": B0, "Dq": D0},
        old_basis=basis32[:split],
        new_basis=basis32[split:],
        affine_stress_batch=affine,
    )
    Kf, Bf, Df, _ = reduced._assemble_reduced_operators(
        phase=phase,
        ori=ori,
        basis=basis64,
    )

    np.testing.assert_allclose(Ki, Kf, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(Bi, Bf, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(Di, Df, rtol=3.0e-13, atol=3.0e-13)
    assert metadata["contraction_dtype"] == "float32"
    assert metadata["contraction_compute_dtype"] == "float64"
    assert metadata["gram_product_dtype"] == "float64"


def test_cuda_ritz_upload_promotes_float32_storage_to_float64():
    values = np.arange(48, dtype=np.float32).reshape(2, 6, 4)
    values_gpu = reduced._gpu_flat_compute(values)
    if values_gpu is None:
        pytest.skip("CUDA is unavailable")

    import cupy as cp

    assert values_gpu.dtype == cp.float64
    stresses_gpu = reduced._gpu_batch_flat_compute(values[None, ...])
    assert stresses_gpu is not None
    assert stresses_gpu.dtype == cp.float64


def test_phase_orientation_permutation_preserves_ritz_operators():
    rng = np.random.default_rng(20260901)
    shape = (5, 4, 3)
    phase = np.zeros(shape, dtype=np.uint8)
    phase.reshape(-1)[::3] = 1
    ori = np.zeros(shape + (3,), dtype=np.float32)
    fiber = np.flatnonzero(phase.reshape(-1))
    ori.reshape(-1, 3)[fiber, np.arange(len(fiber)) % 3] = 1.0
    basis = np.stack(_orthonormal_fields(rng, 8, (6,) + shape)).astype(np.float32)

    K0, B0, D0, _ = reduced._assemble_reduced_operators(
        phase=phase,
        ori=ori,
        basis=basis,
    )
    order = reduced.phase_orientation_voxel_order(phase, ori)
    ordered_basis = np.take(basis.reshape(8, 6, -1), order, axis=2)
    K1, B1, D1, _ = reduced._assemble_reduced_operators(
        phase=phase.reshape(-1)[order],
        ori=ori.reshape(-1, 3)[order],
        basis=ordered_basis,
    )

    np.testing.assert_allclose(K1, K0, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(B1, B0, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(D1, D0, rtol=3.0e-13, atol=3.0e-13)


def test_phase_supported_incremental_ritz_is_exact_and_memory_bounded():
    rng = np.random.default_rng(20260904)
    shape = (6, 5, 4)
    phase = np.zeros(shape, dtype=np.uint8)
    phase.reshape(-1)[::4] = 1
    ori = np.zeros(shape + (3,), dtype=np.float32)
    fiber = np.flatnonzero(phase.reshape(-1))
    ori.reshape(-1, 3)[fiber, np.arange(len(fiber)) % 3] = 1.0
    basis = np.stack(_orthonormal_fields(rng, 9, (6,) + shape)).astype(np.float32)
    order = reduced.phase_orientation_voxel_order(phase, ori)
    ordered_phase = phase.reshape(-1)[order]
    ordered_ori = ori.reshape(-1, 3)[order]
    ordered_basis = np.take(basis.reshape(9, 6, -1), order, axis=2)
    affine = reduced.affine_stress_batch_factory(ordered_phase, ordered_ori)

    K0, B0, D0, _ = reduced._assemble_reduced_operators(
        phase=ordered_phase,
        ori=ordered_ori,
        basis=ordered_basis[:5],
        affine_stress_batch=affine,
    )
    Ki, Bi, Di, metadata = reduced._extend_reduced_operators(
        existing={"Kq": K0, "Bq": B0, "Dq": D0},
        old_basis=ordered_basis[:5],
        new_basis=ordered_basis[5:],
        affine_stress_batch=affine,
    )
    Kf, Bf, Df, _ = reduced._assemble_reduced_operators(
        phase=ordered_phase,
        ori=ordered_ori,
        basis=ordered_basis,
        affine_stress_batch=affine,
    )

    np.testing.assert_allclose(Ki, Kf, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(Bi, Bf, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(Di, Df, rtol=3.0e-13, atol=3.0e-13)
    expected_passes = 2.0 * 0.75 + 5.0 * 0.25
    assert metadata["contraction_mode"] == "phase_supported_blocks"
    assert np.isclose(metadata["full_volume_equivalent_passes"], expected_passes)
    dense_workspace = 7 * len(ordered_basis[5:]) * 6 * phase.size * 4
    assert metadata["stress_workspace_peak_bytes"] < dense_workspace


def test_vectorized_material_and_engineering_maps_match_scalar_references():
    rows = [
        {
            "Em": 2.5,
            "nu_m": 0.31,
            "Ef_L": 71.0,
            "Ef_T": 11.0,
            "G_LT": 4.8,
            "nu_LT": 0.23,
            "nu_TT": 0.36,
        },
        {
            "Em": 7.0,
            "nu_m": 0.44,
            "Ef_L": 210.0,
            "Ef_T": 19.0,
            "G_LT": 7.2,
            "nu_LT": 0.18,
            "nu_TT": 0.29,
        },
    ]
    parameters = np.asarray(
        [[row[name] for name in reduced.MATERIAL_PARAMETER_COLUMNS] for row in rows]
    )
    coefficients = reduced._material_coefficients_batch(parameters)
    expected_coefficients = np.stack(
        [reduced._material_coefficients(row) for row in rows]
    )
    np.testing.assert_allclose(
        coefficients, expected_coefficients, rtol=3.0e-13, atol=3.0e-13
    )

    rng = np.random.default_rng(20260907)
    factors = rng.standard_normal((5, 6, 6))
    matrices = factors @ np.swapaxes(factors, -1, -2) + np.eye(6)[None]
    properties = reduced._engineering_constants_batch(matrices)
    expected_properties = np.asarray(
        [
            [
                reduced.engineering_constants_from_Cmandel(matrix)[name]
                for name in reduced.ENGINEERING_COLUMNS
            ]
            for matrix in matrices
        ]
    )
    np.testing.assert_allclose(
        properties, expected_properties, rtol=3.0e-13, atol=3.0e-13
    )


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
