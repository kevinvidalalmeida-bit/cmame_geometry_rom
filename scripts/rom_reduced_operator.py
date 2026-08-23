#!/usr/bin/env python3
"""Compile a tangential reduced operator from fixed-geometry FFTHomPy fields."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg as scipy_linalg

try:
    from numba import get_num_threads as _numba_get_num_threads
    from numba import njit as _numba_njit
    from numba import prange as _numba_prange
    from numba import set_num_threads as _numba_set_num_threads

    _NUMBA_TRIANGULAR_AVAILABLE = True
except ImportError:
    _NUMBA_TRIANGULAR_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FFT_ROOT = PROJECT_ROOT / "FFT"
FFTHOMPY_PATH = FFT_ROOT / "ffthompy_core" / "ffthompy"

RUN_DEFAULT = (
    PROJECT_ROOT
    / "results"
    / "fixed_geometry_ffthompy"
    / "fixed_geometry_ar15_vf20_sobol8_center_fields"
)

COEFF_NAMES = [
    "matrix_lambda",
    "matrix_mu",
    "fiber_C_TT",
    "fiber_C_TT_cross",
    "fiber_C_LT",
    "fiber_C_LL",
    "fiber_G_LT",
]

DUAL_COEFF_NAMES = [
    "matrix_inv_E",
    "matrix_nu_over_E",
    "fiber_inv_E_L",
    "fiber_inv_E_T",
    "fiber_nu_LT_over_E_L",
    "fiber_nu_TT_over_E_T",
    "fiber_inv_2G_LT",
]

ENGINEERING_COLUMNS = [
    "E1",
    "E2",
    "E3",
    "G12",
    "G13",
    "G23",
    "nu12",
    "nu13",
    "nu23",
]

MATERIAL_PARAMETER_COLUMNS = (
    "Em",
    "nu_m",
    "Ef_L",
    "Ef_T",
    "G_LT",
    "nu_LT",
    "nu_TT",
)


if _NUMBA_TRIANGULAR_AVAILABLE:

    @_numba_njit(cache=True, parallel=True)
    def _forward_lower_numba(lower: np.ndarray, right: np.ndarray) -> np.ndarray:
        result = np.empty_like(right)
        for batch in _numba_prange(lower.shape[0]):
            for row in range(lower.shape[1]):
                for column in range(right.shape[2]):
                    value = right[batch, row, column]
                    for previous in range(row):
                        value -= lower[batch, row, previous] * result[batch, previous, column]
                    result[batch, row, column] = value / lower[batch, row, row]
        return result


    @_numba_njit(cache=True, parallel=True)
    def _backward_lower_transpose_numba(
        lower: np.ndarray, right: np.ndarray
    ) -> np.ndarray:
        result = np.empty_like(right)
        for batch in _numba_prange(lower.shape[0]):
            for row in range(lower.shape[1] - 1, -1, -1):
                for column in range(right.shape[2]):
                    value = right[batch, row, column]
                    for following in range(row + 1, lower.shape[1]):
                        value -= lower[batch, following, row] * result[batch, following, column]
                    result[batch, row, column] = value / lower[batch, row, row]
        return result


def configure_incremental_batch_threads(count: int = 16) -> dict[str, int | str | bool]:
    """Configure the optional parallel triangular kernel used by acquisition."""
    requested = int(count)
    if requested < 1:
        raise ValueError("triangular worker count must be positive.")
    if not _NUMBA_TRIANGULAR_AVAILABLE:
        return {"backend": "numpy", "available": False, "workers": 1}
    _numba_set_num_threads(requested)
    return {
        "backend": "numba_parallel",
        "available": True,
        "workers": int(_numba_get_num_threads()),
    }


def _add_fft_paths() -> None:
    for path in (FFT_ROOT, FFTHOMPY_PATH):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


_add_fft_paths()

from ffthompy_core.ffthompy.RESU import engineering_constants_from_Cmandel
from pipeline.fft_solver import (
    AFFINE_ORIENTATION_QUANTIZATION,
    TI_stiffness_voigt,
    _affine_stiffness_bases,
    _group_quantized_orientations,
    rotate_C_mandel,
    rotation_matrix_from_vector,
    voigt_to_mandel,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(float).eps))


def _isotropic_bases() -> tuple[np.ndarray, np.ndarray]:
    matrix_bases, _ = _affine_stiffness_bases()
    return matrix_bases


def _fiber_local_bases_axis0() -> list[np.ndarray]:
    """Five TI Mandel bases with local axis 0 as the fiber direction."""
    _, fiber_bases = _affine_stiffness_bases()
    return list(fiber_bases)


def _mandel_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return the orthogonal Mandel map from local to global components."""
    R = np.asarray(rotation, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    factors = np.array([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])
    transform = np.empty((6, 6), dtype=np.float64)
    for column, ((i, j), factor) in enumerate(zip(pairs, factors, strict=True)):
        local = np.zeros((3, 3), dtype=np.float64)
        local[i, j] = 1.0 / factor
        local[j, i] = 1.0 / factor
        global_tensor = R @ local @ R.T
        transform[:, column] = np.array(
            [
                global_tensor[0, 0],
                global_tensor[1, 1],
                global_tensor[2, 2],
                np.sqrt(2.0) * global_tensor[1, 2],
                np.sqrt(2.0) * global_tensor[0, 2],
                np.sqrt(2.0) * global_tensor[0, 1],
            ],
            dtype=np.float64,
        )
    return transform


def _isotropic_compliance_bases() -> tuple[np.ndarray, np.ndarray]:
    """Two isotropic compliance bases in Mandel notation."""
    inv_e = np.eye(6, dtype=float)
    nu_over_e = np.zeros((6, 6), dtype=float)
    nu_over_e[:3, :3] = -1.0
    np.fill_diagonal(nu_over_e[:3, :3], 0.0)
    nu_over_e[3, 3] = nu_over_e[4, 4] = nu_over_e[5, 5] = 1.0
    return inv_e, nu_over_e


def _fiber_local_compliance_bases_axis0() -> list[np.ndarray]:
    """Five transversely isotropic compliance bases in Mandel notation."""
    inv_e_l = np.zeros((6, 6), dtype=float)
    inv_e_t = np.zeros((6, 6), dtype=float)
    nu_lt_over_e_l = np.zeros((6, 6), dtype=float)
    nu_tt_over_e_t = np.zeros((6, 6), dtype=float)
    inv_2g_lt = np.zeros((6, 6), dtype=float)

    inv_e_l[0, 0] = 1.0
    inv_e_t[1, 1] = inv_e_t[2, 2] = 1.0
    inv_e_t[3, 3] = 1.0

    nu_lt_over_e_l[0, 1] = nu_lt_over_e_l[1, 0] = -1.0
    nu_lt_over_e_l[0, 2] = nu_lt_over_e_l[2, 0] = -1.0

    nu_tt_over_e_t[1, 2] = nu_tt_over_e_t[2, 1] = -1.0
    nu_tt_over_e_t[3, 3] = 1.0

    inv_2g_lt[4, 4] = inv_2g_lt[5, 5] = 1.0
    return [
        inv_e_l,
        inv_e_t,
        nu_lt_over_e_l,
        nu_tt_over_e_t,
        inv_2g_lt,
    ]


def phase_orientation_voxel_order(phase: np.ndarray, ori: np.ndarray) -> np.ndarray:
    """Return a deterministic phase/orientation order for contiguous kernels."""
    phase_flat = np.asarray(phase).reshape(-1)
    ori_flat = np.asarray(ori).reshape(-1, 3)
    matrix_idx = np.flatnonzero(phase_flat == 0)
    fiber_idx = np.flatnonzero(phase_flat != 0)
    if not len(fiber_idx):
        return matrix_idx.astype(np.int64, copy=False)
    group_ids, _ = _group_quantized_orientations(
        ori_flat[fiber_idx], AFFINE_ORIENTATION_QUANTIZATION
    )
    fiber_order = np.argsort(group_ids, kind="stable")
    return np.concatenate((matrix_idx, fiber_idx[fiber_order])).astype(np.int64, copy=False)


def _contiguous_selector(indices: np.ndarray) -> slice | np.ndarray:
    if not len(indices):
        return slice(0, 0)
    start = int(indices[0])
    if int(indices[-1]) - start + 1 == len(indices):
        return slice(start, start + len(indices))
    return indices


def _exact_spectral_factorization(
    bases_by_group: np.ndarray,
) -> dict[str, Any]:
    """Pack exact symmetric basis factorizations by constitutive coefficient.

    Each local affine basis is represented as ``U diag(weights) U.T``. The
    zero eigendirections are arithmetic zeros of the 6x6 constitutive basis,
    not a reduced-order truncation. Rotations preserve the nonzero spectra,
    which lets all orientation groups share one packed coefficient layout.
    """
    bases = np.asarray(bases_by_group, dtype=np.float64)
    if bases.ndim != 4 or bases.shape[-2:] != (6, 6):
        raise ValueError("factorized affine bases must have shape (g, q, 6, 6)")
    group_count, q_count = bases.shape[:2]
    if group_count < 1:
        return {
            "factors": np.empty((0, 6, 0), dtype=np.float64),
            "weights": np.empty(0, dtype=np.float64),
            "coefficient_slices": tuple(),
            "ranks": tuple(),
        }

    factors_by_q: list[np.ndarray] = []
    weights_by_q: list[np.ndarray] = []
    ranks: list[int] = []
    for q in range(q_count):
        group_factors: list[np.ndarray] = []
        reference_weights: np.ndarray | None = None
        for group in range(group_count):
            symmetric = 0.5 * (bases[group, q] + bases[group, q].T)
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
            scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
            keep = np.abs(eigenvalues) > 256.0 * np.finfo(np.float64).eps * scale
            selected_weights = np.asarray(eigenvalues[keep], dtype=np.float64)
            selected_factors = np.asarray(eigenvectors[:, keep], dtype=np.float64)
            if reference_weights is None:
                reference_weights = selected_weights
            elif not np.allclose(
                selected_weights,
                reference_weights,
                rtol=2.0e-12,
                atol=2.0e-12,
            ):
                raise RuntimeError("rotated affine basis changed its exact spectrum")
            group_factors.append(selected_factors)
        assert reference_weights is not None
        factors_by_q.append(np.stack(group_factors, axis=0))
        weights_by_q.append(reference_weights)
        ranks.append(int(reference_weights.size))

    offsets = np.concatenate(([0], np.cumsum(ranks, dtype=np.int64)))
    packed = np.empty((group_count, 6, int(offsets[-1])), dtype=np.float64)
    coefficient_slices: list[tuple[int, int]] = []
    for q, factors in enumerate(factors_by_q):
        start = int(offsets[q])
        stop = int(offsets[q + 1])
        packed[:, :, start:stop] = factors
        coefficient_slices.append((start, stop))
    return {
        "factors": packed,
        "weights": np.concatenate(weights_by_q),
        "coefficient_slices": tuple(coefficient_slices),
        "ranks": tuple(ranks),
    }


def _affine_tensor_batch_factory(
    phase: np.ndarray,
    ori: np.ndarray,
    *,
    matrix_bases: tuple[np.ndarray, np.ndarray],
    fiber_local_bases: list[np.ndarray],
    coefficient_names: list[str],
    local_frame_snapshots: bool = False,
    gathered_factor_ritz: bool = False,
) -> Any:
    """Build a phase/orientation-aware affine fourth-order tensor map."""
    phase_flat = np.asarray(phase).reshape(-1)
    ori_flat = np.asarray(ori, dtype=float).reshape(-1, 3)
    matrix_idx = np.flatnonzero(phase_flat == 0)
    fiber_idx = np.flatnonzero(phase_flat != 0)
    matrix_selector = _contiguous_selector(matrix_idx)
    fiber_selector = _contiguous_selector(fiber_idx)
    fiber_group_ids = np.empty(0, dtype=np.int32)
    fiber_bases_by_group = np.empty((0, 5, 6, 6), dtype=np.float64)
    fiber_mandel_rotations = np.empty((0, 6, 6), dtype=np.float64)
    if len(fiber_idx):
        inverse, unique_oris = _group_quantized_orientations(
            ori_flat[fiber_idx], AFFINE_ORIENTATION_QUANTIZATION
        )
        group_bases: list[np.ndarray] = []
        group_rotations: list[np.ndarray] = []
        for axis in unique_oris:
            rotation = rotation_matrix_from_vector(axis)
            group_rotations.append(_mandel_rotation_matrix(rotation))
            group_bases.append(np.stack(
                [rotate_C_mandel(basis, rotation) for basis in fiber_local_bases],
                axis=0,
            ))
        fiber_group_ids = np.asarray(inverse, dtype=np.int32)
        fiber_bases_by_group = np.stack(group_bases, axis=0)
        fiber_mandel_rotations = np.stack(group_rotations, axis=0)

    q_count = len(coefficient_names)
    nvox = max(1, int(phase_flat.size))
    averaged_stiffness = np.zeros((q_count, 6, 6), dtype=np.float64)
    matrix_fraction = float(len(matrix_idx)) / float(nvox)
    for q, matrix_basis in enumerate(matrix_bases):
        averaged_stiffness[q] = matrix_fraction * matrix_basis
    if len(fiber_idx):
        group_counts = np.bincount(
            fiber_group_ids,
            minlength=len(fiber_bases_by_group),
        ).astype(np.float64)
        averaged_stiffness[2:] = np.einsum(
            "g,gqab->qab",
            group_counts / float(nvox),
            fiber_bases_by_group,
            optimize=True,
        )
    matrix_bases_by_dtype = {
        np.dtype(np.float32): np.asarray(matrix_bases, dtype=np.float32),
        np.dtype(np.float64): np.asarray(matrix_bases, dtype=np.float64),
    }
    fiber_bases_by_dtype = {
        np.dtype(np.float32): np.asarray(fiber_bases_by_group, dtype=np.float32),
        np.dtype(np.float64): np.asarray(fiber_bases_by_group, dtype=np.float64),
    }
    exact_factorizations = {
        "matrix": _exact_spectral_factorization(
            np.asarray(matrix_bases, dtype=np.float64)[None, ...]
        ),
        "fiber": (
            _exact_spectral_factorization(fiber_bases_by_group)
            if len(fiber_bases_by_group)
            else {
                "factors": np.empty((0, 6, 0), dtype=np.float64),
                "weights": np.empty(0, dtype=np.float64),
                "coefficient_slices": tuple(),
                "ranks": tuple(),
            }
        ),
    }
    local_exact_factorizations = {
        "matrix": exact_factorizations["matrix"],
        "fiber": (
            _exact_spectral_factorization(
                np.asarray(fiber_local_bases, dtype=np.float64)[None, ...]
            )
            if len(fiber_idx)
            else exact_factorizations["fiber"]
        ),
    }
    stiffness_matrix_fast_path = tuple(coefficient_names) == tuple(COEFF_NAMES)
    fiber_group_runs: tuple[tuple[int, int, int], ...] = ()
    if (
        len(fiber_group_ids)
        and isinstance(fiber_selector, slice)
        and np.all(fiber_group_ids[1:] >= fiber_group_ids[:-1])
    ):
        starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.flatnonzero(np.diff(fiber_group_ids)) + 1)
        )
        ends = np.concatenate((starts[1:], np.array([len(fiber_group_ids)])))
        fiber_group_runs = tuple(
            (int(fiber_group_ids[start]), int(start), int(end))
            for start, end in zip(starts, ends, strict=True)
        )
    gpu_basis_cache: dict[str, tuple[Any, Any]] = {}

    def apply(q: int, strains: np.ndarray) -> np.ndarray:
        values = np.asarray(strains)
        dtype = values.dtype if values.dtype in matrix_bases_by_dtype else np.dtype(np.float64)
        values = np.asarray(values, dtype=dtype)
        stress = np.zeros_like(values)
        if q < 2:
            if len(matrix_idx):
                stress[:, :, matrix_selector] = np.einsum(
                    "ab,lbn->lan",
                    matrix_bases_by_dtype[dtype][q],
                    values[:, :, matrix_selector],
                    optimize=True,
                )
        else:
            local_q = int(q) - 2
            if len(fiber_idx):
                voxel_stiffness = fiber_bases_by_dtype[dtype][:, local_q][fiber_group_ids]
                stress[:, :, fiber_selector] = np.einsum(
                    "nab,lbn->lan",
                    voxel_stiffness,
                    values[:, :, fiber_selector],
                    optimize=True,
                )
        return stress

    def apply_indices(q_indices: Any, strains: np.ndarray) -> np.ndarray:
        selected = np.asarray(q_indices, dtype=np.intp).reshape(-1)
        if np.any((selected < 0) | (selected >= q_count)):
            raise ValueError("affine coefficient index is out of range.")
        values = np.asarray(strains)
        dtype = values.dtype if values.dtype in matrix_bases_by_dtype else np.dtype(np.float64)
        values = np.asarray(values, dtype=dtype)
        if not isinstance(matrix_selector, slice) or not isinstance(fiber_selector, slice):
            return np.stack([apply(int(q), values) for q in selected])
        stresses = np.zeros((len(selected),) + values.shape, dtype=dtype)
        if len(matrix_idx):
            matrix_values = values[:, :, matrix_selector]
            if stiffness_matrix_fast_path:
                for output_index, q in enumerate(selected):
                    if q == 0:
                        normal_sum = np.sum(matrix_values[:, :3], axis=1, dtype=dtype)
                        stresses[output_index, :, :3, matrix_selector] = normal_sum[:, None]
                    elif q == 1:
                        np.multiply(
                            matrix_values,
                            dtype.type(2.0),
                            out=stresses[output_index, :, :, matrix_selector],
                        )
            else:
                matrix_outputs = np.flatnonzero(selected < 2)
                matrix_indices = selected[matrix_outputs]
                if len(matrix_outputs):
                    matrix_applied = np.einsum(
                        "qab,lbn->qlan",
                        matrix_bases_by_dtype[dtype][matrix_indices],
                        matrix_values,
                        optimize=True,
                    )
                    stresses[matrix_outputs, :, :, matrix_selector] = matrix_applied
        if len(fiber_idx):
            fiber_values = values[:, :, fiber_selector]
            output_indices = np.flatnonzero(selected >= 2)
            local_indices = selected[output_indices] - 2
            contiguous_outputs = bool(len(output_indices)) and np.array_equal(
                output_indices,
                np.arange(output_indices[0], output_indices[-1] + 1),
            )
            if fiber_group_runs and contiguous_outputs:
                fiber_stresses = stresses[:, :, :, fiber_selector]
                output_slice = slice(int(output_indices[0]), int(output_indices[-1]) + 1)
                group_bases = fiber_bases_by_dtype[dtype]
                for group, start, end in fiber_group_runs:
                    np.einsum(
                        "qab,lbn->qlan",
                        group_bases[group, local_indices],
                        fiber_values[:, :, start:end],
                        optimize=True,
                        out=fiber_stresses[output_slice, :, :, start:end],
                    )
            else:
                for output_index, local_q in zip(
                    output_indices, local_indices, strict=True
                ):
                    voxel_stiffness = fiber_bases_by_dtype[dtype][:, local_q][fiber_group_ids]
                    np.einsum(
                        "nab,lbn->lan",
                        voxel_stiffness,
                        fiber_values,
                        optimize=True,
                        out=stresses[output_index, :, :, fiber_selector],
                    )
        return stresses

    def apply_supported_indices(
        q_indices: Any,
        strains: np.ndarray,
        support: str,
    ) -> np.ndarray:
        """Apply coefficients only where they are nonzero in the sorted phase field."""
        selected = np.asarray(q_indices, dtype=np.intp).reshape(-1)
        values = np.asarray(strains)
        dtype = values.dtype if values.dtype in matrix_bases_by_dtype else np.dtype(np.float64)
        values = np.asarray(values, dtype=dtype)
        component_major = values.ndim == 3 and values.shape[1] == 6
        voxel_major = values.ndim == 3 and values.shape[-1] == 6
        if not component_major and not voxel_major:
            raise ValueError("supported strains must have a six-component axis.")
        stresses = np.zeros((len(selected),) + values.shape, dtype=dtype)
        if support == "matrix":
            if np.any((selected < 0) | (selected >= 2)):
                raise ValueError("matrix support only accepts coefficients 0 and 1.")
            support_voxels = values.shape[-1] if component_major else values.shape[1]
            if support_voxels != len(matrix_idx):
                raise ValueError("matrix support has an incompatible voxel count.")
            if stiffness_matrix_fast_path:
                for output_index, q in enumerate(selected):
                    if q == 0 and component_major:
                        normal_sum = np.sum(values[:, :3], axis=1, dtype=dtype)
                        stresses[output_index, :, :3] = normal_sum[:, None]
                    elif q == 0:
                        normal_sum = np.sum(values[:, :, :3], axis=2, dtype=dtype)
                        stresses[output_index, :, :, :3] = normal_sum[:, :, None]
                    else:
                        np.multiply(values, dtype.type(2.0), out=stresses[output_index])
            else:
                if component_major:
                    stresses[:] = np.einsum(
                        "qab,lbn->qlan",
                        matrix_bases_by_dtype[dtype][selected],
                        values,
                        optimize=True,
                    )
                else:
                    stresses[:] = np.einsum(
                        "qab,lnb->qlna",
                        matrix_bases_by_dtype[dtype][selected],
                        values,
                        optimize=True,
                    )
            return stresses
        if support != "fiber":
            raise ValueError("support must be 'matrix' or 'fiber'.")
        if np.any((selected < 2) | (selected >= q_count)):
            raise ValueError("fiber support only accepts coefficients 2 and above.")
        support_voxels = values.shape[-1] if component_major else values.shape[1]
        if support_voxels != len(fiber_idx):
            raise ValueError("fiber support has an incompatible voxel count.")
        local_indices = selected - 2
        if fiber_group_runs:
            group_bases = fiber_bases_by_dtype[dtype]
            for group, start, end in fiber_group_runs:
                if component_major:
                    np.einsum(
                        "qab,lbn->qlan",
                        group_bases[group, local_indices],
                        values[:, :, start:end],
                        optimize=True,
                        out=stresses[:, :, :, start:end],
                    )
                else:
                    np.einsum(
                        "qab,lnb->qlna",
                        group_bases[group, local_indices],
                        values[:, start:end, :],
                        optimize=True,
                        out=stresses[:, :, start:end, :],
                    )
        else:
            for output_index, local_q in enumerate(local_indices):
                voxel_stiffness = fiber_bases_by_dtype[dtype][:, local_q][fiber_group_ids]
                if component_major:
                    np.einsum(
                        "nab,lbn->lan",
                        voxel_stiffness,
                        values,
                        optimize=True,
                        out=stresses[output_index],
                    )
                else:
                    np.einsum(
                        "nab,lnb->lna",
                        voxel_stiffness,
                        values,
                        optimize=True,
                        out=stresses[output_index],
                    )
        return stresses

    def apply_supported_chunk(
        q_indices: Any,
        strains: np.ndarray,
        support: str,
        support_offset: int,
    ) -> np.ndarray:
        """Apply affine tensors to a contiguous support chunk.

        ``strains`` may be component-major ``(r, 6, n)`` or voxel-major
        ``(r, n, 6)``.  The arithmetic follows the storage dtype (normally
        float32 for streamed fields); all global contractions are accumulated
        separately in float64.
        """
        selected = np.asarray(q_indices, dtype=np.intp).reshape(-1)
        values = np.asarray(strains)
        dtype = values.dtype if values.dtype in matrix_bases_by_dtype else np.dtype(np.float64)
        values = np.asarray(values, dtype=dtype)
        component_major = values.ndim == 3 and values.shape[1] == 6
        voxel_major = values.ndim == 3 and values.shape[-1] == 6
        if not component_major and not voxel_major:
            raise ValueError("supported chunk must have a six-component axis")
        n_chunk = values.shape[-1] if component_major else values.shape[1]
        offset = int(support_offset)
        if offset < 0 or n_chunk < 0:
            raise ValueError("invalid support chunk offset")
        stresses = np.zeros((len(selected),) + values.shape, dtype=dtype)

        if support == "matrix":
            if np.any((selected < 0) | (selected >= 2)):
                raise ValueError("matrix support only accepts coefficients 0 and 1")
            if offset + n_chunk > len(matrix_idx):
                raise ValueError("matrix support chunk exceeds available voxels")
            if stiffness_matrix_fast_path:
                for output_index, q in enumerate(selected):
                    if q == 0 and component_major:
                        normal_sum = np.sum(values[:, :3], axis=1, dtype=dtype)
                        stresses[output_index, :, :3] = normal_sum[:, None]
                    elif q == 0:
                        normal_sum = np.sum(values[:, :, :3], axis=2, dtype=dtype)
                        stresses[output_index, :, :, :3] = normal_sum[:, :, None]
                    else:
                        np.multiply(values, dtype.type(2.0), out=stresses[output_index])
            elif component_major:
                stresses[:] = np.einsum(
                    "qab,lbn->qlan",
                    matrix_bases_by_dtype[dtype][selected],
                    values,
                    optimize=True,
                )
            else:
                stresses[:] = np.einsum(
                    "qab,lnb->qlna",
                    matrix_bases_by_dtype[dtype][selected],
                    values,
                    optimize=True,
                )
            return stresses

        if support != "fiber":
            raise ValueError("support must be 'matrix' or 'fiber'")
        if np.any((selected < 2) | (selected >= q_count)):
            raise ValueError("fiber support only accepts coefficients 2 and above")
        if offset + n_chunk > len(fiber_idx):
            raise ValueError("fiber support chunk exceeds available voxels")
        local_indices = selected - 2
        chunk_group_ids = fiber_group_ids[offset : offset + n_chunk]
        if not len(chunk_group_ids):
            return stresses

        # The geometry is phase/orientation sorted, so each orientation is a
        # contiguous run.  Process runs directly without allocating a
        # per-voxel 6x6 stiffness tensor.
        starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.flatnonzero(np.diff(chunk_group_ids)) + 1)
        )
        ends = np.concatenate((starts[1:], np.array([len(chunk_group_ids)])))
        group_bases = fiber_bases_by_dtype[dtype]
        for start, end in zip(starts, ends, strict=True):
            group = int(chunk_group_ids[int(start)])
            if component_major:
                np.einsum(
                    "qab,lbn->qlan",
                    group_bases[group, local_indices],
                    values[:, :, int(start) : int(end)],
                    optimize=True,
                    out=stresses[:, :, :, int(start) : int(end)],
                )
            else:
                np.einsum(
                    "qab,lnb->qlna",
                    group_bases[group, local_indices],
                    values[:, int(start) : int(end), :],
                    optimize=True,
                    out=stresses[:, :, int(start) : int(end), :],
                )
        return stresses

    def apply_supported_chunk_gpu(
        q_indices: Any,
        strains_gpu: Any,
        support: str,
        support_offset: int,
    ) -> Any:
        """GPU counterpart of ``apply_supported_chunk`` for resident Ritz chunks."""
        import cupy as cp

        selected = np.asarray(q_indices, dtype=np.intp).reshape(-1)
        values = cp.asarray(strains_gpu)
        if values.ndim == 2:
            if values.shape[1] % 6:
                raise ValueError("flattened GPU strains lack a six-component axis")
            values = values.reshape(values.shape[0], 6, values.shape[1] // 6)
        if values.ndim != 3 or values.shape[1] != 6:
            raise ValueError("GPU supported strains must be component-major")
        n_chunk = int(values.shape[2])
        offset = int(support_offset)
        if offset < 0:
            raise ValueError("invalid support chunk offset")
        dtype_key = "float32" if values.dtype == cp.float32 else "float64"
        if dtype_key not in gpu_basis_cache:
            np_dtype = np.float32 if dtype_key == "float32" else np.float64
            gpu_basis_cache[dtype_key] = (
                cp.asarray(matrix_bases_by_dtype[np.dtype(np_dtype)]),
                cp.asarray(fiber_bases_by_dtype[np.dtype(np_dtype)]),
            )
        matrix_gpu, fiber_gpu = gpu_basis_cache[dtype_key]

        if support == "matrix":
            if np.any((selected < 0) | (selected >= 2)):
                raise ValueError("matrix support only accepts coefficients 0 and 1")
            if offset + n_chunk > len(matrix_idx):
                raise ValueError("matrix support chunk exceeds available voxels")
            stresses = cp.empty((len(selected),) + values.shape, dtype=values.dtype)
            if stiffness_matrix_fast_path:
                for output_index, q in enumerate(selected):
                    if q == 0:
                        stresses[output_index].fill(0)
                        normal_sum = cp.sum(values[:, :3], axis=1, dtype=values.dtype)
                        stresses[output_index, :, :3] = normal_sum[:, None, :]
                    else:
                        cp.multiply(values, values.dtype.type(2.0), out=stresses[output_index])
            else:
                stresses[:] = cp.einsum(
                    "qab,lbn->qlan",
                    matrix_gpu[selected],
                    values,
                    optimize=True,
                )
            return stresses

        if support != "fiber":
            raise ValueError("support must be 'matrix' or 'fiber'")
        if np.any((selected < 2) | (selected >= q_count)):
            raise ValueError("fiber support only accepts coefficients 2 and above")
        if offset + n_chunk > len(fiber_idx):
            raise ValueError("fiber support chunk exceeds available voxels")
        local_indices = selected - 2
        chunk_group_ids = fiber_group_ids[offset : offset + n_chunk]
        if not len(chunk_group_ids):
            return cp.empty((len(selected),) + values.shape, dtype=values.dtype)
        starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.flatnonzero(np.diff(chunk_group_ids)) + 1)
        )
        ends = np.concatenate((starts[1:], np.array([len(chunk_group_ids)])))
        if len(starts) == 1:
            group = int(chunk_group_ids[0])
            return cp.einsum(
                "qab,lbn->qlan",
                fiber_gpu[group, local_indices],
                values,
                optimize=True,
            )
        stresses = cp.empty((len(selected),) + values.shape, dtype=values.dtype)
        for start, end in zip(starts, ends, strict=True):
            group = int(chunk_group_ids[int(start)])
            stresses[:, :, :, int(start) : int(end)] = cp.einsum(
                "qab,lbn->qlan",
                fiber_gpu[group, local_indices],
                values[:, :, int(start) : int(end)],
                optimize=True,
            )
        return stresses

    def apply_all(strains: np.ndarray) -> np.ndarray:
        return apply_indices(np.arange(q_count), strains)

    apply.averaged_stiffness = averaged_stiffness
    apply.unique_fiber_orientations = int(len(fiber_bases_by_group))
    apply.orientation_kernel = "grouped" if fiber_group_runs else "voxelwise"
    apply.coefficient_names = tuple(coefficient_names)
    apply.exact_factorizations = exact_factorizations
    apply.local_exact_factorizations = local_exact_factorizations
    apply.local_frame_snapshots = bool(local_frame_snapshots)
    apply.gathered_factor_ritz = bool(gathered_factor_ritz)
    apply.fiber_group_ids = fiber_group_ids
    apply.fiber_group_runs = fiber_group_runs
    apply.fiber_mandel_rotations = fiber_mandel_rotations
    apply.apply_indices = apply_indices
    apply.apply_all = apply_all
    apply.apply_supported_chunk = apply_supported_chunk
    if isinstance(matrix_selector, slice) and isinstance(fiber_selector, slice):
        apply.support_blocks = (
            ("matrix", np.arange(0, 2, dtype=np.intp), matrix_selector),
            ("fiber", np.arange(2, q_count, dtype=np.intp), fiber_selector),
        )
        apply.apply_supported_indices = apply_supported_indices
        apply.apply_supported_chunk_gpu = apply_supported_chunk_gpu
    return apply


def affine_stress_batch_factory(
    phase: np.ndarray,
    ori: np.ndarray,
    *,
    local_frame_snapshots: bool = False,
    gathered_factor_ritz: bool = False,
) -> Any:
    """Build the affine stiffness action used by primal Ritz compilation."""
    return _affine_tensor_batch_factory(
        phase,
        ori,
        matrix_bases=_isotropic_bases(),
        fiber_local_bases=_fiber_local_bases_axis0(),
        coefficient_names=COEFF_NAMES,
        local_frame_snapshots=bool(local_frame_snapshots),
        gathered_factor_ritz=bool(gathered_factor_ritz),
    )


_LOCAL_FRAME_KERNEL_CACHE: dict[str, Any] = {}


def _local_frame_kernel(dtype: np.dtype) -> Any:
    """Compile the voxelwise global-to-local Mandel rotation kernel."""
    import cupy as cp

    key = np.dtype(dtype).name
    if key not in ("float32", "float64"):
        raise ValueError("local-frame snapshots require float32 or float64 fields")
    if key not in _LOCAL_FRAME_KERNEL_CACHE:
        scalar = "float" if key == "float32" else "double"
        name = f"mandel_global_to_local_{key}"
        source = f"""
        extern "C" __global__
        void {name}(
            const {scalar}* values,
            {scalar}* output,
            const int* group_ids,
            const {scalar}* rotations,
            const int rows,
            const int voxels)
        {{
            const long long index =
                (long long)blockDim.x * blockIdx.x + threadIdx.x;
            const long long total = (long long)rows * 6 * voxels;
            if (index >= total) return;
            const int voxel = (int)(index % voxels);
            const int local_component = (int)((index / voxels) % 6);
            const int row = (int)(index / ((long long)6 * voxels));
            const int group = group_ids[voxel];
            {scalar} value = ({scalar})0;
            #pragma unroll
            for (int global_component = 0; global_component < 6; ++global_component) {{
                const {scalar} rotation = rotations[
                    ((long long)group * 6 + global_component) * 6
                    + local_component
                ];
                value += rotation * values[
                    ((long long)row * 6 + global_component) * voxels + voxel
                ];
            }}
            output[index] = value;
        }}
        """
        _LOCAL_FRAME_KERNEL_CACHE[key] = cp.RawKernel(source, name)
    return _LOCAL_FRAME_KERNEL_CACHE[key]


def _localize_snapshot_fields_inplace(
    fields: np.ndarray,
    affine: Any,
    *,
    max_chunk_voxels: int = 1_000_000,
) -> dict[str, Any]:
    """Rotate ordered fiber snapshots once into their local Mandel frames."""
    values = np.asarray(fields)
    if values.ndim != 3 or values.shape[1] != 6:
        raise ValueError("snapshot fields must have shape (r, 6, nvox)")
    if not bool(getattr(affine, "local_frame_snapshots", False)):
        raise ValueError("affine map is not configured for local-frame snapshots")
    support_blocks = getattr(affine, "support_blocks", ())
    fiber_selector = next(
        (selector for support, _, selector in support_blocks if support == "fiber"),
        slice(0, 0),
    )
    if not isinstance(fiber_selector, slice):
        raise ValueError("local-frame snapshots require contiguous fiber support")
    fiber_start = int(fiber_selector.start or 0)
    fiber_stop = int(fiber_selector.stop or values.shape[2])
    fiber_count = max(0, fiber_stop - fiber_start)
    group_ids = np.asarray(getattr(affine, "fiber_group_ids"), dtype=np.int32)
    rotations = np.asarray(
        getattr(affine, "fiber_mandel_rotations"), dtype=np.float64
    )
    if fiber_count != len(group_ids):
        raise ValueError("fiber group map does not match ordered snapshot fields")
    if not fiber_count:
        return {
            "local_frame_backend": "none",
            "local_frame_transform_wall_s": 0.0,
            "local_frame_chunk_voxels": 0,
            "local_frame_workspace_peak_bytes": 0,
        }

    started = time.perf_counter()
    chunk_voxels = min(int(max_chunk_voxels), fiber_count)
    workspace_peak_bytes = 0
    backend = "cpu"
    try:
        import cupy as cp

        cp_dtype = cp.float32 if values.dtype == np.float32 else cp.float64
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        bytes_per_voxel = values.shape[0] * 6 * values.dtype.itemsize * 2 + 4
        gpu_limit = max(4096, int(0.18 * free_bytes) // max(bytes_per_voxel, 1))
        chunk_voxels = max(4096, min(chunk_voxels, gpu_limit, fiber_count))
        rotations_gpu = cp.asarray(rotations, dtype=cp_dtype)
        kernel = _local_frame_kernel(values.dtype)
        for relative_start in range(0, fiber_count, chunk_voxels):
            relative_stop = min(relative_start + chunk_voxels, fiber_count)
            global_start = fiber_start + relative_start
            global_stop = fiber_start + relative_stop
            host_chunk = np.ascontiguousarray(values[:, :, global_start:global_stop])
            input_gpu = cp.asarray(host_chunk, dtype=cp_dtype)
            output_gpu = cp.empty_like(input_gpu)
            groups_gpu = cp.asarray(group_ids[relative_start:relative_stop])
            count = int(relative_stop - relative_start)
            total = int(values.shape[0] * 6 * count)
            kernel(
                ((total + 255) // 256,),
                (256,),
                (
                    input_gpu,
                    output_gpu,
                    groups_gpu,
                    rotations_gpu,
                    np.int32(values.shape[0]),
                    np.int32(count),
                ),
            )
            output_gpu.get(out=host_chunk)
            values[:, :, global_start:global_stop] = host_chunk
            workspace_peak_bytes = max(
                workspace_peak_bytes,
                int(host_chunk.nbytes + input_gpu.nbytes + output_gpu.nbytes + groups_gpu.nbytes),
            )
            del host_chunk, input_gpu, output_gpu, groups_gpu
        cp.cuda.get_current_stream().synchronize()
        backend = "gpu_raw_kernel"
    except Exception as exc:
        raise RuntimeError(
            "local-frame Ritz snapshot rotation requires CUDA/CuPy"
        ) from exc
    return {
        "local_frame_backend": backend,
        "local_frame_transform_wall_s": float(time.perf_counter() - started),
        "local_frame_chunk_voxels": int(chunk_voxels),
        "local_frame_workspace_peak_bytes": int(workspace_peak_bytes),
    }


def affine_compliance_batch_factory(
    phase: np.ndarray,
    ori: np.ndarray,
) -> Any:
    """Build the affine compliance action used by dual Ritz compilation."""
    return _affine_tensor_batch_factory(
        phase,
        ori,
        matrix_bases=_isotropic_compliance_bases(),
        fiber_local_bases=_fiber_local_compliance_bases_axis0(),
        coefficient_names=DUAL_COEFF_NAMES,
    )


def _material_coefficients(row: dict[str, Any]) -> np.ndarray:
    em = float(row["Em"])
    nu_m = float(row["nu_m"])
    lam_m = em * nu_m / ((1.0 + nu_m) * (1.0 - 2.0 * nu_m))
    mu_m = em / (2.0 * (1.0 + nu_m))

    cf_voigt = TI_stiffness_voigt(
        float(row["Ef_L"]),
        float(row["Ef_T"]),
        float(row["nu_LT"]),
        float(row["nu_TT"]),
        float(row["G_LT"]),
    )
    cf_local = voigt_to_mandel(cf_voigt)
    coeffs = np.array(
        [
            lam_m,
            mu_m,
            cf_local[1, 1],
            cf_local[1, 2],
            cf_local[0, 1],
            cf_local[0, 0],
            cf_local[4, 4] / 2.0,
        ],
        dtype=float,
    )
    return coeffs


def _dual_material_coefficients(row: dict[str, Any]) -> np.ndarray:
    """Seven exact affine coefficients of the phase compliance field."""
    em = float(row["Em"])
    ef_l = float(row["Ef_L"])
    ef_t = float(row["Ef_T"])
    g_lt = float(row["G_LT"])
    return np.asarray(
        [
            1.0 / em,
            float(row["nu_m"]) / em,
            1.0 / ef_l,
            1.0 / ef_t,
            float(row["nu_LT"]) / ef_l,
            float(row["nu_TT"]) / ef_t,
            1.0 / (2.0 * g_lt),
        ],
        dtype=np.float64,
    )


def _material_coefficients_batch(parameters: np.ndarray) -> np.ndarray:
    """Vectorized seven-term affine coefficients for material batches."""
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(MATERIAL_PARAMETER_COLUMNS):
        raise ValueError(
            "parameters must have columns " + ", ".join(MATERIAL_PARAMETER_COLUMNS)
        )
    Em, nu_m, Ef_L, Ef_T, G_LT, nu_LT, nu_TT = values.T
    lam_m = Em * nu_m / ((1.0 + nu_m) * (1.0 - 2.0 * nu_m))
    mu_m = Em / (2.0 * (1.0 + nu_m))
    compliance = np.zeros((len(values), 6, 6), dtype=np.float64)
    compliance[:, 0, 0] = 1.0 / Ef_L
    compliance[:, 1, 1] = 1.0 / Ef_T
    compliance[:, 2, 2] = 1.0 / Ef_T
    compliance[:, 0, 1] = compliance[:, 1, 0] = -nu_LT / Ef_L
    compliance[:, 0, 2] = compliance[:, 2, 0] = -nu_LT / Ef_L
    compliance[:, 1, 2] = compliance[:, 2, 1] = -nu_TT / Ef_T
    compliance[:, 3, 3] = 2.0 * (1.0 + nu_TT) / Ef_T
    compliance[:, 4, 4] = 1.0 / G_LT
    compliance[:, 5, 5] = 1.0 / G_LT
    voigt = np.linalg.inv(compliance)
    factors = np.array(
        (1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0))
    )
    mandel = voigt * factors[None, :, None] * factors[None, None, :]
    return np.column_stack(
        (
            lam_m,
            mu_m,
            mandel[:, 1, 1],
            mandel[:, 1, 2],
            mandel[:, 0, 1],
            mandel[:, 0, 0],
            mandel[:, 4, 4] / 2.0,
        )
    )


def _dual_material_coefficients_batch(parameters: np.ndarray) -> np.ndarray:
    """Vectorized seven-term affine compliance coefficients."""
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(MATERIAL_PARAMETER_COLUMNS):
        raise ValueError(
            "parameters must have columns " + ", ".join(MATERIAL_PARAMETER_COLUMNS)
        )
    em, nu_m, ef_l, ef_t, g_lt, nu_lt, nu_tt = values.T
    return np.column_stack(
        (
            1.0 / em,
            nu_m / em,
            1.0 / ef_l,
            1.0 / ef_t,
            nu_lt / ef_l,
            nu_tt / ef_t,
            1.0 / (2.0 * g_lt),
        )
    )


def _engineering_constants_batch(C_mandel: np.ndarray) -> np.ndarray:
    """Return engineering constants in ``ENGINEERING_COLUMNS`` order."""
    matrices = np.asarray(C_mandel, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (6, 6):
        raise ValueError("C_mandel must have shape (count, 6, 6).")
    matrices = 0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    compliance = np.linalg.inv(matrices)
    return np.column_stack(
        (
            1.0 / compliance[:, 0, 0],
            1.0 / compliance[:, 1, 1],
            1.0 / compliance[:, 2, 2],
            1.0 / (2.0 * compliance[:, 5, 5]),
            1.0 / (2.0 * compliance[:, 4, 4]),
            1.0 / (2.0 * compliance[:, 3, 3]),
            -compliance[:, 0, 1] / compliance[:, 0, 0],
            -compliance[:, 0, 2] / compliance[:, 0, 0],
            -compliance[:, 1, 2] / compliance[:, 1, 1],
        )
    )


def _check_local_basis(row: dict[str, Any]) -> float:
    coeffs = _material_coefficients(row)
    local = sum(
        coeffs[q] * basis
        for q, basis in enumerate(
            [*_isotropic_bases(), *_fiber_local_bases_axis0()]
        )
    )
    # Only the fiber block is meaningful for this check.
    cf_local = voigt_to_mandel(
        TI_stiffness_voigt(
            float(row["Ef_L"]),
            float(row["Ef_T"]),
            float(row["nu_LT"]),
            float(row["nu_TT"]),
            float(row["G_LT"]),
        )
    )
    recon = sum(
        coeffs[q + 2] * basis
        for q, basis in enumerate(_fiber_local_bases_axis0())
    )
    del local
    return _relative_frobenius(recon, cf_local)


def affine_constitutive_reconstruction_error(
    phase: np.ndarray,
    ori: np.ndarray,
    row: dict[str, Any],
) -> dict[str, float | int]:
    """Verify the shared affine stiffness representation over all voxel groups."""
    phase_flat = np.asarray(phase).reshape(-1)
    ori_flat = np.asarray(ori, dtype=np.float64).reshape(-1, 3)
    coefficients = _material_coefficients(row)
    matrix_bases, fiber_bases = _affine_stiffness_bases()

    em = float(row["Em"])
    nu_m = float(row["nu_m"])
    matrix_reference = voigt_to_mandel(
        TI_stiffness_voigt(
            em,
            em,
            nu_m,
            nu_m,
            em / (2.0 * (1.0 + nu_m)),
        )
    )
    matrix_affine = sum(coefficients[q] * matrix_bases[q] for q in range(2))
    matrix_error = _relative_frobenius(matrix_affine, matrix_reference)

    fiber_reference_local = voigt_to_mandel(
        TI_stiffness_voigt(
            float(row["Ef_L"]),
            float(row["Ef_T"]),
            float(row["nu_LT"]),
            float(row["nu_TT"]),
            float(row["G_LT"]),
        )
    )
    fiber_affine_local = sum(
        coefficients[q + 2] * fiber_bases[q] for q in range(len(fiber_bases))
    )

    matrix_count = float(np.count_nonzero(phase_flat == 0))
    squared_difference = matrix_count * float(
        np.linalg.norm(matrix_affine - matrix_reference, ord="fro") ** 2
    )
    squared_reference = matrix_count * float(
        np.linalg.norm(matrix_reference, ord="fro") ** 2
    )
    maximum_local_error = matrix_error
    group_count = 0
    fiber_mask = phase_flat != 0
    if np.any(fiber_mask):
        inverse, means = _group_quantized_orientations(
            ori_flat[fiber_mask], AFFINE_ORIENTATION_QUANTIZATION
        )
        counts = np.bincount(inverse, minlength=len(means))
        group_count = int(len(means))
        for group_id, axis in enumerate(means):
            rotation = rotation_matrix_from_vector(axis)
            direct = rotate_C_mandel(fiber_reference_local, rotation)
            reconstructed = rotate_C_mandel(fiber_affine_local, rotation)
            local_error = _relative_frobenius(reconstructed, direct)
            maximum_local_error = max(maximum_local_error, local_error)
            weight = float(counts[group_id])
            squared_difference += weight * float(
                np.linalg.norm(reconstructed - direct, ord="fro") ** 2
            )
            squared_reference += weight * float(np.linalg.norm(direct, ord="fro") ** 2)

    global_error = math.sqrt(
        squared_difference / max(squared_reference, np.finfo(float).tiny)
    )
    return {
        "relative_frobenius_error": float(global_error),
        "maximum_voxel_group_relative_error": float(maximum_local_error),
        "orientation_group_count": group_count,
        "voxel_count": int(phase_flat.size),
    }


def _discover_snapshot_ids(run_dir: Path, results_df: pd.DataFrame) -> list[int]:
    ids: list[int] = []
    if "solution_fields_path" in results_df.columns:
        for _, row in results_df.iterrows():
            path_raw = row.get("solution_fields_path", "")
            if isinstance(path_raw, str) and path_raw.strip():
                path = Path(path_raw)
                if not path.is_absolute():
                    path = PROJECT_ROOT / path
                if path.is_file():
                    ids.append(int(row["material_id"]))
    if ids:
        return sorted(set(ids))
    for path in sorted(run_dir.glob("material_*/solution_fields.npz")):
        ids.append(int(path.parent.name.split("_")[-1]))
    return sorted(set(ids))


def _snapshot_path(run_dir: Path, material_id: int) -> Path:
    return run_dir / f"material_{int(material_id):04d}" / "solution_fields.npz"


def _load_snapshot_fields(
    run_dir: Path,
    material_ids: list[int],
    *,
    dtype: str | np.dtype = np.float32,
) -> list[np.ndarray]:
    field_dtype = np.dtype(dtype)
    if field_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("snapshot dtype must be float32 or float64.")
    fields: list[np.ndarray] = []
    for material_id in material_ids:
        path = _snapshot_path(run_dir, material_id)
        if not path.is_file():
            raise FileNotFoundError(f"No existe snapshot tangencial: {path}")
        with np.load(path) as payload:
            for load_id in range(6):
                key = f"fluctuation_load{load_id}"
                if key not in payload:
                    raise KeyError(f"{path} no contiene {key}")
                fields.append(np.asarray(payload[key], dtype=field_dtype))
    return fields


def _inner(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a * b))


def _orthonormalize(fields: list[np.ndarray], tol: float) -> list[np.ndarray]:
    basis: list[np.ndarray] = []
    for field in fields:
        vector = np.asarray(field, dtype=np.float64).copy()
        for base in basis:
            vector -= _inner(base, vector) * base
        for base in basis:
            vector -= _inner(base, vector) * base
        norm = math.sqrt(max(_inner(vector, vector), 0.0))
        if norm > tol:
            basis.append(vector / norm)
    if not basis:
        raise RuntimeError("No se pudo construir una base reducida no vacia.")
    return basis


def _field_component_voxel_view(field: np.ndarray, nvox: int) -> np.ndarray:
    """Return one field as ``(6, nvox)`` without forcing float64 storage."""
    value = np.asarray(field)
    if value.ndim == 2 and value.shape == (6, nvox):
        return value
    if value.ndim == 2 and value.shape == (nvox, 6):
        return value.T
    if value.size != 6 * nvox:
        raise ValueError(f"field has {value.size} entries; expected {6 * nvox}")
    if value.shape[0] == 6:
        return value.reshape(6, nvox)
    if value.shape[-1] == 6:
        return np.moveaxis(value, -1, 0).reshape(6, nvox)
    return value.reshape(6, nvox)


def _basis_rank_nvox(fields: np.ndarray | list[np.ndarray], nvox: int) -> int:
    if isinstance(fields, np.ndarray):
        if fields.ndim < 2:
            raise ValueError("basis array must have at least two dimensions")
        return int(fields.shape[0])
    return int(len(fields))


def _basis_chunk(
    fields: np.ndarray | list[np.ndarray],
    *,
    nvox: int,
    start: int,
    end: int,
) -> np.ndarray:
    """Materialize only a spatial chunk in component-major layout.

    Basis fields stay resident in their storage dtype (normally float32).
    The returned chunk is contiguous and is the only large temporary used by
    the streaming Ritz compiler.
    """
    if start < 0 or end < start or end > nvox:
        raise ValueError("invalid spatial chunk bounds")
    if isinstance(fields, np.ndarray):
        values = np.asarray(fields)
        r = int(values.shape[0])
        if values.ndim == 3 and values.shape[1:] == (6, nvox):
            return np.ascontiguousarray(values[:, :, start:end])
        if values.ndim == 3 and values.shape[1:] == (nvox, 6):
            return np.ascontiguousarray(np.moveaxis(values[:, start:end, :], -1, 1))
        return np.ascontiguousarray(values.reshape(r, 6, nvox)[:, :, start:end])

    if not fields:
        return np.empty((0, 6, end - start), dtype=np.float32)
    first = _field_component_voxel_view(fields[0], nvox)
    out = np.empty((len(fields), 6, end - start), dtype=first.dtype)
    out[0] = first[:, start:end]
    for idx, field in enumerate(fields[1:], start=1):
        out[idx] = _field_component_voxel_view(field, nvox)[:, start:end]
    return out


def _basis_chunk_into(
    fields: np.ndarray | list[np.ndarray],
    out: np.ndarray,
    *,
    nvox: int,
    start: int,
    end: int,
) -> None:
    """Fill a preallocated component-major chunk, zero-padding its tail."""
    if out.ndim != 3 or out.shape[1] != 6:
        raise ValueError("preallocated basis chunk must have shape (r, 6, n)")
    count = int(end) - int(start)
    if count < 0 or start < 0 or end > nvox or count > out.shape[2]:
        raise ValueError("invalid preallocated basis chunk bounds")
    rank = _basis_rank_nvox(fields, nvox)
    if rank != out.shape[0]:
        raise ValueError("preallocated basis chunk has the wrong rank")
    if isinstance(fields, np.ndarray):
        values = np.asarray(fields)
        if values.ndim == 3 and values.shape[1:] == (6, nvox):
            np.copyto(out[:, :, :count], values[:, :, start:end])
        elif values.ndim == 3 and values.shape[1:] == (nvox, 6):
            np.copyto(out[:, :, :count], np.moveaxis(values[:, start:end], -1, 1))
        else:
            np.copyto(
                out[:, :, :count], values.reshape(rank, 6, nvox)[:, :, start:end]
            )
    else:
        for index, field in enumerate(fields):
            out[index, :, :count] = _field_component_voxel_view(field, nvox)[
                :, start:end
            ]
    if count < out.shape[2]:
        out[:, :, count:].fill(0)


def _reduced_field_values(
    fields: np.ndarray | list[np.ndarray],
    *,
    count: int,
    nvox: int,
) -> tuple[np.ndarray | list[np.ndarray], str]:
    """Compatibility helper that avoids stacking a list of giant fields."""
    if isinstance(fields, np.ndarray):
        values = np.asarray(fields)
        if values.ndim == 3 and values.shape == (count, nvox, 6):
            return values, "voxel_component"
        if values.ndim >= 3 and values.shape[0] == count:
            return values.reshape(count, 6, nvox), "component_voxel"
        raise ValueError("basis array has an incompatible shape")
    if len(fields) != count:
        raise ValueError("basis list length does not match count")
    return fields, "field_list"


def _supported_values(
    values: np.ndarray | list[np.ndarray],
    selector: slice,
    layout: str,
) -> np.ndarray | list[np.ndarray]:
    if layout == "field_list":
        return [np.asarray(value)[..., selector] for value in values]
    return values[:, selector, :] if layout == "voxel_component" else values[:, :, selector]


def _flatten_supported_values(values: np.ndarray, layout: str) -> np.ndarray:
    if layout != "voxel_component":
        return np.ascontiguousarray(values).reshape(len(values), -1)
    return np.ascontiguousarray(np.moveaxis(values, -1, 1)).reshape(len(values), -1)


def _supported_stress_mean(stresses: np.ndarray, layout: str, nvox: int) -> np.ndarray:
    voxel_axis = 2 if layout == "voxel_component" else 3
    return np.sum(stresses, axis=voxel_axis, dtype=np.float64) / float(nvox)


def _stream_chunk_voxels(
    *,
    ranks: tuple[int, ...],
    q_block_size: int,
    nvox: int,
    storage_itemsize: int = 4,
    max_chunk_voxels: int = 1_000_000,
) -> int:
    """Choose a RAM/VRAM-safe spatial chunk from the live reduced ranks."""
    # CPU temporaries: basis chunks + q stress chunks in storage precision.
    live_rows = max(sum(max(int(r), 0) for r in ranks), 1)
    q_live = max(int(q_block_size), 1)
    cpu_target = 1.0 * 1024**3
    cpu_bytes_per_voxel = 6 * storage_itemsize * (live_rows + q_live * max(ranks[-1], 1))
    cpu_limit = max(4096, int(cpu_target // max(cpu_bytes_per_voxel, 1)))

    # GPU contractions use float64.  Limit to a conservative fraction of free
    # VRAM when CuPy is available.
    gpu_limit = int(max_chunk_voxels)
    try:
        import cupy as cp
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        gpu_target = max(256 * 1024**2, int(0.22 * free_bytes))
        gpu_rows = max(live_rows + max(ranks[-1], 1), 1)
        gpu_bytes_per_voxel = 6 * 8 * gpu_rows
        gpu_limit = max(4096, int(gpu_target // max(gpu_bytes_per_voxel, 1)))
    except Exception:
        pass
    return max(4096, min(int(nvox), int(max_chunk_voxels), cpu_limit, gpu_limit))


def _gpu_flat64(values: np.ndarray) -> Any | None:
    """Upload one component-major chunk once for repeated contractions."""
    try:
        import cupy as cp
        flat = np.ascontiguousarray(values).reshape(values.shape[0], -1)
        return cp.asarray(flat, dtype=cp.float64)
    except Exception:
        return None


def _gpu_batch_flat64(values: np.ndarray) -> Any | None:
    """Upload a batched component-major chunk as ``(batch, rows, dofs)``."""
    try:
        import cupy as cp
        array = np.ascontiguousarray(values)
        if array.ndim < 3:
            raise ValueError("batched contraction expects at least three dimensions")
        flat = array.reshape(array.shape[0], array.shape[1], -1)
        return cp.asarray(flat, dtype=cp.float64)
    except Exception:
        return None


def _ritz_compute_dtype(value: str | np.dtype) -> np.dtype:
    dtype = np.dtype(value)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Ritz contraction dtype must be float32 or float64.")
    return dtype


def _gpu_flat_compute(
    values: np.ndarray, *, compute_dtype: str | np.dtype = np.float64
) -> Any | None:
    """Upload a flattened chunk in the requested Ritz compute precision."""
    try:
        import cupy as cp
        array = np.ascontiguousarray(values)
        dtype = _ritz_compute_dtype(compute_dtype)
        cp_dtype = cp.float32 if dtype == np.dtype(np.float32) else cp.float64
        return cp.asarray(array.reshape(array.shape[0], -1), dtype=cp_dtype)
    except Exception:
        return None


def _gpu_batch_flat_compute(
    values: np.ndarray, *, compute_dtype: str | np.dtype = np.float64
) -> Any | None:
    """Upload batched stresses in the requested Ritz compute precision."""
    try:
        import cupy as cp
        array = np.ascontiguousarray(values)
        if array.ndim < 3:
            raise ValueError("batched contraction expects at least three dimensions")
        dtype = _ritz_compute_dtype(compute_dtype)
        cp_dtype = cp.float32 if dtype == np.dtype(np.float32) else cp.float64
        flat = array.reshape(array.shape[0], array.shape[1], -1)
        return cp.asarray(flat, dtype=cp_dtype)
    except Exception:
        return None


def _contract_preloaded_batch(
    left: np.ndarray,
    right: np.ndarray,
    left_gpu: Any | None = None,
    right_gpu: Any | None = None,
    *,
    compute_dtype: str | np.dtype = np.float64,
) -> np.ndarray:
    """Return ``left @ right[q].T`` for all affine coefficients at once.

    ``right`` has shape ``(q, rows, 6, n)``.  Keeping the coefficient axis
    inside one GPU contraction avoids one host-to-device transfer and one
    kernel launch per affine coefficient.  The NumPy path follows the same
    contraction and remains the fallback when CUDA is unavailable.
    """
    if left.ndim != 3 or right.ndim != 4:
        raise ValueError("batched contraction expects (r, 6, n) and (q, r, 6, n)")
    if left.shape[1] != 6 or right.shape[2] != 6 or left.shape[2] != right.shape[3]:
        raise ValueError("batched contraction arrays have incompatible shapes")
    left_flat = np.ascontiguousarray(left).reshape(left.shape[0], -1)
    right_flat = np.ascontiguousarray(right).reshape(
        right.shape[0], right.shape[1], -1
    )

    try:
        import cupy as cp

        owns_left = left_gpu is None
        owns_right = right_gpu is None
        requested_dtype = _ritz_compute_dtype(compute_dtype)
        cp_compute_dtype = (
            cp.float32
            if requested_dtype == np.dtype(np.float32)
            else cp.float64
        )
        if left_gpu is None:
            left_gpu = cp.asarray(left_flat, dtype=cp_compute_dtype)
        if right_gpu is None:
            right_gpu = cp.asarray(right_flat, dtype=cp_compute_dtype)
        result = cp.einsum("am,qbm->qab", left_gpu, right_gpu, optimize=True)
        result = cp.asnumpy(result)
        if owns_left:
            del left_gpu
        if owns_right:
            del right_gpu
        return np.asarray(result, dtype=np.float64)
    except Exception:
        return np.einsum(
            "am,qbm->qab",
            np.asarray(left_flat, dtype=np.float64),
            np.asarray(right_flat, dtype=np.float64),
            optimize=True,
        )


def _contract_and_sum_gpu_batch(
    left_gpu: Any,
    right_gpu: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Contract resident stresses and reduce their voxel mean in float64."""
    import cupy as cp

    left = cp.asarray(left_gpu)
    right = cp.asarray(right_gpu)
    if left.ndim != 2 or right.ndim != 4 or right.shape[2] != 6:
        raise ValueError("resident contraction expects (r, 6*n) and (q, r, 6, n)")
    right_flat = right.reshape(right.shape[0], right.shape[1], -1)
    if left.shape[1] != right_flat.shape[2]:
        raise ValueError("resident contraction operands have incompatible dimensions")
    products = cp.einsum("am,qbm->qab", left, right_flat, optimize=True)
    sums = cp.sum(right, axis=3, dtype=cp.float64)
    return (
        np.asarray(cp.asnumpy(products), dtype=np.float64),
        np.asarray(cp.asnumpy(sums), dtype=np.float64),
    )


def _contract_gpu_batch(left_gpu: Any, right_gpu: Any) -> np.ndarray:
    """Contract a second resident basis against an existing stress batch."""
    import cupy as cp

    left = cp.asarray(left_gpu)
    right = cp.asarray(right_gpu)
    right_flat = right.reshape(right.shape[0], right.shape[1], -1)
    if left.ndim != 2 or left.shape[1] != right_flat.shape[2]:
        raise ValueError("resident contraction operands have incompatible dimensions")
    products = cp.einsum("am,qbm->qab", left, right_flat, optimize=True)
    return np.asarray(cp.asnumpy(products), dtype=np.float64)


def _factorized_chunk_gpu(
    *,
    affine: Any,
    support: str,
    support_offset: int,
    left_gpu: Any | None,
    right_gpu: Any,
) -> tuple[Any | None, Any, Any, int]:
    """Return exact affine Ritz contributions using constitutive-rank factors."""
    import cupy as cp

    use_local_fiber = bool(
        support == "fiber" and getattr(affine, "local_frame_snapshots", False)
    )
    use_gathered_fiber = bool(
        support == "fiber" and getattr(affine, "gathered_factor_ritz", False)
    )
    factorizations = getattr(
        affine,
        "local_exact_factorizations" if use_local_fiber else "exact_factorizations",
        None,
    )
    if not isinstance(factorizations, dict) or support not in factorizations:
        raise NotImplementedError("affine map does not expose exact factorizations")
    factorization = factorizations[support]
    right_values = cp.asarray(right_gpu)
    if right_values.ndim == 2 and right_values.shape[1] % 6 == 0:
        right = right_values.reshape(
            right_values.shape[0], 6, right_values.shape[1] // 6
        )
    elif right_values.ndim == 3 and right_values.shape[1] == 6:
        right = right_values
    else:
        raise ValueError("factorized Ritz expects component-major fields")
    left = None
    if left_gpu is not None:
        left_values = cp.asarray(left_gpu)
        if left_values.ndim == 2 and left_values.shape[1] == 6 * right.shape[2]:
            left = left_values.reshape(left_values.shape[0], 6, right.shape[2])
        elif (
            left_values.ndim == 3
            and left_values.shape[1] == 6
            and left_values.shape[2] == right.shape[2]
        ):
            left = left_values
        else:
            raise ValueError("factorized Ritz left/right chunks are incompatible")

    factors_cpu = np.asarray(factorization["factors"], dtype=np.float64)
    weights_cpu = np.asarray(factorization["weights"], dtype=np.float64)
    coefficient_slices = tuple(factorization["coefficient_slices"])
    if not coefficient_slices:
        raise ValueError(f"support {support!r} has no factorized coefficients")
    dtype_name = "float32" if right.dtype == cp.float32 else "float64"
    cache = factorization.setdefault("_gpu_cache", {})
    if dtype_name not in cache:
        compute_dtype = cp.float32 if dtype_name == "float32" else cp.float64
        cache[dtype_name] = (
            cp.asarray(factors_cpu, dtype=compute_dtype),
            cp.asarray(weights_cpu, dtype=compute_dtype),
            cp.asarray(factors_cpu, dtype=cp.float64),
            cp.asarray(weights_cpu, dtype=cp.float64),
        )
    factors, weights, factors64, weights64 = cache[dtype_name]

    n_chunk = int(right.shape[2])
    if support == "matrix":
        group_ids = np.zeros(n_chunk, dtype=np.int32)
    elif support == "fiber":
        all_group_ids = np.asarray(getattr(affine, "fiber_group_ids"), dtype=np.int32)
        offset = int(support_offset)
        group_ids = all_group_ids[offset : offset + n_chunk]
        if len(group_ids) != n_chunk:
            raise ValueError("factorized fiber chunk exceeds orientation groups")
    else:
        raise ValueError("support must be 'matrix' or 'fiber'")

    total_factor_rank = int(weights.shape[0])
    right_features = cp.empty(
        (right.shape[0], total_factor_rank, n_chunk), dtype=right.dtype
    )
    left_features = (
        cp.empty((left.shape[0], total_factor_rank, n_chunk), dtype=left.dtype)
        if left is not None
        else None
    )
    starts = np.concatenate(
        (np.array([0], dtype=np.int64), np.flatnonzero(np.diff(group_ids)) + 1)
    )
    ends = np.concatenate((starts[1:], np.array([n_chunk], dtype=np.int64)))
    sums = cp.zeros(
        (len(coefficient_slices), right.shape[0], 6), dtype=cp.float64
    )
    sum_workspace_peak_bytes = 0
    if use_local_fiber:
        right_features[:] = cp.einsum(
            "ak,ran->rkn",
            factors[0],
            right,
            optimize=True,
        )
        if left_features is not None and left is not None:
            left_features[:] = cp.einsum(
                "ak,ran->rkn",
                factors[0],
                left,
                optimize=True,
            )
        rotations_cpu = np.asarray(
            getattr(affine, "fiber_mandel_rotations"), dtype=np.float64
        )
        global_factors_cpu = np.einsum(
            "gab,bk->gak",
            rotations_cpu,
            factors_cpu[0],
            optimize=True,
        )
        rotation_cache = factorization.setdefault("_global_factor_gpu_cache", {})
        if dtype_name not in rotation_cache:
            compute_dtype = cp.float32 if dtype_name == "float32" else cp.float64
            rotation_cache[dtype_name] = cp.asarray(
                global_factors_cpu, dtype=compute_dtype
            )
        global_factors = rotation_cache[dtype_name]
        group_ids_gpu = cp.asarray(group_ids)
        for local_q, (factor_start, factor_stop) in enumerate(coefficient_slices):
            voxel_factors = global_factors[
                group_ids_gpu, :, factor_start:factor_stop
            ]
            global_stress = cp.einsum(
                "nak,k,rkn->ran",
                voxel_factors,
                weights[factor_start:factor_stop],
                right_features[:, factor_start:factor_stop],
                optimize=True,
            )
            sums[local_q] = cp.sum(global_stress, axis=2, dtype=cp.float64)
            sum_workspace_peak_bytes = max(
                sum_workspace_peak_bytes,
                int(voxel_factors.nbytes + global_stress.nbytes + group_ids_gpu.nbytes),
            )
            del voxel_factors, global_stress
    elif use_gathered_fiber:
        group_ids_gpu = cp.asarray(group_ids)
        voxel_factors = factors[group_ids_gpu]
        right_features[:] = cp.einsum(
            "nak,ran->rkn",
            voxel_factors,
            right,
            optimize=True,
        )
        if left_features is not None and left is not None:
            left_features[:] = cp.einsum(
                "nak,ran->rkn",
                voxel_factors,
                left,
                optimize=True,
            )
        for local_q, (factor_start, factor_stop) in enumerate(coefficient_slices):
            global_stress = cp.einsum(
                "nak,k,rkn->ran",
                voxel_factors[:, :, factor_start:factor_stop],
                weights[factor_start:factor_stop],
                right_features[:, factor_start:factor_stop],
                optimize=True,
            )
            sums[local_q] = cp.sum(global_stress, axis=2, dtype=cp.float64)
            sum_workspace_peak_bytes = max(
                sum_workspace_peak_bytes,
                int(voxel_factors.nbytes + global_stress.nbytes + group_ids_gpu.nbytes),
            )
            del global_stress
        del voxel_factors, group_ids_gpu
    else:
        for start_value, end_value in zip(starts, ends, strict=True):
            start = int(start_value)
            end = int(end_value)
            group = int(group_ids[start])
            group_factors = factors[group]
            right_features[:, :, start:end] = cp.einsum(
                "ak,ran->rkn",
                group_factors,
                right[:, :, start:end],
                optimize=True,
            )
            if left_features is not None and left is not None:
                left_features[:, :, start:end] = cp.einsum(
                    "ak,ran->rkn",
                    group_factors,
                    left[:, :, start:end],
                    optimize=True,
                )
            for local_q, (factor_start, factor_stop) in enumerate(coefficient_slices):
                feature_sum = cp.sum(
                    right_features[:, factor_start:factor_stop, start:end],
                    axis=2,
                    dtype=cp.float64,
                )
                sums[local_q] += cp.einsum(
                    "ak,k,rk->ra",
                    factors64[group, :, factor_start:factor_stop],
                    weights64[factor_start:factor_stop],
                    feature_sum,
                    optimize=True,
                )

    diagonal = cp.empty(
        (len(coefficient_slices), right.shape[0], right.shape[0]),
        dtype=cp.float64,
    )
    cross = (
        cp.empty(
            (len(coefficient_slices), left.shape[0], right.shape[0]),
            dtype=cp.float64,
        )
        if left is not None
        else None
    )
    weighted_peak_bytes = 0
    for local_q, (factor_start, factor_stop) in enumerate(coefficient_slices):
        right_q = right_features[:, factor_start:factor_stop]
        weighted = right_q * weights[factor_start:factor_stop][None, :, None]
        weighted_peak_bytes = max(weighted_peak_bytes, int(weighted.nbytes))
        weighted_flat = weighted.reshape(weighted.shape[0], -1)
        diagonal[local_q] = (
            right_q.reshape(right_q.shape[0], -1) @ weighted_flat.T
        ).astype(cp.float64)
        if cross is not None and left_features is not None:
            left_q = left_features[:, factor_start:factor_stop]
            cross[local_q] = (
                left_q.reshape(left_q.shape[0], -1) @ weighted_flat.T
            ).astype(cp.float64)
        del weighted, weighted_flat

    workspace_bytes = (
        int(right_features.nbytes)
        + weighted_peak_bytes
        + sum_workspace_peak_bytes
    )
    if left_features is not None:
        workspace_bytes += int(left_features.nbytes)
    return cross, diagonal, sums, workspace_bytes


def _factorized_raw_gpu(
    *,
    affine: Any,
    old_basis: np.ndarray | list[np.ndarray] | None,
    new_basis: np.ndarray | list[np.ndarray],
    nvox: int,
    compute_dtype: str | np.dtype,
    async_transfers: bool = False,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict[str, Any]]:
    """Stream exact factorized contributions with GPU-resident accumulators."""
    import cupy as cp

    requested_dtype = _ritz_compute_dtype(compute_dtype)
    old_rank = 0 if old_basis is None else _basis_rank_nvox(old_basis, nvox)
    new_rank = _basis_rank_nvox(new_basis, nvox)
    coefficient_names = tuple(getattr(affine, "coefficient_names", COEFF_NAMES))
    q_count = len(coefficient_names)
    support_blocks = getattr(affine, "support_blocks", None)
    if support_blocks is None or getattr(affine, "exact_factorizations", None) is None:
        raise NotImplementedError("factorized Ritz requires ordered supported blocks")

    first = new_basis[0] if isinstance(new_basis, list) else new_basis[0]
    storage_dtype = _field_component_voxel_view(first, nvox).dtype
    storage_itemsize = int(storage_dtype.itemsize)
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    live_rank = max(old_rank + new_rank, 1)
    compute_itemsize = int(requested_dtype.itemsize)
    # Snapshots plus packed constitutive features (11 fiber channels) and the
    # largest temporary weighted coefficient block (six channels).
    gpu_bytes_per_voxel = compute_itemsize * (
        6 * live_rank + 11 * live_rank + 6 * max(new_rank, 1)
    )
    gpu_target = max(256 * 1024**2, int(0.22 * free_bytes))
    gpu_limit = max(4096, int(gpu_target // max(gpu_bytes_per_voxel, 1)))
    cpu_bytes_per_voxel = storage_itemsize * 6 * live_rank
    cpu_limit = max(4096, int((1.0 * 1024**3) // max(cpu_bytes_per_voxel, 1)))
    chunk_voxels = max(4096, min(int(nvox), 1_000_000, gpu_limit, cpu_limit))
    if bool(async_transfers):
        if storage_dtype != requested_dtype:
            raise ValueError(
                "asynchronous Ritz requires matching storage and compute dtypes"
            )
        pinned_limit = max(
            4096,
            int((1.0 * 1024**3) // max(2 * cpu_bytes_per_voxel, 1)),
        )
        chunk_voxels = max(4096, min(chunk_voxels, pinned_limit))

    cross_accumulator = (
        cp.zeros((q_count, old_rank, new_rank), dtype=cp.float64)
        if old_rank
        else None
    )
    diagonal_accumulator = cp.zeros(
        (q_count, new_rank, new_rank), dtype=cp.float64
    )
    b_accumulator = cp.zeros((q_count, new_rank, 6), dtype=cp.float64)
    upload_wall_s = 0.0
    kernel_enqueue_wall_s = 0.0
    workspace_peak_bytes = 0
    basis_gpu_uploads = 0
    chunks = 0
    pinned_bytes = 0
    host_prepare_wall_s = 0.0
    asynchronous_overlap = False
    started = time.perf_counter()
    tasks: list[tuple[str, slice, int, int, int]] = []
    for support, support_indices, selector in support_blocks:
        support_start = int(selector.start or 0)
        support_stop = int(selector.stop or nvox)
        q_indices = np.asarray(support_indices, dtype=np.intp)
        q_slice = slice(int(q_indices[0]), int(q_indices[-1]) + 1)
        for global_start in range(support_start, support_stop, chunk_voxels):
            tasks.append(
                (
                    support,
                    q_slice,
                    support_start,
                    global_start,
                    min(global_start + chunk_voxels, support_stop),
                )
            )

    if bool(async_transfers) and tasks:
        cp_dtype = cp.float32 if requested_dtype == np.dtype(np.float32) else cp.float64
        transfer_stream = cp.cuda.Stream(non_blocking=True)
        compute_stream = cp.cuda.Stream(non_blocking=True)
        cp.cuda.get_current_stream().synchronize()
        slots: list[dict[str, Any]] = []
        for _ in range(2):
            new_memory = cp.cuda.alloc_pinned_memory(
                new_rank * 6 * chunk_voxels * storage_itemsize
            )
            new_host = np.ndarray(
                (new_rank, 6, chunk_voxels),
                dtype=storage_dtype,
                buffer=new_memory,
            )
            old_memory = None
            old_host = None
            old_gpu = None
            if old_rank:
                old_memory = cp.cuda.alloc_pinned_memory(
                    old_rank * 6 * chunk_voxels * storage_itemsize
                )
                old_host = np.ndarray(
                    (old_rank, 6, chunk_voxels),
                    dtype=storage_dtype,
                    buffer=old_memory,
                )
                old_gpu = cp.empty(
                    (old_rank, 6, chunk_voxels), dtype=cp_dtype
                )
            slots.append(
                {
                    "new_memory": new_memory,
                    "new_host": new_host,
                    "new_gpu": cp.empty(
                        (new_rank, 6, chunk_voxels), dtype=cp_dtype
                    ),
                    "old_memory": old_memory,
                    "old_host": old_host,
                    "old_gpu": old_gpu,
                    "ready": cp.cuda.Event(disable_timing=True),
                    "done": cp.cuda.Event(disable_timing=True),
                }
            )
        pinned_bytes = int(
            2 * (old_rank + new_rank) * 6 * chunk_voxels * storage_itemsize
        )

        def prepare_slot(slot: dict[str, Any], task: tuple[str, slice, int, int, int]) -> None:
            nonlocal host_prepare_wall_s
            _, _, _, start, end = task
            prepare_started = time.perf_counter()
            if old_basis is not None and slot["old_host"] is not None:
                _basis_chunk_into(
                    old_basis,
                    slot["old_host"],
                    nvox=nvox,
                    start=start,
                    end=end,
                )
            _basis_chunk_into(
                new_basis,
                slot["new_host"],
                nvox=nvox,
                start=start,
                end=end,
            )
            host_prepare_wall_s += float(time.perf_counter() - prepare_started)

        def enqueue_transfer(
            slot: dict[str, Any], *, wait_for_compute: bool
        ) -> None:
            nonlocal upload_wall_s, basis_gpu_uploads
            upload_started = time.perf_counter()
            with transfer_stream:
                if wait_for_compute:
                    transfer_stream.wait_event(slot["done"])
                if slot["old_gpu"] is not None:
                    slot["old_gpu"].set(slot["old_host"], stream=transfer_stream)
                slot["new_gpu"].set(slot["new_host"], stream=transfer_stream)
                slot["ready"].record(transfer_stream)
            upload_wall_s += float(time.perf_counter() - upload_started)
            basis_gpu_uploads += 1 + int(slot["old_gpu"] is not None)

        prepare_slot(slots[0], tasks[0])
        enqueue_transfer(slots[0], wait_for_compute=False)
        for task_index, task in enumerate(tasks):
            support, q_slice, support_start, global_start, global_end = task
            slot = slots[task_index % 2]
            count = int(global_end - global_start)
            enqueue_started = time.perf_counter()
            with compute_stream:
                compute_stream.wait_event(slot["ready"])
                cross, diagonal, b_block, workspace_bytes = (
                    _factorized_chunk_gpu(
                        affine=affine,
                        support=support,
                        support_offset=global_start - support_start,
                        left_gpu=(
                            None
                            if slot["old_gpu"] is None
                            else slot["old_gpu"][:, :, :count]
                        ),
                        right_gpu=slot["new_gpu"][:, :, :count],
                    )
                )
                diagonal_accumulator[q_slice] += diagonal
                b_accumulator[q_slice] += b_block
                if cross_accumulator is not None and cross is not None:
                    cross_accumulator[q_slice] += cross
                slot["done"].record(compute_stream)
            kernel_enqueue_wall_s += float(time.perf_counter() - enqueue_started)
            workspace_peak_bytes = max(workspace_peak_bytes, int(workspace_bytes))
            chunks += 1
            del diagonal, b_block
            if cross is not None:
                del cross

            next_index = task_index + 1
            if next_index < len(tasks):
                next_slot = slots[next_index % 2]
                if next_index >= 2:
                    next_slot["ready"].synchronize()
                prepare_slot(next_slot, tasks[next_index])
                enqueue_transfer(next_slot, wait_for_compute=next_index >= 2)
        compute_stream.synchronize()
        asynchronous_overlap = len(tasks) > 1
    else:
        for support, q_slice, support_start, global_start, global_end in tasks:
            old_values = (
                None
                if old_basis is None
                else _basis_chunk(
                    old_basis, nvox=nvox, start=global_start, end=global_end
                )
            )
            new_values = _basis_chunk(
                new_basis, nvox=nvox, start=global_start, end=global_end
            )
            upload_started = time.perf_counter()
            old_gpu = (
                None
                if old_values is None
                else _gpu_flat_compute(old_values, compute_dtype=requested_dtype)
            )
            new_gpu = _gpu_flat_compute(new_values, compute_dtype=requested_dtype)
            if new_gpu is None or (old_values is not None and old_gpu is None):
                raise RuntimeError("factorized Ritz requires CUDA/CuPy")
            upload_wall_s += float(time.perf_counter() - upload_started)
            basis_gpu_uploads += 1 + int(old_gpu is not None)

            enqueue_started = time.perf_counter()
            cross, diagonal, b_block, workspace_bytes = (
                _factorized_chunk_gpu(
                    affine=affine,
                    support=support,
                    support_offset=global_start - support_start,
                    left_gpu=old_gpu,
                    right_gpu=new_gpu,
                )
            )
            diagonal_accumulator[q_slice] += diagonal
            b_accumulator[q_slice] += b_block
            if cross_accumulator is not None and cross is not None:
                cross_accumulator[q_slice] += cross
            kernel_enqueue_wall_s += float(time.perf_counter() - enqueue_started)
            workspace_peak_bytes = max(workspace_peak_bytes, int(workspace_bytes))
            chunks += 1
            del diagonal, b_block, new_gpu, new_values
            if cross is not None:
                del cross
            if old_gpu is not None:
                del old_gpu
            if old_values is not None:
                del old_values

        cp.cuda.get_current_stream().synchronize()
    cross_host = (
        None
        if cross_accumulator is None
        else np.asarray(cp.asnumpy(cross_accumulator), dtype=np.float64) / float(nvox)
    )
    diagonal_host = (
        np.asarray(cp.asnumpy(diagonal_accumulator), dtype=np.float64) / float(nvox)
    )
    b_host = np.asarray(cp.asnumpy(b_accumulator), dtype=np.float64) / float(nvox)
    rank_factorizations = (
        getattr(affine, "local_exact_factorizations")
        if bool(getattr(affine, "local_frame_snapshots", False))
        else getattr(affine, "exact_factorizations")
    )
    factorization_ranks = {
        support: list(rank_factorizations[support]["ranks"])
        for support in ("matrix", "fiber")
    }
    snapshot_buffer_copies = 2 if bool(async_transfers) else 1
    gpu_snapshot_buffer_bytes = int(
        snapshot_buffer_copies
        * (old_rank + new_rank)
        * 6
        * chunk_voxels
        * compute_itemsize
    )
    accumulator_bytes = int(diagonal_accumulator.nbytes + b_accumulator.nbytes)
    if cross_accumulator is not None:
        accumulator_bytes += int(cross_accumulator.nbytes)
    return cross_host, diagonal_host, b_host, {
        "factorized_gpu_wall_s": float(time.perf_counter() - started),
        "factorized_upload_wall_s": float(upload_wall_s),
        "factorized_kernel_enqueue_wall_s": float(kernel_enqueue_wall_s),
        "factorized_host_prepare_wall_s": float(host_prepare_wall_s),
        "factorized_chunk_count": int(chunks),
        "factorized_constitutive_ranks": factorization_ranks,
        "chunk_voxels": int(chunk_voxels),
        "basis_gpu_uploads": int(basis_gpu_uploads),
        "stress_workspace_peak_bytes": int(workspace_peak_bytes),
        "gpu_resident_reduced_accumulation": True,
        "local_frame_snapshots": bool(
            getattr(affine, "local_frame_snapshots", False)
        ),
        "gathered_factor_ritz": bool(
            getattr(affine, "gathered_factor_ritz", False)
        ),
        "async_pinned_double_buffer": bool(asynchronous_overlap),
        "async_pinned_bytes": int(pinned_bytes),
        "gpu_snapshot_buffer_bytes": gpu_snapshot_buffer_bytes,
        "gpu_reduced_accumulator_bytes": accumulator_bytes,
        "estimated_gpu_peak_bytes": int(
            workspace_peak_bytes + gpu_snapshot_buffer_bytes + accumulator_bytes
        ),
        "reduced_d2h_transfers": 3 if cross_accumulator is not None else 2,
    }


def _contract_component_major(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return A B^T over component+voxel coordinates, accumulated in float64."""
    return _contract_component_major_compute(A, B, compute_dtype=np.float64)


def _contract_component_major_compute(
    A: np.ndarray,
    B: np.ndarray,
    *,
    compute_dtype: str | np.dtype,
) -> np.ndarray:
    """Return A B^T using the requested product precision.

    The small contracted block is always returned in float64 so spatial
    chunks can be accumulated accurately even when the products use float32.
    """
    if A.ndim != 3 or B.ndim != 3 or A.shape[1] != 6 or B.shape[1] != 6:
        raise ValueError("contraction expects component-major (r, 6, n) arrays")
    if A.shape[2] != B.shape[2]:
        raise ValueError("contraction arrays must share the spatial chunk")
    requested_dtype = _ritz_compute_dtype(compute_dtype)
    a2 = np.ascontiguousarray(A).reshape(A.shape[0], -1)
    b2 = np.ascontiguousarray(B).reshape(B.shape[0], -1)
    try:
        import cupy as cp
        cp_dtype = cp.float32 if requested_dtype == np.dtype(np.float32) else cp.float64
        a_gpu = cp.asarray(a2, dtype=cp_dtype)
        if A is B:
            b_gpu = a_gpu
        else:
            b_gpu = cp.asarray(b2, dtype=cp_dtype)
        result = cp.asnumpy(a_gpu @ b_gpu.T)
        del a_gpu
        if A is not B:
            del b_gpu
        return np.asarray(result, dtype=np.float64)
    except Exception:
        return np.asarray(
            np.asarray(a2, dtype=requested_dtype)
            @ np.asarray(b2, dtype=requested_dtype).T,
            dtype=np.float64,
        )


def _contract_component_major_cpu(
    A: np.ndarray,
    B: np.ndarray,
    *,
    compute_dtype: str | np.dtype,
) -> np.ndarray:
    """Return A B^T on the CPU using the requested product precision."""
    if A.ndim != 3 or B.ndim != 3 or A.shape[1] != 6 or B.shape[1] != 6:
        raise ValueError("contraction expects component-major (r, 6, n) arrays")
    if A.shape[2] != B.shape[2]:
        raise ValueError("contraction arrays must share the spatial chunk")
    requested_dtype = _ritz_compute_dtype(compute_dtype)
    a2 = np.asarray(
        np.ascontiguousarray(A).reshape(A.shape[0], -1), dtype=requested_dtype
    )
    b2 = a2 if A is B else np.asarray(
        np.ascontiguousarray(B).reshape(B.shape[0], -1), dtype=requested_dtype
    )
    return np.asarray(a2 @ b2.T, dtype=np.float64)


def _timed_cpu_gram(
    values: np.ndarray,
    *,
    compute_dtype: str | np.dtype,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    product = _contract_component_major_cpu(
        values,
        values,
        compute_dtype=compute_dtype,
    )
    return product, float(time.perf_counter() - started)


def _timed_cpu_gram_extension(
    old_values: np.ndarray,
    new_values: np.ndarray,
    *,
    compute_dtype: str | np.dtype,
) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    cross = _contract_component_major_cpu(
        old_values,
        new_values,
        compute_dtype=compute_dtype,
    )
    diagonal = _contract_component_major_cpu(
        new_values,
        new_values,
        compute_dtype=compute_dtype,
    )
    return cross, diagonal, float(time.perf_counter() - started)


def _ritz_gram_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError("Gram backend must be auto, cpu, or gpu.")
    return backend


def _gpu_contract_dense(
    A: np.ndarray | list[np.ndarray],
    B: np.ndarray | list[np.ndarray],
    spatial_chunk_size: int | None = None,
) -> np.ndarray:
    """Compatibility dense contraction with robust layout handling."""
    def info(values: np.ndarray | list[np.ndarray]) -> tuple[int, int]:
        if isinstance(values, list):
            if not values:
                return 0, 0
            first = np.asarray(values[0])
            if first.ndim != 2:
                raise ValueError("list contraction expects 2-D fields")
            n = first.shape[1] if first.shape[0] == 6 else first.shape[0]
            return len(values), int(n)
        arr = np.asarray(values)
        if arr.ndim != 3:
            raise ValueError("array contraction expects a 3-D basis")
        n = arr.shape[2] if arr.shape[1] == 6 else arr.shape[1]
        return int(arr.shape[0]), int(n)

    rA, nA = info(A)
    rB, nB = info(B)
    if nA != nB:
        raise ValueError("contraction inputs have different voxel counts")
    nvox = nA
    if spatial_chunk_size is None:
        spatial_chunk_size = _stream_chunk_voxels(
            ranks=(rA, rB), q_block_size=1, nvox=nvox, max_chunk_voxels=1_000_000
        )
    result = np.zeros((rA, rB), dtype=np.float64)
    for start in range(0, nvox, int(spatial_chunk_size)):
        end = min(start + int(spatial_chunk_size), nvox)
        a_chunk = _basis_chunk(A, nvox=nvox, start=start, end=end)
        b_chunk = a_chunk if A is B else _basis_chunk(B, nvox=nvox, start=start, end=end)
        result += _contract_component_major(a_chunk, b_chunk)
    return result


def _orthonormal_transform_from_gram(
    G: np.ndarray,
    *,
    rank_rtol: float = 1.0e-11,
    allow_rank_reveal: bool = False,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Return T such that T G T.T = I, optionally dropping dependent rows."""
    gram = 0.5 * (np.asarray(G, dtype=np.float64) + np.asarray(G, dtype=np.float64).T)
    eigvals = scipy_linalg.eigvalsh(gram, check_finite=False)
    largest = float(eigvals[-1]) if len(eigvals) else 0.0
    smallest = float(eigvals[0]) if len(eigvals) else 0.0
    if not np.isfinite(largest) or largest <= 0.0:
        raise np.linalg.LinAlgError("reduced Gram matrix is not positive")
    relative_min = smallest / largest
    dependent = smallest <= 0.0 or relative_min <= float(rank_rtol)
    if dependent and not allow_rank_reveal:
        raise np.linalg.LinAlgError(
            "Reduced basis is numerically rank deficient: "
            f"lambda_min/lambda_max={relative_min:.3e}. "
            "Rank-reveal the incoming block before assembly."
        )
    eigenvalues, eigenvectors = scipy_linalg.eigh(gram, check_finite=False)
    keep = eigenvalues > largest * float(rank_rtol) if dependent else np.ones_like(
        eigenvalues, dtype=bool
    )
    if not np.any(keep):
        raise np.linalg.LinAlgError("reduced Gram matrix has zero numerical rank")
    kept_vectors = np.array(eigenvectors[:, keep], dtype=np.float64, copy=True)
    pivot_rows = np.argmax(np.abs(kept_vectors), axis=0)
    pivot_signs = np.sign(kept_vectors[pivot_rows, np.arange(kept_vectors.shape[1])])
    pivot_signs[pivot_signs == 0.0] = 1.0
    kept_vectors *= pivot_signs
    T = (kept_vectors / np.sqrt(eigenvalues[keep])).T
    transform_mode = "eigh_rank_reveal" if dependent else "eigh_full_rank"
    meta = {
        "gram_lambda_min": smallest,
        "gram_lambda_max": largest,
        "gram_condition": largest / smallest,
        "gram_relative_min": relative_min,
        "effective_rank": int(T.shape[0]),
        "discarded_rank": int(G.shape[0] - T.shape[0]),
        "gram_transform_mode": transform_mode,
    }
    return T, meta


def _transform_raw_operators(
    raw_Kq: np.ndarray,
    raw_Bq: np.ndarray,
    G: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | str]]:
    return _transform_raw_operators_with_rank_policy(
        raw_Kq, raw_Bq, G, allow_rank_reveal=False
    )


def _transform_raw_operators_with_rank_policy(
    raw_Kq: np.ndarray,
    raw_Bq: np.ndarray,
    G: np.ndarray,
    *,
    allow_rank_reveal: bool,
    rank_rtol: float = 1.0e-11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | str]]:
    T, gram_meta = _orthonormal_transform_from_gram(
        G,
        rank_rtol=float(rank_rtol),
        allow_rank_reveal=bool(allow_rank_reveal),
    )
    rank = int(T.shape[0])
    Kq = np.empty((raw_Kq.shape[0], rank, rank), dtype=np.float64)
    Bq = np.empty((raw_Bq.shape[0], rank, raw_Bq.shape[2]), dtype=np.float64)
    for q in range(raw_Kq.shape[0]):
        Kq[q] = T @ raw_Kq[q] @ T.T
        Kq[q] = 0.5 * (Kq[q] + Kq[q].T)
        Bq[q] = T @ raw_Bq[q]
    # Historical name retained for cache compatibility: invR = T.T.
    return Kq, Bq, T.T, gram_meta


def _householder_r(matrix: np.ndarray) -> np.ndarray:
    """Return the thin Householder-QR R factor without materializing Q."""
    values = np.asarray(matrix, dtype=np.float64, order="F")
    if values.ndim != 2 or values.shape[0] < values.shape[1]:
        raise ValueError("Householder QR requires a tall matrix")
    factored = scipy_linalg.qr(
        values,
        mode="r",
        overwrite_a=True,
        check_finite=False,
    )[0]
    rank = int(values.shape[1])
    return np.array(np.triu(factored[:rank]), dtype=np.float64, copy=True)


def _experimental_tsqr_factor(
    basis: np.ndarray | list[np.ndarray],
    *,
    nvox: int,
    block_max_gib: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Factor the normalized tall snapshot matrix by blocked Householder TSQR."""
    started = time.perf_counter()
    rank = _basis_rank_nvox(basis, int(nvox))
    if rank < 1:
        raise ValueError("TSQR requires at least one snapshot")
    if not np.isfinite(float(block_max_gib)) or float(block_max_gib) <= 0.0:
        raise ValueError("TSQR block_max_gib must be finite and positive")

    first = _field_component_voxel_view(
        basis[0] if isinstance(basis, list) else basis[0], int(nvox)
    )
    storage_itemsize = int(first.dtype.itemsize)
    block_budget_bytes = int(float(block_max_gib) * (1024**3))
    bytes_per_voxel = rank * 6 * (storage_itemsize + np.dtype(np.float64).itemsize)
    minimum_voxels = max(1, math.ceil(rank / 6))
    block_voxels = max(
        minimum_voxels,
        min(int(nvox), block_budget_bytes // max(bytes_per_voxel, 1)),
    )
    if block_voxels * bytes_per_voxel > block_budget_bytes:
        raise MemoryError(
            "TSQR workspace cap cannot hold one full-rank Householder block"
        )

    scale = 1.0 / math.sqrt(float(nvox))
    accumulated_r: np.ndarray | None = None
    local_factor_wall_s = 0.0
    merge_factor_wall_s = 0.0
    block_count = 0
    peak_snapshot_chunk_bytes = 0
    peak_qr_matrix_bytes = 0
    for start in range(0, int(nvox), int(block_voxels)):
        end = min(start + int(block_voxels), int(nvox))
        values = _basis_chunk(basis, nvox=int(nvox), start=start, end=end)
        peak_snapshot_chunk_bytes = max(
            peak_snapshot_chunk_bytes, int(values.nbytes)
        )
        tall = np.array(
            values.reshape(rank, -1).T,
            dtype=np.float64,
            order="F",
            copy=True,
        )
        tall *= scale
        peak_qr_matrix_bytes = max(peak_qr_matrix_bytes, int(tall.nbytes))
        local_started = time.perf_counter()
        local_r = _householder_r(tall)
        local_factor_wall_s += float(time.perf_counter() - local_started)
        del tall, values

        if accumulated_r is None:
            accumulated_r = local_r
        else:
            merge_started = time.perf_counter()
            accumulated_r = _householder_r(
                np.asfortranarray(np.vstack((accumulated_r, local_r)))
            )
            merge_factor_wall_s += float(time.perf_counter() - merge_started)
        block_count += 1

    if accumulated_r is None:
        raise RuntimeError("TSQR produced no local factors")
    diagonal = np.diag(accumulated_r).copy()
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    accumulated_r *= signs[:, None]
    diagonal = np.diag(accumulated_r)
    if not np.all(np.isfinite(accumulated_r)) or np.any(diagonal == 0.0):
        raise np.linalg.LinAlgError(
            "TSQR encountered an arithmetic-zero snapshot direction"
        )

    metadata = {
        "qr_method": "blocked_householder_tsqr",
        "qr_compute_dtype": "float64",
        "qr_forms_explicit_q": False,
        "qr_rank": int(rank),
        "qr_block_count": int(block_count),
        "qr_block_voxels": int(block_voxels),
        "qr_block_max_gib": float(block_max_gib),
        "qr_peak_snapshot_chunk_bytes": int(peak_snapshot_chunk_bytes),
        "qr_peak_factor_matrix_bytes": int(peak_qr_matrix_bytes),
        "qr_estimated_peak_temporary_bytes": int(
            peak_snapshot_chunk_bytes + peak_qr_matrix_bytes
        ),
        "qr_local_factor_wall_s": float(local_factor_wall_s),
        "qr_merge_factor_wall_s": float(merge_factor_wall_s),
        "qr_factor_wall_s": float(time.perf_counter() - started),
        "qr_r_condition": float(np.linalg.cond(accumulated_r)),
        "qr_r_diagonal_min_abs": float(np.min(np.abs(diagonal))),
        "qr_r_diagonal_max_abs": float(np.max(np.abs(diagonal))),
        "qr_r_diagonal_relative_min": float(
            np.min(np.abs(diagonal))
            / max(np.max(np.abs(diagonal)), np.finfo(np.float64).tiny)
        ),
    }
    return accumulated_r, metadata


def _experimental_tsqr_recompile(
    *,
    basis: np.ndarray | list[np.ndarray],
    raw_Kq: np.ndarray,
    raw_Bq: np.ndarray,
    Dq: np.ndarray,
    G: np.ndarray,
    nvox: int,
    block_max_gib: float = 2.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Recompile raw affine blocks in full-span TSQR coordinates."""
    total_started = time.perf_counter()
    R, metadata = _experimental_tsqr_factor(
        basis,
        nvox=int(nvox),
        block_max_gib=float(block_max_gib),
    )
    rank = int(R.shape[0])
    raw_K = np.asarray(raw_Kq, dtype=np.float64)
    raw_B = np.asarray(raw_Bq, dtype=np.float64)
    if raw_K.shape[1:] != (rank, rank) or raw_B.shape[1] != rank:
        raise ValueError("raw affine blocks do not match the TSQR snapshot rank")

    transform_started = time.perf_counter()
    Kq = np.empty_like(raw_K)
    Bq = np.empty_like(raw_B)
    for q in range(raw_K.shape[0]):
        left = scipy_linalg.solve_triangular(
            R.T,
            raw_K[q],
            lower=True,
            check_finite=False,
        )
        Kq[q] = scipy_linalg.solve_triangular(
            R.T,
            left.T,
            lower=True,
            check_finite=False,
        ).T
        Kq[q] = 0.5 * (Kq[q] + Kq[q].T)
        Bq[q] = scipy_linalg.solve_triangular(
            R.T,
            raw_B[q],
            lower=True,
            check_finite=False,
        )
    transform_wall_s = float(time.perf_counter() - transform_started)

    identity = np.eye(rank, dtype=np.float64)
    T = scipy_linalg.solve_triangular(
        R.T,
        identity,
        lower=True,
        check_finite=False,
    )
    gram = 0.5 * (np.asarray(G, dtype=np.float64) + np.asarray(G, dtype=np.float64).T)
    qr_gram = R.T @ R
    orthogonality = T @ gram @ T.T
    metadata.update(
        {
            "qr_transform_wall_s": transform_wall_s,
            "qr_total_wall_s": float(time.perf_counter() - total_started),
            "qr_gram_reconstruction_relative_error": float(
                np.linalg.norm(qr_gram - gram, ord="fro")
                / max(np.linalg.norm(gram, ord="fro"), np.finfo(float).tiny)
            ),
            "qr_orthogonality_frobenius_error": float(
                np.linalg.norm(orthogonality - identity, ord="fro")
            ),
            "qr_orthogonality_spectral_error": float(
                np.linalg.norm(orthogonality - identity, ord=2)
            ),
            "qr_discarded_rank": 0,
        }
    )
    operators = {
        "Kq": Kq,
        "Bq": Bq,
        "Dq": np.asarray(Dq, dtype=np.float64).copy(),
        "R": R,
    }
    return operators, metadata


def _reference_energy_qr_recompile(
    *,
    raw_Kq: np.ndarray,
    raw_Bq: np.ndarray,
    Dq: np.ndarray,
    reference_coefficients: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Whiten the full snapshot span in a reference-energy inner product.

    The raw affine contractions already contain ``S.T @ K_q @ S``.  Their
    coercive reference combination therefore supplies a weighted QR factor
    without another voxel-scale pass or an explicit Q basis.
    """
    total_started = time.perf_counter()
    raw_K = np.asarray(raw_Kq, dtype=np.float64)
    raw_B = np.asarray(raw_Bq, dtype=np.float64)
    coefficients = np.asarray(reference_coefficients, dtype=np.float64)
    if raw_K.ndim != 3 or raw_K.shape[1] != raw_K.shape[2]:
        raise ValueError("raw_Kq must contain square affine blocks")
    if raw_B.shape[:2] != raw_K.shape[:2]:
        raise ValueError("raw_Bq does not match raw_Kq")
    if coefficients.shape != (raw_K.shape[0],):
        raise ValueError("reference coefficients do not match affine blocks")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("reference coefficients must be finite")

    reference = np.einsum("q,qij->ij", coefficients, raw_K, optimize=True)
    reference = 0.5 * (reference + reference.T)
    factor_started = time.perf_counter()
    try:
        R = scipy_linalg.cholesky(
            reference,
            lower=False,
            overwrite_a=False,
            check_finite=False,
        )
    except np.linalg.LinAlgError as exc:
        eigenvalues = scipy_linalg.eigvalsh(reference, check_finite=False)
        raise np.linalg.LinAlgError(
            "reference-energy QR requires a positive-definite raw Ritz block; "
            f"lambda_min={float(eigenvalues[0]):.3e}"
        ) from exc
    factor_wall_s = float(time.perf_counter() - factor_started)

    transform_started = time.perf_counter()
    Kq = np.empty_like(raw_K)
    Bq = np.empty_like(raw_B)
    for q in range(raw_K.shape[0]):
        left = scipy_linalg.solve_triangular(
            R.T,
            raw_K[q],
            lower=True,
            check_finite=False,
        )
        Kq[q] = scipy_linalg.solve_triangular(
            R.T,
            left.T,
            lower=True,
            check_finite=False,
        ).T
        Kq[q] = 0.5 * (Kq[q] + Kq[q].T)
        Bq[q] = scipy_linalg.solve_triangular(
            R.T,
            raw_B[q],
            lower=True,
            check_finite=False,
        )
    transform_wall_s = float(time.perf_counter() - transform_started)

    rank = int(R.shape[0])
    identity = np.eye(rank, dtype=np.float64)
    normalized_reference = np.einsum(
        "q,qij->ij", coefficients, Kq, optimize=True
    )
    normalized_reference = 0.5 * (
        normalized_reference + normalized_reference.T
    )
    identity_error = normalized_reference - identity
    diagonal = np.diag(R)
    metadata = {
        "energy_qr_principal_route": True,
        "energy_qr_method": "reference_energy_cholesky_qr",
        "energy_qr_compute_dtype": "float64",
        "energy_qr_forms_explicit_q": False,
        "energy_qr_uses_snapshot_gram": False,
        "energy_qr_additional_voxel_passes": 0,
        "energy_qr_rank": rank,
        "energy_qr_discarded_rank": 0,
        "energy_qr_reference_coefficients": coefficients.copy(),
        "energy_qr_reference_condition": float(np.linalg.cond(reference)),
        "energy_qr_r_condition": float(np.linalg.cond(R)),
        "energy_qr_r_diagonal_min_abs": float(np.min(np.abs(diagonal))),
        "energy_qr_r_diagonal_max_abs": float(np.max(np.abs(diagonal))),
        "energy_qr_factor_wall_s": factor_wall_s,
        "energy_qr_transform_wall_s": transform_wall_s,
        "energy_qr_total_wall_s": float(time.perf_counter() - total_started),
        "energy_qr_reference_identity_frobenius_error": float(
            np.linalg.norm(identity_error, ord="fro")
        ),
        "energy_qr_reference_identity_spectral_error": float(
            np.linalg.norm(identity_error, ord=2)
        ),
    }
    operators = {
        "Kq": Kq,
        "Bq": Bq,
        "Dq": np.asarray(Dq, dtype=np.float64).copy(),
        "R": R,
        "invR": scipy_linalg.solve_triangular(
            R,
            identity,
            lower=False,
            check_finite=False,
        ),
        "reference_coefficients": coefficients.copy(),
    }
    return operators, metadata


def _transform_raw_operators_with_energy_retention(
    raw_Kq: np.ndarray,
    raw_Bq: np.ndarray,
    G: np.ndarray,
    *,
    retention: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Construct a conventional POD space using cumulative snapshot energy."""
    requested = float(retention)
    if not 0.0 < requested <= 1.0:
        raise ValueError("POD energy retention must lie in (0, 1].")
    gram = 0.5 * (
        np.asarray(G, dtype=np.float64) + np.asarray(G, dtype=np.float64).T
    )
    eigenvalues, eigenvectors = scipy_linalg.eigh(gram, check_finite=False)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(np.sum(eigenvalues))
    if not np.isfinite(total) or total <= 0.0:
        raise np.linalg.LinAlgError("snapshot Gram matrix has no positive POD energy")
    descending = eigenvalues[::-1]
    rank = int(np.searchsorted(np.cumsum(descending) / total, requested) + 1)
    rank = min(rank, len(eigenvalues))
    selected_values = eigenvalues[-rank:]
    positive = selected_values > np.finfo(np.float64).eps * eigenvalues[-1]
    if not np.any(positive):
        raise np.linalg.LinAlgError("selected POD energy space has zero numerical rank")
    selected_vectors = eigenvectors[:, -rank:][:, positive]
    selected_values = selected_values[positive]
    T = (selected_vectors / np.sqrt(selected_values)).T
    effective_rank = int(T.shape[0])
    Kq = np.empty((raw_Kq.shape[0], effective_rank, effective_rank), dtype=np.float64)
    Bq = np.empty((raw_Bq.shape[0], effective_rank, raw_Bq.shape[2]), dtype=np.float64)
    for q in range(raw_Kq.shape[0]):
        Kq[q] = T @ np.asarray(raw_Kq[q], dtype=np.float64) @ T.T
        Kq[q] = 0.5 * (Kq[q] + Kq[q].T)
        Bq[q] = T @ np.asarray(raw_Bq[q], dtype=np.float64)
    retained_fraction = float(np.sum(selected_values) / total)
    metadata = {
        "pod_energy_retention_requested": requested,
        "pod_energy_retention_realized": retained_fraction,
        "effective_rank": effective_rank,
        "discarded_rank": int(len(eigenvalues) - effective_rank),
        "gram_lambda_min": float(eigenvalues[0]),
        "gram_lambda_max": float(eigenvalues[-1]),
        "gram_transform_mode": "conventional_pod_energy",
    }
    return Kq, Bq, T.T, metadata


def _assemble_dense_operators(
    *,
    phase: np.ndarray,
    ori: np.ndarray,
    basis: np.ndarray | list[np.ndarray],
    affine_stress_batch: Any | None,
    preserve_raw_coordinates: bool = False,
    gram_rank_reveal: bool = False,
    gram_rank_rtol: float = 1.0e-11,
    contraction_compute_dtype: str | np.dtype = np.float64,
    gram_compute_dtype: str | np.dtype = np.float64,
    gram_backend: str = "auto",
    overlap_cpu_gram_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compatibility assembly for non phase-contiguous layouts."""
    t0 = time.perf_counter()
    nvox = int(np.asarray(phase).size)
    r = _basis_rank_nvox(basis, nvox)
    if r < 1:
        raise ValueError("basis must contain at least one field")
    affine = affine_stress_batch or affine_stress_batch_factory(phase, ori)
    coefficient_names = tuple(getattr(affine, "coefficient_names", COEFF_NAMES))
    q_count = len(coefficient_names)
    values = _basis_chunk(basis, nvox=nvox, start=0, end=nvox)
    compute_dtype = _ritz_compute_dtype(contraction_compute_dtype)
    gram_dtype = _ritz_compute_dtype(gram_compute_dtype)
    selected_gram_backend = _ritz_gram_backend(gram_backend)
    raw_Kq = np.zeros((q_count, r, r), dtype=np.float64)
    raw_Bq = np.zeros((q_count, r, 6), dtype=np.float64)
    G = None
    if not bool(preserve_raw_coordinates):
        gram_contract = (
            _contract_component_major_cpu
            if selected_gram_backend == "cpu"
            else _contract_component_major_compute
        )
        G = gram_contract(values, values, compute_dtype=gram_dtype) / float(nvox)
    stress_started = time.perf_counter()
    stresses = np.asarray(affine.apply_all(np.asarray(values, dtype=compute_dtype)))
    affine_stress_wall_s = float(time.perf_counter() - stress_started)
    contraction_started = time.perf_counter()
    values_gpu = _gpu_flat_compute(values, compute_dtype=compute_dtype)
    stresses_gpu = _gpu_batch_flat_compute(stresses, compute_dtype=compute_dtype)
    raw_Kq[:] = _contract_preloaded_batch(
        values,
        stresses,
        values_gpu,
        stresses_gpu,
        compute_dtype=compute_dtype,
    ) / float(nvox)
    raw_Bq[:] = np.sum(stresses, axis=3, dtype=np.float64) / float(nvox)
    if values_gpu is not None:
        del values_gpu
    if stresses_gpu is not None:
        del stresses_gpu
    contraction_wall_s = float(time.perf_counter() - contraction_started)
    Dq = np.asarray(affine.averaged_stiffness, dtype=np.float64).copy()
    if G is not None:
        G = 0.5 * (G + G.T)
    for q in range(q_count):
        raw_Kq[q] = 0.5 * (raw_Kq[q] + raw_Kq[q].T)
        Dq[q] = 0.5 * (Dq[q] + Dq[q].T)
    if preserve_raw_coordinates:
        Kq = raw_Kq.copy()
        Bq = raw_Bq.copy()
        invR = np.eye(r, dtype=np.float64)
        gram_meta = {
            "gram_lambda_min": 0.0,
            "gram_lambda_max": 0.0,
            "gram_condition": 1.0,
            "gram_relative_min": 1.0,
            "effective_rank": int(r),
            "discarded_rank": 0,
            "gram_transform_mode": "raw_coordinates_no_gram",
        }
    else:
        assert G is not None
        Kq, Bq, invR, gram_meta = _transform_raw_operators_with_rank_policy(
            raw_Kq,
            raw_Bq,
            G,
            allow_rank_reveal=bool(gram_rank_reveal),
            rank_rtol=float(gram_rank_rtol),
        )
    metadata = {
        "assembly_wall_s": float(time.perf_counter() - t0),
        "assembly_mode": "batched_affine_cpu",
        "contraction_mode": "dense_full_support",
        "contraction_dtype": str(values.dtype),
        "contraction_compute_dtype": str(compute_dtype),
        "gram_product_dtype": (
            "not_computed" if bool(preserve_raw_coordinates) else str(gram_dtype)
        ),
        "gram_product_backend": (
            "none" if bool(preserve_raw_coordinates) else selected_gram_backend
        ),
        "gram_product_wall_s": 0.0,
        "gram_overlap_wait_wall_s": 0.0,
        "gram_overlap_hidden_wall_s": 0.0,
        "gram_overlap_requested": False,
        "gram_overlap_enabled": False,
        "affine_stress_wall_s": affine_stress_wall_s,
        "contraction_wall_s": contraction_wall_s,
        "stress_workspace_peak_bytes": int(stresses.nbytes),
        "full_volume_equivalent_passes": float(q_count),
        "raw_Kq": raw_Kq,
        "raw_Bq": raw_Bq,
        "invR": invR,
        **gram_meta,
    }
    if G is not None:
        metadata["G"] = G
    return Kq, Bq, Dq, metadata


def _assemble_reduced_operators(
    *,
    phase: np.ndarray,
    ori: np.ndarray,
    basis: np.ndarray | list[np.ndarray],
    affine_stress_batch: Any | None = None,
    affine_q_block_size: int | None = None,
    gram_rank_reveal: bool = False,
    gram_rank_rtol: float = 1.0e-11,
    contraction_compute_dtype: str | np.dtype = np.float64,
    gram_compute_dtype: str | np.dtype = np.float64,
    gram_backend: str = "auto",
    overlap_cpu_gram_gpu: bool = False,
    preserve_raw_coordinates: bool = False,
    factorized_ritz: bool = False,
    async_ritz: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Stream an exact reduced Ritz assembly without stacking the global basis.

    Large fields remain in their storage precision (normally float32). Only
    small spatial chunks are materialized; products use the requested compute
    precision and reduced blocks are accumulated in float64.
    """
    t0 = time.perf_counter()
    nvox = int(np.asarray(phase).size)
    r = _basis_rank_nvox(basis, nvox)
    if r < 1:
        raise ValueError("basis must contain at least one field")
    compute_dtype = _ritz_compute_dtype(contraction_compute_dtype)
    gram_dtype = _ritz_compute_dtype(gram_compute_dtype)
    selected_gram_backend = _ritz_gram_backend(gram_backend)
    external_affine = affine_stress_batch is not None
    affine = affine_stress_batch or affine_stress_batch_factory(phase, ori)
    apply_chunk = getattr(affine, "apply_supported_chunk", None)
    apply_chunk_gpu = getattr(affine, "apply_supported_chunk_gpu", None)
    support_blocks = getattr(affine, "support_blocks", None)
    if apply_chunk is None or support_blocks is None:
        return _assemble_dense_operators(
            phase=phase,
            ori=ori,
            basis=basis,
            affine_stress_batch=affine,
            preserve_raw_coordinates=(
                bool(preserve_raw_coordinates)
                or (external_affine and isinstance(basis, list))
            ),
            gram_rank_reveal=bool(gram_rank_reveal),
            gram_rank_rtol=float(gram_rank_rtol),
            contraction_compute_dtype=compute_dtype,
            gram_compute_dtype=gram_dtype,
            gram_backend=selected_gram_backend,
            overlap_cpu_gram_gpu=bool(overlap_cpu_gram_gpu),
        )

    coefficient_names = tuple(getattr(affine, "coefficient_names", COEFF_NAMES))
    q_count = len(coefficient_names)
    requested_q_block_size = (
        q_count if affine_q_block_size is None else int(affine_q_block_size)
    )
    q_block_size = max(1, requested_q_block_size)
    q_block_size = min(q_block_size, q_count)
    first_dtype = _field_component_voxel_view(
        basis[0] if isinstance(basis, list) else basis[0], nvox
    ).dtype
    if bool(factorized_ritz):
        if not bool(preserve_raw_coordinates):
            raise ValueError(
                "factorized_ritz currently requires raw Ritz coordinates"
            )
        _, raw_Kq, raw_Bq, factorized_meta = _factorized_raw_gpu(
            affine=affine,
            old_basis=None,
            new_basis=basis,
            nvox=nvox,
            compute_dtype=compute_dtype,
            async_transfers=bool(async_ritz),
        )
        averaged = getattr(affine, "averaged_stiffness", None)
        Dq = np.asarray(averaged, dtype=np.float64).copy()
        for q in range(q_count):
            raw_Kq[q] = 0.5 * (raw_Kq[q] + raw_Kq[q].T)
            Dq[q] = 0.5 * (Dq[q] + Dq[q].T)
        invR = np.eye(r, dtype=np.float64)
        metadata = {
            "assembly_wall_s": float(time.perf_counter() - t0),
            "assembly_mode": "factorized_gpu",
            "contraction_mode": "exact_constitutive_rank_factorization",
            "contraction_dtype": str(first_dtype),
            "gram_product_dtype": "not_computed",
            "gram_product_backend": "none",
            "gram_overlap_requested": False,
            "gram_overlap_enabled": False,
            "gram_overlap_used_chunks": 0,
            "gram_product_wall_s": 0.0,
            "gram_overlap_wait_wall_s": 0.0,
            "gram_overlap_hidden_wall_s": 0.0,
            "contraction_compute_dtype": str(compute_dtype),
            "reduced_accumulation_dtype": "float64_gpu_resident",
            "affine_stress_wall_s": 0.0,
            "contraction_wall_s": float(factorized_meta["factorized_gpu_wall_s"]),
            "full_volume_equivalent_passes": float(
                sum(
                    len(indices)
                    * (int(selector.stop or nvox) - int(selector.start or 0))
                    / float(nvox)
                    for _, indices, selector in support_blocks
                )
            ),
            "q_block_size": int(q_block_size),
            "batched_contraction": True,
            "avoided_duplicate_basis_gpu_uploads": int(
                factorized_meta["basis_gpu_uploads"]
            ),
            "affine_stress_backend": "gpu_factorized_exact",
            "gpu_affine_chunks": int(factorized_meta["factorized_chunk_count"]),
            "cpu_affine_chunks": 0,
            "gpu_affine_fallback": "",
            "raw_Kq": raw_Kq,
            "raw_Bq": raw_Bq,
            "invR": invR,
            "gram_lambda_min": 0.0,
            "gram_lambda_max": 0.0,
            "gram_condition": 1.0,
            "gram_relative_min": 1.0,
            "effective_rank": int(r),
            "discarded_rank": 0,
            "gram_transform_mode": "raw_coordinates_no_gram",
            "factorized_ritz": True,
            "async_ritz": bool(async_ritz),
            **factorized_meta,
        }
        return raw_Kq.copy(), raw_Bq.copy(), Dq, metadata
    chunk_voxels = _stream_chunk_voxels(
        ranks=(r,), q_block_size=q_block_size, nvox=nvox,
        storage_itemsize=max(
            int(first_dtype.itemsize),
            int(compute_dtype.itemsize),
            int(gram_dtype.itemsize),
        ),
        max_chunk_voxels=1_000_000,
    )

    raw_Kq = np.zeros((q_count, r, r), dtype=np.float64)
    raw_Bq = np.zeros((q_count, r, 6), dtype=np.float64)
    G = (
        None
        if bool(preserve_raw_coordinates)
        else np.zeros((r, r), dtype=np.float64)
    )
    affine_stress_wall_s = 0.0
    contraction_wall_s = 0.0
    stress_workspace_peak_bytes = 0
    full_volume_equivalent_passes = 0.0
    basis_gpu_uploads = 0
    gpu_affine_chunks = 0
    cpu_affine_chunks = 0
    gpu_affine_fallback = ""
    gram_product_wall_s = 0.0
    gram_overlap_wait_wall_s = 0.0
    gram_overlap_used_chunks = 0
    gram_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="ritz-gram")
        if (
            not bool(preserve_raw_coordinates)
            and bool(overlap_cpu_gram_gpu)
            and selected_gram_backend == "cpu"
        )
        else None
    )

    for support, support_indices, selector in support_blocks:
        support_start = int(selector.start or 0)
        support_stop = int(selector.stop or nvox)
        support_voxels = max(0, support_stop - support_start)
        full_volume_equivalent_passes += (
            float(len(support_indices)) * float(support_voxels) / float(nvox)
        )
        for global_start in range(support_start, support_stop, chunk_voxels):
            global_end = min(global_start + chunk_voxels, support_stop)
            values = _basis_chunk(basis, nvox=nvox, start=global_start, end=global_end)
            contraction_started = time.perf_counter()
            values_gpu = _gpu_flat_compute(values, compute_dtype=compute_dtype)
            gram_future: Future[tuple[np.ndarray, float]] | None = None
            gram_gpu = None
            if not bool(preserve_raw_coordinates):
                gram_gpu = (
                    values_gpu
                    if selected_gram_backend != "cpu" and gram_dtype == compute_dtype
                    else (
                        _gpu_flat_compute(values, compute_dtype=gram_dtype)
                        if selected_gram_backend != "cpu"
                        else None
                    )
                )
            separate_gram_gpu = gram_gpu is not None and gram_gpu is not values_gpu
            if gram_gpu is not None:
                import cupy as cp
                gram_started = time.perf_counter()
                basis_gpu_uploads += 1 + int(separate_gram_gpu)
                assert G is not None
                G += cp.asnumpy(gram_gpu @ gram_gpu.T) / float(nvox)
                gram_product_wall_s += float(time.perf_counter() - gram_started)
            elif gram_executor is not None and values_gpu is not None and apply_chunk_gpu is not None:
                gram_future = gram_executor.submit(
                    _timed_cpu_gram,
                    values,
                    compute_dtype=gram_dtype,
                )
                gram_overlap_used_chunks += 1
            elif not bool(preserve_raw_coordinates):
                gram_contract = (
                    _contract_component_major_cpu
                    if selected_gram_backend == "cpu"
                    else _contract_component_major_compute
                )
                gram_started = time.perf_counter()
                assert G is not None
                G += gram_contract(values, values, compute_dtype=gram_dtype) / float(nvox)
                gram_product_wall_s += float(time.perf_counter() - gram_started)
            contraction_wall_s += float(time.perf_counter() - contraction_started)
            support_offset = global_start - support_start

            for q_start in range(0, len(support_indices), q_block_size):
                q_indices = np.asarray(
                    support_indices[q_start : q_start + q_block_size], dtype=np.intp
                )
                stress_started = time.perf_counter()
                stresses_gpu = None
                if values_gpu is not None and apply_chunk_gpu is not None:
                    try:
                        stresses_gpu = apply_chunk_gpu(
                            q_indices, values_gpu, support, support_offset
                        )
                        import cupy as cp

                        cp.cuda.get_current_stream().synchronize()
                        gpu_affine_chunks += 1
                        stress_workspace_peak_bytes = max(
                            stress_workspace_peak_bytes, int(stresses_gpu.nbytes)
                        )
                    except Exception as exc:
                        gpu_affine_fallback = f"{type(exc).__name__}: {exc}"
                        apply_chunk_gpu = None
                        stresses_gpu = None
                if stresses_gpu is None:
                    stresses = apply_chunk(
                        q_indices,
                        np.asarray(values, dtype=compute_dtype),
                        support,
                        support_offset,
                    )
                    cpu_affine_chunks += 1
                    stress_workspace_peak_bytes = max(
                        stress_workspace_peak_bytes, int(stresses.nbytes)
                    )
                affine_stress_wall_s += float(time.perf_counter() - stress_started)
                contraction_started = time.perf_counter()
                if stresses_gpu is not None:
                    contracted, summed = _contract_and_sum_gpu_batch(
                        values_gpu, stresses_gpu
                    )
                    raw_Kq[q_indices] += contracted / float(nvox)
                    raw_Bq[q_indices] += summed / float(nvox)
                else:
                    uploaded_stresses = _gpu_batch_flat_compute(
                        stresses, compute_dtype=compute_dtype
                    )
                    raw_Kq[q_indices] += _contract_preloaded_batch(
                        values,
                        stresses,
                        values_gpu,
                        uploaded_stresses,
                        compute_dtype=compute_dtype,
                    ) / float(nvox)
                    raw_Bq[q_indices] += (
                        np.sum(stresses, axis=3, dtype=np.float64) / float(nvox)
                    )
                    if uploaded_stresses is not None:
                        del uploaded_stresses
                contraction_wall_s += float(time.perf_counter() - contraction_started)
                if stresses_gpu is not None:
                    del stresses_gpu
                else:
                    del stresses
            if gram_future is not None:
                wait_started = time.perf_counter()
                gram_block, gram_elapsed = gram_future.result()
                wait_elapsed = float(time.perf_counter() - wait_started)
                assert G is not None
                G += gram_block / float(nvox)
                gram_product_wall_s += float(gram_elapsed)
                gram_overlap_wait_wall_s += wait_elapsed
                contraction_wall_s += wait_elapsed
            if separate_gram_gpu:
                del gram_gpu
            if values_gpu is not None:
                del values_gpu
            del values

    if gram_executor is not None:
        gram_executor.shutdown(wait=True)

    averaged = getattr(affine, "averaged_stiffness", None)
    Dq = np.asarray(averaged, dtype=np.float64).copy()
    if G is not None:
        G = 0.5 * (G + G.T)
    for q in range(q_count):
        raw_Kq[q] = 0.5 * (raw_Kq[q] + raw_Kq[q].T)
        Dq[q] = 0.5 * (Dq[q] + Dq[q].T)

    if bool(preserve_raw_coordinates):
        Kq = raw_Kq.copy()
        Bq = raw_Bq.copy()
        invR = np.eye(r, dtype=np.float64)
        gram_meta = {
            "gram_lambda_min": 0.0,
            "gram_lambda_max": 0.0,
            "gram_condition": 1.0,
            "gram_relative_min": 1.0,
            "effective_rank": int(r),
            "discarded_rank": 0,
            "gram_transform_mode": "raw_coordinates_no_gram",
        }
    else:
        assert G is not None
        Kq, Bq, invR, gram_meta = _transform_raw_operators_with_rank_policy(
            raw_Kq,
            raw_Bq,
            G,
            allow_rank_reveal=bool(gram_rank_reveal),
            rank_rtol=float(gram_rank_rtol),
        )
    metadata = {
        "assembly_wall_s": float(time.perf_counter() - t0),
        "assembly_mode": "phase_supported_blocks",
        "contraction_mode": "phase_supported_blocks",
        "contraction_dtype": str(first_dtype),
        "gram_product_dtype": (
            "not_computed" if bool(preserve_raw_coordinates) else str(gram_dtype)
        ),
        "gram_product_backend": (
            "none" if bool(preserve_raw_coordinates) else selected_gram_backend
        ),
        "gram_overlap_requested": bool(overlap_cpu_gram_gpu)
        and not bool(preserve_raw_coordinates),
        "gram_overlap_enabled": bool(gram_overlap_used_chunks),
        "gram_overlap_used_chunks": int(gram_overlap_used_chunks),
        "gram_product_wall_s": float(gram_product_wall_s),
        "gram_overlap_wait_wall_s": float(gram_overlap_wait_wall_s),
        "gram_overlap_hidden_wall_s": float(
            max(0.0, gram_product_wall_s - gram_overlap_wait_wall_s)
            if gram_overlap_used_chunks
            else 0.0
        ),
        "contraction_compute_dtype": str(compute_dtype),
        "reduced_accumulation_dtype": "float64",
        "affine_stress_wall_s": affine_stress_wall_s,
        "contraction_wall_s": contraction_wall_s,
        "stress_workspace_peak_bytes": int(stress_workspace_peak_bytes),
        "full_volume_equivalent_passes": float(full_volume_equivalent_passes),
        "chunk_voxels": int(chunk_voxels),
        "q_block_size": int(q_block_size),
        "batched_contraction": True,
        "basis_gpu_uploads": int(basis_gpu_uploads),
        "avoided_duplicate_basis_gpu_uploads": int(basis_gpu_uploads),
        "affine_stress_backend": (
            "gpu_resident" if gpu_affine_chunks and not cpu_affine_chunks else "cpu"
        ),
        "gpu_affine_chunks": int(gpu_affine_chunks),
        "cpu_affine_chunks": int(cpu_affine_chunks),
        "gpu_affine_fallback": gpu_affine_fallback,
        "raw_Kq": raw_Kq,
        "raw_Bq": raw_Bq,
        "invR": invR,
        **gram_meta,
    }
    if G is not None:
        metadata["G"] = G
    return Kq, Bq, Dq, metadata


def _extend_reduced_operators(
    *,
    existing: dict[str, Any],
    old_basis: np.ndarray | list[np.ndarray],
    new_basis: np.ndarray | list[np.ndarray],
    affine_stress_batch: Any,
    affine_q_block_size: int | None = None,
    gram_rank_reveal: bool = False,
    gram_rank_rtol: float = 1.0e-11,
    contraction_compute_dtype: str | np.dtype = np.float64,
    gram_compute_dtype: str | np.dtype = np.float64,
    gram_backend: str = "auto",
    overlap_cpu_gram_gpu: bool = False,
    preserve_raw_coordinates: bool = False,
    factorized_ritz: bool = False,
    async_ritz: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Exact streaming extension of raw Ritz blocks, followed by re-whitening."""
    t0 = time.perf_counter()
    compute_dtype = _ritz_compute_dtype(contraction_compute_dtype)
    gram_dtype = _ritz_compute_dtype(gram_compute_dtype)
    selected_gram_backend = _ritz_gram_backend(gram_backend)
    raw_cache_available = (
        "raw_Kq" in existing
        and "raw_Bq" in existing
        and (bool(preserve_raw_coordinates) or "G" in existing)
    )
    if not raw_cache_available:
        if isinstance(old_basis, np.ndarray) or isinstance(new_basis, np.ndarray):
            old_array = np.asarray(old_basis)
            new_array = np.asarray(new_basis)
            combined: np.ndarray | list[np.ndarray] = np.concatenate(
                (old_array, new_array), axis=0
            )
        else:
            combined = [*old_basis, *new_basis]
        Kq, Bq, Dq, metadata = _assemble_reduced_operators(
            phase=np.zeros(
                int(np.asarray(combined[0] if isinstance(combined, list) else combined[0]).size // 6),
                dtype=np.uint8,
            ),
            ori=np.zeros(
                (
                    int(np.asarray(combined[0] if isinstance(combined, list) else combined[0]).size // 6),
                    3,
                ),
                dtype=np.float64,
            ),
            basis=combined,
            affine_stress_batch=affine_stress_batch,
            affine_q_block_size=affine_q_block_size,
            gram_rank_reveal=bool(gram_rank_reveal),
            gram_rank_rtol=float(gram_rank_rtol),
            contraction_compute_dtype=compute_dtype,
            gram_compute_dtype=gram_dtype,
            gram_backend=selected_gram_backend,
            overlap_cpu_gram_gpu=bool(overlap_cpu_gram_gpu),
            preserve_raw_coordinates=bool(preserve_raw_coordinates),
            factorized_ritz=bool(factorized_ritz),
            async_ritz=bool(async_ritz),
        )
        metadata.update({
            "assembly_mode": "incremental",
            "incremental_mode": "full_reassembly_without_raw_cache",
            "assembly_wall_s": float(time.perf_counter() - t0),
        })
        return Kq, Bq, Dq, metadata

    raw_K_old = np.asarray(existing["raw_Kq"], dtype=np.float64)
    raw_B_old = np.asarray(existing["raw_Bq"], dtype=np.float64)
    G_old = (
        None
        if bool(preserve_raw_coordinates)
        else np.asarray(existing["G"], dtype=np.float64)
    )
    Dq = np.asarray(existing["Dq"], dtype=np.float64).copy()
    old_rank = int(raw_K_old.shape[1])
    count = int(len(new_basis) if isinstance(new_basis, list) else new_basis.shape[0])
    if count < 1:
        return (
            np.asarray(existing["Kq"], dtype=np.float64),
            np.asarray(existing["Bq"], dtype=np.float64),
            Dq,
            {"assembly_wall_s": 0.0, **existing},
        )
    nvox = int(
        np.asarray(new_basis[0] if isinstance(new_basis, list) else new_basis[0]).size // 6
    )
    if _basis_rank_nvox(old_basis, nvox) != old_rank:
        raise ValueError("old basis rank does not match cached raw operators")
    new_rank = old_rank + count

    apply_chunk = getattr(affine_stress_batch, "apply_supported_chunk", None)
    apply_chunk_gpu = getattr(affine_stress_batch, "apply_supported_chunk_gpu", None)
    support_blocks = getattr(affine_stress_batch, "support_blocks", None)
    if apply_chunk is None or support_blocks is None:
        raise NotImplementedError("streaming extension requires supported chunk kernels")
    coefficient_names = tuple(getattr(affine_stress_batch, "coefficient_names", COEFF_NAMES))
    q_count = len(coefficient_names)
    requested_q_block_size = (
        q_count if affine_q_block_size is None else int(affine_q_block_size)
    )
    q_block_size = max(1, min(requested_q_block_size, q_count))
    first_dtype = _field_component_voxel_view(
        new_basis[0] if isinstance(new_basis, list) else new_basis[0], nvox
    ).dtype
    if bool(factorized_ritz):
        if not bool(preserve_raw_coordinates):
            raise ValueError(
                "factorized_ritz currently requires raw Ritz coordinates"
            )
        cross, diagonal, appended_Bq, factorized_meta = (
            _factorized_raw_gpu(
                affine=affine_stress_batch,
                old_basis=old_basis,
                new_basis=new_basis,
                nvox=nvox,
                compute_dtype=compute_dtype,
                async_transfers=bool(async_ritz),
            )
        )
        assert cross is not None
        raw_Kq = np.zeros((q_count, new_rank, new_rank), dtype=np.float64)
        raw_Bq = np.zeros((q_count, new_rank, 6), dtype=np.float64)
        raw_Kq[:, :old_rank, :old_rank] = raw_K_old
        raw_Kq[:, :old_rank, old_rank:] = cross
        raw_Kq[:, old_rank:, :old_rank] = np.swapaxes(cross, 1, 2)
        raw_Kq[:, old_rank:, old_rank:] = diagonal
        raw_Bq[:, :old_rank] = raw_B_old
        raw_Bq[:, old_rank:] = appended_Bq
        for q in range(q_count):
            raw_Kq[q] = 0.5 * (raw_Kq[q] + raw_Kq[q].T)
            Dq[q] = 0.5 * (Dq[q] + Dq[q].T)
        invR = np.eye(new_rank, dtype=np.float64)
        metadata = {
            "assembly_wall_s": float(time.perf_counter() - t0),
            "assembly_mode": "incremental_factorized_gpu",
            "incremental_mode": "raw_block_extension_factorized_gpu",
            "contraction_mode": "exact_constitutive_rank_factorization",
            "contraction_dtype": str(first_dtype),
            "gram_product_dtype": "not_computed",
            "gram_product_backend": "none",
            "gram_overlap_requested": False,
            "gram_overlap_enabled": False,
            "gram_overlap_used_chunks": 0,
            "gram_product_wall_s": 0.0,
            "gram_overlap_wait_wall_s": 0.0,
            "gram_overlap_hidden_wall_s": 0.0,
            "contraction_compute_dtype": str(compute_dtype),
            "reduced_accumulation_dtype": "float64_gpu_resident",
            "affine_stress_wall_s": 0.0,
            "contraction_wall_s": float(factorized_meta["factorized_gpu_wall_s"]),
            "full_volume_equivalent_passes": float(
                sum(
                    len(indices)
                    * (int(selector.stop or nvox) - int(selector.start or 0))
                    / float(nvox)
                    for _, indices, selector in support_blocks
                )
            ),
            "q_block_size": int(q_block_size),
            "batched_contraction": True,
            "avoided_duplicate_basis_gpu_uploads": int(
                factorized_meta["basis_gpu_uploads"]
            ),
            "affine_stress_backend": "gpu_factorized_exact",
            "gpu_affine_chunks": int(factorized_meta["factorized_chunk_count"]),
            "cpu_affine_chunks": 0,
            "gpu_affine_fallback": "",
            "raw_Kq": raw_Kq,
            "raw_Bq": raw_Bq,
            "invR": invR,
            "gram_lambda_min": 0.0,
            "gram_lambda_max": 0.0,
            "gram_condition": 1.0,
            "gram_relative_min": 1.0,
            "effective_rank": int(new_rank),
            "discarded_rank": 0,
            "gram_transform_mode": "raw_coordinates_no_gram",
            "factorized_ritz": True,
            "async_ritz": bool(async_ritz),
            **factorized_meta,
        }
        return raw_Kq.copy(), raw_Bq.copy(), Dq, metadata
    chunk_voxels = _stream_chunk_voxels(
        ranks=(old_rank, count), q_block_size=q_block_size, nvox=nvox,
        storage_itemsize=max(
            int(first_dtype.itemsize),
            int(compute_dtype.itemsize),
            int(gram_dtype.itemsize),
        ),
        max_chunk_voxels=1_000_000,
    )

    raw_Kq = np.zeros((q_count, new_rank, new_rank), dtype=np.float64)
    raw_Bq = np.zeros((q_count, new_rank, 6), dtype=np.float64)
    G = (
        None
        if bool(preserve_raw_coordinates)
        else np.zeros((new_rank, new_rank), dtype=np.float64)
    )
    raw_Kq[:, :old_rank, :old_rank] = raw_K_old
    raw_Bq[:, :old_rank] = raw_B_old
    if G is not None and G_old is not None:
        G[:old_rank, :old_rank] = G_old
    affine_stress_wall_s = 0.0
    contraction_wall_s = 0.0
    stress_workspace_peak_bytes = 0
    full_volume_equivalent_passes = 0.0
    basis_gpu_uploads = 0
    gpu_affine_chunks = 0
    cpu_affine_chunks = 0
    gpu_affine_fallback = ""
    gram_product_wall_s = 0.0
    gram_overlap_wait_wall_s = 0.0
    gram_overlap_used_chunks = 0
    gram_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="ritz-gram")
        if (
            not bool(preserve_raw_coordinates)
            and bool(overlap_cpu_gram_gpu)
            and selected_gram_backend == "cpu"
        )
        else None
    )

    for support, support_indices, selector in support_blocks:
        support_start = int(selector.start or 0)
        support_stop = int(selector.stop or nvox)
        support_voxels = max(0, support_stop - support_start)
        full_volume_equivalent_passes += (
            float(len(support_indices)) * float(support_voxels) / float(nvox)
        )
        for global_start in range(support_start, support_stop, chunk_voxels):
            global_end = min(global_start + chunk_voxels, support_stop)
            old_values = _basis_chunk(old_basis, nvox=nvox, start=global_start, end=global_end)
            new_values = _basis_chunk(new_basis, nvox=nvox, start=global_start, end=global_end)
            contraction_started = time.perf_counter()
            old_gpu = _gpu_flat_compute(old_values, compute_dtype=compute_dtype)
            new_gpu = _gpu_flat_compute(new_values, compute_dtype=compute_dtype)
            gram_future: Future[tuple[np.ndarray, np.ndarray, float]] | None = None
            old_gram_gpu = None
            new_gram_gpu = None
            if not bool(preserve_raw_coordinates):
                old_gram_gpu = (
                    old_gpu
                    if selected_gram_backend != "cpu" and gram_dtype == compute_dtype
                    else (
                        _gpu_flat_compute(old_values, compute_dtype=gram_dtype)
                        if selected_gram_backend != "cpu"
                        else None
                    )
                )
                new_gram_gpu = (
                    new_gpu
                    if selected_gram_backend != "cpu" and gram_dtype == compute_dtype
                    else (
                        _gpu_flat_compute(new_values, compute_dtype=gram_dtype)
                        if selected_gram_backend != "cpu"
                        else None
                    )
                )
            separate_old_gram_gpu = (
                old_gram_gpu is not None and old_gram_gpu is not old_gpu
            )
            separate_new_gram_gpu = (
                new_gram_gpu is not None and new_gram_gpu is not new_gpu
            )
            if old_gram_gpu is not None and new_gram_gpu is not None:
                import cupy as cp
                gram_started = time.perf_counter()
                basis_gpu_uploads += 2
                basis_gpu_uploads += int(separate_old_gram_gpu)
                basis_gpu_uploads += int(separate_new_gram_gpu)
                assert G is not None
                G[:old_rank, old_rank:] += cp.asnumpy(
                    old_gram_gpu @ new_gram_gpu.T
                ) / float(nvox)
                G[old_rank:, old_rank:] += cp.asnumpy(
                    new_gram_gpu @ new_gram_gpu.T
                ) / float(nvox)
                gram_product_wall_s += float(time.perf_counter() - gram_started)
            elif (
                gram_executor is not None
                and old_gpu is not None
                and new_gpu is not None
                and apply_chunk_gpu is not None
            ):
                gram_future = gram_executor.submit(
                    _timed_cpu_gram_extension,
                    old_values,
                    new_values,
                    compute_dtype=gram_dtype,
                )
                gram_overlap_used_chunks += 1
            elif not bool(preserve_raw_coordinates):
                gram_contract = (
                    _contract_component_major_cpu
                    if selected_gram_backend == "cpu"
                    else _contract_component_major_compute
                )
                gram_started = time.perf_counter()
                assert G is not None
                G[:old_rank, old_rank:] += (
                    gram_contract(
                        old_values, new_values, compute_dtype=gram_dtype
                    ) / float(nvox)
                )
                G[old_rank:, old_rank:] += (
                    gram_contract(
                        new_values, new_values, compute_dtype=gram_dtype
                    ) / float(nvox)
                )
                gram_product_wall_s += float(time.perf_counter() - gram_started)
            contraction_wall_s += float(time.perf_counter() - contraction_started)
            support_offset = global_start - support_start

            for q_start in range(0, len(support_indices), q_block_size):
                q_indices = np.asarray(
                    support_indices[q_start : q_start + q_block_size], dtype=np.intp
                )
                stress_started = time.perf_counter()
                stresses_gpu = None
                if new_gpu is not None and apply_chunk_gpu is not None:
                    try:
                        stresses_gpu = apply_chunk_gpu(
                            q_indices, new_gpu, support, support_offset
                        )
                        import cupy as cp

                        cp.cuda.get_current_stream().synchronize()
                        gpu_affine_chunks += 1
                        stress_workspace_peak_bytes = max(
                            stress_workspace_peak_bytes, int(stresses_gpu.nbytes)
                        )
                    except Exception as exc:
                        gpu_affine_fallback = f"{type(exc).__name__}: {exc}"
                        apply_chunk_gpu = None
                        stresses_gpu = None
                if stresses_gpu is None:
                    stresses = apply_chunk(
                        q_indices,
                        np.asarray(new_values, dtype=compute_dtype),
                        support,
                        support_offset,
                    )
                    cpu_affine_chunks += 1
                    stress_workspace_peak_bytes = max(
                        stress_workspace_peak_bytes, int(stresses.nbytes)
                    )
                affine_stress_wall_s += float(time.perf_counter() - stress_started)
                contraction_started = time.perf_counter()
                if stresses_gpu is not None:
                    new_contracted, summed = _contract_and_sum_gpu_batch(
                        new_gpu, stresses_gpu
                    )
                    old_contracted = _contract_gpu_batch(old_gpu, stresses_gpu)
                    raw_Kq[q_indices, :old_rank, old_rank:] += (
                        old_contracted / float(nvox)
                    )
                    raw_Kq[q_indices, old_rank:, old_rank:] += (
                        new_contracted / float(nvox)
                    )
                    raw_Bq[q_indices, old_rank:] += summed / float(nvox)
                else:
                    uploaded_stresses = _gpu_batch_flat_compute(
                        stresses, compute_dtype=compute_dtype
                    )
                    raw_Kq[q_indices, :old_rank, old_rank:] += (
                        _contract_preloaded_batch(
                            old_values,
                            stresses,
                            old_gpu,
                            uploaded_stresses,
                            compute_dtype=compute_dtype,
                        ) / float(nvox)
                    )
                    raw_Kq[q_indices, old_rank:, old_rank:] += (
                        _contract_preloaded_batch(
                            new_values,
                            stresses,
                            new_gpu,
                            uploaded_stresses,
                            compute_dtype=compute_dtype,
                        ) / float(nvox)
                    )
                    raw_Bq[q_indices, old_rank:] += (
                        np.sum(stresses, axis=3, dtype=np.float64) / float(nvox)
                    )
                    if uploaded_stresses is not None:
                        del uploaded_stresses
                contraction_wall_s += float(time.perf_counter() - contraction_started)
                if stresses_gpu is not None:
                    del stresses_gpu
                else:
                    del stresses
            if gram_future is not None:
                wait_started = time.perf_counter()
                gram_cross, gram_diagonal, gram_elapsed = gram_future.result()
                wait_elapsed = float(time.perf_counter() - wait_started)
                assert G is not None
                G[:old_rank, old_rank:] += gram_cross / float(nvox)
                G[old_rank:, old_rank:] += gram_diagonal / float(nvox)
                gram_product_wall_s += float(gram_elapsed)
                gram_overlap_wait_wall_s += wait_elapsed
                contraction_wall_s += wait_elapsed
            if separate_old_gram_gpu:
                del old_gram_gpu
            if separate_new_gram_gpu:
                del new_gram_gpu
            if old_gpu is not None:
                del old_gpu
            if new_gpu is not None:
                del new_gpu
            del old_values, new_values

    if gram_executor is not None:
        gram_executor.shutdown(wait=True)

    if G is not None:
        G[old_rank:, :old_rank] = G[:old_rank, old_rank:].T
        G = 0.5 * (G + G.T)
    for q in range(q_count):
        raw_Kq[q, old_rank:, :old_rank] = raw_Kq[q, :old_rank, old_rank:].T
        raw_Kq[q] = 0.5 * (raw_Kq[q] + raw_Kq[q].T)

    if bool(preserve_raw_coordinates):
        Kq = raw_Kq.copy()
        Bq = raw_Bq.copy()
        invR = np.eye(new_rank, dtype=np.float64)
        gram_meta = {
            "gram_lambda_min": 0.0,
            "gram_lambda_max": 0.0,
            "gram_condition": 1.0,
            "gram_relative_min": 1.0,
            "effective_rank": int(new_rank),
            "discarded_rank": 0,
            "gram_transform_mode": "raw_coordinates_no_gram",
        }
    else:
        assert G is not None
        Kq, Bq, invR, gram_meta = _transform_raw_operators_with_rank_policy(
            raw_Kq,
            raw_Bq,
            G,
            allow_rank_reveal=bool(gram_rank_reveal),
            rank_rtol=float(gram_rank_rtol),
        )
    metadata = {
        "assembly_wall_s": float(time.perf_counter() - t0),
        "assembly_mode": "incremental",
        "contraction_mode": "phase_supported_blocks",
        "contraction_dtype": str(first_dtype),
        "gram_product_dtype": (
            "not_computed" if bool(preserve_raw_coordinates) else str(gram_dtype)
        ),
        "gram_product_backend": (
            "none" if bool(preserve_raw_coordinates) else selected_gram_backend
        ),
        "gram_overlap_requested": bool(overlap_cpu_gram_gpu)
        and not bool(preserve_raw_coordinates),
        "gram_overlap_enabled": bool(gram_overlap_used_chunks),
        "gram_overlap_used_chunks": int(gram_overlap_used_chunks),
        "gram_product_wall_s": float(gram_product_wall_s),
        "gram_overlap_wait_wall_s": float(gram_overlap_wait_wall_s),
        "gram_overlap_hidden_wall_s": float(
            max(0.0, gram_product_wall_s - gram_overlap_wait_wall_s)
            if gram_overlap_used_chunks
            else 0.0
        ),
        "contraction_compute_dtype": str(compute_dtype),
        "reduced_accumulation_dtype": "float64",
        "affine_stress_wall_s": affine_stress_wall_s,
        "contraction_wall_s": contraction_wall_s,
        "stress_workspace_peak_bytes": int(stress_workspace_peak_bytes),
        "full_volume_equivalent_passes": float(full_volume_equivalent_passes),
        "chunk_voxels": int(chunk_voxels),
        "q_block_size": int(q_block_size),
        "batched_contraction": True,
        "basis_gpu_uploads": int(basis_gpu_uploads),
        "avoided_duplicate_basis_gpu_uploads": int(basis_gpu_uploads),
        "affine_stress_backend": (
            "gpu_resident" if gpu_affine_chunks and not cpu_affine_chunks else "cpu"
        ),
        "gpu_affine_chunks": int(gpu_affine_chunks),
        "cpu_affine_chunks": int(cpu_affine_chunks),
        "gpu_affine_fallback": gpu_affine_fallback,
        "raw_Kq": raw_Kq,
        "raw_Bq": raw_Bq,
        "invR": invR,
        **gram_meta,
    }
    if G is not None:
        metadata["G"] = G
    return Kq, Bq, Dq, metadata


def _solve_spd_reduced(K: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve K a = -B in float64 without modifying the Ritz operator."""
    stiffness = 0.5 * (np.asarray(K, dtype=np.float64) + np.asarray(K, dtype=np.float64).T)
    rhs = -np.asarray(B, dtype=np.float64)
    try:
        return np.linalg.solve(stiffness, rhs)
    except np.linalg.LinAlgError as exc:
        eigvals = scipy_linalg.eigvalsh(stiffness, check_finite=False)
        raise np.linalg.LinAlgError(
            "Reduced Ritz solve failed; refusing silent regularization. "
            f"lambda_min={float(eigvals[0]):.3e}, lambda_max={float(eigvals[-1]):.3e}."
        ) from exc


def _rom_ceff(
    coeffs: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    coeffs64 = np.asarray(coeffs, dtype=np.float64)
    K = np.tensordot(coeffs64, np.asarray(Kq, dtype=np.float64), axes=(0, 0))
    B = np.tensordot(coeffs64, np.asarray(Bq, dtype=np.float64), axes=(0, 0))
    D = np.tensordot(coeffs64, np.asarray(Dq, dtype=np.float64), axes=(0, 0))
    amplitudes = _solve_spd_reduced(K, B)
    C = D + B.T @ amplitudes
    C = 0.5 * (C + C.T)
    return C, amplitudes, float(time.perf_counter() - t0)


def _rom_ceff_batch(
    coeffs_batch: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate independent Ritz systems in float64 with one definition everywhere."""
    started = time.perf_counter()
    coeffs = np.asarray(coeffs_batch, dtype=np.float64)
    if coeffs.ndim != 2 or coeffs.shape[1] != Kq.shape[0]:
        raise ValueError("coeffs_batch must have shape (n_candidates, n_coefficients).")
    K = np.einsum("nq,qij->nij", coeffs, np.asarray(Kq, dtype=np.float64), optimize=True)
    B = np.einsum("nq,qij->nij", coeffs, np.asarray(Bq, dtype=np.float64), optimize=True)
    D = np.einsum("nq,qij->nij", coeffs, np.asarray(Dq, dtype=np.float64), optimize=True)
    amplitudes = np.empty_like(B, dtype=np.float64)
    for i in range(len(K)):
        amplitudes[i] = _solve_spd_reduced(K[i], B[i])
    C = D + np.einsum("nri,nrj->nij", B, amplitudes, optimize=True)
    C = 0.5 * (C + np.swapaxes(C, -1, -2))
    return C, amplitudes, float(time.perf_counter() - started)

class GpuAffineBatchEvaluator:
    """Keep affine operators resident on CUDA for repeated float64 ROM batches."""

    def __init__(self, Kq: np.ndarray, Bq: np.ndarray, Dq: np.ndarray) -> None:
        import cupy as cp
        started = time.perf_counter()
        self.cp = cp
        self.Kq = cp.asarray(Kq, dtype=cp.float64)
        self.Bq = cp.asarray(Bq, dtype=cp.float64)
        self.Dq = cp.asarray(Dq, dtype=cp.float64)
        cp.cuda.Stream.null.synchronize()
        self.operator_transfer_wall_s = float(time.perf_counter() - started)

    def evaluate(
        self,
        coeffs_batch: np.ndarray,
        *,
        return_amplitudes: bool = False,
    ) -> tuple[np.ndarray, float] | tuple[np.ndarray, np.ndarray, float]:
        cp = self.cp
        started = time.perf_counter()
        coeffs = cp.asarray(coeffs_batch, dtype=cp.float64)
        if coeffs.ndim != 2 or coeffs.shape[1] != self.Kq.shape[0]:
            raise ValueError("coeffs_batch must have shape (n_candidates, n_coefficients).")
        K = cp.einsum("nq,qij->nij", coeffs, self.Kq, optimize=True)
        K = 0.5 * (K + cp.swapaxes(K, -1, -2))
        B = cp.einsum("nq,qij->nij", coeffs, self.Bq, optimize=True)
        D = cp.einsum("nq,qij->nij", coeffs, self.Dq, optimize=True)
        try:
            amplitudes = cp.linalg.solve(K, -B)
        except Exception as exc:
            raise np.linalg.LinAlgError(
                "GPU reduced Ritz solve failed; no Tikhonov regularization is applied."
            ) from exc
        C = D + cp.einsum("nri,nrj->nij", B, amplitudes, optimize=True)
        C = 0.5 * (C + cp.swapaxes(C, -1, -2))
        result = cp.asnumpy(C)
        amp_result = cp.asnumpy(amplitudes) if return_amplitudes else None
        cp.cuda.Stream.null.synchronize()
        wall = float(time.perf_counter() - started)
        if return_amplitudes:
            return result, amp_result, wall
        return result, wall

class IncrementalAffineBatchEvaluator:
    """Maintain exact batched Ritz solves as the basis grows by small blocks."""

    def __init__(
        self,
        coefficients: np.ndarray,
        Kq: np.ndarray,
        Bq: np.ndarray,
        Dq: np.ndarray,
    ) -> None:
        started = time.perf_counter()
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        if self.coefficients.ndim != 2 or self.coefficients.shape[1] != Kq.shape[0]:
            raise ValueError("coefficients must have shape (n_candidates, n_coefficients).")
        self.rank = int(Kq.shape[1])
        if self.rank < 1 or Kq.shape[2] != self.rank or Bq.shape[1] != self.rank:
            raise ValueError("Kq and Bq must define a non-empty square reduced system.")
        self._assemble_and_solve(Kq, Bq, Dq)
        self.initialization_wall_s = float(time.perf_counter() - started)
        self.last_update_wall_s = self.initialization_wall_s

    def _assemble_and_solve(
        self,
        Kq: np.ndarray,
        Bq: np.ndarray,
        Dq: np.ndarray,
    ) -> None:
        self.B = np.einsum("nq,qij->nij", self.coefficients, Bq, optimize=True)
        self.D = np.einsum("nq,qij->nij", self.coefficients, Dq, optimize=True)
        self.stiffness = np.einsum(
            "nq,qij->nij", self.coefficients, Kq, optimize=True
        )
        self.stiffness = 0.5 * (
            self.stiffness + np.swapaxes(self.stiffness, -1, -2)
        )
        try:
            self.amplitudes = np.linalg.solve(self.stiffness, -self.B)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "A reduced Ritz solve failed; no regularization is applied."
            ) from error

    def extend(
        self,
        Kq: np.ndarray,
        Bq: np.ndarray,
        Dq: np.ndarray,
    ) -> dict[str, float | int | str]:
        """Replace the affine blocks and solve the enlarged dense systems."""
        started = time.perf_counter()
        new_rank = int(Kq.shape[1])
        if Kq.shape[2] != new_rank or Bq.shape[1] != new_rank:
            raise ValueError("Kq and Bq must define a square reduced system.")
        if new_rank < self.rank:
            raise ValueError("Incremental evaluator cannot shrink the reduced basis.")
        if new_rank == self.rank:
            self._assemble_and_solve(Kq, Bq, Dq)
            return {
                "update_mode": "dense_resolve",
                "old_rank": self.rank,
                "new_rank": new_rank,
                "update_wall_s": float(time.perf_counter() - started),
            }

        old_rank = self.rank
        block_rank = new_rank - old_rank
        self.rank = new_rank
        self._assemble_and_solve(Kq, Bq, Dq)
        self.last_update_wall_s = float(time.perf_counter() - started)
        return {
            "update_mode": "dense_resolve",
            "old_rank": old_rank,
            "appended_rank": block_rank,
            "new_rank": new_rank,
            "update_wall_s": self.last_update_wall_s,
        }

    def evaluate(self, indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return current full Ritz stiffnesses and amplitudes for selected rows."""
        if indices is None:
            amplitudes = self.amplitudes
            B = self.B
            D = self.D
        else:
            selected = np.asarray(indices, dtype=np.intp)
            amplitudes = self.amplitudes[selected]
            B = self.B[selected]
            D = self.D[selected]
        effective = D + np.einsum("nri,nrj->nij", B, amplitudes, optimize=True)
        effective = 0.5 * (effective + np.swapaxes(effective, -1, -2))
        return effective, amplitudes


def _full_ceff_from_row(row: pd.Series) -> np.ndarray:
    matrix = np.zeros((6, 6), dtype=float)
    for ii in range(6):
        for jj in range(6):
            column = f"Ceff_{ii + 1}{jj + 1}"
            if column not in row:
                raise KeyError(f"Falta columna {column} en resultados full-order.")
            matrix[ii, jj] = float(row[column])
    return matrix


def _evaluate_rom(
    *,
    results_df: pd.DataFrame,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> pd.DataFrame:
    # Build coefficients and FOM tensors in batch
    n = len(results_df)
    coeffs_all = np.empty((n, len(COEFF_NAMES)), dtype=np.float64)
    C_fom_all = np.empty((n, 6, 6), dtype=np.float64)
    material_ids = np.empty(n, dtype=int)
    material_labels: list[str] = []
    for idx, (_, row) in enumerate(results_df.iterrows()):
        row_dict = row.to_dict()
        coeffs_all[idx] = _material_coefficients(row_dict)
        C_fom_all[idx] = 0.5 * (_full_ceff_from_row(row) + _full_ceff_from_row(row).T)
        material_ids[idx] = int(row["material_id"])
        material_labels.append(str(row.get("material_label", "")))

    # Use the batched path normally, then isolate failures per material if needed.
    numerical_failures = np.zeros(n, dtype=bool)
    failure_messages = [""] * n
    try:
        C_rom_all, amplitudes_all, batch_online_s = _rom_ceff_batch(
            coeffs_all, Kq, Bq, Dq
        )
    except np.linalg.LinAlgError:
        C_rom_all = np.full((n, 6, 6), np.nan, dtype=np.float64)
        amplitudes_all = np.full((n, Kq.shape[1], 6), np.nan, dtype=np.float64)
        batch_online_s = 0.0
        for idx in range(n):
            try:
                C_rom_all[idx], amplitudes_all[idx], wall_s = _rom_ceff(
                    coeffs_all[idx], Kq, Bq, Dq
                )
                batch_online_s += float(wall_s)
            except np.linalg.LinAlgError as exc:
                numerical_failures[idx] = True
                failure_messages[idx] = str(exc)
    per_material_s = float(batch_online_s) / max(n, 1)

    # Vectorized diagnostics for successful materials; failed rows remain NaN.
    valid = ~numerical_failures
    diff_all = C_rom_all - C_fom_all
    fom_norms = np.linalg.norm(C_fom_all.reshape(n, -1), axis=1)
    fom_norms = np.maximum(fom_norms, np.finfo(float).eps)
    diff_norms = np.full(n, np.nan, dtype=np.float64)
    rel_errors = np.full(n, np.nan, dtype=np.float64)
    eig_rom_all = np.full((n, 6), np.nan, dtype=np.float64)
    eig_diff_all = np.full((n, 6), np.nan, dtype=np.float64)
    schur_eta = np.full(n, np.nan, dtype=np.float64)
    if np.any(valid):
        diff_norms[valid] = np.linalg.norm(diff_all[valid].reshape(valid.sum(), -1), axis=1)
        rel_errors[valid] = diff_norms[valid] / fom_norms[valid]
        eig_rom_all[valid] = np.linalg.eigvalsh(C_rom_all[valid])
    diff_sym = 0.5 * (diff_all + np.swapaxes(diff_all, -1, -2))
    if np.any(valid):
        eig_diff_all[valid] = np.linalg.eigvalsh(diff_sym[valid])
    eig_fom_all = np.linalg.eigvalsh(C_fom_all)
    fom_spectral_norms = np.max(np.abs(eig_fom_all), axis=1)
    fom_spectral_norms = np.maximum(fom_spectral_norms, np.finfo(float).eps)
    schur_eta[valid] = eig_diff_all[valid, 0] / fom_spectral_norms[valid]
    K_all = np.einsum(
        "nq,qij->nij", coeffs_all, np.asarray(Kq, dtype=np.float64), optimize=True
    )
    K_all = 0.5 * (K_all + np.swapaxes(K_all, -1, -2))
    eig_K_all = np.linalg.eigvalsh(K_all)
    spectral_spd_margin = eig_K_all[:, 0] / np.maximum(
        eig_K_all[:, -1], np.finfo(float).tiny
    )

    rows: list[dict[str, Any]] = []
    for idx in range(n):
        C_rom = C_rom_all[idx]
        props = (
            engineering_constants_from_Cmandel(C_rom)
            if valid[idx]
            else {key: np.nan for key in ENGINEERING_COLUMNS}
        )
        out = {
            "material_id": int(material_ids[idx]),
            "material_label": material_labels[idx],
            "rom_numerical_failure": bool(numerical_failures[idx]),
            "rom_failure_message": failure_messages[idx],
            "relative_frobenius_error": float(rel_errors[idx]),
            "absolute_frobenius_error": float(diff_norms[idx]),
            "rom_online_s": per_material_s,
            "rom_min_eig": float(eig_rom_all[idx, 0]),
            "rom_max_eig": float(eig_rom_all[idx, -1]),
            "reduced_K_min_eig": float(eig_K_all[idx, 0]),
            "reduced_K_max_eig": float(eig_K_all[idx, -1]),
            "reduced_K_spectral_spd_margin": float(spectral_spd_margin[idx]),
            "min_eig_Crom_minus_Cfom": float(eig_diff_all[idx, 0]),
            "max_eig_Crom_minus_Cfom": float(eig_diff_all[idx, -1]),
            "schur_eta": float(schur_eta[idx]),
            "amplitude_frobenius_norm": float(np.linalg.norm(amplitudes_all[idx])),
        }
        for q, name in enumerate(COEFF_NAMES):
            out[f"xi_{name}"] = float(coeffs_all[idx, q])
        for key in ENGINEERING_COLUMNS:
            out[f"rom_{key}"] = float(props.get(key, np.nan))
        for ii in range(6):
            for jj in range(6):
                out[f"Crom_{ii + 1}{jj + 1}"] = float(C_rom[ii, jj])
        rows.append(out)
    return pd.DataFrame(rows)


def _make_out_dir(run_dir: Path, snapshot_ids: list[int], out_name: str | None) -> Path:
    if out_name:
        out_dir = run_dir / out_name
    else:
        tag = "_".join(f"m{mid}" for mid in snapshot_ids)
        out_dir = run_dir / f"rom_tangential_{tag}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compila Kq/Bq/Dq reducidos desde campos tangenciales FFTHomPy "
            "guardados sobre una geometria fija."
        )
    )
    parser.add_argument("run_dir", type=Path, nargs="?", default=RUN_DEFAULT)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--snapshot-material-ids", type=int, nargs="*", default=None)
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--orthonormal-tol", type=float, default=1e-10)
    parser.add_argument(
        "--save-basis-fields",
        action="store_true",
        help="Guarda la base voxelizada; util para indicadores residuales.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No existe run_dir: {run_dir}")
    results_csv = (
        args.results_csv
        if args.results_csv is not None
        else run_dir / "fixed_geometry_ffthompy_results.csv"
    )
    if not results_csv.is_file():
        raise FileNotFoundError(f"No existe CSV full-order: {results_csv}")

    geometry_dir = run_dir / "_fixed_geometry"
    phase = np.load(geometry_dir / "phase.npy").astype(np.uint8)
    ori = np.load(geometry_dir / "ori.npy").astype(np.float64)
    geometry_manifest = _load_json(geometry_dir / "geometry_manifest.json")
    results_df = pd.read_csv(results_csv)

    snapshot_ids = (
        sorted(set(int(value) for value in args.snapshot_material_ids))
        if args.snapshot_material_ids is not None
        else _discover_snapshot_ids(run_dir, results_df)
    )
    if not snapshot_ids:
        raise RuntimeError("No se encontraron solution_fields.npz para snapshots.")

    print(
        "[ROM] cargando snapshots | "
        f"run={run_dir.name} | snapshot_ids={snapshot_ids}",
        flush=True,
    )
    fields = _load_snapshot_fields(run_dir, snapshot_ids)
    basis = _orthonormalize(fields, tol=float(args.orthonormal_tol))

    print(
        "[ROM] ensamblando operadores reducidos | "
        f"r={len(basis)} | grid={phase.shape}",
        flush=True,
    )
    Kq, Bq, Dq, assembly_meta = _assemble_reduced_operators(
        phase=phase,
        ori=ori,
        basis=basis,
    )
    validation_df = _evaluate_rom(
        results_df=results_df,
        Kq=Kq,
        Bq=Bq,
        Dq=Dq,
    )

    out_dir = _make_out_dir(run_dir, snapshot_ids, args.out_name)
    np.savez_compressed(
        out_dir / "reduced_operators.npz",
        Kq=Kq,
        Bq=Bq,
        Dq=Dq,
        coefficient_names=np.array(COEFF_NAMES),
        snapshot_material_ids=np.array(snapshot_ids, dtype=np.int64),
    )
    if args.save_basis_fields:
        np.savez_compressed(
            out_dir / "basis_fields.npz",
            **{f"basis_{idx:04d}": basis_field.astype(np.float32) for idx, basis_field in enumerate(basis)},
        )
    validation_df.to_csv(out_dir / "rom_validation_results.csv", index=False)
    validation_df.to_excel(out_dir / "rom_validation_results.xlsx", index=False)

    errors = validation_df["relative_frobenius_error"].to_numpy(dtype=float)
    summary = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "snapshot_material_ids": snapshot_ids,
        "basis_rank": int(len(basis)),
        "coefficient_names": COEFF_NAMES,
        "geometry_hash": (
            str(geometry_manifest.get("phase_sha256", ""))[:12]
            + "_"
            + str(geometry_manifest.get("ori_sha256", ""))[:12]
        ),
        "assembly": assembly_meta,
        "material_count": int(len(validation_df)),
        "mean_relative_error": float(np.mean(errors)),
        "median_relative_error": float(np.median(errors)),
        "p95_relative_error": float(np.quantile(errors, 0.95)),
        "max_relative_error": float(np.max(errors)),
        "worst_material_id": int(
            validation_df.iloc[int(np.argmax(errors))]["material_id"]
        ),
        "median_online_s": float(np.median(validation_df["rom_online_s"])),
        "min_rom_eig": float(validation_df["rom_min_eig"].min()),
        "min_eig_Crom_minus_Cfom": float(
            validation_df["min_eig_Crom_minus_Cfom"].min()
        ),
        "local_fiber_basis_reconstruction_error_center": float(
            _check_local_basis(results_df.iloc[0].to_dict())
        ),
    }
    _write_json(out_dir / "rom_manifest.json", summary)

    text = f"""# Tangential Reduced Operator

- Source run: `{run_dir}`
- Snapshot material IDs: `{snapshot_ids}`
- Reduced rank: `{len(basis)}`
- Affine coefficients: `{len(COEFF_NAMES)}`
- Mean relative tensor error: `{summary['mean_relative_error']:.6e}`
- P95 relative tensor error: `{summary['p95_relative_error']:.6e}`
- Max relative tensor error: `{summary['max_relative_error']:.6e}`
- Worst material ID: `{summary['worst_material_id']}`
- Median online time: `{summary['median_online_s']:.6e} s`
- Min ROM eigenvalue: `{summary['min_rom_eig']:.6e}`
- Min eig(Crom - Cfom): `{summary['min_eig_Crom_minus_Cfom']:.6e}`
"""
    (out_dir / "rom_summary.md").write_text(text, encoding="utf-8")

    print(
        "[ROM] listo | "
        f"out={out_dir} | r={len(basis)} | "
        f"mean={summary['mean_relative_error']:.3e} | "
        f"max={summary['max_relative_error']:.3e} | "
        f"worst={summary['worst_material_id']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
