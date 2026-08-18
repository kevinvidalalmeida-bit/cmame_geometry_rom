#!/usr/bin/env python3
"""Compile a tangential reduced operator from fixed-geometry FFTHomPy fields."""

from __future__ import annotations

import argparse
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
    TI_stiffness_voigt,
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
    lam = np.zeros((6, 6), dtype=float)
    mu = np.zeros((6, 6), dtype=float)
    lam[:3, :3] = 1.0
    mu[0, 0] = mu[1, 1] = mu[2, 2] = 2.0
    mu[3, 3] = mu[4, 4] = mu[5, 5] = 2.0
    return lam, mu


def _fiber_local_bases_axis0() -> list[np.ndarray]:
    """Five TI Mandel bases with local axis 0 as the fiber direction."""
    c_tt = np.zeros((6, 6), dtype=float)
    c_tt_cross = np.zeros((6, 6), dtype=float)
    c_lt = np.zeros((6, 6), dtype=float)
    c_ll = np.zeros((6, 6), dtype=float)
    g_lt = np.zeros((6, 6), dtype=float)

    c_tt[1, 1] = c_tt[2, 2] = 1.0
    c_tt[3, 3] = 1.0

    c_tt_cross[1, 2] = c_tt_cross[2, 1] = 1.0
    c_tt_cross[3, 3] = -1.0

    c_lt[0, 1] = c_lt[1, 0] = 1.0
    c_lt[0, 2] = c_lt[2, 0] = 1.0

    c_ll[0, 0] = 1.0

    g_lt[4, 4] = 2.0
    g_lt[5, 5] = 2.0

    return [c_tt, c_tt_cross, c_lt, c_ll, g_lt]


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
    rounded = np.round(ori_flat[fiber_idx], decimals=12)
    _, group_ids = np.unique(rounded, axis=0, return_inverse=True)
    fiber_order = np.argsort(group_ids, kind="stable")
    return np.concatenate((matrix_idx, fiber_idx[fiber_order])).astype(np.int64, copy=False)


def _contiguous_selector(indices: np.ndarray) -> slice | np.ndarray:
    if not len(indices):
        return slice(0, 0)
    start = int(indices[0])
    if int(indices[-1]) - start + 1 == len(indices):
        return slice(start, start + len(indices))
    return indices


def _affine_tensor_batch_factory(
    phase: np.ndarray,
    ori: np.ndarray,
    *,
    matrix_bases: tuple[np.ndarray, np.ndarray],
    fiber_local_bases: list[np.ndarray],
    coefficient_names: list[str],
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
    if len(fiber_idx):
        rounded = np.round(ori_flat[fiber_idx], decimals=12)
        unique_oris, inverse = np.unique(rounded, axis=0, return_inverse=True)
        group_bases: list[np.ndarray] = []
        for axis in unique_oris:
            rotation = rotation_matrix_from_vector(axis)
            group_bases.append(np.stack(
                [rotate_C_mandel(basis, rotation) for basis in fiber_local_bases],
                axis=0,
            ))
        fiber_group_ids = np.asarray(inverse, dtype=np.int32)
        fiber_bases_by_group = np.stack(group_bases, axis=0)

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
) -> Any:
    """Build the affine stiffness action used by primal Ritz compilation."""
    return _affine_tensor_batch_factory(
        phase,
        ori,
        matrix_bases=_isotropic_bases(),
        fiber_local_bases=_fiber_local_bases_axis0(),
        coefficient_names=COEFF_NAMES,
    )


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


def _contract_component_major(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return A B^T over component+voxel coordinates, accumulated in float64."""
    if A.ndim != 3 or B.ndim != 3 or A.shape[1] != 6 or B.shape[1] != 6:
        raise ValueError("contraction expects component-major (r, 6, n) arrays")
    if A.shape[2] != B.shape[2]:
        raise ValueError("contraction arrays must share the spatial chunk")
    a2 = np.ascontiguousarray(A).reshape(A.shape[0], -1)
    b2 = np.ascontiguousarray(B).reshape(B.shape[0], -1)
    try:
        import cupy as cp
        a_gpu = cp.asarray(a2, dtype=cp.float64)
        if A is B:
            b_gpu = a_gpu
        else:
            b_gpu = cp.asarray(b2, dtype=cp.float64)
        result = cp.asnumpy(a_gpu @ b_gpu.T)
        del a_gpu
        if A is not B:
            del b_gpu
        return np.asarray(result, dtype=np.float64)
    except Exception:
        return np.asarray(a2, dtype=np.float64) @ np.asarray(b2, dtype=np.float64).T


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
    if dependent:
        eigenvalues, eigenvectors = scipy_linalg.eigh(gram, check_finite=False)
        keep = eigenvalues > largest * float(rank_rtol)
        if not np.any(keep):
            raise np.linalg.LinAlgError("reduced Gram matrix has zero numerical rank")
        T = (
            eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
        ).T
        transform_mode = "eigh_rank_reveal"
    else:
        L = scipy_linalg.cholesky(gram, lower=True, check_finite=False)
        T = scipy_linalg.solve_triangular(
            L,
            np.eye(L.shape[0], dtype=np.float64),
            lower=True,
            check_finite=False,
        )
        transform_mode = "cholesky_full_rank"
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
    raw_Kq = np.zeros((q_count, r, r), dtype=np.float64)
    raw_Bq = np.zeros((q_count, r, 6), dtype=np.float64)
    G = _contract_component_major(values, values) / float(nvox)
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
    G = 0.5 * (G + G.T)
    for q in range(q_count):
        raw_Kq[q] = 0.5 * (raw_Kq[q] + raw_Kq[q].T)
        Dq[q] = 0.5 * (Dq[q] + Dq[q].T)
    if preserve_raw_coordinates:
        Kq = raw_Kq.copy()
        Bq = raw_Bq.copy()
        invR = np.eye(r, dtype=np.float64)
        gram_meta = {
            "gram_lambda_min": float(np.linalg.eigvalsh(G)[0]),
            "gram_lambda_max": float(np.linalg.eigvalsh(G)[-1]),
            "gram_condition": float(np.linalg.cond(G)),
            "gram_relative_min": float(
                np.linalg.eigvalsh(G)[0] / max(np.linalg.eigvalsh(G)[-1], np.finfo(float).eps)
            ),
            "effective_rank": int(r),
        }
    else:
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
        "gram_product_dtype": "float64",
        "affine_stress_wall_s": affine_stress_wall_s,
        "contraction_wall_s": contraction_wall_s,
        "stress_workspace_peak_bytes": int(stresses.nbytes),
        "full_volume_equivalent_passes": float(q_count),
        "raw_Kq": raw_Kq,
        "raw_Bq": raw_Bq,
        "G": G,
        "invR": invR,
        **gram_meta,
    }
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
            preserve_raw_coordinates=external_affine and isinstance(basis, list),
            gram_rank_reveal=bool(gram_rank_reveal),
            gram_rank_rtol=float(gram_rank_rtol),
            contraction_compute_dtype=compute_dtype,
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
    chunk_voxels = _stream_chunk_voxels(
        ranks=(r,), q_block_size=q_block_size, nvox=nvox,
        storage_itemsize=max(int(first_dtype.itemsize), int(compute_dtype.itemsize)),
        max_chunk_voxels=1_000_000,
    )

    raw_Kq = np.zeros((q_count, r, r), dtype=np.float64)
    raw_Bq = np.zeros((q_count, r, 6), dtype=np.float64)
    G = np.zeros((r, r), dtype=np.float64)
    affine_stress_wall_s = 0.0
    contraction_wall_s = 0.0
    stress_workspace_peak_bytes = 0
    full_volume_equivalent_passes = 0.0
    basis_gpu_uploads = 0
    gpu_affine_chunks = 0
    cpu_affine_chunks = 0
    gpu_affine_fallback = ""

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
            if values_gpu is not None:
                import cupy as cp
                basis_gpu_uploads += 1
                G += cp.asnumpy(values_gpu @ values_gpu.T) / float(nvox)
            else:
                G += _contract_component_major(values, values) / float(nvox)
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
            if values_gpu is not None:
                del values_gpu
            del values

    averaged = getattr(affine, "averaged_stiffness", None)
    Dq = np.asarray(averaged, dtype=np.float64).copy()
    G = 0.5 * (G + G.T)
    for q in range(q_count):
        raw_Kq[q] = 0.5 * (raw_Kq[q] + raw_Kq[q].T)
        Dq[q] = 0.5 * (Dq[q] + Dq[q].T)

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
        "gram_product_dtype": str(compute_dtype),
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
        "G": G,
        "invR": invR,
        **gram_meta,
    }
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Exact streaming extension of raw Ritz blocks, followed by re-whitening."""
    t0 = time.perf_counter()
    compute_dtype = _ritz_compute_dtype(contraction_compute_dtype)
    if "raw_Kq" not in existing or "raw_Bq" not in existing or "G" not in existing:
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
        )
        metadata.update({
            "assembly_mode": "incremental",
            "incremental_mode": "full_reassembly_without_raw_cache",
            "assembly_wall_s": float(time.perf_counter() - t0),
        })
        return Kq, Bq, Dq, metadata

    raw_K_old = np.asarray(existing["raw_Kq"], dtype=np.float64)
    raw_B_old = np.asarray(existing["raw_Bq"], dtype=np.float64)
    G_old = np.asarray(existing["G"], dtype=np.float64)
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
    chunk_voxels = _stream_chunk_voxels(
        ranks=(old_rank, count), q_block_size=q_block_size, nvox=nvox,
        storage_itemsize=max(int(first_dtype.itemsize), int(compute_dtype.itemsize)),
        max_chunk_voxels=1_000_000,
    )

    raw_Kq = np.zeros((q_count, new_rank, new_rank), dtype=np.float64)
    raw_Bq = np.zeros((q_count, new_rank, 6), dtype=np.float64)
    G = np.zeros((new_rank, new_rank), dtype=np.float64)
    raw_Kq[:, :old_rank, :old_rank] = raw_K_old
    raw_Bq[:, :old_rank] = raw_B_old
    G[:old_rank, :old_rank] = G_old
    affine_stress_wall_s = 0.0
    contraction_wall_s = 0.0
    stress_workspace_peak_bytes = 0
    full_volume_equivalent_passes = 0.0
    basis_gpu_uploads = 0
    gpu_affine_chunks = 0
    cpu_affine_chunks = 0
    gpu_affine_fallback = ""

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
            if old_gpu is not None and new_gpu is not None:
                import cupy as cp
                basis_gpu_uploads += 2
                G[:old_rank, old_rank:] += cp.asnumpy(
                    old_gpu @ new_gpu.T
                ) / float(nvox)
                G[old_rank:, old_rank:] += cp.asnumpy(
                    new_gpu @ new_gpu.T
                ) / float(nvox)
            else:
                G[:old_rank, old_rank:] += (
                    _contract_component_major(old_values, new_values) / float(nvox)
                )
                G[old_rank:, old_rank:] += (
                    _contract_component_major(new_values, new_values) / float(nvox)
                )
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
            if old_gpu is not None:
                del old_gpu
            if new_gpu is not None:
                del new_gpu
            del old_values, new_values

    G[old_rank:, :old_rank] = G[:old_rank, old_rank:].T
    G = 0.5 * (G + G.T)
    for q in range(q_count):
        raw_Kq[q, old_rank:, :old_rank] = raw_Kq[q, :old_rank, old_rank:].T
        raw_Kq[q] = 0.5 * (raw_Kq[q] + raw_Kq[q].T)

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
        "gram_product_dtype": str(compute_dtype),
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
        "G": G,
        "invR": invR,
        **gram_meta,
    }
    return Kq, Bq, Dq, metadata


def _solve_spd_reduced(K: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve K a = -B in float64 without modifying the Ritz operator."""
    stiffness = 0.5 * (np.asarray(K, dtype=np.float64) + np.asarray(K, dtype=np.float64).T)
    rhs = -np.asarray(B, dtype=np.float64)
    try:
        factor = scipy_linalg.cho_factor(
            stiffness, lower=True, overwrite_a=False, check_finite=False
        )
        return scipy_linalg.cho_solve(factor, rhs, check_finite=False)
    except scipy_linalg.LinAlgError as exc:
        eigvals = scipy_linalg.eigvalsh(stiffness, check_finite=False)
        raise np.linalg.LinAlgError(
            "Reduced Ritz stiffness lost SPD; refusing silent regularization. "
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
    """Maintain exact batched Ritz solves as the basis grows by small blocks.

    For a fixed candidate pool, extending a reduced basis from ``r`` to
    ``r + b`` only appends a block to every SPD system.  The block Cholesky
    identity updates each candidate in ``O(r**2 b)`` instead of solving every
    dense system again in ``O(r**3)``.  It is deliberately an algebraic cache:
    all returned amplitudes are those of the full current Ritz system.
    """

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
        self.B = np.einsum("nq,qij->nij", self.coefficients, Bq, optimize=True)
        self.D = np.einsum("nq,qij->nij", self.coefficients, Dq, optimize=True)
        stiffness = np.einsum("nq,qij->nij", self.coefficients, Kq, optimize=True)
        stiffness = 0.5 * (stiffness + np.swapaxes(stiffness, -1, -2))
        try:
            self.cholesky = np.linalg.cholesky(stiffness)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "The reduced systems must be SPD for incremental Cholesky updates."
            ) from error
        self.amplitudes = self._solve_from_cholesky(self.cholesky, -self.B)
        self.initialization_wall_s = float(time.perf_counter() - started)
        self.last_update_wall_s = self.initialization_wall_s

    @staticmethod
    def _forward_lower(lower: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Batched forward substitution for lower triangular matrices."""
        L = np.asarray(lower, dtype=np.float64)
        rhs = np.asarray(right, dtype=np.float64)
        if L.ndim != 3 or rhs.ndim != 3 or L.shape[0] != rhs.shape[0] or L.shape[1] != rhs.shape[1]:
            raise ValueError("Incompatible batched triangular solve shapes.")
        if _NUMBA_TRIANGULAR_AVAILABLE:
            return _forward_lower_numba(
                np.ascontiguousarray(L), np.ascontiguousarray(rhs)
            )
        result = np.empty_like(rhs)
        for row in range(L.shape[1]):
            value = rhs[:, row]
            if row:
                value = value - np.einsum(
                    "ni,nij->nj", L[:, row, :row], result[:, :row], optimize=True
                )
            result[:, row] = value / L[:, row, row, None]
        return result

    @staticmethod
    def _backward_lower_transpose(lower: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Batched backward substitution for ``L.T x = right``."""
        L = np.asarray(lower, dtype=np.float64)
        rhs = np.asarray(right, dtype=np.float64)
        if L.ndim != 3 or rhs.ndim != 3 or L.shape[0] != rhs.shape[0] or L.shape[1] != rhs.shape[1]:
            raise ValueError("Incompatible batched triangular solve shapes.")
        if _NUMBA_TRIANGULAR_AVAILABLE:
            return _backward_lower_transpose_numba(
                np.ascontiguousarray(L), np.ascontiguousarray(rhs)
            )
        result = np.empty_like(rhs)
        for row in range(L.shape[1] - 1, -1, -1):
            value = rhs[:, row]
            if row + 1 < L.shape[1]:
                value = value - np.einsum(
                    "ni,nij->nj", L[:, row + 1 :, row], result[:, row + 1 :], optimize=True
                )
            result[:, row] = value / L[:, row, row, None]
        return result

    @classmethod
    def _solve_from_cholesky(cls, lower: np.ndarray, right: np.ndarray) -> np.ndarray:
        return cls._backward_lower_transpose(lower, cls._forward_lower(lower, right))

    def extend(
        self,
        Kq: np.ndarray,
        Bq: np.ndarray,
        Dq: np.ndarray,
    ) -> dict[str, float | int | str]:
        """Append new Ritz coordinates using an exact block Cholesky update."""
        started = time.perf_counter()
        new_rank = int(Kq.shape[1])
        if Kq.shape[2] != new_rank or Bq.shape[1] != new_rank:
            raise ValueError("Kq and Bq must define a square reduced system.")
        if new_rank < self.rank:
            raise ValueError("Incremental evaluator cannot shrink the reduced basis.")
        if new_rank == self.rank:
            self.D = np.einsum("nq,qij->nij", self.coefficients, Dq, optimize=True)
            return {
                "update_mode": "unchanged",
                "old_rank": self.rank,
                "new_rank": new_rank,
                "update_wall_s": float(time.perf_counter() - started),
            }

        old_rank = self.rank
        block_rank = new_rank - old_rank
        cross = np.einsum(
            "nq,qij->nij", self.coefficients, Kq[:, :old_rank, old_rank:new_rank], optimize=True
        )
        diagonal = np.einsum(
            "nq,qij->nij", self.coefficients, Kq[:, old_rank:new_rank, old_rank:new_rank], optimize=True
        )
        new_B = np.einsum(
            "nq,qij->nij", self.coefficients, Bq[:, old_rank:new_rank], optimize=True)
        W = self._forward_lower(self.cholesky, cross)
        schur = diagonal - np.einsum("nri,nrj->nij", W, W, optimize=True)
        schur = 0.5 * (schur + np.swapaxes(schur, -1, -2))
        try:
            new_lower = np.linalg.cholesky(schur)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "A Schur block lost positive definiteness during the Ritz update."
            ) from error
        right = -(new_B + np.einsum("nri,nrj->nij", cross, self.amplitudes, optimize=True))
        new_amplitudes = self._solve_from_cholesky(new_lower, right)
        correction = self._backward_lower_transpose(
            self.cholesky,
            np.einsum("nrb,nbj->nrj", W, new_amplitudes, optimize=True),
        )
        all_amplitudes = np.empty(
            (len(self.coefficients), new_rank, 6), dtype=np.float64
        )
        all_amplitudes[:, :old_rank] = self.amplitudes - correction
        all_amplitudes[:, old_rank:] = new_amplitudes
        all_lower = np.zeros(
            (len(self.coefficients), new_rank, new_rank), dtype=np.float64
        )
        all_lower[:, :old_rank, :old_rank] = self.cholesky
        all_lower[:, old_rank:, :old_rank] = np.swapaxes(W, -1, -2)
        all_lower[:, old_rank:, old_rank:] = new_lower
        self.rank = new_rank
        self.B = np.concatenate((self.B, new_B), axis=1)
        self.D = np.einsum("nq,qij->nij", self.coefficients, Dq, optimize=True)
        self.cholesky = all_lower
        self.amplitudes = all_amplitudes
        self.last_update_wall_s = float(time.perf_counter() - started)
        return {
            "update_mode": "block_cholesky",
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

    # Build all affine systems once and use the batched SPD solver.
    C_rom_all, amplitudes_all, batch_online_s = _rom_ceff_batch(
        coeffs_all, Kq, Bq, Dq
    )
    per_material_s = float(batch_online_s) / max(n, 1)

    # Vectorized error and eigenvalue computation
    diff_all = C_rom_all - C_fom_all
    fom_norms = np.linalg.norm(C_fom_all.reshape(n, -1), axis=1)
    fom_norms = np.maximum(fom_norms, np.finfo(float).eps)
    diff_norms = np.linalg.norm(diff_all.reshape(n, -1), axis=1)
    rel_errors = diff_norms / fom_norms
    eig_rom_all = np.linalg.eigvalsh(C_rom_all)
    diff_sym = 0.5 * (diff_all + np.swapaxes(diff_all, -1, -2))
    eig_diff_all = np.linalg.eigvalsh(diff_sym)

    rows: list[dict[str, Any]] = []
    for idx in range(n):
        C_rom = C_rom_all[idx]
        props = engineering_constants_from_Cmandel(C_rom)
        out = {
            "material_id": int(material_ids[idx]),
            "material_label": material_labels[idx],
            "relative_frobenius_error": float(rel_errors[idx]),
            "absolute_frobenius_error": float(diff_norms[idx]),
            "rom_online_s": per_material_s,
            "rom_min_eig": float(eig_rom_all[idx, 0]),
            "rom_max_eig": float(eig_rom_all[idx, -1]),
            "min_eig_Crom_minus_Cfom": float(eig_diff_all[idx, 0]),
            "max_eig_Crom_minus_Cfom": float(eig_diff_all[idx, -1]),
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
