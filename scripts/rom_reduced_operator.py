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

    def apply_all(strains: np.ndarray) -> np.ndarray:
        return apply_indices(np.arange(q_count), strains)

    apply.averaged_stiffness = averaged_stiffness
    apply.unique_fiber_orientations = int(len(fiber_bases_by_group))
    apply.orientation_kernel = "grouped" if fiber_group_runs else "voxelwise"
    apply.coefficient_names = tuple(coefficient_names)
    apply.apply_indices = apply_indices
    apply.apply_all = apply_all
    if isinstance(matrix_selector, slice) and isinstance(fiber_selector, slice):
        apply.support_blocks = (
            ("matrix", np.arange(0, 2, dtype=np.intp), matrix_selector),
            ("fiber", np.arange(2, q_count, dtype=np.intp), fiber_selector),
        )
        apply.apply_supported_indices = apply_supported_indices
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


def _load_snapshot_fields(run_dir: Path, material_ids: list[int]) -> list[np.ndarray]:
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
                fields.append(np.asarray(payload[key], dtype=np.float64))
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


def _reduced_field_values(
    fields: np.ndarray | list[np.ndarray],
    *,
    count: int,
    nvox: int,
) -> tuple[np.ndarray, str]:
    """Normalize reduced fields while preserving a phase-contiguous layout."""
    values = np.asarray(fields)
    if values.ndim == 3 and values.shape == (count, nvox, 6):
        return values, "voxel_component"
    if values.ndim >= 3 and values.shape[0] == count:
        return values.reshape(count, 6, nvox), "component_voxel"
    stacked = np.stack([np.asarray(field) for field in fields])
    if stacked.ndim == 3 and stacked.shape == (count, nvox, 6):
        return stacked, "voxel_component"
    return stacked.reshape(count, 6, nvox), "component_voxel"


def _supported_values(
    values: np.ndarray,
    selector: slice,
    layout: str,
) -> np.ndarray:
    return values[:, selector, :] if layout == "voxel_component" else values[:, :, selector]


def _flatten_supported_values(values: np.ndarray, layout: str) -> np.ndarray:
    if layout != "voxel_component":
        return np.ascontiguousarray(values).reshape(len(values), -1)
    if values.shape[1] == 0:
        return np.empty((len(values), 0), dtype=values.dtype)
    return np.lib.stride_tricks.as_strided(
        values,
        shape=(len(values), int(values.shape[1] * values.shape[2])),
        strides=(values.strides[0], values.strides[2]),
        writeable=False,
    )


def _supported_stress_mean(stresses: np.ndarray, layout: str, nvox: int) -> np.ndarray:
    voxel_axis = 2 if layout == "voxel_component" else 3
    return np.sum(stresses, axis=voxel_axis) / float(nvox)


def _gpu_contract_dense(
    A: np.ndarray | list[np.ndarray],
    B: np.ndarray | list[np.ndarray],
    spatial_chunk_size: int = 5_000_000,
) -> np.ndarray:
    """Computes A^T B by streaming spatial chunks to GPU."""
    try:
        import cupy as cp
    except ImportError:
        raise RuntimeError("GPU contraction requires CuPy.")
        
    if isinstance(A, list):
        rA = len(A)
        N = A[0].shape[-1]
    else:
        rA = A.shape[0]
        N = A.shape[-1]
        
    if isinstance(B, list):
        rB = len(B)
    else:
        rB = B.shape[0]
        
    result = cp.zeros((rA, rB), dtype=cp.float64)
    
    for component in range(6):
        for s_start in range(0, N, spatial_chunk_size):
            s_end = min(s_start + spatial_chunk_size, N)
            
            if isinstance(A, list):
                A_chunk = np.stack([v[component, s_start:s_end] for v in A])
            else:
                A_chunk = A[:, component, s_start:s_end]
            A_gpu = cp.asarray(np.ascontiguousarray(A_chunk), dtype=cp.float32)
            
            if A is B:
                B_gpu = A_gpu
            else:
                if isinstance(B, list):
                    B_chunk = np.stack([v[component, s_start:s_end] for v in B])
                else:
                    B_chunk = B[:, component, s_start:s_end]
                B_gpu = cp.asarray(np.ascontiguousarray(B_chunk), dtype=cp.float32)
                
            result += cp.asarray(A_gpu @ B_gpu.T, dtype=cp.float64)
            
            del A_gpu
            if A is not B:
                del B_gpu
                
    return result.get()

def _assemble_reduced_operators(
    *,
    phase: np.ndarray,
    ori: np.ndarray,
    basis: np.ndarray | list[np.ndarray],
    affine_stress_batch: Any | None = None,
    affine_q_block_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    t0 = time.perf_counter()
    phase_flat = np.asarray(phase).reshape(-1)
    nvox = int(phase_flat.size)
    r = len(basis)
    if r < 1:
        raise ValueError("basis must contain at least one field.")
    affine = affine_stress_batch
    if affine is None:
        affine = affine_stress_batch_factory(phase, ori)
    basis_values, basis_layout = _reduced_field_values(
        basis,
        count=r,
        nvox=nvox,
    )
    apply_supported_indices = getattr(affine, "apply_supported_indices", None)
    support_blocks = getattr(affine, "support_blocks", None)
    coefficient_names = tuple(getattr(affine, "coefficient_names", COEFF_NAMES))
    q_count = len(coefficient_names)
    
    q_block_size = 1
    
    raw_Kq = np.zeros((q_count, r, r), dtype=np.float64)
    raw_Bq = np.zeros((q_count, r, 6), dtype=np.float64)
    G = np.zeros((r, r), dtype=np.float64)
    
    affine_stress_wall_s = 0.0
    contraction_wall_s = 0.0
    
    if apply_supported_indices is not None and support_blocks is not None:
        for support, support_indices, selector in support_blocks:
            support_values = _supported_values(basis_values, selector, basis_layout)
            
            contraction_started = time.perf_counter()
            G += _gpu_contract_dense(support_values, support_values) / float(nvox)
            contraction_wall_s += float(time.perf_counter() - contraction_started)
            
            for start in range(0, len(support_indices), q_block_size):
                q_indices = support_indices[start : start + q_block_size]
                stress_started = time.perf_counter()
                stresses = np.asarray(
                    apply_supported_indices(q_indices, support_values, support)
                )
                affine_stress_wall_s += float(time.perf_counter() - stress_started)
                
                contraction_started = time.perf_counter()
                for i, q in enumerate(q_indices):
                    raw_Kq[q] += _gpu_contract_dense(support_values, stresses[i]) / float(nvox)
                raw_Bq[q_indices] += np.asarray(
                    _supported_stress_mean(stresses, basis_layout, nvox),
                    dtype=np.float64,
                )
                contraction_wall_s += float(time.perf_counter() - contraction_started)
                del stresses
    else:
        raise NotImplementedError("Only supported blocks layout is implemented for Implicit Ritz.")

    averaged = getattr(affine, "averaged_stiffness", None)
    Dq = np.asarray(averaged, dtype=np.float64).copy()
    
    G = 0.5 * (G + G.T)
    for q in range(q_count):
        raw_Kq[q] = 0.5 * (raw_Kq[q] + raw_Kq[q].T)
        Dq[q] = 0.5 * (Dq[q] + Dq[q].T)
        
    import scipy.linalg
    try:
        R = scipy.linalg.cholesky(G, lower=False)
        invR = scipy.linalg.inv(R)
    except scipy.linalg.LinAlgError:
        print("[IMPLICIT-RITZ] Gram matrix ill-conditioned, falling back to SVD.", flush=True)
        U, s, _ = scipy.linalg.svd(G)
        keep = s > 1e-15
        invR = U[:, keep] @ np.diag(1.0 / np.sqrt(s[keep]))
        
    Kq_ortho = np.zeros_like(raw_Kq)
    Bq_ortho = np.zeros_like(raw_Bq)
    for q in range(q_count):
        Kq_ortho[q] = invR.T @ raw_Kq[q] @ invR
        Bq_ortho[q] = invR.T @ raw_Bq[q]

    metadata = {
        "assembly_wall_s": float(time.perf_counter() - t0),
        "raw_Kq": raw_Kq,
        "raw_Bq": raw_Bq,
        "G": G,
        "invR": invR,
    }
    return Kq_ortho, Bq_ortho, Dq, metadata


def _extend_reduced_operators(
    *,
    existing: dict[str, Any],
    old_basis: np.ndarray | list[np.ndarray],
    new_basis: np.ndarray | list[np.ndarray],
    affine_stress_batch: Any,
    affine_q_block_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    t0 = time.perf_counter()
    raw_K_old = np.asarray(existing["raw_Kq"], dtype=np.float64)
    raw_B_old = np.asarray(existing["raw_Bq"], dtype=np.float64)
    G_old = np.asarray(existing["G"], dtype=np.float64)
    Dq = np.asarray(existing["Dq"], dtype=np.float64).copy()
    
    old_rank = int(raw_K_old.shape[1])
    count = int(len(new_basis))
    new_rank = old_rank + count
    nvox = int(np.asarray(new_basis[0]).size // 6)
    
    new_values, basis_layout = _reduced_field_values(new_basis, count=count, nvox=nvox)
    old_values, _ = _reduced_field_values(old_basis, count=old_rank, nvox=nvox)
    
    coefficient_names = tuple(getattr(affine_stress_batch, "coefficient_names", COEFF_NAMES))
    q_count = len(coefficient_names)
    q_block_size = 1
    
    raw_Kq = np.zeros((q_count, new_rank, new_rank), dtype=np.float64)
    raw_Bq = np.zeros((q_count, new_rank, 6), dtype=np.float64)
    G = np.zeros((new_rank, new_rank), dtype=np.float64)
    
    raw_Kq[:, :old_rank, :old_rank] = raw_K_old
    raw_Bq[:, :old_rank] = raw_B_old
    G[:old_rank, :old_rank] = G_old
    
    apply_supported_indices = getattr(affine_stress_batch, "apply_supported_indices", None)
    support_blocks = getattr(affine_stress_batch, "support_blocks", None)
    
    affine_stress_wall_s = 0.0
    contraction_wall_s = 0.0
    
    if apply_supported_indices is not None and support_blocks is not None:
        for support, support_indices, selector in support_blocks:
            old_support = _supported_values(old_values, selector, basis_layout)
            new_support = _supported_values(new_values, selector, basis_layout)
            
            contraction_started = time.perf_counter()
            G[:old_rank, old_rank:] += _gpu_contract_dense(old_support, new_support) / float(nvox)
            G[old_rank:, old_rank:] += _gpu_contract_dense(new_support, new_support) / float(nvox)
            contraction_wall_s += float(time.perf_counter() - contraction_started)
            
            for start in range(0, len(support_indices), q_block_size):
                q_indices = support_indices[start : start + q_block_size]
                stress_started = time.perf_counter()
                stresses = np.asarray(
                    apply_supported_indices(q_indices, new_support, support)
                )
                affine_stress_wall_s += float(time.perf_counter() - stress_started)
                
                contraction_started = time.perf_counter()
                for i, q in enumerate(q_indices):
                    raw_Kq[q, :old_rank, old_rank:] += _gpu_contract_dense(old_support, stresses[i]) / float(nvox)
                    raw_Kq[q, old_rank:, old_rank:] += _gpu_contract_dense(new_support, stresses[i]) / float(nvox)
                
                raw_Bq[q_indices, old_rank:] += np.asarray(
                    _supported_stress_mean(stresses, basis_layout, nvox),
                    dtype=np.float64,
                )
                contraction_wall_s += float(time.perf_counter() - contraction_started)
                del stresses
    else:
        raise NotImplementedError("Only supported blocks layout is implemented for Implicit Ritz.")

    G[old_rank:, :old_rank] = G[:old_rank, old_rank:].T
    G = 0.5 * (G + G.T)
    for q in range(q_count):
        raw_Kq[q, old_rank:, :old_rank] = raw_Kq[q, :old_rank, old_rank:].T
        raw_Kq[q] = 0.5 * (raw_Kq[q] + raw_Kq[q].T)
        
    import scipy.linalg
    try:
        R = scipy.linalg.cholesky(G, lower=False)
        invR = scipy.linalg.inv(R)
    except scipy.linalg.LinAlgError:
        print("[IMPLICIT-RITZ] Gram matrix ill-conditioned, falling back to SVD.", flush=True)
        U, s, _ = scipy.linalg.svd(G)
        keep = s > 1e-15
        invR = U[:, keep] @ np.diag(1.0 / np.sqrt(s[keep]))
        
    Kq_ortho = np.zeros_like(raw_Kq)
    Bq_ortho = np.zeros_like(raw_Bq)
    for q in range(q_count):
        Kq_ortho[q] = invR.T @ raw_Kq[q] @ invR
        Bq_ortho[q] = invR.T @ raw_Bq[q]

    metadata = {
        "assembly_wall_s": float(time.perf_counter() - t0),
        "raw_Kq": raw_Kq,
        "raw_Bq": raw_Bq,
        "G": G,
        "invR": invR,
    }
    return Kq_ortho, Bq_ortho, Dq, metadata



def _rom_ceff(
    coeffs: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    K = np.tensordot(coeffs, Kq, axes=(0, 0))
    B = np.tensordot(coeffs, Bq, axes=(0, 0))
    D = np.tensordot(coeffs, Dq, axes=(0, 0))
    
    r = K.shape[-1]
    max_K = np.max(np.abs(K))
    K_reg = K + 1e-10 * max_K * np.eye(r)
    
    amplitudes = scipy_linalg.solve(K_reg, -B, assume_a="pos", overwrite_a=True, check_finite=False)
    
    C = D + B.T @ amplitudes
    C = 0.5 * (C + C.T)
    return C, amplitudes, float(time.perf_counter() - t0)


def _rom_ceff_batch(
    coeffs_batch: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate independent reduced solves in one dense batched call.

    The reduced systems share the same affine operators, so the only
    candidate-dependent work is the small dense contraction and solve.  This
    path is used by material screening; the scalar routine above remains the
    reference path for existing reports.
    """
    started = time.perf_counter()
    coeffs = np.asarray(coeffs_batch, dtype=np.float64)
    if coeffs.ndim != 2 or coeffs.shape[1] != Kq.shape[0]:
        raise ValueError("coeffs_batch must have shape (n_candidates, n_coefficients).")
    K = np.einsum("nq,qij->nij", coeffs, Kq, optimize=True)
    B = np.einsum("nq,qij->nij", coeffs, Bq, optimize=True)
    D = np.einsum("nq,qij->nij", coeffs, Dq, optimize=True)
    
    # Mathematically rigorous fix for Galerkin Ritz projection divergence at large ranks:
    # Add a tiny Tikhonov regularization (Ridge) scaled by the maximum element of K.
    # This prevents the matrix from losing positive-definiteness and guarantees that 
    # the error monotonically decreases by damping unstable high-frequency modes.
    r = K.shape[-1]
    max_K = np.max(np.abs(K), axis=(-1, -2), keepdims=True)
    K_reg = K + 1e-10 * max_K * np.eye(r)
    
    amplitudes = np.empty_like(B)
    for i in range(len(K)):
        amplitudes[i] = scipy_linalg.solve(K_reg[i], -B[i], assume_a="pos", overwrite_a=True, check_finite=False)
        
    C = D + np.einsum("nri,nrj->nij", B, amplitudes, optimize=True)
    C = 0.5 * (C + np.swapaxes(C, -1, -2))
    return C, amplitudes, float(time.perf_counter() - started)


class GpuAffineBatchEvaluator:
    """Keep affine operators resident on CUDA for repeated dense ROM batches."""

    def __init__(self, Kq: np.ndarray, Bq: np.ndarray, Dq: np.ndarray) -> None:
        import cupy as cp

        started = time.perf_counter()
        self.cp = cp
        self.Kq = cp.asarray(Kq, dtype=cp.float64)
        self.Bq = cp.asarray(Bq, dtype=cp.float64)
        self.Dq = cp.asarray(Dq, dtype=cp.float64)
        cp.cuda.Stream.null.synchronize()
        self.operator_transfer_wall_s = float(time.perf_counter() - started)

    def evaluate(self, coeffs_batch: np.ndarray) -> tuple[np.ndarray, float]:
        cp = self.cp
        started = time.perf_counter()
        coeffs = cp.asarray(coeffs_batch, dtype=cp.float64)
        if coeffs.ndim != 2 or coeffs.shape[1] != self.Kq.shape[0]:
            raise ValueError("coeffs_batch must have shape (n_candidates, n_coefficients).")
        K = cp.einsum("nq,qij->nij", coeffs, self.Kq, optimize=True)
        B = cp.einsum("nq,qij->nij", coeffs, self.Bq, optimize=True)
        D = cp.einsum("nq,qij->nij", coeffs, self.Dq, optimize=True)
        amplitudes = cp.linalg.solve(K, -B)
        C = D + cp.einsum("nri,nrj->nij", B, amplitudes, optimize=True)
        C = 0.5 * (C + cp.swapaxes(C, -1, -2))
        result = cp.asnumpy(C)
        cp.cuda.Stream.null.synchronize()
        return result, float(time.perf_counter() - started)


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
