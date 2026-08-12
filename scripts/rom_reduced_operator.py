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


def _apply_dense(C: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.einsum("ab,jbn->jan", C, values, optimize=True)


def _accumulate_block(
    *,
    C: np.ndarray,
    values: np.ndarray,
    total_voxels: int,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    q: int,
) -> None:
    if values.shape[2] == 0:
        return
    # Pre-compute C @ values once and reuse for both Kq and Bq
    Cv = np.einsum("ab,jbn->jan", C, values, optimize=True)  # (r, 6, n_vox)
    Kq[q] += np.einsum("ian,jan->ij", values, Cv, optimize=True)
    Bq[q] += np.sum(Cv, axis=2)  # equivalent to einsum("ian,am->im", values, C)
    Dq[q] += C * values.shape[2]
    if total_voxels <= 0:
        raise ValueError("total_voxels debe ser positivo.")


def _assemble_reduced_operators(
    *,
    phase: np.ndarray,
    ori: np.ndarray,
    basis: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    t0 = time.perf_counter()
    phase_flat = np.asarray(phase).reshape(-1)
    ori_flat = np.asarray(ori, dtype=float).reshape(-1, 3)
    nvox = int(phase_flat.size)
    r = len(basis)
    values_all = np.stack([field.reshape(6, nvox) for field in basis], axis=0)

    Kq_sum = np.zeros((len(COEFF_NAMES), r, r), dtype=np.float64)
    Bq_sum = np.zeros((len(COEFF_NAMES), r, 6), dtype=np.float64)
    Dq_sum = np.zeros((len(COEFF_NAMES), 6, 6), dtype=np.float64)

    matrix_mask = phase_flat == 0
    matrix_values = values_all[:, :, matrix_mask]
    for q, C in enumerate(_isotropic_bases()):
        _accumulate_block(
            C=C,
            values=matrix_values,
            total_voxels=nvox,
            Kq=Kq_sum,
            Bq=Bq_sum,
            Dq=Dq_sum,
            q=q,
        )

    fiber_indices = np.flatnonzero(phase_flat != 0)
    unique_orientation_count = 0
    fiber_bases = _fiber_local_bases_axis0()
    if len(fiber_indices):
        rounded = np.round(ori_flat[fiber_indices], decimals=12)
        unique_oris, inverse = np.unique(rounded, axis=0, return_inverse=True)
        unique_orientation_count = int(len(unique_oris))
        for group_id, axis in enumerate(unique_oris):
            indices = fiber_indices[inverse == group_id]
            if len(indices) == 0:
                continue
            values = values_all[:, :, indices]
            R = rotation_matrix_from_vector(axis)
            for local_q, local_basis in enumerate(fiber_bases, start=2):
                C = rotate_C_mandel(local_basis, R)
                _accumulate_block(
                    C=C,
                    values=values,
                    total_voxels=nvox,
                    Kq=Kq_sum,
                    Bq=Bq_sum,
                    Dq=Dq_sum,
                    q=local_q,
                )

    Kq = Kq_sum / float(nvox)
    Bq = Bq_sum / float(nvox)
    Dq = Dq_sum / float(nvox)
    for q in range(Kq.shape[0]):
        Kq[q] = 0.5 * (Kq[q] + Kq[q].T)
        Dq[q] = 0.5 * (Dq[q] + Dq[q].T)

    metadata = {
        "assembly_wall_s": float(time.perf_counter() - t0),
        "nvox": int(nvox),
        "basis_rank": int(r),
        "matrix_voxels": int(np.sum(matrix_mask)),
        "fiber_voxels": int(len(fiber_indices)),
        "unique_fiber_orientations": int(unique_orientation_count),
        "coefficient_names": COEFF_NAMES,
    }
    return Kq, Bq, Dq, metadata


def _extend_reduced_operators(
    *,
    existing: dict[str, np.ndarray],
    old_basis: list[np.ndarray],
    new_basis: list[np.ndarray],
    affine_stress_batch: Any,
    basis_block_size: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Append only the new rows and columns of the affine Ritz operators."""
    started = time.perf_counter()
    K_old = np.asarray(existing["Kq"], dtype=np.float64)
    B_old = np.asarray(existing["Bq"], dtype=np.float64)
    Dq = np.asarray(existing["Dq"], dtype=np.float64).copy()
    old_rank = int(K_old.shape[1])
    count = int(len(new_basis))
    if old_rank != len(old_basis):
        raise ValueError("existing operator rank and old_basis are inconsistent.")
    if count < 1:
        raise ValueError("new_basis must contain at least one field.")

    new_rank = old_rank + count
    nvox = int(np.asarray(new_basis[0]).size // 6)
    new_values = np.stack(
        [np.asarray(field, dtype=np.float64).reshape(6, nvox) for field in new_basis]
    )
    new_flat = new_values.reshape(count, -1)
    old_flat = np.stack(
        [np.asarray(field, dtype=np.float64).reshape(-1) for field in old_basis]
    )
    Kq = np.zeros((len(COEFF_NAMES), new_rank, new_rank), dtype=np.float64)
    Bq = np.zeros((len(COEFF_NAMES), new_rank, 6), dtype=np.float64)
    Kq[:, :old_rank, :old_rank] = K_old
    Bq[:, :old_rank] = B_old
    block_size = max(1, int(basis_block_size))
    stresses = np.stack(
        [
            np.asarray(affine_stress_batch(q, new_values), dtype=np.float64)
            for q in range(len(COEFF_NAMES))
        ]
    )
    if stresses.shape != (len(COEFF_NAMES),) + new_values.shape:
        raise ValueError("affine_stress_batch returned an incompatible shape.")
    stress_flat = stresses.reshape(len(COEFF_NAMES), count, -1)
    cross_all = old_flat @ stress_flat.reshape(len(COEFF_NAMES) * count, -1).T
    cross_all /= float(nvox)
    cross_all = cross_all.reshape(old_rank, len(COEFF_NAMES), count)
    new_new_all = np.einsum(
        "id,qjd->qij", new_flat, stress_flat, optimize=True
    ) / float(nvox)

    for q in range(len(COEFF_NAMES)):
        stress = stresses[q]
        if stress.shape != new_values.shape:
            raise ValueError("affine_stress_batch returned an incompatible shape.")
        Bq[q, old_rank:] = np.mean(stress, axis=2)
        new_new = new_new_all[q]
        Kq[q, old_rank:, old_rank:] = 0.5 * (new_new + new_new.T)
        cross = cross_all[:, q]
        Kq[q, :old_rank, old_rank:] = cross
        Kq[q, old_rank:, :old_rank] = cross.T

    metadata = {
        "assembly_wall_s": float(time.perf_counter() - started),
        "assembly_mode": "incremental",
        "old_rank": old_rank,
        "appended_rank": count,
        "basis_rank": new_rank,
        "nvox": nvox,
        "basis_block_size": block_size,
        "coefficient_names": COEFF_NAMES,
    }
    return Kq, Bq, Dq, metadata


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
    amplitudes = np.linalg.solve(K, -B)
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
    amplitudes = np.linalg.solve(K, -B)
    C = D + np.einsum("nri,nrj->nij", B, amplitudes, optimize=True)
    C = 0.5 * (C + np.swapaxes(C, -1, -2))
    return C, amplitudes, float(time.perf_counter() - started)


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

    # Batched ROM solve — single dense contraction + batched np.linalg.solve
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
