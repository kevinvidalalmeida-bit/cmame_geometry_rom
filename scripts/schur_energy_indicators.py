#!/usr/bin/env python3
"""Rank fixed-geometry material candidates with a QoI/energy residual indicator."""

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
from scipy.stats import qmc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
RUN_DEFAULT = (
    PROJECT_ROOT
    / "results"
    / "fixed_geometry_ffthompy"
    / "fixed_geometry_ar15_vf20_sobol8_center_fields"
)
ROM_DEFAULT = RUN_DEFAULT / "rom_tangential_r48_center_m1_m2_m3_m5_m7_v13_v3_basis"

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import fft_homogenization_solver as sweep
import rom_reduced_operator as reduced
from constitutive_transfer.schur_estimator import (
    TwoKernelEstimator,
    optimize_isotropic_reference,
    optimize_isotropic_reference_batch,
)


MANDEL_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
MANDEL_FACTORS = np.array([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])
PHYSICAL_NAMES = ["Em", "nu_m", "Ef_L", "Ef_T", "G_LT", "nu_LT", "nu_TT"]


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


def _make_out_dir(
    run_dir: Path,
    out_name: str | None,
    rom_dir: Path,
    *,
    overwrite: bool,
) -> Path:
    if out_name:
        out_dir = run_dir / out_name
    else:
        out_dir = run_dir / f"qoi_indicator_{rom_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=bool(overwrite))
    return out_dir


def _round(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def _build_candidates(n_points: int, seed: int) -> pd.DataFrame:
    if n_points < 1:
        raise ValueError("--candidate-points debe ser >= 1.")
    names = list(sweep.MATERIAL_BOUNDS.keys())
    sampler = qmc.Sobol(d=len(names), scramble=True, seed=int(seed))
    power = int(math.ceil(math.log2(int(n_points))))
    unit = sampler.random_base2(m=power)[: int(n_points)]
    lower = [sweep.MATERIAL_BOUNDS[name][0] for name in names]
    upper = [sweep.MATERIAL_BOUNDS[name][1] for name in names]
    scaled = qmc.scale(unit, lower, upper)

    rows: list[dict[str, Any]] = []
    for material_id, values in enumerate(scaled):
        sampled = {name: float(value) for name, value in zip(names, values)}
        material = sweep._material_derived(sampled)
        sweep._validate_material(material)
        rows.append(
            {
                "material_id": int(material_id),
                "material_label": f"candidate_sobol_{material_id:04d}",
                **{key: _round(value) for key, value in material.items()},
            }
        )
    return pd.DataFrame(rows)[sweep.MATERIAL_COLUMNS].copy()


def _load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    if args.candidate_results_csv is not None:
        df = pd.read_csv(args.candidate_results_csv)
        if "material_id" not in df.columns:
            df = df.copy()
            df.insert(0, "material_id", np.arange(len(df), dtype=int))
        return df
    return _build_candidates(
        n_points=int(args.candidate_points),
        seed=int(args.candidate_seed),
    )


def _load_basis_fields(rom_dir: Path) -> np.ndarray:
    path = rom_dir / "basis_fields.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Recompila el ROM con --save-basis-fields."
        )
    with np.load(path) as payload:
        names = sorted(name for name in payload.files if name.startswith("basis_"))
        if not names:
            raise RuntimeError(f"{path} no contiene basis_XXXX.")
        return np.stack([np.asarray(payload[name], dtype=np.float32) for name in names], axis=0)


def _full_ceff_from_row(row: pd.Series) -> np.ndarray | None:
    required = [f"Ceff_{ii + 1}{jj + 1}" for ii in range(6) for jj in range(6)]
    if not all(column in row for column in required):
        return None
    matrix = np.zeros((6, 6), dtype=float)
    for ii in range(6):
        for jj in range(6):
            matrix[ii, jj] = float(row[f"Ceff_{ii + 1}{jj + 1}"])
    return 0.5 * (matrix + matrix.T)


def _sampled_from_row(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    return {name: float(row[name]) for name in PHYSICAL_NAMES}


def _material_row_from_sampled(
    sampled: dict[str, float],
    *,
    material_id: int,
    label: str,
) -> dict[str, Any]:
    material = sweep._material_derived(sampled)
    sweep._validate_material(material)
    return {
        "material_id": int(material_id),
        "material_label": str(label),
        **material,
    }


def _perturb_physical_row(
    row: pd.Series | dict[str, Any],
    *,
    name: str,
    rel_step: float,
) -> tuple[pd.Series, pd.Series, float, float]:
    if name not in sweep.MATERIAL_BOUNDS:
        raise KeyError(f"No existe parametro fisico en MATERIAL_BOUNDS: {name}")
    sampled = _sampled_from_row(row)
    low, high = sweep.MATERIAL_BOUNDS[name]
    value = float(sampled[name])
    step = float(rel_step) * max(abs(value), 1.0)
    max_centered = 0.45 * min(value - low, high - value)
    if max_centered > 0.0:
        step = min(step, max_centered)
    if step <= 0.0:
        raise ValueError(
            f"No se puede perturbar {name}={value:g} dentro del dominio "
            f"[{low:g}, {high:g}]."
        )
    plus = dict(sampled)
    minus = dict(sampled)
    plus[name] = value + step
    minus[name] = value - step
    material_id = int(row.get("material_id", -1)) if hasattr(row, "get") else -1
    label = str(row.get("material_label", "")) if hasattr(row, "get") else ""
    plus_row = _material_row_from_sampled(
        plus,
        material_id=material_id,
        label=f"{label}_{name}_plus",
    )
    minus_row = _material_row_from_sampled(
        minus,
        material_id=material_id,
        label=f"{label}_{name}_minus",
    )
    return pd.Series(plus_row), pd.Series(minus_row), step, high - low


def _energy_matrix_from_record(record: dict[str, Any], *, relative: bool) -> np.ndarray:
    prefix = "energy_matrix_rel" if relative else "energy_matrix"
    matrix = np.zeros((6, 6), dtype=float)
    for ii in range(6):
        for jj in range(6):
            matrix[ii, jj] = float(record[f"{prefix}_{ii + 1}{jj + 1}"])
    return 0.5 * (matrix + matrix.T)


def _mandel_flat_to_tensor_field(field: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    flat = np.asarray(field).reshape(6, -1)
    tensor = np.zeros((3, 3) + shape, dtype=flat.dtype)
    for comp, (ii, jj) in enumerate(MANDEL_PAIRS):
        values = (flat[comp] / MANDEL_FACTORS[comp]).reshape(shape)
        tensor[ii, jj] = values
        tensor[jj, ii] = values
    return tensor


def _tensor_field_to_mandel_flat(tensor: np.ndarray) -> np.ndarray:
    spatial = int(np.prod(tensor.shape[2:]))
    out = np.empty((6, spatial), dtype=tensor.dtype)
    for comp, (ii, jj) in enumerate(MANDEL_PAIRS):
        out[comp] = (MANDEL_FACTORS[comp] * tensor[ii, jj]).reshape(spatial)
    return out


def _mandel_loads_to_tensor_field(
    fields: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    flat = np.asarray(fields)
    if flat.ndim != 3 or flat.shape[1] != 6:
        raise ValueError("fields debe tener forma (n_loads, 6, nvox).")
    tensor = np.zeros((flat.shape[0], 3, 3) + shape, dtype=flat.dtype)
    for comp, (ii, jj) in enumerate(MANDEL_PAIRS):
        values = (flat[:, comp] / MANDEL_FACTORS[comp]).reshape((flat.shape[0],) + shape)
        tensor[:, ii, jj] = values
        tensor[:, jj, ii] = values
    return tensor


def _tensor_loads_to_mandel_flat(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim != 6 or tensor.shape[1:3] != (3, 3):
        raise ValueError("tensor debe tener forma (n_loads, 3, 3, nx, ny, nz).")
    spatial = int(np.prod(tensor.shape[3:]))
    out = np.empty((tensor.shape[0], 6, spatial), dtype=tensor.dtype)
    for comp, (ii, jj) in enumerate(MANDEL_PAIRS):
        out[:, comp] = (MANDEL_FACTORS[comp] * tensor[:, ii, jj]).reshape(
            tensor.shape[0],
            spatial,
        )
    return out


def _frequency_unit_vectors(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    grids = np.meshgrid(
        *[np.fft.fftfreq(size) for size in shape],
        indexing="ij",
    )
    freq = np.asarray(grids, dtype=float)
    norm = np.sqrt(np.sum(freq * freq, axis=0))
    nonzero = norm > 0.0
    nvec = np.zeros_like(freq)
    nvec[:, nonzero] = freq[:, nonzero] / norm[nonzero]
    return nvec, nonzero


def _project_compatible(
    field: np.ndarray,
    *,
    shape: tuple[int, int, int],
    nvec: np.ndarray,
    nonzero: np.ndarray,
) -> np.ndarray:
    tensor = _mandel_flat_to_tensor_field(field, shape)
    fourier = np.fft.fftn(tensor, axes=(2, 3, 4))
    traction = np.einsum("ijxyz,jxyz->ixyz", fourier, nvec, optimize=True)
    parallel_scalar = np.einsum("ixyz,ixyz->xyz", traction, nvec, optimize=True)
    parallel = nvec * parallel_scalar[None, ...]
    disp = 2.0 * (traction - parallel) + parallel
    projected = 0.5 * (
        np.einsum("ixyz,jxyz->ijxyz", nvec, disp, optimize=True)
        + np.einsum("ixyz,jxyz->ijxyz", disp, nvec, optimize=True)
    )
    projected[:, :, ~nonzero] = 0.0
    out = np.fft.ifftn(projected, axes=(2, 3, 4)).real
    return _tensor_field_to_mandel_flat(out).astype(np.float64, copy=False)


def _project_compatible_fourier(
    fields: np.ndarray,
    *,
    shape: tuple[int, int, int],
    nvec: np.ndarray,
    nonzero: np.ndarray,
) -> np.ndarray:
    tensor = _mandel_loads_to_tensor_field(fields, shape)
    fourier = np.fft.fftn(tensor, axes=(3, 4, 5))
    traction = np.einsum("bijxyz,jxyz->bixyz", fourier, nvec, optimize=True)
    parallel_scalar = np.einsum("bixyz,ixyz->bxyz", traction, nvec, optimize=True)
    parallel = nvec[None, ...] * parallel_scalar[:, None, ...]
    disp = 2.0 * (traction - parallel) + parallel
    projected = 0.5 * (
        np.einsum("ixyz,bjxyz->bijxyz", nvec, disp, optimize=True)
        + np.einsum("bixyz,jxyz->bijxyz", disp, nvec, optimize=True)
    )
    projected[:, :, :, ~nonzero] = 0.0
    return projected


def _reference_inverse(
    field: np.ndarray,
    *,
    shape: tuple[int, int, int],
    nvec: np.ndarray,
    nonzero: np.ndarray,
    lam0: float,
    mu0: float,
) -> np.ndarray:
    tensor = _mandel_flat_to_tensor_field(field, shape)
    fourier = np.fft.fftn(tensor, axes=(2, 3, 4))
    traction = np.einsum("ijxyz,jxyz->ixyz", fourier, nvec, optimize=True)
    parallel_scalar = np.einsum("ixyz,ixyz->xyz", traction, nvec, optimize=True)
    parallel = nvec * parallel_scalar[None, ...]
    denom_parallel = max(float(lam0) + 2.0 * float(mu0), np.finfo(float).eps)
    denom_shear = max(float(mu0), np.finfo(float).eps)
    disp = (traction - parallel) / denom_shear + parallel / denom_parallel
    strain = 0.5 * (
        np.einsum("ixyz,jxyz->ijxyz", nvec, disp, optimize=True)
        + np.einsum("ixyz,jxyz->ijxyz", disp, nvec, optimize=True)
    )
    strain[:, :, ~nonzero] = 0.0
    out = np.fft.ifftn(strain, axes=(2, 3, 4)).real
    return _tensor_field_to_mandel_flat(out).astype(np.float64, copy=False)


def _reference_inverse_fourier(
    residual_fourier: np.ndarray,
    *,
    nvec: np.ndarray,
    nonzero: np.ndarray,
    lam0: float,
    mu0: float,
) -> np.ndarray:
    traction = np.einsum("bijxyz,jxyz->bixyz", residual_fourier, nvec, optimize=True)
    parallel_scalar = np.einsum("bixyz,ixyz->bxyz", traction, nvec, optimize=True)
    parallel = nvec[None, ...] * parallel_scalar[:, None, ...]
    denom_parallel = max(float(lam0) + 2.0 * float(mu0), np.finfo(float).eps)
    denom_shear = max(float(mu0), np.finfo(float).eps)
    disp = (traction - parallel) / denom_shear + parallel / denom_parallel
    strain = 0.5 * (
        np.einsum("ixyz,bjxyz->bijxyz", nvec, disp, optimize=True)
        + np.einsum("bixyz,jxyz->bijxyz", disp, nvec, optimize=True)
    )
    strain[:, :, :, ~nonzero] = 0.0
    return strain


def _geometry_groups(
    phase: np.ndarray,
    ori: np.ndarray,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    phase_flat = np.asarray(phase).reshape(-1)
    ori_flat = np.asarray(ori, dtype=float).reshape(-1, 3)
    matrix_idx = np.flatnonzero(phase_flat == 0)
    fiber_idx = np.flatnonzero(phase_flat != 0)
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    if len(fiber_idx):
        rounded = np.round(ori_flat[fiber_idx], decimals=12)
        unique_oris, inverse = np.unique(rounded, axis=0, return_inverse=True)
        local_bases = reduced._fiber_local_bases_axis0()
        for group_id, axis in enumerate(unique_oris):
            idx = fiber_idx[inverse == group_id]
            if len(idx) == 0:
                continue
            rotation = reduced.rotation_matrix_from_vector(axis)
            bases = np.stack(
                [reduced.rotate_C_mandel(basis, rotation) for basis in local_bases],
                axis=0,
            )
            groups.append((idx, bases))
    return matrix_idx, groups


def _apply_material_to_strain_batch(
    strain: np.ndarray,
    *,
    coeffs: np.ndarray,
    matrix_idx: np.ndarray,
    fiber_groups: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    stress = np.zeros_like(strain, dtype=np.float64)
    matrix_bases = reduced._isotropic_bases()
    matrix_C = coeffs[0] * matrix_bases[0] + coeffs[1] * matrix_bases[1]
    if len(matrix_idx):
        stress[:, :, matrix_idx] = np.einsum(
            "ab,lbn->lan",
            matrix_C,
            strain[:, :, matrix_idx],
            optimize=True,
        )
    for idx, bases in fiber_groups:
        fiber_C = np.tensordot(coeffs[2:], bases, axes=(0, 0))
        stress[:, :, idx] = np.einsum(
            "ab,lbn->lan",
            fiber_C,
            strain[:, :, idx],
            optimize=True,
        )
    return stress


def _apply_material_to_strain(
    strain: np.ndarray,
    *,
    coeffs: np.ndarray,
    matrix_idx: np.ndarray,
    fiber_groups: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    stress = np.zeros_like(strain, dtype=np.float64)
    matrix_bases = reduced._isotropic_bases()
    matrix_C = coeffs[0] * matrix_bases[0] + coeffs[1] * matrix_bases[1]
    if len(matrix_idx):
        stress[:, matrix_idx] = matrix_C @ strain[:, matrix_idx]
    for idx, bases in fiber_groups:
        fiber_C = np.tensordot(coeffs[2:], bases, axes=(0, 0))
        stress[:, idx] = fiber_C @ strain[:, idx]
    return stress


def _candidate_strain_loads(
    amplitudes: np.ndarray,
    basis_flat: np.ndarray,
) -> np.ndarray:
    strains = np.einsum(
        "rl,rcn->lcn",
        amplitudes.astype(np.float64, copy=False),
        basis_flat,
        optimize=True,
    ).astype(np.float64, copy=False)
    for load_id in range(6):
        strains[load_id, load_id] += 1.0
    return strains


def _energy_matrix_from_fourier(
    residual_fft_mandel: np.ndarray,
    correction_fft_mandel: np.ndarray,
    *,
    nvox: int,
) -> np.ndarray:
    energy = np.einsum(
        "icn,jcn->ij",
        residual_fft_mandel,
        np.conjugate(correction_fft_mandel),
        optimize=True,
    ).real
    energy /= float(nvox) ** 2
    return 0.5 * (energy + energy.T)


def _load_residual_norms_from_fourier(
    residual_fft_mandel: np.ndarray,
    *,
    nvox: int,
) -> list[float]:
    power = np.sum(np.abs(residual_fft_mandel) ** 2, axis=(1, 2))
    power /= float(6 * nvox * nvox)
    return [float(math.sqrt(max(value, 0.0))) for value in power]


def _reference_lame(coeffs: np.ndarray) -> tuple[float, float]:
    lam_m = float(coeffs[0])
    mu_m = float(coeffs[1])
    local_fiber = sum(
        coeffs[q + 2] * basis
        for q, basis in enumerate(reduced._fiber_local_bases_axis0())
    )
    bulk_m = lam_m + 2.0 * mu_m / 3.0
    bulk_f = (
        local_fiber[0, 0]
        + local_fiber[1, 1]
        + local_fiber[2, 2]
        + 2.0 * (local_fiber[0, 1] + local_fiber[0, 2] + local_fiber[1, 2])
    ) / 9.0
    mu_f = float(np.mean(np.diag(local_fiber)[3:]) / 2.0)
    bulk0 = math.sqrt(max(bulk_m, np.finfo(float).eps) * max(bulk_f, np.finfo(float).eps))
    mu0 = math.sqrt(max(mu_m, np.finfo(float).eps) * max(mu_f, np.finfo(float).eps))
    lam0 = bulk0 - 2.0 * mu0 / 3.0
    return float(lam0), float(mu0)


def _local_phase_stiffnesses(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix_bases = reduced._isotropic_bases()
    matrix = coeffs[0] * matrix_bases[0] + coeffs[1] * matrix_bases[1]
    fiber = sum(
        coeffs[q + 2] * basis
        for q, basis in enumerate(reduced._fiber_local_bases_axis0())
    )
    return 0.5 * (matrix + matrix.T), 0.5 * (fiber + fiber.T)


def _local_phase_stiffnesses_batch(
    coeffs_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build matrix/fiber stiffnesses for a candidate batch."""
    coeffs = np.asarray(coeffs_batch, dtype=np.float64)
    if coeffs.ndim != 2 or coeffs.shape[1] != 7:
        raise ValueError("coeffs_batch must have shape (n_candidates, 7).")
    matrix_bases = np.stack(reduced._isotropic_bases(), axis=0)
    fiber_bases = np.stack(reduced._fiber_local_bases_axis0(), axis=0)
    matrix = np.einsum("nq,qij->nij", coeffs[:, :2], matrix_bases, optimize=True)
    fiber = np.einsum("nq,qij->nij", coeffs[:, 2:], fiber_bases, optimize=True)
    return (
        0.5 * (matrix + np.swapaxes(matrix, -1, -2)),
        0.5 * (fiber + np.swapaxes(fiber, -1, -2)),
    )


def _score_candidate(
    row: pd.Series,
    *,
    basis_flat: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    shape: tuple[int, int, int],
    nvec: np.ndarray,
    nonzero: np.ndarray,
    matrix_idx: np.ndarray,
    fiber_groups: list[tuple[np.ndarray, np.ndarray]],
    indicator_mode: str,
    load_batch_size: int,
    score_mode: str,
    sensitivity_rel_step: float,
    sensitivity_scale: str,
    mixed_energy_weight: float,
    mixed_sensitivity_weight: float,
    online_estimator: TwoKernelEstimator | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    row_dict = row.to_dict()
    coeffs = reduced._material_coefficients(row_dict)
    C_rom, amplitudes, online_s = reduced._rom_ceff(coeffs, Kq, Bq, Dq)
    lam0, mu0 = _reference_lame(coeffs)
    kernel_matrices: np.ndarray | None = None
    upper_bound_matrix: np.ndarray | None = None
    reference_strategy = "geometric_phase_average"
    beta = np.nan
    beta_safe = np.nan
    reference_index = -1
    reference_poisson0 = np.nan

    if indicator_mode == "dense":
        if online_estimator is None:
            raise RuntimeError("indicator_mode='dense' requiere online_estimator.npz.")
        kernel_matrices = online_estimator.kernel_energy_matrices(coeffs, amplitudes)
        selection = optimize_isotropic_reference(
            kernel_matrices,
            _local_phase_stiffnesses(coeffs),
            beta_margin=1.0e-10,
        )
        energy_matrix = selection.energy_matrix
        upper_bound_matrix = selection.upper_bound_matrix
        lam0 = selection.lambda0
        mu0 = selection.mu0
        beta = selection.beta
        beta_safe = selection.beta_safe
        reference_index = selection.index
        reference_poisson0 = selection.poisson0
        reference_strategy = "optimized_129_isotropic_shapes"
        load_residual_norms = [np.nan] * 6
    elif indicator_mode == "legacy":
        residuals: list[np.ndarray] = []
        corrections: list[np.ndarray] = []
        load_residual_norms: list[float] = []
        for load_id in range(6):
            strain = np.tensordot(
                amplitudes[:, load_id].astype(np.float64, copy=False),
                basis_flat,
                axes=(0, 0),
            ).astype(np.float64, copy=False)
            strain[load_id] += 1.0
            stress = _apply_material_to_strain(
                strain,
                coeffs=coeffs,
                matrix_idx=matrix_idx,
                fiber_groups=fiber_groups,
            )
            residual = _project_compatible(
                stress,
                shape=shape,
                nvec=nvec,
                nonzero=nonzero,
            )
            correction = _reference_inverse(
                residual,
                shape=shape,
                nvec=nvec,
                nonzero=nonzero,
                lam0=lam0,
                mu0=mu0,
            )
            residuals.append(residual)
            corrections.append(correction)
            load_residual_norms.append(float(np.sqrt(np.mean(residual * residual))))

        energy_matrix = np.zeros((6, 6), dtype=float)
        for ii in range(6):
            for jj in range(6):
                energy_matrix[ii, jj] = float(
                    np.mean(np.sum(residuals[ii] * corrections[jj], axis=0))
                )
        energy_matrix = 0.5 * (energy_matrix + energy_matrix.T)
    else:
        nvox = int(np.prod(shape))
        batch = max(1, min(int(load_batch_size), 6))
        strains = _candidate_strain_loads(amplitudes, basis_flat)
        residual_fft_mandel = np.empty((6, 6, nvox), dtype=np.complex128)
        correction_fft_mandel = np.empty_like(residual_fft_mandel)
        for start in range(0, 6, batch):
            stop = min(start + batch, 6)
            stress = _apply_material_to_strain_batch(
                strains[start:stop],
                coeffs=coeffs,
                matrix_idx=matrix_idx,
                fiber_groups=fiber_groups,
            )
            residual_fourier = _project_compatible_fourier(
                stress,
                shape=shape,
                nvec=nvec,
                nonzero=nonzero,
            )
            correction_fourier = _reference_inverse_fourier(
                residual_fourier,
                nvec=nvec,
                nonzero=nonzero,
                lam0=lam0,
                mu0=mu0,
            )
            residual_fft_mandel[start:stop] = _tensor_loads_to_mandel_flat(
                residual_fourier
            )
            correction_fft_mandel[start:stop] = _tensor_loads_to_mandel_flat(
                correction_fourier
            )
        energy_matrix = _energy_matrix_from_fourier(
            residual_fft_mandel,
            correction_fft_mandel,
            nvox=nvox,
        )
        load_residual_norms = _load_residual_norms_from_fourier(
            residual_fft_mandel,
            nvox=nvox,
        )

    energy_abs = float(np.linalg.norm(energy_matrix))
    rom_norm = float(max(np.linalg.norm(C_rom), np.finfo(float).eps))
    bound_abs = (
        float(np.linalg.norm(upper_bound_matrix))
        if upper_bound_matrix is not None
        else float("nan")
    )
    field_residual = (
        float(np.linalg.norm(load_residual_norms))
        if np.all(np.isfinite(load_residual_norms))
        else float("nan")
    )

    fom = _full_ceff_from_row(row)
    true_error = np.nan
    if fom is not None:
        true_error = reduced._relative_frobenius(C_rom, fom)

    out: dict[str, Any] = {
        "material_id": int(row.get("material_id", -1)),
        "material_label": str(row.get("material_label", "")),
        "energy_indicator_abs": energy_abs,
        "energy_indicator_rel": energy_abs / rom_norm,
        "schur_bound_indicator_abs": bound_abs,
        "schur_bound_indicator_rel": bound_abs / rom_norm,
        "field_residual_norm": field_residual,
        "field_residual_rel": field_residual / rom_norm,
        "true_relative_error": float(true_error),
        "rom_online_s": float(online_s),
        "indicator_wall_s": float(time.perf_counter() - t0),
        "indicator_mode": str(indicator_mode),
        "load_batch_size": int(load_batch_size),
        "reference_lambda0": float(lam0),
        "reference_mu0": float(mu0),
        "reference_poisson0": float(reference_poisson0),
        "reference_index": int(reference_index),
        "reference_strategy": reference_strategy,
        "beta": float(beta),
        "beta_safe": float(beta_safe),
        "rom_min_eig": float(np.linalg.eigvalsh(C_rom).min()),
        "rom_max_eig": float(np.linalg.eigvalsh(C_rom).max()),
    }
    energy_matrix_rel = energy_matrix / rom_norm
    for ii in range(6):
        for jj in range(6):
            out[f"energy_matrix_{ii + 1}{jj + 1}"] = float(energy_matrix[ii, jj])
            out[f"energy_matrix_rel_{ii + 1}{jj + 1}"] = float(
                energy_matrix_rel[ii, jj]
            )
            if kernel_matrices is not None:
                out[f"kernel_transverse_{ii + 1}{jj + 1}"] = float(
                    kernel_matrices[0, ii, jj]
                )
                out[f"kernel_longitudinal_{ii + 1}{jj + 1}"] = float(
                    kernel_matrices[1, ii, jj]
                )
            if upper_bound_matrix is not None:
                out[f"upper_bound_matrix_{ii + 1}{jj + 1}"] = float(
                    upper_bound_matrix[ii, jj]
                )

    if score_mode in {"sensitivity", "mixed"}:
        sensitivity_values: list[float] = []
        sensitivity_abs_values: list[float] = []
        for name in PHYSICAL_NAMES:
            plus_row, minus_row, step, domain_width = _perturb_physical_row(
                row,
                name=name,
                rel_step=float(sensitivity_rel_step),
            )
            plus = _score_candidate(
                plus_row,
                basis_flat=basis_flat,
                Kq=Kq,
                Bq=Bq,
                Dq=Dq,
                shape=shape,
                nvec=nvec,
                nonzero=nonzero,
                matrix_idx=matrix_idx,
                fiber_groups=fiber_groups,
                indicator_mode=indicator_mode,
                load_batch_size=load_batch_size,
                score_mode="energy",
                sensitivity_rel_step=sensitivity_rel_step,
                sensitivity_scale=sensitivity_scale,
                mixed_energy_weight=mixed_energy_weight,
                mixed_sensitivity_weight=mixed_sensitivity_weight,
                online_estimator=online_estimator,
            )
            minus = _score_candidate(
                minus_row,
                basis_flat=basis_flat,
                Kq=Kq,
                Bq=Bq,
                Dq=Dq,
                shape=shape,
                nvec=nvec,
                nonzero=nonzero,
                matrix_idx=matrix_idx,
                fiber_groups=fiber_groups,
                indicator_mode=indicator_mode,
                load_batch_size=load_batch_size,
                score_mode="energy",
                sensitivity_rel_step=sensitivity_rel_step,
                sensitivity_scale=sensitivity_scale,
                mixed_energy_weight=mixed_energy_weight,
                mixed_sensitivity_weight=mixed_sensitivity_weight,
                online_estimator=online_estimator,
            )
            d_abs = (
                _energy_matrix_from_record(plus, relative=False)
                - _energy_matrix_from_record(minus, relative=False)
            ) / (2.0 * step)
            d_rel = (
                _energy_matrix_from_record(plus, relative=True)
                - _energy_matrix_from_record(minus, relative=True)
            ) / (2.0 * step)
            scale = (
                float(domain_width)
                if sensitivity_scale == "domain"
                else max(abs(float(row_dict[name])), 1.0)
            )
            eta_abs = float(scale * np.linalg.norm(d_abs))
            eta_rel = float(scale * np.linalg.norm(d_rel))
            sensitivity_abs_values.append(eta_abs)
            sensitivity_values.append(eta_rel)
            out[f"sensitivity_indicator_abs_{name}"] = eta_abs
            out[f"sensitivity_indicator_rel_{name}"] = eta_rel
            out[f"sensitivity_fd_step_{name}"] = float(step)
            out[f"sensitivity_scale_{name}"] = float(scale)

        sensitivity_abs = float(max(sensitivity_abs_values))
        sensitivity_rel = float(max(sensitivity_values))
        out["sensitivity_indicator_abs"] = sensitivity_abs
        out["sensitivity_indicator_rel"] = sensitivity_rel
        out["sensitivity_indicator_aggregate"] = "max"
    else:
        out["sensitivity_indicator_abs"] = np.nan
        out["sensitivity_indicator_rel"] = np.nan
        out["sensitivity_indicator_aggregate"] = "none"

    tensor_score = (
        float(out["schur_bound_indicator_rel"])
        if math.isfinite(float(out["schur_bound_indicator_rel"]))
        else float(out["energy_indicator_rel"])
    )
    mixed = math.sqrt(
        (float(mixed_energy_weight) * tensor_score) ** 2
        + (
            float(mixed_sensitivity_weight)
            * float(np.nan_to_num(out["sensitivity_indicator_rel"], nan=0.0))
        )
        ** 2
    )
    out["mixed_indicator_rel"] = float(mixed)
    out["score_mode"] = str(score_mode)
    out["candidate_score"] = float(
        {
            "energy": out["energy_indicator_rel"],
            "bound": tensor_score,
            "sensitivity": out["sensitivity_indicator_rel"],
            "mixed": out["mixed_indicator_rel"],
        }[score_mode]
    )
    out["indicator_wall_s"] = float(time.perf_counter() - t0)
    for q, name in enumerate(reduced.COEFF_NAMES):
        out[f"xi_{name}"] = float(coeffs[q])
    for key in sweep.MATERIAL_COLUMNS:
        if key in row:
            out[key] = row[key]
    return out


def _spearman(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    return float(a[mask].corr(b[mask], method="spearman"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evalua candidatos de material con un indicador residual energetico "
            "sin resolver de nuevo FFTHomPy full-order."
        )
    )
    parser.add_argument("run_dir", type=Path, nargs="?", default=RUN_DEFAULT)
    parser.add_argument("--rom-dir", type=Path, default=ROM_DEFAULT)
    parser.add_argument("--candidate-results-csv", type=Path, default=None)
    parser.add_argument(
        "--candidate-material-ids",
        type=int,
        nargs="*",
        default=None,
        help="Filtra material_id dentro del CSV de candidatos.",
    )
    parser.add_argument("--candidate-points", type=int, default=64)
    parser.add_argument("--candidate-seed", type=int, default=20261031)
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--indicator-mode",
        choices=("auto", "dense", "fourier", "legacy"),
        default="auto",
        help=(
            "dense usa BB/BK/KK sin voxels; fourier valida la ruta directa; "
            "legacy reproduce la ruta fisica anterior; auto prefiere dense."
        ),
    )
    parser.add_argument(
        "--load-batch-size",
        type=int,
        default=6,
        help="Numero de cargas macroscopicas procesadas juntas en modo fourier.",
    )
    parser.add_argument(
        "--score-mode",
        choices=("energy", "bound", "sensitivity", "mixed"),
        default="energy",
        help=(
            "energy reproduce el ranking anterior; sensitivity ordena por la "
            "derivada finita del estimador; mixed combina ambos."
        ),
    )
    parser.add_argument(
        "--sensitivity-rel-step",
        type=float,
        default=5e-3,
        help="Paso relativo para derivar el estimador respecto a parametros fisicos.",
    )
    parser.add_argument(
        "--sensitivity-scale",
        choices=("domain", "local"),
        default="domain",
        help=(
            "Escala la derivada de sensibilidad por ancho del dominio fisico "
            "o por la magnitud local del parametro."
        ),
    )
    parser.add_argument("--mixed-energy-weight", type=float, default=1.0)
    parser.add_argument("--mixed-sensitivity-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rom_dir = args.rom_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No existe run_dir: {run_dir}")
    if not rom_dir.is_dir():
        raise FileNotFoundError(f"No existe rom_dir: {rom_dir}")

    out_dir = _make_out_dir(
        run_dir,
        args.out_name,
        rom_dir,
        overwrite=bool(args.overwrite),
    )
    geometry_dir = run_dir / "_fixed_geometry"
    phase = np.load(geometry_dir / "phase.npy").astype(np.uint8)
    ori = np.load(geometry_dir / "ori.npy").astype(np.float64)
    shape = tuple(int(value) for value in phase.shape)

    operators = np.load(rom_dir / "reduced_operators.npz")
    Kq = np.asarray(operators["Kq"], dtype=float)
    Bq = np.asarray(operators["Bq"], dtype=float)
    Dq = np.asarray(operators["Dq"], dtype=float)
    basis_fields = _load_basis_fields(rom_dir)
    basis_flat = basis_fields.reshape(basis_fields.shape[0], 6, -1)
    estimator_path = rom_dir / "online_estimator.npz"
    indicator_mode = str(args.indicator_mode)
    if indicator_mode == "auto":
        indicator_mode = "dense" if estimator_path.is_file() else "fourier"
    online_estimator = (
        TwoKernelEstimator.load(estimator_path) if indicator_mode == "dense" else None
    )
    if online_estimator is not None and online_estimator.rank != basis_flat.shape[0]:
        raise RuntimeError(
            f"Rango inconsistente: estimator={online_estimator.rank}, basis={basis_flat.shape[0]}."
        )

    candidates = _load_candidates(args)
    if args.candidate_material_ids:
        wanted = {int(value) for value in args.candidate_material_ids}
        candidates = candidates.loc[
            candidates["material_id"].astype(int).isin(wanted)
        ].copy()
        missing = sorted(wanted - set(candidates["material_id"].astype(int)))
        if missing:
            raise KeyError(f"material_id no encontrado en candidatos: {missing}")
    if candidates.empty:
        raise RuntimeError("No hay candidatos para evaluar.")
    candidates.to_csv(out_dir / "candidate_materials.csv", index=False)
    matrix_idx, fiber_groups = _geometry_groups(phase, ori)
    nvec, nonzero = _frequency_unit_vectors(shape)

    print(
        "[QOI] ranking candidatos | "
        f"rom={rom_dir.name} | candidates={len(candidates)} | "
        f"r={basis_flat.shape[0]} | score={args.score_mode}",
        flush=True,
    )
    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for position, (_, row) in enumerate(candidates.iterrows(), start=1):
        record = _score_candidate(
            row,
            basis_flat=basis_flat,
            Kq=Kq,
            Bq=Bq,
            Dq=Dq,
            shape=shape,
            nvec=nvec,
            nonzero=nonzero,
            matrix_idx=matrix_idx,
            fiber_groups=fiber_groups,
            indicator_mode=indicator_mode,
            load_batch_size=int(args.load_batch_size),
            score_mode=str(args.score_mode),
            sensitivity_rel_step=float(args.sensitivity_rel_step),
            sensitivity_scale=str(args.sensitivity_scale),
            mixed_energy_weight=float(args.mixed_energy_weight),
            mixed_sensitivity_weight=float(args.mixed_sensitivity_weight),
            online_estimator=online_estimator,
        )
        records.append(record)
        print(
            "[QOI] "
            f"{position}/{len(candidates)} | material={record['material_id']} | "
            f"eta={record['energy_indicator_rel']:.3e} | "
            f"eta_s={record['sensitivity_indicator_rel']:.3e} | "
            f"score={record['candidate_score']:.3e} | "
            f"res={record['field_residual_rel']:.3e}",
            flush=True,
        )

    ranking = pd.DataFrame(records).sort_values(
        "candidate_score",
        ascending=False,
    )
    ranking.insert(0, "score_rank", np.arange(1, len(ranking) + 1, dtype=int))
    ranking["energy_rank"] = (
        ranking["energy_indicator_rel"].rank(method="first", ascending=False).astype(int)
    )
    ranking["sensitivity_indicator_rank"] = (
        ranking["sensitivity_indicator_rel"].rank(method="first", ascending=False).astype(int)
        if ranking["sensitivity_indicator_rel"].notna().any()
        else np.nan
    )
    ranking["mixed_indicator_rank"] = (
        ranking["mixed_indicator_rel"].rank(method="first", ascending=False).astype(int)
    )
    ranking["field_residual_rank"] = (
        ranking["field_residual_rel"].rank(method="first", ascending=False).astype(int)
        if ranking["field_residual_rel"].notna().any()
        else np.nan
    )
    if ranking["true_relative_error"].notna().any():
        ranking["true_error_rank"] = (
            ranking["true_relative_error"].rank(method="first", ascending=False).astype(int)
        )
    ranking.to_csv(out_dir / "qoi_indicator_ranking.csv", index=False)
    ranking.to_excel(out_dir / "qoi_indicator_ranking.xlsx", index=False)

    top_k = max(1, min(int(args.top_k), len(ranking)))
    top = ranking.head(top_k).copy()
    summary = {
        "run_dir": str(run_dir),
        "rom_dir": str(rom_dir),
        "out_dir": str(out_dir),
        "candidate_count": int(len(ranking)),
        "basis_rank": int(basis_flat.shape[0]),
        "grid_shape": list(shape),
        "indicator_mode": indicator_mode,
        "score_mode": str(args.score_mode),
        "load_batch_size": int(args.load_batch_size),
        "sensitivity_rel_step": float(args.sensitivity_rel_step),
        "sensitivity_scale": str(args.sensitivity_scale),
        "mixed_energy_weight": float(args.mixed_energy_weight),
        "mixed_sensitivity_weight": float(args.mixed_sensitivity_weight),
        "wall_s": float(time.perf_counter() - t0),
        "top_k": int(top_k),
        "top_score_material_ids": [int(value) for value in top["material_id"].tolist()],
        "top_energy_material_ids": [
            int(value)
            for value in ranking.sort_values(
                "energy_indicator_rel",
                ascending=False,
            )
            .head(top_k)["material_id"]
            .tolist()
        ],
        "top_sensitivity_material_ids": (
            [
                int(value)
                for value in ranking.sort_values(
                    "sensitivity_indicator_rel",
                    ascending=False,
                )
                .head(top_k)["material_id"]
                .tolist()
            ]
            if ranking["sensitivity_indicator_rel"].notna().any()
            else []
        ),
        "max_energy_indicator_rel": float(ranking["energy_indicator_rel"].max()),
        "max_sensitivity_indicator_rel": float(
            ranking["sensitivity_indicator_rel"].max()
        )
        if ranking["sensitivity_indicator_rel"].notna().any()
        else float("nan"),
        "max_mixed_indicator_rel": float(ranking["mixed_indicator_rel"].max()),
        "max_candidate_score": float(ranking["candidate_score"].max()),
        "median_indicator_wall_s": float(ranking["indicator_wall_s"].median()),
        "spearman_score_vs_true_error": _spearman(
            ranking["candidate_score"],
            ranking["true_relative_error"],
        ),
        "spearman_energy_vs_true_error": _spearman(
            ranking["energy_indicator_rel"],
            ranking["true_relative_error"],
        ),
        "spearman_sensitivity_vs_true_error": _spearman(
            ranking["sensitivity_indicator_rel"],
            ranking["true_relative_error"],
        ),
        "spearman_mixed_vs_true_error": _spearman(
            ranking["mixed_indicator_rel"],
            ranking["true_relative_error"],
        ),
        "spearman_field_residual_vs_true_error": _spearman(
            ranking["field_residual_rel"],
            ranking["true_relative_error"],
        ),
    }
    if ranking["true_relative_error"].notna().any():
        worst_true = ranking.sort_values("true_relative_error", ascending=False).iloc[0]
        summary["worst_true_material_id"] = int(worst_true["material_id"])
        summary["worst_true_relative_error"] = float(worst_true["true_relative_error"])
        summary["worst_true_score_rank"] = int(worst_true["score_rank"])
        summary["worst_true_energy_rank"] = int(worst_true["energy_rank"])
        if not pd.isna(worst_true["sensitivity_indicator_rank"]):
            summary["worst_true_sensitivity_rank"] = int(
                worst_true["sensitivity_indicator_rank"]
            )
        summary["worst_true_mixed_rank"] = int(worst_true["mixed_indicator_rank"])
        if not pd.isna(worst_true["field_residual_rank"]):
            summary["worst_true_field_residual_rank"] = int(
                worst_true["field_residual_rank"]
            )
    _write_json(out_dir / "qoi_indicator_summary.json", summary)

    text = f"""# QoI/Energy Residual Candidate Ranking

- Source fixed-geometry run: `{run_dir}`
- ROM: `{rom_dir}`
- Candidate count: `{len(ranking)}`
- Basis rank: `{basis_flat.shape[0]}`
- Indicator mode: `{indicator_mode}`
- Score mode: `{args.score_mode}`
- Load batch size: `{args.load_batch_size}`
- Top score material IDs: `{summary['top_score_material_ids']}`
- Top energy-indicator material IDs: `{summary['top_energy_material_ids']}`
- Top sensitivity-indicator material IDs: `{summary['top_sensitivity_material_ids']}`
- Max relative energy indicator: `{summary['max_energy_indicator_rel']:.6e}`
- Max relative sensitivity indicator: `{summary['max_sensitivity_indicator_rel']:.6e}`
- Max mixed indicator: `{summary['max_mixed_indicator_rel']:.6e}`
- Median indicator time: `{summary['median_indicator_wall_s']:.6e} s`
- Spearman selected score vs true error: `{summary['spearman_score_vs_true_error']:.6e}`
- Spearman energy indicator vs true error: `{summary['spearman_energy_vs_true_error']:.6e}`
- Spearman sensitivity indicator vs true error: `{summary['spearman_sensitivity_vs_true_error']:.6e}`
- Spearman mixed indicator vs true error: `{summary['spearman_mixed_vs_true_error']:.6e}`
- Spearman field residual vs true error: `{summary['spearman_field_residual_vs_true_error']:.6e}`
"""
    if "worst_true_material_id" in summary:
        text += (
            f"- Worst true-error material ID: `{summary['worst_true_material_id']}`\n"
            f"- Worst true-error value: `{summary['worst_true_relative_error']:.6e}`\n"
            f"- Score rank of worst true-error point: `{summary['worst_true_score_rank']}`\n"
            f"- Energy rank of worst true-error point: `{summary['worst_true_energy_rank']}`\n"
        )
        if "worst_true_sensitivity_rank" in summary:
            text += (
                "- Sensitivity rank of worst true-error point: "
                f"`{summary['worst_true_sensitivity_rank']}`\n"
            )
        text += (
            f"- Mixed rank of worst true-error point: `{summary['worst_true_mixed_rank']}`\n"
        )
    (out_dir / "qoi_indicator_summary.md").write_text(text, encoding="utf-8")

    print(
        "[QOI] listo | "
        f"out={out_dir} | top={summary['top_score_material_ids'][:3]} | "
        f"rho_score={summary['spearman_score_vs_true_error']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
