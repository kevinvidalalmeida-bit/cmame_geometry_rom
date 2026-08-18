#!/usr/bin/env python3
"""Adaptive anchor-tangent ROM construction for a single CMAME geometry.

Pipeline:
  Geometry → Anchor γ⁽¹⁾ → FOM X⁽¹⁾ + tangents ∂X/∂γ⁽¹⁾ → V₁ = orth(S₁)
           → project Kq_r, Bq_r → Sobol monitor (ROM only, cheap)
           → if max error ≤ ε_target: STOP
           → else: worst point → new anchor + tangents → enrich V → repeat

Anchors are chosen greedily based on the worst ROM monitor error.  Tangent
fields are computed by exact discrete affine sensitivity solves by default;
finite-difference tangents remain available for diagnostics.  The number of
anchors is NOT prescribed: it is determined by the target tolerance.

For each anchor a, we build:
  S_a = [X⁽ᵃ⁾, X_{,1}⁽ᵃ⁾, …, X_{,Q}⁽ᵃ⁾]

and enrich the global basis:
  V_{new} = orth([V_old, S_a])

Then we update (or extend) the affine reduced operators Kq_r, Bq_r, Dq, and
re-evaluate the Sobol monitor purely from the ROM — no FOM is needed for the
monitor points.

Final independent FOM validation is run once at the end to confirm the
tolerance on a fresh set of materials.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import linalg as scipy_linalg
from scipy.stats import qmc


SCRIPT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DEFAULT = SCRIPT_DIR / "campaign_config.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from env_bootstrap import ensure_configured_venv

ensure_configured_venv(CONFIG_DEFAULT)

ROOT = SCRIPT_DIR.parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cmame_campaign_common as common
import fft_homogenization_solver as sweep
import rom_reduced_operator as reduced
import rom_validation_utils as validate


OUT_ROOT = ROOT / "results" / "cmame_method" / "interpretable_vf05_25_ar5_20"
BASE_RUN_STEM = "run_sobol_pod_20260815_010726"
MATERIAL_NAMES = tuple(sweep.MATERIAL_BOUNDS)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(common.jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def center_material() -> dict[str, Any]:
    """Build the center-of-domain material point."""
    sampled = {
        name: 0.5 * (bounds[0] + bounds[1])
        for name, bounds in sweep.MATERIAL_BOUNDS.items()
    }
    material = sweep._material_derived(sampled)
    sweep._validate_material(material)
    return {"material_id": 0, "material_label": "gamma_anchor_center", **material}


def material_from_gamma(
    gamma: np.ndarray,
    *,
    material_id: int,
    label: str,
) -> dict[str, Any]:
    """Reconstruct a full material dict from gamma coefficients."""
    gamma = np.asarray(gamma, dtype=np.float64)
    lam_m, mu_m = gamma[:2]
    if mu_m <= 0.0 or lam_m + mu_m <= 0.0:
        raise ValueError("invalid isotropic matrix affine coefficients")
    Em = mu_m * (3.0 * lam_m + 2.0 * mu_m) / (lam_m + mu_m)
    nu_m = lam_m / (2.0 * (lam_m + mu_m))

    # Rebuild local fiber stiffness from gamma[2:]
    c_tt, c_tt_cross, c_lt, c_ll, g_lt = gamma[2:]
    C_f = np.zeros((6, 6), dtype=np.float64)
    C_f[0, 0] = c_ll
    C_f[0, 1] = C_f[1, 0] = c_lt
    C_f[0, 2] = C_f[2, 0] = c_lt
    C_f[1, 1] = C_f[2, 2] = c_tt
    C_f[1, 2] = C_f[2, 1] = c_tt_cross
    C_f[3, 3] = c_tt - c_tt_cross
    C_f[4, 4] = C_f[5, 5] = 2.0 * g_lt

    eigvals = np.linalg.eigvalsh(0.5 * (C_f + C_f.T))
    if np.min(eigvals) <= 0.0:
        raise ValueError("fiber local stiffness is not SPD")
    props = reduced._engineering_constants_batch(C_f[None, :, :])[0]
    Ef_L = float(props[0])
    Ef_T = float(0.5 * (props[1] + props[2]))
    G_LT = float(0.5 * (props[3] + props[4]))
    nu_LT = float(0.5 * (props[6] + props[7]))
    nu_TT = float(props[8])
    sampled = {
        "Em": float(Em),
        "nu_m": float(nu_m),
        "Ef_L": Ef_L,
        "Ef_T": Ef_T,
        "G_LT": G_LT,
        "nu_LT": nu_LT,
        "nu_TT": nu_TT,
    }
    for name, value in sampled.items():
        low, high = sweep.MATERIAL_BOUNDS[name]
        tol = 1.0e-10 * max(abs(low), abs(high), 1.0)
        if value < low - tol or value > high + tol:
            raise ValueError(f"{name}={value:.6g} is outside [{low}, {high}]")
        sampled[name] = min(max(value, low), high)
    material = sweep._material_derived(sampled)
    sweep._validate_material(material)
    return {"material_id": int(material_id), "material_label": label, **material}


def gamma_scales(candidate_count: int, seed: int) -> np.ndarray:
    """Estimate gamma coefficient scales from a large Sobol sample."""
    candidates = validate._build_independent_materials(int(candidate_count), int(seed))
    params = candidates.loc[:, MATERIAL_NAMES].to_numpy(dtype=np.float64)
    gammas = reduced._material_coefficients_batch(params)
    span = np.ptp(gammas, axis=0)
    center = np.abs(reduced._material_coefficients(center_material()))
    return np.maximum(span, np.maximum(center, 1.0))


# ---------------------------------------------------------------------------
# Tangent field computation via central finite differences
# ---------------------------------------------------------------------------

def _build_tangent_materials(
    *,
    gamma0: np.ndarray,
    scales: np.ndarray,
    rel_step: float,
    min_rel_step: float,
    anchor_id: int,
    fd_mode: str = "forward",  # "forward" (7 solves) or "central" (14 solves)
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Build perturbed materials for forward (8 solves total) or central (15 solves total) tangents."""
    step_rows: list[dict[str, Any]] = []
    perturbed_materials: list[dict[str, Any]] = []
    
    for q, name in enumerate(reduced.COEFF_NAMES):
        used_step = float(rel_step) * float(scales[q])
        
        if fd_mode == "forward":
            # Forward difference: 1 material per tangent direction
            accepted: dict[str, Any] | None = None
            while abs(used_step) >= float(min_rel_step) * float(scales[q]):
                trial = gamma0.copy()
                trial[q] += used_step
                try:
                    mat = material_from_gamma(
                        trial,
                        material_id=1000 * anchor_id + q,
                        label=f"gamma_anchor{anchor_id}_fd_{name}_plus",
                    )
                    accepted = mat
                    break
                except ValueError:
                    used_step *= 0.5
            if accepted is None:
                raise RuntimeError(f"Could not build forward FD material for coefficient {name}.")
            perturbed_materials.append(accepted)
            step_rows.append({
                "coefficient_index": q,
                "coefficient_name": name,
                "gamma0": float(gamma0[q]),
                "scale": float(scales[q]),
                "signed_step": float(used_step),
                "relative_step_used": float(abs(used_step) / float(scales[q])),
                "anchor_id": int(anchor_id),
                "fd_mode": "forward",
            })
        else:
            # Central difference: 2 materials per tangent direction
            accepted_plus: dict[str, Any] | None = None
            accepted_minus: dict[str, Any] | None = None
            while abs(used_step) >= float(min_rel_step) * float(scales[q]):
                for sign_label, sign in [("plus", 1), ("minus", -1)]:
                    trial = gamma0.copy()
                    trial[q] += sign * used_step
                    try:
                        mat = material_from_gamma(
                            trial,
                            material_id=1000 * anchor_id + 2 * q + (0 if sign > 0 else 1),
                            label=f"gamma_anchor{anchor_id}_fd_{name}_{sign_label}",
                        )
                        if sign > 0:
                            accepted_plus = mat
                        else:
                            accepted_minus = mat
                    except ValueError:
                        continue
                if accepted_plus is not None and accepted_minus is not None:
                    break
                accepted_plus = None
                accepted_minus = None
                used_step *= 0.5
            if accepted_plus is None or accepted_minus is None:
                raise RuntimeError(f"Could not build central FD materials for coefficient {name}.")
            perturbed_materials.append(accepted_plus)
            perturbed_materials.append(accepted_minus)
            step_rows.append({
                "coefficient_index": q,
                "coefficient_name": name,
                "gamma0": float(gamma0[q]),
                "scale": float(scales[q]),
                "signed_plus_step": float(used_step),
                "relative_step_used": float(abs(used_step) / float(scales[q])),
                "anchor_id": int(anchor_id),
                "fd_mode": "central",
            })
    return perturbed_materials, pd.DataFrame(step_rows)


def _linear_chunk_size(
    *,
    rows_a: int,
    rows_b: int,
    length: int,
    requested: int = 2_000_000,
) -> int:
    """Choose a conservative float64 GPU chunk for tall-skinny products."""
    limit = int(requested)
    try:
        import cupy as cp
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        target = max(256 * 1024**2, int(0.22 * free_bytes))
        bytes_per_scalar = 8 * max(int(rows_a) + int(rows_b), 1)
        limit = min(limit, max(65536, int(target // bytes_per_scalar)))
    except Exception:
        pass
    return max(65536, min(int(length), int(limit)))


def _gpu_spatial_chunked_gram(
    incoming_matrix: np.ndarray,
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    """Compute A A.T with float64 accumulation while A stays in float32 RAM."""
    k, d = incoming_matrix.shape
    effective = _linear_chunk_size(rows_a=k, rows_b=k, length=d, requested=chunk_size)
    try:
        import cupy as cp
        G_gpu = cp.zeros((k, k), dtype=cp.float64)
        for start in range(0, d, effective):
            end = min(start + effective, d)
            block_gpu = cp.asarray(incoming_matrix[:, start:end], dtype=cp.float64)
            G_gpu += block_gpu @ block_gpu.T
            del block_gpu
        G = cp.asnumpy(G_gpu)
        cp.cuda.Stream.null.synchronize()
        return 0.5 * (G + G.T)
    except Exception:
        A = np.asarray(incoming_matrix, dtype=np.float64)
        G = A @ A.T
        return 0.5 * (G + G.T)


def _gpu_spatial_chunked_project(
    incoming_matrix: np.ndarray,
    v_old_matrix: np.ndarray,
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    """Compute A V_old.T with float64 accumulation and bounded VRAM."""
    k, d = incoming_matrix.shape
    r_old = v_old_matrix.shape[0]
    effective = _linear_chunk_size(
        rows_a=k, rows_b=r_old, length=d, requested=chunk_size
    )
    try:
        import cupy as cp
        C_gpu = cp.zeros((k, r_old), dtype=cp.float64)
        for start in range(0, d, effective):
            end = min(start + effective, d)
            a_chunk = cp.asarray(incoming_matrix[:, start:end], dtype=cp.float64)
            v_chunk = cp.asarray(v_old_matrix[:, start:end], dtype=cp.float64)
            C_gpu += a_chunk @ v_chunk.T
            del a_chunk, v_chunk
        C = cp.asnumpy(C_gpu)
        cp.cuda.Stream.null.synchronize()
        return C
    except Exception:
        return (
            np.asarray(incoming_matrix, dtype=np.float64)
            @ np.asarray(v_old_matrix, dtype=np.float64).T
        )


def _apply_left_transform_in_place(
    matrix: np.ndarray,
    transform: np.ndarray,
    *,
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    """Apply a small float64 left transform while storing the result in float32."""
    out_rows = int(transform.shape[0])
    d = int(matrix.shape[1])
    effective = _linear_chunk_size(
        rows_a=max(matrix.shape[0], out_rows), rows_b=out_rows,
        length=d, requested=chunk_size,
    )
    try:
        import cupy as cp
        T_gpu = cp.asarray(transform, dtype=cp.float64)
        for start in range(0, d, effective):
            end = min(start + effective, d)
            block_gpu = cp.asarray(matrix[:, start:end], dtype=cp.float64)
            out_gpu = T_gpu @ block_gpu
            matrix[:out_rows, start:end] = cp.asnumpy(out_gpu).astype(
                matrix.dtype, copy=False
            )
            del block_gpu, out_gpu
        cp.cuda.Stream.null.synchronize()
    except Exception:
        for start in range(0, d, effective):
            end = min(start + effective, d)
            block = np.asarray(matrix[:, start:end], dtype=np.float64)
            matrix[:out_rows, start:end] = (transform @ block).astype(
                matrix.dtype, copy=False
            )
    return matrix[:out_rows]


def append_block_orthonormal(
    existing_basis: list[np.ndarray],
    new_fields: list[np.ndarray],
    *,
    tolerance: float = 1.0e-12,
    verbose: bool = True,
    chunk_size: int = 2_000_000,
    rank_rtol: float = 1.0e-12,
) -> list[np.ndarray]:
    """Append a rank-revealed block with mixed-precision storage.

    Large fields remain float32 in RAM.  Projection coefficients, Gram
    matrices, eigendecompositions and normalization transforms are computed in
    float64.  Near-dependent tangent directions are rejected *before* they can
    pollute the Ritz system.
    """
    if not new_fields:
        return []
    first = np.asarray(new_fields[0])
    if first.ndim != 3 or first.shape[1] != 6:
        raise ValueError("new tangent blocks must have shape (loads, 6, nvox)")
    nvox = int(first.shape[2])
    d = 6 * nvox
    count = sum(int(np.asarray(block).shape[0]) for block in new_fields)

    # Storage is intentionally float32.  The block list is consumed as it is
    # copied, so peak RAM is roughly old basis + incoming snapshots + one
    # float32 incoming buffer, never a duplicate float64 global matrix.
    incoming = np.empty((count, d), dtype=np.float32)
    row = 0
    while new_fields:
        block = np.asarray(new_fields.pop(0), dtype=np.float32)
        for local in range(block.shape[0]):
            incoming[row] = block[local].reshape(-1)
            row += 1
        del block
    gc.collect()

    # Project against the existing approximately orthonormal basis.  Existing
    # fields are float32 storage; dot products and coefficients are float64.
    if existing_basis:
        old_group = 4
        for start_old in range(0, len(existing_basis), old_group):
            group = existing_basis[start_old : start_old + old_group]
            old_matrix = np.stack(
                [np.asarray(v, dtype=np.float32).reshape(-1) for v in group], axis=0
            )
            for _ in range(2):
                coeffs = _gpu_spatial_chunked_project(incoming, old_matrix, chunk_size)
                effective = _linear_chunk_size(
                    rows_a=incoming.shape[0], rows_b=old_matrix.shape[0],
                    length=d, requested=chunk_size,
                )
                for start in range(0, d, effective):
                    end = min(start + effective, d)
                    correction = coeffs @ np.asarray(
                        old_matrix[:, start:end], dtype=np.float64
                    )
                    incoming[:, start:end] -= correction.astype(np.float32)
            del old_matrix

    # Rank reveal the residual block.  Gram eigenvalues are squared singular
    # values; a relative 1e-12 cutoff corresponds to ~1e-6 singular-value
    # resolution, appropriate for float32 field storage and a 1e-4 ROM target.
    G = _gpu_spatial_chunked_gram(incoming, chunk_size)
    eigvals, U = scipy_linalg.eigh(G, check_finite=False)
    largest = float(eigvals[-1]) if len(eigvals) else 0.0
    if largest <= 0.0 or not np.isfinite(largest):
        return []
    cutoff = max(
        float(rank_rtol) * largest,
        np.finfo(np.float64).eps * max(count, 1) * largest,
        float(tolerance) ** 2,
    )
    keep = eigvals > cutoff
    kept = int(np.count_nonzero(keep))
    if kept == 0:
        if verbose:
            print(
                f"[BASIS-RANK] candidates={count} kept=0 "
                f"lambda_max={largest:.3e} cutoff={cutoff:.3e}",
                flush=True,
            )
        return []

    U_keep = U[:, keep]
    lam_keep = eigvals[keep]
    transform = (U_keep / np.sqrt(lam_keep)[None, :]).T
    incoming = _apply_left_transform_in_place(
        incoming, transform, chunk_size=chunk_size
    )

    # One reduced re-normalization pass removes the rounding introduced when
    # the normalized fields are stored back as float32.
    G2 = _gpu_spatial_chunked_gram(incoming, chunk_size)
    e2, U2 = scipy_linalg.eigh(G2, check_finite=False)
    if float(e2[0]) <= 0.0:
        raise np.linalg.LinAlgError("rank-revealed basis lost positive Gram eigenvalue")
    correction = (U2 / np.sqrt(e2)[None, :]).T
    incoming = _apply_left_transform_in_place(
        incoming, correction, chunk_size=chunk_size
    )

    # Compact rejected rows so retained basis fields do not keep the larger
    # candidate allocation alive.
    if kept < count:
        compact = np.empty((kept, d), dtype=np.float32)
        effective = min(chunk_size, d)
        for start in range(0, d, effective):
            end = min(start + effective, d)
            compact[:, start:end] = incoming[:, start:end]
        del incoming
        incoming = compact
        gc.collect()

    appended = [incoming[i].reshape(6, nvox) for i in range(kept)]
    existing_basis.extend(appended)

    if verbose:
        gram_final = _gpu_spatial_chunked_gram(incoming, chunk_size)
        ortho_err = float(np.max(np.abs(gram_final - np.eye(kept))))
        cross_err = 0.0
        if len(existing_basis) > kept:
            old = existing_basis[:-kept]
            for start_old in range(0, len(old), 4):
                old_matrix = np.stack(
                    [np.asarray(v, dtype=np.float32).reshape(-1) for v in old[start_old:start_old+4]],
                    axis=0,
                )
                cross = _gpu_spatial_chunked_project(incoming, old_matrix, chunk_size)
                cross_err = max(cross_err, float(np.max(np.abs(cross))))
                del old_matrix
        print(
            f"[BASIS-RANK] candidates={count} kept={kept} rejected={count-kept} "
            f"lambda_min_kept={float(lam_keep[0]):.3e} cutoff={cutoff:.3e}",
            flush=True,
        )
        print(
            f"[TSQR-CHECKS] r={len(existing_basis)} | "
            f"new_gram_err={ortho_err:.2e} | old_new_cross={cross_err:.2e} | "
            "storage=float32 compute=float64",
            flush=True,
        )
    return appended

# ---------------------------------------------------------------------------
# Field solve and ordered extraction

# ---------------------------------------------------------------------------
# Field solve and ordered extraction
# ---------------------------------------------------------------------------

def solve_ordered_fields(
    *,
    run_dir: Path,
    geometry: common.GeometryData,
    runtime: dict[str, Any],
    material: dict[str, Any],
    seed: int,
    profile: str,
    basis_dtype: np.dtype,
    voxel_order: np.ndarray,
    quiet_solver: bool,
    initial_solution_fields: Any = None,
    return_raw_fields: bool = False,
    return_sensitivity_fields: bool = False,
    field_block_consumer: Callable[[np.ndarray], None] | None = None,
    sensitivity_block_consumer: Callable[[int, str, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]] | tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Solve the FOM for one material, returning reordered fields (6, 6, nvox)."""
    nvox = int(voxel_order.size)
    fields = np.empty((6, 6, nvox), dtype=basis_dtype)
    raw_fields = np.empty((6, 6, nvox), dtype=basis_dtype) if return_raw_fields else None
    field_loads_seen: set[int] = set()
    sensitivity_fields = (
        np.empty((len(reduced.COEFF_NAMES), 6, 6, nvox), dtype=basis_dtype)
        if return_sensitivity_fields
        else None
    )
    sensitivity_blocks: dict[int, np.ndarray] = {}
    sensitivity_loads_seen: dict[int, set[int]] = {}

    def consume(load_id: int, field: np.ndarray) -> None:
        arr = np.asarray(field).reshape(6, nvox)
        if return_raw_fields and raw_fields is not None:
            raw_fields[int(load_id)] = arr
        np.take(
            arr,
            voxel_order,
            axis=1,
            out=fields[int(load_id)],
        )
        field_loads_seen.add(int(load_id))
        if field_block_consumer is not None and field_loads_seen == set(range(6)):
            field_block_consumer(fields)

    def consume_sensitivity(
        coefficient_index: int,
        coefficient_name: str,
        load_id: int,
        field: np.ndarray,
    ) -> None:
        if sensitivity_fields is None and sensitivity_block_consumer is None:
            raise RuntimeError("sensitivity consumer called without storage.")
        arr = np.asarray(field).reshape(6, nvox)
        q = int(coefficient_index)
        iL = int(load_id)
        if return_sensitivity_fields and sensitivity_fields is not None:
            out = sensitivity_fields[q, iL]
        else:
            block = sensitivity_blocks.get(q)
            if block is None:
                block = np.empty((6, 6, nvox), dtype=basis_dtype)
                sensitivity_blocks[q] = block
                sensitivity_loads_seen[q] = set()
            out = block[iL]
        np.take(
            arr,
            voxel_order,
            axis=1,
            out=out,
        )
        if sensitivity_block_consumer is not None:
            sensitivity_loads_seen.setdefault(q, set()).add(iL)
            if sensitivity_loads_seen[q] == set(range(6)):
                block = sensitivity_blocks.pop(q)
                sensitivity_loads_seen.pop(q, None)
                sensitivity_block_consumer(q, str(coefficient_name), block)
                del block

    material_dir = run_dir / "training_truth" / f"material_{int(material['material_id']):04d}"
    with open(Path("/dev/null"), "w", encoding="utf-8") as sink:
        if quiet_solver:
            from contextlib import redirect_stderr, redirect_stdout
            with redirect_stdout(sink), redirect_stderr(sink):
                record = common.solve_material(
                    material_row=material,
                    material_dir=material_dir,
                    geometry=geometry,
                    runtime=runtime,
                    profile=profile,
                    seed=int(seed) + int(material["material_id"]),
                    save_solution_fields=True,
                    solution_field_dtype=basis_dtype,
                    solution_field_consumer=consume,
                    initial_solution_fields=initial_solution_fields,
                    solution_sensitivity_dtype=basis_dtype,
                    solution_sensitivity_consumer=(
                        consume_sensitivity
                        if return_sensitivity_fields or sensitivity_block_consumer is not None
                        else None
                    ),
                )
        else:
            record = common.solve_material(
                material_row=material,
                material_dir=material_dir,
                geometry=geometry,
                runtime=runtime,
                profile=profile,
                seed=int(seed) + int(material["material_id"]),
                save_solution_fields=True,
                solution_field_dtype=basis_dtype,
                solution_field_consumer=consume,
                initial_solution_fields=initial_solution_fields,
                solution_sensitivity_dtype=basis_dtype,
                solution_sensitivity_consumer=(
                    consume_sensitivity
                    if return_sensitivity_fields or sensitivity_block_consumer is not None
                    else None
                ),
            )
    if return_raw_fields and return_sensitivity_fields:
        if raw_fields is None or sensitivity_fields is None:
            raise RuntimeError("internal field storage was not allocated.")
        return fields, record, raw_fields, sensitivity_fields
    if return_raw_fields:
        return fields, record, raw_fields
    if return_sensitivity_fields:
        if sensitivity_fields is None:
            raise RuntimeError("internal sensitivity storage was not allocated.")
        return fields, record, sensitivity_fields
    return fields, record


# ---------------------------------------------------------------------------
# Sobol monitor sampling
# ---------------------------------------------------------------------------

def build_sobol_monitor_gammas(
    n_monitor: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return Sobol monitor gamma coefficients and material parameters."""
    materials = validate._build_independent_materials(n_monitor, seed)
    params = materials.loc[:, MATERIAL_NAMES].to_numpy(dtype=np.float64)
    gammas = reduced._material_coefficients_batch(params)
    return gammas, materials


def _rom_batch_for_monitor(
    gammas: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        evaluator = reduced.GpuAffineBatchEvaluator(Kq, Bq, Dq)
        return evaluator.evaluate(gammas, return_amplitudes=True)
    except Exception:
        return reduced._rom_ceff_batch(gammas, Kq, Bq, Dq)


def _truth_ceff_array(truth: pd.DataFrame) -> np.ndarray:
    values = np.empty((len(truth), 6, 6), dtype=np.float64)
    for row_index, (_, row) in enumerate(truth.iterrows()):
        values[row_index] = 0.5 * (
            reduced._full_ceff_from_row(row) + reduced._full_ceff_from_row(row).T
        )
    return values


def _validate_truth_frame(truth: pd.DataFrame) -> dict[str, Any]:
    """Validate convergence metadata without rejecting useful legacy truth."""
    info: dict[str, Any] = {"count": int(len(truth)), "high_precision": False}
    if "solver_all_converged" in truth.columns:
        if not bool(truth["solver_all_converged"].astype(bool).all()):
            raise RuntimeError("truth contains non-converged FOM rows")
    if "solver_real_dtype" in truth.columns:
        dtypes = set(truth["solver_real_dtype"].astype(str))
        info["solver_real_dtypes"] = sorted(dtypes)
        info["high_precision"] = dtypes == {"float64"}
    if "solver_rtol" in truth.columns:
        info["solver_rtol_max"] = float(truth["solver_rtol"].astype(float).max())
    return info


def _truth_matches_profile(truth: pd.DataFrame, profile: str) -> bool:
    if profile not in common.SOLVER_PROFILES:
        raise ValueError(f"unknown truth profile: {profile}")
    settings = common.SOLVER_PROFILES[profile]
    required_dtype = str(settings["solver_real_dtype"])
    required_rtol = float(settings["solver_rtol"])
    if "solver_real_dtype" not in truth.columns or "solver_rtol" not in truth.columns:
        return False
    if set(truth["solver_real_dtype"].astype(str)) != {required_dtype}:
        return False
    rtol = truth["solver_rtol"].astype(float).to_numpy()
    return bool(np.all(np.isclose(rtol, required_rtol, rtol=0.0, atol=1.0e-15)))


def _reference_materials(
    *,
    source_csv: Path,
    count: int,
    seed: int,
    label_prefix: str,
) -> pd.DataFrame:
    """Return the material set to validate against, independent of Ceff quality."""
    if source_csv.is_file():
        source = pd.read_csv(source_csv)
        missing = [name for name in MATERIAL_NAMES if name not in source.columns]
        if missing:
            raise KeyError(f"{source_csv} is missing material columns: {missing}")
        rows: list[dict[str, Any]] = []
        for idx, row in source.iterrows():
            sampled = {name: float(row[name]) for name in MATERIAL_NAMES}
            material = sweep._material_derived(sampled)
            sweep._validate_material(material)
            rows.append({
                "material_id": int(row.get("material_id", idx)),
                "material_label": str(row.get("material_label", f"{label_prefix}_{idx:04d}")),
                **material,
            })
        return pd.DataFrame(rows)[sweep.MATERIAL_COLUMNS].reset_index(drop=True)
    materials = validate._build_independent_materials(int(count), int(seed)).copy()
    if "material_label" in materials.columns:
        materials["material_label"] = [
            f"{label_prefix}_{idx:04d}" for idx in range(len(materials))
        ]
    return materials


def _solve_truth_materials(
    *,
    csv_path: Path,
    truth_root: Path,
    materials: pd.DataFrame,
    geometry: common.GeometryData,
    runtime: dict[str, Any],
    profile: str,
    seed: int,
    index_column: str,
    quiet_solver: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build or reuse FOM reference tensors with the requested solver profile."""
    if csv_path.is_file():
        cached = pd.read_csv(csv_path)
        if len(cached) == len(materials) and _truth_matches_profile(cached, profile):
            info = _validate_truth_frame(cached)
            info.update({
                "csv": str(csv_path),
                "profile": str(profile),
                "source": "run_cache",
            })
            return cached, info

    truth_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def solve_one(material: dict[str, Any]) -> dict[str, Any]:
        material_id = int(material["material_id"])
        material_dir = truth_root / f"material_{material_id:04d}"
        return common.solve_material(
            material_row=material,
            material_dir=material_dir,
            geometry=geometry,
            runtime=runtime,
            profile=profile,
            seed=int(seed) + material_id,
            save_solution_fields=False,
        )

    with open(Path("/dev/null"), "w", encoding="utf-8") as sink:
        for idx, material in enumerate(materials.to_dict(orient="records")):
            print(
                f"[ANCHOR-TANGENT-ADAPTIVE] truth {index_column}={idx} | "
                f"profile={profile} | material={int(material['material_id'])}",
                flush=True,
            )
            if quiet_solver:
                from contextlib import redirect_stderr, redirect_stdout

                with redirect_stdout(sink), redirect_stderr(sink):
                    record = solve_one(material)
            else:
                record = solve_one(material)
            rows.append({index_column: int(idx), **record})

    truth = pd.DataFrame(rows)
    _validate_truth_frame(truth)
    truth.to_csv(csv_path, index=False)
    info = _validate_truth_frame(truth)
    info.update({
        "csv": str(csv_path),
        "profile": str(profile),
        "source": "computed",
    })
    return truth, info


def evaluate_monitor_rom(
    gammas: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return amplitude indicator, ROM tensors and wall time."""
    C_rom, amplitudes, wall_s = _rom_batch_for_monitor(gammas, Kq, Bq, Dq)
    amplitude_norms = np.linalg.norm(amplitudes.reshape(len(amplitudes), -1), axis=1)
    return amplitude_norms, C_rom, wall_s


def evaluate_monitor_with_fom_truth(
    gammas: np.ndarray,
    materials: pd.DataFrame,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    truth: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate the *same* monitor materials used for greedy anchor selection."""
    C_rom, amplitudes, wall_s = _rom_batch_for_monitor(gammas, Kq, Bq, Dq)
    if truth is None:
        indicator = np.linalg.norm(amplitudes.reshape(len(amplitudes), -1), axis=1)
        return indicator, C_rom, wall_s
    if len(truth) != len(gammas):
        raise RuntimeError("truth rows and monitor gammas are not the same set")
    C_fom = _truth_ceff_array(truth)
    diff = C_rom - C_fom
    denom = np.maximum(
        np.linalg.norm(C_fom.reshape(len(C_fom), -1), axis=1),
        np.finfo(np.float64).eps,
    )
    errors = np.linalg.norm(diff.reshape(len(diff), -1), axis=1) / denom
    return errors, C_rom, wall_s

# ---------------------------------------------------------------------------
# Main adaptive pipeline

# ---------------------------------------------------------------------------
# Main adaptive pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-id", type=int, default=3)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")

    # Tolerance and limits
    parser.add_argument(
        "--tolerance", type=float, default=1.0e-4,
        help="Target max relative Frobenius error for monitor convergence.",
    )
    parser.add_argument(
        "--max-anchors", type=int, default=10,
        help="Safety limit on number of anchors.",
    )

    # Monitor and validation
    parser.add_argument("--monitor-count", type=int, default=512)
    parser.add_argument("--monitor-seed", type=int, default=20260822)
    parser.add_argument("--final-validation-count", type=int, default=16)
    parser.add_argument("--final-validation-seed", type=int, default=20260824)

    # Tangent settings
    parser.add_argument(
        "--tangent-method",
        choices=("fd", "sensitivity"),
        default="sensitivity",
        help="sensitivity solves exact discrete affine tangent equations; fd uses finite differences for diagnostics.",
    )
    parser.add_argument("--fd-mode", choices=("forward", "central"), default="forward",
                        help="forward = 8 solves/anchor (1 anchor + 7 tangents); central = 15 solves/anchor")
    parser.add_argument("--rel-step", type=float, default=1.0e-3)
    parser.add_argument("--min-rel-step", type=float, default=1.0e-5)
    parser.add_argument("--candidate-count", type=int, default=1024)
    parser.add_argument("--candidate-seed", type=int, default=20260821)

    # Solver settings
    parser.add_argument("--training-seed", type=int, default=20260821)
    parser.add_argument("--profile", choices=tuple(common.SOLVER_PROFILES), default="snapshot",
                        help="Anchor FOM profile. snapshot=float32/1e-5 is the mixed-precision default.")
    parser.add_argument("--tangent-profile", choices=tuple(common.SOLVER_PROFILES), default="snapshot",
                        help="FD-only tangent FOM profile. Ignored by --tangent-method=sensitivity.")
    parser.add_argument("--truth-profile", choices=tuple(common.SOLVER_PROFILES), default="reference",
                        help="Reference Ceff profile. Default uses float64/1e-6, not float64/1e-10.")
    parser.add_argument("--basis-dtype", choices=("float32", "float64"), default="float32",
                        help="Large field storage dtype. Reduced algebra remains float64.")
    parser.add_argument("--basis-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--fft-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--geometry-backend", choices=("numba", "cupy", "auto"), default="numba")
    parser.add_argument("--generator-cores", default="auto")
    parser.add_argument("--affine-q-block-size", type=int, default=1)
    parser.add_argument("--load-batch-size", type=int, default=6,
                        help="Number of load channels to solve in parallel on GPU (default: 6).")
    parser.add_argument(
        "--cleanup-training-truth",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--quiet-solver",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= int(args.geometry_id) <= 9:
        raise ValueError("geometry-id must be in [0, 9].")
    if float(args.tolerance) <= 0.0:
        raise ValueError("tolerance must be positive.")

    out_root = Path(args.out_root).resolve()
    geometry_dir = out_root / "geometries" / f"geometry_{int(args.geometry_id):02d}"
    geometry = common.load_fixed_geometry(geometry_dir)
    run_name = args.run_name or f"run_anchor_tangent_adaptive_geometry_{int(args.geometry_id):02d}"
    run_dir = out_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=bool(args.overwrite))

    cpu_info = common.cpu_resource_info()
    generator_cores = common.resolve_cpu_workers(args.generator_cores, resource_info=cpu_info)

    common.prepare_runtime(
        Path(sys.prefix),
        Path(__file__),
        marker_name="CMAME_ANCHOR_TANGENT_ADAPTIVE_CUDA_READY",
    )

    runtime = common.configure_runtime(
        geometry_backend=str(args.geometry_backend),
        generator_cores=generator_cores,
        solver_tol=float(common.SOLVER_PROFILES[str(args.profile)]["solver_rtol"]),
        fft_backend=str(args.fft_backend),
    )
    runtime["load_batch_size"] = int(args.load_batch_size)

    dtype = np.dtype(args.basis_dtype)
    tolerance = float(args.tolerance)
    max_anchors = int(args.max_anchors)

    # Deterministic voxel ordering for contiguous basis operations
    voxel_order = reduced.phase_orientation_voxel_order(geometry.phase, geometry.ori)
    operator_phase = geometry.phase.reshape(-1)[voxel_order]
    operator_ori = geometry.ori.reshape(-1, 3)[voxel_order]
    nvox = int(voxel_order.size)

    # Precompute gamma scales only when finite-difference tangent steps need them.
    scales = (
        gamma_scales(int(args.candidate_count), int(args.candidate_seed))
        if str(args.tangent_method) == "fd"
        else np.ones(len(reduced.COEFF_NAMES), dtype=np.float64)
    )

    # Affine stress factory (reused across all assemblies)
    affine = reduced.affine_stress_batch_factory(operator_phase, operator_ori)

    base_monitor_truth_csv = (
        out_root / "runs"
        / f"{BASE_RUN_STEM}_geometry_{int(args.geometry_id):02d}"
        / "monitor_truth_results.csv"
    )
    base_final_truth_csv = (
        out_root / "runs"
        / f"{BASE_RUN_STEM}_geometry_{int(args.geometry_id):02d}"
        / "final_validation_truth_results.csv"
    )
    monitor_truth_csv = run_dir / "monitor_truth_results.csv"
    final_truth_csv = run_dir / "final_validation_truth_results.csv"

    # Use one and the same material set for greedy selection and error
    # evaluation, but do not reuse the historical rom_floor Ceff as truth.
    # Its material set is fine; its Ceff tolerance is too loose for a 1e-4 ROM.
    monitor_materials = _reference_materials(
        source_csv=base_monitor_truth_csv,
        count=int(args.monitor_count),
        seed=int(args.monitor_seed),
        label_prefix="monitor_sobol",
    )
    monitor_truth, truth_info = _solve_truth_materials(
        csv_path=monitor_truth_csv,
        truth_root=run_dir / f"monitor_truth_{args.truth_profile}",
        materials=monitor_materials,
        geometry=geometry,
        runtime=runtime,
        profile=str(args.truth_profile),
        seed=int(args.monitor_seed),
        index_column="monitor_id",
        quiet_solver=bool(args.quiet_solver),
    )
    monitor_params = monitor_materials.loc[:, MATERIAL_NAMES].to_numpy(dtype=np.float64)
    monitor_gammas = reduced._material_coefficients_batch(monitor_params)
    print(
        f"[ANCHOR-TANGENT-ADAPTIVE] monitor=FOM-truth ({len(monitor_truth)} points) "
        f"profile={args.truth_profile} high_precision={bool(truth_info.get('high_precision', False))}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Orthonormal basis fields storage
    # -----------------------------------------------------------------------
    basis_fields: list[np.ndarray] = []

    # State tracking
    anchor_records: list[dict[str, Any]] = []
    iteration_records: list[dict[str, Any]] = []
    all_step_dfs: list[pd.DataFrame] = []
    operators: dict[str, np.ndarray] | None = None
    converged = False
    total_fom_solves = 0
    total_solve_wall_s = 0.0
    previous_monitor_C: np.ndarray | None = None
    previous_error_max: float | None = None

    print(
        f"[ANCHOR-TANGENT-ADAPTIVE] geometry={int(args.geometry_id):02d} "
        f"tol={tolerance:.2e} max_anchors={max_anchors} "
        f"monitor={int(args.monitor_count)}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Initial anchor: center of domain
    # -----------------------------------------------------------------------
    gamma_center = reduced._material_coefficients(center_material())
    # For the first iteration, use the center; for subsequent ones, use worst monitor point
    next_anchor_gamma = gamma_center.copy()
    next_anchor_label = "center"

    for anchor_idx in range(max_anchors):
        anchor_started = time.perf_counter()

        # -------------------------------------------------------------------
        # 1. Build anchor material from gamma
        # -------------------------------------------------------------------
        if anchor_idx == 0:
            anchor_material = center_material()
            anchor_material["material_id"] = anchor_idx
            anchor_material["material_label"] = f"anchor_{anchor_idx}_center"
        else:
            try:
                anchor_material = material_from_gamma(
                    next_anchor_gamma,
                    material_id=anchor_idx,
                    label=f"anchor_{anchor_idx}_{next_anchor_label}",
                )
            except ValueError as exc:
                print(
                    f"[ANCHOR-TANGENT-ADAPTIVE] WARNING: could not reconstruct "
                    f"material for anchor {anchor_idx}: {exc}. Stopping.",
                    flush=True,
                )
                break

        anchor_gamma = reduced._material_coefficients(anchor_material)
        print(
            f"[ANCHOR-TANGENT-ADAPTIVE] anchor {anchor_idx} | "
            f"label={anchor_material['material_label']}",
            flush=True,
        )

        # -------------------------------------------------------------------
        # 2. FOM solve for anchor
        # -------------------------------------------------------------------
        old_rank = len(basis_fields)
        appended_fields: list[np.ndarray] = []
        basis_wall_s = 0.0
        basis_streamed = str(args.tangent_method) == "sensitivity"

        if basis_streamed:
            def append_streamed_block(label: str, block: np.ndarray) -> None:
                nonlocal basis_wall_s
                started = time.perf_counter()
                new_fields = append_block_orthonormal(
                    existing_basis=basis_fields,
                    new_fields=[block],
                    tolerance=float(args.basis_tolerance),
                )
                appended_fields.extend(new_fields)
                basis_wall_s += float(time.perf_counter() - started)
                print(
                    f"[ANCHOR-TANGENT-ADAPTIVE]   streamed basis block "
                    f"{label}: +{len(new_fields)}",
                    flush=True,
                )

            def consume_anchor_block(block: np.ndarray) -> None:
                append_streamed_block("anchor", block)

            def consume_sensitivity_block(q: int, name: str, block: np.ndarray) -> None:
                append_streamed_block(f"d{name}/dgamma_{q}", block)

            anchor_fields, anchor_record = solve_ordered_fields(
                run_dir=run_dir,
                geometry=geometry,
                runtime=runtime,
                material=anchor_material,
                seed=int(args.training_seed),
                profile=str(args.profile),
                basis_dtype=dtype,
                voxel_order=voxel_order,
                quiet_solver=bool(args.quiet_solver),
                field_block_consumer=consume_anchor_block,
                sensitivity_block_consumer=consume_sensitivity_block,
            )
            anchor_raw_fields = None
            sensitivity_fields = None
        else:
            anchor_fields, anchor_record, anchor_raw_fields = solve_ordered_fields(
                run_dir=run_dir,
                geometry=geometry,
                runtime=runtime,
                material=anchor_material,
                seed=int(args.training_seed),
                profile=str(args.profile),
                basis_dtype=dtype,
                voxel_order=voxel_order,
                quiet_solver=bool(args.quiet_solver),
                return_raw_fields=True,
            )
            sensitivity_fields = None
        total_fom_solves += 1
        total_solve_wall_s += float(anchor_record.get("solve_wall_s", 0.0))
        print(
            f"[ANCHOR-TANGENT-ADAPTIVE]   anchor FOM solve: "
            f"{float(anchor_record.get('solve_wall_s', 0.0)):.2f}s",
            flush=True,
        )

        # -------------------------------------------------------------------
        # 3. Build tangent snapshots
        # -------------------------------------------------------------------
        tangent_blocks = [] if basis_streamed else [anchor_fields]  # anchor itself
        if str(args.tangent_method) == "sensitivity":
            step_df = pd.DataFrame([
                {
                    "anchor_id": int(anchor_idx),
                    "coefficient_index": int(q),
                    "coefficient_name": str(name),
                    "gamma0": float(anchor_gamma[q]),
                    "tangent_method": "sensitivity",
                    "fd_mode": "",
                    "signed_step": 0.0,
                }
                for q, name in enumerate(reduced.COEFF_NAMES)
            ])
            step_df.to_csv(
                run_dir / f"anchor_{anchor_idx}_sensitivity_tangents.csv",
                index=False,
            )
            all_step_dfs.append(step_df)
            gc.collect()
        else:
            fd_mode = str(args.fd_mode)
            perturbed_materials, step_df = _build_tangent_materials(
                gamma0=anchor_gamma,
                scales=scales,
                rel_step=float(args.rel_step),
                min_rel_step=float(args.min_rel_step),
                anchor_id=anchor_idx,
                fd_mode=fd_mode,
            )
            step_df.to_csv(
                run_dir / f"anchor_{anchor_idx}_fd_steps.csv", index=False,
            )
            all_step_dfs.append(step_df)

            perturbed_fields: dict[int, np.ndarray] = {}
            for mat in perturbed_materials:
                print(
                    f"[ANCHOR-TANGENT-ADAPTIVE]   FOM tangent | "
                    f"material={int(mat['material_id'])} {mat['material_label']}",
                    flush=True,
                )
                fields, record = solve_ordered_fields(
                    run_dir=run_dir,
                    geometry=geometry,
                    runtime=runtime,
                    material=mat,
                    seed=int(args.training_seed),
                    profile=str(args.tangent_profile or args.profile),
                    basis_dtype=dtype,
                    voxel_order=voxel_order,
                    quiet_solver=bool(args.quiet_solver),
                    initial_solution_fields=anchor_raw_fields,
                )
                perturbed_fields[int(mat["material_id"])] = fields
                total_fom_solves += 1
                total_solve_wall_s += float(record.get("solve_wall_s", 0.0))
            gc.collect()

            for row in step_df.to_dict(orient="records"):
                q = int(row["coefficient_index"])
                if fd_mode == "forward":
                    step = float(row["signed_step"])
                    plus_id = 1000 * anchor_idx + q
                    plus_fields = perturbed_fields[plus_id]
                    tangent = (plus_fields - anchor_fields) / step
                else:
                    step = float(row["signed_plus_step"])
                    plus_id = 1000 * anchor_idx + 2 * q
                    minus_id = 1000 * anchor_idx + 2 * q + 1
                    plus_fields = perturbed_fields[plus_id]
                    minus_fields = perturbed_fields[minus_id]
                    tangent = (plus_fields - minus_fields) / (2.0 * step)
                tangent_blocks.append(tangent)

            del perturbed_fields
            gc.collect()

        # -------------------------------------------------------------------
        # 4. Enrich the basis using double-pass MGS
        # -------------------------------------------------------------------
        if not basis_streamed:
            basis_started = time.perf_counter()
            appended_fields = append_block_orthonormal(
                existing_basis=basis_fields,
                new_fields=tangent_blocks,
                tolerance=float(args.basis_tolerance),
            )
            basis_wall_s = float(time.perf_counter() - basis_started)
        new_rank = len(basis_fields)
        print(
            f"[ANCHOR-TANGENT-ADAPTIVE]   basis: {old_rank} → {new_rank} "
            f"(+{new_rank - old_rank}) in {basis_wall_s:.2f}s",
            flush=True,
        )
        if new_rank == old_rank:
            print(
                "[ANCHOR-TANGENT-ADAPTIVE]   WARNING: anchor added zero new "
                "basis vectors. Stopping enrichment.",
                flush=True,
            )
            break

        # -------------------------------------------------------------------
        # 6. Assemble / extend reduced operators
        # -------------------------------------------------------------------
        assembly_started = time.perf_counter()
        if operators is None:
            # Full assembly from scratch
            operators, assembly_meta = common.update_reduced_operators(
                phase=operator_phase,
                ori=operator_ori,
                basis=basis_fields,
                existing=None,
                new_fields=basis_fields,
                affine_stress_batch=affine,
                affine_q_block_size=int(args.affine_q_block_size),
            )
        else:
            # Incremental extension
            operators, assembly_meta = common.update_reduced_operators(
                phase=operator_phase,
                ori=operator_ori,
                basis=basis_fields,
                existing=operators,
                new_fields=appended_fields,
                affine_stress_batch=affine,
                affine_q_block_size=int(args.affine_q_block_size),
            )
        assembly_wall_s = float(time.perf_counter() - assembly_started)
        print(
            f"[ANCHOR-TANGENT-ADAPTIVE]   assembly: {assembly_wall_s:.2f}s",
            flush=True,
        )

        # -------------------------------------------------------------------
        # 7. Evaluate monitor set (ROM-only: cheap)
        # -------------------------------------------------------------------
        Kq = operators["Kq"]
        Bq = operators["Bq"]
        Dq = operators["Dq"]

        monitor_errors, monitor_C, monitor_wall_s = evaluate_monitor_with_fom_truth(
            monitor_gammas, monitor_materials, Kq, Bq, Dq, monitor_truth,
        )

        # Nested Ritz spaces must make C_r decrease in Loewner order.  This
        # check is independent of FOM truth and immediately catches corrupted
        # basis/assembly/update paths.
        monotonic_min_eig_rel = 0.0
        if previous_monitor_C is not None:
            delta = previous_monitor_C - monitor_C
            delta = 0.5 * (delta + np.swapaxes(delta, -1, -2))
            eig_delta = np.linalg.eigvalsh(delta)
            scale = np.maximum(
                np.max(np.abs(np.linalg.eigvalsh(previous_monitor_C)), axis=1), 1.0
            )
            monotonic_min_eig_rel = float(np.min(eig_delta[:, 0] / scale))
            if monotonic_min_eig_rel < -5.0e-7:
                raise RuntimeError(
                    "Nested Ritz monotonicity violated: "
                    f"min_eig(C_prev-C_new)/scale={monotonic_min_eig_rel:.3e}. "
                    "Stop before G09; basis or reduced assembly is inconsistent."
                )
        previous_monitor_C = monitor_C.copy()

        error_max = float(np.max(monitor_errors))
        error_mean = float(np.mean(monitor_errors))
        error_p95 = float(np.quantile(monitor_errors, 0.95))
        worst_idx = int(np.argmax(monitor_errors))

        anchor_wall_s = float(time.perf_counter() - anchor_started)
        has_truth = monitor_truth is not None
        status = "CONVERGED" if has_truth and error_max <= tolerance else "iterating"

        iteration_record = {
            "anchor_id": anchor_idx,
            "anchor_label": anchor_material["material_label"],
            "basis_rank": new_rank,
            "basis_rank_added": new_rank - old_rank,
            "error_max": error_max,
            "error_mean": error_mean,
            "error_p95": error_p95,
            "worst_monitor_idx": worst_idx,
            "total_fom_solves": total_fom_solves,
            "total_solve_wall_s": total_solve_wall_s,
            "anchor_wall_s": anchor_wall_s,
            "assembly_wall_s": assembly_wall_s,
            "basis_wall_s": basis_wall_s,
            "status": status,
            "monitor_has_fom_truth": bool(monitor_truth is not None),
            "ritz_monotonic_min_eig_rel": monotonic_min_eig_rel,
        }
        iteration_records.append(iteration_record)
        anchor_records.append(anchor_material)

        # Save iteration curve
        pd.DataFrame(iteration_records).to_csv(
            run_dir / "adaptive_convergence_curve.csv", index=False,
        )

        print(
            f"[ANCHOR-TANGENT-ADAPTIVE] ── anchor={anchor_idx} r={new_rank} "
            f"fom_solves={total_fom_solves} "
            f"mean={error_mean:.3e} p95={error_p95:.3e} max={error_max:.3e} "
            f"[{status}]",
            flush=True,
        )

        if monitor_truth is not None and error_max <= tolerance:
            converged = True
            break

        # -------------------------------------------------------------------
        # 8. Select next anchor: worst monitor point
        # -------------------------------------------------------------------
        next_anchor_gamma = monitor_gammas[worst_idx].copy()
        next_anchor_label = f"worst_monitor_{worst_idx}"

    # -----------------------------------------------------------------------
    # Save final reduced operators
    # -----------------------------------------------------------------------
    if operators is not None:
        save_dict = {
            "Kq": operators["Kq"],
            "Bq": operators["Bq"],
            "Dq": operators["Dq"],
            "coefficient_names": np.asarray(reduced.COEFF_NAMES),
        }
        if "raw_Kq" in operators:
            save_dict.update({
                "raw_Kq": operators["raw_Kq"],
                "raw_Bq": operators["raw_Bq"],
                "G": operators["G"],
                "invR": operators["invR"],
            })
        np.savez_compressed(run_dir / "reduced_operators.npz", **save_dict)

    # -----------------------------------------------------------------------
    # Final independent FOM validation with the same requested reference profile.
    # -----------------------------------------------------------------------
    final_summary: dict[str, Any] | None = None
    final_truth_info: dict[str, Any] | None = None
    if operators is not None:
        final_materials = _reference_materials(
            source_csv=base_final_truth_csv,
            count=int(args.final_validation_count),
            seed=int(args.final_validation_seed),
            label_prefix="final_validation_sobol",
        )
        final_truth, final_truth_info = _solve_truth_materials(
            csv_path=final_truth_csv,
            truth_root=run_dir / f"final_validation_truth_{args.truth_profile}",
            materials=final_materials,
            geometry=geometry,
            runtime=runtime,
            profile=str(args.truth_profile),
            seed=int(args.final_validation_seed),
            index_column="final_validation_id",
            quiet_solver=bool(args.quiet_solver),
        )
        final_rom = reduced._evaluate_rom(
            results_df=final_truth,
            Kq=operators["Kq"],
            Bq=operators["Bq"],
            Dq=operators["Dq"],
        )
        final_rom.to_csv(run_dir / "final_validation_rom_results.csv", index=False)
        final_errors = final_rom["relative_frobenius_error"].to_numpy(dtype=float)
        final_summary = {
            "count": int(len(final_errors)),
            "error_mean": float(np.mean(final_errors)),
            "error_median": float(np.median(final_errors)),
            "error_p95": float(np.quantile(final_errors, 0.95)),
            "error_max": float(np.max(final_errors)),
        }
        print(
            f"[ANCHOR-TANGENT-ADAPTIVE] final validation: "
            f"mean={final_summary['error_mean']:.3e} "
            f"p95={final_summary['error_p95']:.3e} "
            f"max={final_summary['error_max']:.3e}",
            flush=True,
        )

    # Also evaluate monitor ROM results if operators exist
    monitor_summary: dict[str, Any] | None = None
    if operators is not None:
        monitor_rom = reduced._evaluate_rom(
            results_df=monitor_truth,
            Kq=operators["Kq"],
            Bq=operators["Bq"],
            Dq=operators["Dq"],
        )
        monitor_rom.to_csv(run_dir / "monitor_rom_results.csv", index=False)
        monitor_errors_final = monitor_rom["relative_frobenius_error"].to_numpy(dtype=float)
        monitor_summary = {
            "count": int(len(monitor_errors_final)),
            "error_mean": float(np.mean(monitor_errors_final)),
            "error_p95": float(np.quantile(monitor_errors_final, 0.95)),
            "error_max": float(np.max(monitor_errors_final)),
        }

    # -----------------------------------------------------------------------
    # Sobol-POD reference for comparison
    # -----------------------------------------------------------------------
    sobol_reference: dict[str, Any] | None = None
    sobol_curve_path = (
        out_root / "runs"
        / f"{BASE_RUN_STEM}_geometry_{int(args.geometry_id):02d}"
        / "sobol_pod_error_curve.csv"
    )
    if sobol_curve_path.is_file() and operators is not None:
        try:
            curve = pd.read_csv(sobol_curve_path).sort_values("pod_rank")
            exact = curve.loc[curve["pod_rank"] == len(basis_fields)]
            if not exact.empty:
                ref_row = exact.iloc[0]
            else:
                ref_row = curve.iloc[
                    int(np.argmin(np.abs(curve["pod_rank"].to_numpy() - len(basis_fields))))
                ]
            sobol_reference = {
                "reference_rank": int(ref_row["pod_rank"]),
                "reference_training_materials": int(ref_row["training_materials"]),
                "reference_monitor_error_max": float(ref_row["monitor_error_max"]),
            }
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Summary manifest
    # -----------------------------------------------------------------------
    n_anchors = len(anchor_records)
    if str(args.tangent_method) == "sensitivity":
        method_note = (
            "Adaptive greedy anchor selection with exact discrete affine "
            "sensitivity tangents. Anchors are chosen by worst ROM monitor "
            "error until tolerance is met."
        )
        solves_per_anchor = (
            f"1 anchor FOM + {len(reduced.COEFF_NAMES)} same-operator "
            "sensitivity RHS blocks"
        )
    else:
        method_note = (
            "Adaptive greedy anchor selection with finite-difference gamma "
            "tangents. Anchors are chosen by worst ROM monitor error until "
            "tolerance is met."
        )
        fd_tangent_count = (
            len(reduced.COEFF_NAMES)
            if args.fd_mode == "forward"
            else 2 * len(reduced.COEFF_NAMES)
        )
        solves_per_anchor = (
            f"1 anchor + {fd_tangent_count} tangent FD = "
            f"{1 + fd_tangent_count} per anchor"
        )
    summary = {
        "method": "anchor_tangent_adaptive_greedy",
        "note": method_note,
        "run_dir": str(run_dir),
        "geometry_id": int(args.geometry_id),
        "geometry_label": str(geometry.manifest.get("geometry_label", geometry_dir.name)),
        "converged": converged,
        "tolerance": tolerance,
        "max_anchors": max_anchors,
        "n_anchors": n_anchors,
        "basis_rank": int(operators["Kq"].shape[1]) if operators is not None else len(basis_fields),
        "basis_storage_fields": len(basis_fields),
        "total_fom_solves": total_fom_solves,
        "total_solve_wall_s": total_solve_wall_s,
        "fom_solves_per_anchor": solves_per_anchor,
        "monitor_count": int(args.monitor_count),
        "monitor_seed": int(args.monitor_seed),
        "monitor_summary": monitor_summary,
        "final_validation_summary": final_summary,
        "sobol_reference_near_same_rank": sobol_reference,
        "anchors": [
            {
                "anchor_id": int(rec.get("material_id", idx)),
                "label": str(rec.get("material_label", "")),
            }
            for idx, rec in enumerate(anchor_records)
        ],
        "iteration_records": iteration_records,
        "basis_dtype": str(dtype),
        "reduced_compute_dtype": "float64",
        "solver_profile": str(args.profile),
        "tangent_solver_profile": str(args.tangent_profile or args.profile),
        "tangent_method": str(args.tangent_method),
        "fd_mode": str(args.fd_mode) if str(args.tangent_method) == "fd" else "",
        "truth_profile_requested": str(args.truth_profile),
        "monitor_truth_info": truth_info,
        "final_truth_info": final_truth_info,
        "monitor_truth_csv": str(monitor_truth_csv),
        "final_validation_truth_csv": str(final_truth_csv),
        "legacy_monitor_material_source_csv": str(base_monitor_truth_csv),
        "legacy_final_material_source_csv": str(base_final_truth_csv),
        "fft_backend": str(args.fft_backend),
        "gamma_fd_rel_step_requested": (
            float(args.rel_step) if str(args.tangent_method) == "fd" else None
        ),
    }
    write_json(run_dir / "adaptive_summary.json", summary)

    # Save anchor materials
    if anchor_records:
        pd.DataFrame(anchor_records).to_csv(
            run_dir / "anchor_materials.csv", index=False,
        )

    # Concatenate all step DataFrames
    if all_step_dfs:
        all_tangent_steps = pd.concat(all_step_dfs, ignore_index=True)
        all_tangent_steps.to_csv(run_dir / "all_tangent_steps.csv", index=False)
        if str(args.tangent_method) == "fd":
            all_tangent_steps.to_csv(run_dir / "all_fd_steps.csv", index=False)

    # Cleanup large training truth fields if requested
    if bool(args.cleanup_training_truth):
        shutil.rmtree(run_dir / "training_truth", ignore_errors=True)

    # -----------------------------------------------------------------------
    # Final banner
    # -----------------------------------------------------------------------
    print(
        f"[ANCHOR-TANGENT-ADAPTIVE] {'CONVERGED' if converged else 'MAX-ANCHORS-REACHED'} | "
        f"anchors={n_anchors} | r={len(basis_fields)} | "
        f"fom_solves={total_fom_solves} | "
        f"out={run_dir}",
        flush=True,
    )
    if final_summary:
        print(
            f"[ANCHOR-TANGENT-ADAPTIVE] final_max={final_summary['error_max']:.3e} "
            f"final_p95={final_summary['error_p95']:.3e}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
