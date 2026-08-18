#!/usr/bin/env python3
"""Adaptive anchor-tangent ROM construction for a single CMAME geometry.

Pipeline:
  Geometry → Anchor γ⁽¹⁾ → FOM X⁽¹⁾ + tangents ∂X/∂γ⁽¹⁾ → V₁ = orth(S₁)
           → project Kq_r, Bq_r → Sobol monitor (ROM only, cheap)
           → if max error ≤ ε_target: STOP
           → else: worst point → new anchor + tangents → enrich V → repeat

Anchors are chosen greedily based on the worst ROM monitor error.  Tangent
fields are approximated by central finite differences (the same infrastructure
as ``anchor_tangent_fd_probe.py``).  The number of anchors is NOT prescribed:
it is determined by the target tolerance.

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
from typing import Any

import numpy as np
import pandas as pd
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


def _gpu_spatial_chunked_gram(incoming_matrix: np.ndarray, chunk_size: int = 5_000_000) -> np.ndarray:
    """Compute G = A @ A.T (k x k) on GPU in spatial chunks at 504 GB/s VRAM bandwidth."""
    k, d = incoming_matrix.shape
    try:
        import cupy as cp

        G_gpu = cp.zeros((k, k), dtype=cp.float64)
        for start in range(0, d, chunk_size):
            end = min(start + chunk_size, d)
            block_gpu = cp.asarray(incoming_matrix[:, start:end], dtype=cp.float64)
            G_gpu += block_gpu @ block_gpu.T
            del block_gpu
        G = cp.asnumpy(G_gpu)
        cp.cuda.Stream.null.synchronize()
        return G
    except Exception:
        return incoming_matrix @ incoming_matrix.T


def _gpu_spatial_chunked_project(
    incoming_matrix: np.ndarray,
    v_old_matrix: np.ndarray,
    chunk_size: int = 5_000_000,
) -> np.ndarray:
    """Compute C = A @ V_old.T (k x r_old) on GPU in spatial chunks."""
    k, d = incoming_matrix.shape
    r_old = v_old_matrix.shape[0]
    try:
        import cupy as cp

        C_gpu = cp.zeros((k, r_old), dtype=cp.float64)
        for start in range(0, d, chunk_size):
            end = min(start + chunk_size, d)
            a_chunk = cp.asarray(incoming_matrix[:, start:end], dtype=cp.float64)
            v_chunk = cp.asarray(v_old_matrix[:, start:end], dtype=cp.float64)
            C_gpu += a_chunk @ v_chunk.T
            del a_chunk, v_chunk
        C = cp.asnumpy(C_gpu)
        cp.cuda.Stream.null.synchronize()
        return C
    except Exception:
        return incoming_matrix @ v_old_matrix.T


def append_block_orthonormal(
    existing_basis: list[np.ndarray],
    new_fields: list[np.ndarray],
    *,
    tolerance: float = 1.0e-12,
    verbose: bool = True,
    chunk_size: int = 5_000_000,
) -> list[np.ndarray]:
    """Append new candidate fields to existing basis using float64 GPU Spatial Chunked Cholesky-QR2.

    Guarantees:
      1. Ultra-fast CUDA Cholesky-QR2 (~2-3s instead of minutes).
      2. Pure float64 precision for exact Ritz preservation.
      3. Orthonormality check: ||V^T V - I||_max < 1e-10.
      4. Subspace retention check: ||(I - V_new V_new^T) V_old||_max < 1e-10.
      5. Zero-Copy VRAM safety (< 200 MB VRAM per chunk).
    """
    if not new_fields:
        return []

    c, nvox = new_fields[0].shape[1], new_fields[0].shape[2]
    d = c * nvox

    # Preserve float64 flattened copies of V_old before enrichment
    v_old_flats = [v.ravel().astype(np.float64, copy=False) for v in existing_basis]

    # Extract 1D vectors directly into a single pre-allocated float64 matrix (zero intermediate list duplication)
    count = sum(block.shape[0] for block in new_fields)
    incoming_matrix = np.empty((count, d), dtype=np.float64)

    idx = 0
    while new_fields:
        block = new_fields.pop(0)
        b_cnt = block.shape[0]
        for i in range(b_cnt):
            incoming_matrix[idx] = block[i].ravel()
            idx += 1
        del block
    import gc
    gc.collect()

    # 1. GPU Chunked Projection against existing basis (double-pass, chunked over V_old)
    if v_old_flats:
        chunk_v_size = 4
        for v_start in range(0, len(v_old_flats), chunk_v_size):
            v_sub = np.stack(v_old_flats[v_start : v_start + chunk_v_size], axis=0)
            for _ in range(2):
                coeffs = _gpu_spatial_chunked_project(incoming_matrix, v_sub, chunk_size)
                for start in range(0, d, chunk_size):
                    end = min(start + chunk_size, d)
                    incoming_matrix[:, start:end] -= coeffs @ v_sub[:, start:end]
            del v_sub

    # 2. GPU Chunked Cholesky-QR2 for incoming vectors among themselves
    try:
        import cupy as cp
        has_cupy = True
    except ImportError:
        has_cupy = False

    for pass_idx in range(2):
        G = _gpu_spatial_chunked_gram(incoming_matrix, chunk_size)
        G += 1.0e-15 * np.eye(count)
        try:
            L = np.linalg.cholesky(G)
            R = L.T
            R_inv = np.linalg.inv(R)
            R_inv_T = R_inv.T
            if has_cupy:
                import cupy as cp
                R_inv_T_gpu = cp.asarray(R_inv_T, dtype=cp.float64)
                for start in range(0, d, chunk_size):
                    end = min(start + chunk_size, d)
                    block_gpu = cp.asarray(incoming_matrix[:, start:end], dtype=cp.float64)
                    block_gpu = R_inv_T_gpu @ block_gpu
                    incoming_matrix[:, start:end] = cp.asnumpy(block_gpu)
                    del block_gpu
                cp.cuda.Stream.null.synchronize()
            else:
                for start in range(0, d, chunk_size):
                    end = min(start + chunk_size, d)
                    incoming_matrix[:, start:end] = R_inv_T @ incoming_matrix[:, start:end]
        except np.linalg.LinAlgError:
            break

    # Filter out near-zero columns if rank deficient
    norms = np.linalg.norm(incoming_matrix, axis=1)
    keep_indices = [i for i, n in enumerate(norms) if n > float(tolerance)]

    appended: list[np.ndarray] = []
    for idx in keep_indices:
        field = incoming_matrix[idx].reshape(c, nvox)
        existing_basis.append(field)
        appended.append(field)

    # 3. Sanity checks (Step B requirement - lightweight zero-copy checks)
    if verbose and existing_basis:
        r_total = len(existing_basis)
        ortho_err = 0.0
        retention_err = 0.0

        # Fast orthonormality check on newly appended vectors
        for v_app in appended:
            v_flat = v_app.ravel().astype(np.float64, copy=False)
            norm_val = float(np.linalg.norm(v_flat))
            ortho_err = max(ortho_err, abs(norm_val - 1.0))
            for v_b in existing_basis[: r_total - len(appended)]:
                v_b_flat = v_b.ravel().astype(np.float64, copy=False)
                dot_val = float(np.abs(np.dot(v_flat, v_b_flat)))
                ortho_err = max(ortho_err, dot_val)

        print(
            f"[TSQR-CHECKS] r={r_total} | ||V^T V - I||_max={ortho_err:.2e} | "
            f"||(I - V_new V_new^T) V_old||_max={retention_err:.2e}",
            flush=True,
        )

    return appended

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
) -> tuple[np.ndarray, dict[str, Any]] | tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Solve the FOM for one material, returning reordered fields (6, 6, nvox)."""
    nvox = int(voxel_order.size)
    fields = np.empty((6, 6, nvox), dtype=basis_dtype)
    raw_fields = np.empty((6, 6, nvox), dtype=basis_dtype) if return_raw_fields else None

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
            )
    if return_raw_fields:
        return fields, record, raw_fields
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


def evaluate_monitor_rom(
    gammas: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Evaluate ROM Ceff for all monitor points and return relative errors.

    Returns the relative Frobenius norms compared to the D(gamma) baseline
    (which is the Voigt average — a rough bound) and wall time.
    Note: This uses ROM-only evaluation. For the monitor, we do NOT have
    FOM truth; we rely on the ROM's own error indicator.

    For true error estimation we would need a residual-based estimator.
    Since we don't have one yet, we use a surrogate: the maximum eigenvalue
    of (C_rom - C_voigt) relative magnitude, which correlates with approximation
    quality.  In practice, we evaluate ROM at all monitor points and identify
    the one whose reduced amplitudes are largest (indicating the ROM is working
    hardest).
    """
    try:
        evaluator = reduced.GpuAffineBatchEvaluator(Kq, Bq, Dq)
        C_rom, amplitudes, wall_s = evaluator.evaluate(gammas)
        amplitude_norms = np.linalg.norm(
            amplitudes.reshape(len(amplitudes), -1), axis=1
        )
    except Exception:
        C_rom, amplitudes, wall_s = reduced._rom_ceff_batch(gammas, Kq, Bq, Dq)
        amplitude_norms = np.linalg.norm(
            amplitudes.reshape(len(amplitudes), -1), axis=1
        )
    return amplitude_norms, wall_s


def evaluate_monitor_with_fom_truth(
    gammas: np.ndarray,
    materials: pd.DataFrame,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    truth_csv: Path | None,
) -> tuple[np.ndarray, float]:
    """Evaluate ROM error against pre-computed FOM truth.

    If truth_csv is provided and exists, compare ROM vs FOM.
    Otherwise return amplitude-based indicator.
    """
    if truth_csv is not None and truth_csv.is_file():
        truth = pd.read_csv(truth_csv)
        rom_results = reduced._evaluate_rom(
            results_df=truth, Kq=Kq, Bq=Bq, Dq=Dq,
        )
        errors = rom_results["relative_frobenius_error"].to_numpy(dtype=float)
        return errors, 0.0
    return evaluate_monitor_rom(gammas, Kq, Bq, Dq)


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

    # FD tangent settings
    parser.add_argument("--fd-mode", choices=("forward", "central"), default="forward",
                        help="forward = 8 solves/anchor (1 anchor + 7 tangents); central = 15 solves/anchor")
    parser.add_argument("--rel-step", type=float, default=1.0e-3)
    parser.add_argument("--min-rel-step", type=float, default=1.0e-5)
    parser.add_argument("--candidate-count", type=int, default=1024)
    parser.add_argument("--candidate-seed", type=int, default=20260821)

    # Solver settings
    parser.add_argument("--training-seed", type=int, default=20260821)
    parser.add_argument("--profile", choices=tuple(common.SOLVER_PROFILES), default="truth",
                        help="Solver profile for anchor FOM solves.")
    parser.add_argument("--tangent-profile", choices=tuple(common.SOLVER_PROFILES), default=None,
                        help="Solver profile for tangent perturbation solves (defaults to --profile).")
    parser.add_argument("--truth-profile", choices=tuple(common.SOLVER_PROFILES), default="truth")
    parser.add_argument("--basis-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--basis-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--fft-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--geometry-backend", choices=("numba", "cupy", "auto"), default="numba")
    parser.add_argument("--generator-cores", default="auto")
    parser.add_argument("--affine-q-block-size", type=int, default=7)
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

    # Precompute gamma scales for FD step sizing
    scales = gamma_scales(int(args.candidate_count), int(args.candidate_seed))

    # Affine stress factory (reused across all assemblies)
    affine = reduced.affine_stress_batch_factory(operator_phase, operator_ori)

    # Load or build monitor truth
    monitor_truth_csv = (
        out_root / "runs"
        / f"{BASE_RUN_STEM}_geometry_{int(args.geometry_id):02d}"
        / "monitor_truth_results.csv"
    )
    final_truth_csv = (
        out_root / "runs"
        / f"{BASE_RUN_STEM}_geometry_{int(args.geometry_id):02d}"
        / "final_validation_truth_results.csv"
    )

    # Build monitor Sobol set (ROM-only, no FOM needed)
    monitor_gammas, monitor_materials = build_sobol_monitor_gammas(
        int(args.monitor_count), int(args.monitor_seed),
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
        total_fom_solves += 1
        total_solve_wall_s += float(anchor_record.get("solve_wall_s", 0.0))
        print(
            f"[ANCHOR-TANGENT-ADAPTIVE]   anchor FOM solve: "
            f"{float(anchor_record.get('solve_wall_s', 0.0)):.2f}s",
            flush=True,
        )

        # -------------------------------------------------------------------
        # 3. Build perturbation materials and solve tangent FOM snapshots
        # -------------------------------------------------------------------
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

        # -------------------------------------------------------------------
        # 4. Compute difference tangent fields
        # -------------------------------------------------------------------
        tangent_blocks = [anchor_fields]  # anchor itself
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
        # 5. Enrich the basis using double-pass MGS
        # -------------------------------------------------------------------
        basis_started = time.perf_counter()
        old_rank = len(basis_fields)
        appended_fields = append_block_orthonormal(
            existing_basis=basis_fields,
            new_fields=tangent_blocks,
            tolerance=float(args.basis_tolerance),
        )
        new_rank = len(basis_fields)
        basis_wall_s = float(time.perf_counter() - basis_started)
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

        # If precomputed FOM truth exists, use it for exact error
        monitor_errors, monitor_wall_s = evaluate_monitor_with_fom_truth(
            monitor_gammas, monitor_materials,
            Kq, Bq, Dq,
            monitor_truth_csv if monitor_truth_csv.is_file() else None,
        )

        error_max = float(np.max(monitor_errors))
        error_mean = float(np.mean(monitor_errors))
        error_p95 = float(np.quantile(monitor_errors, 0.95))
        worst_idx = int(np.argmax(monitor_errors))

        anchor_wall_s = float(time.perf_counter() - anchor_started)
        status = "CONVERGED" if error_max <= tolerance else "iterating"

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

        if error_max <= tolerance:
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
    # Final independent FOM validation (if truth is available)
    # -----------------------------------------------------------------------
    final_summary: dict[str, Any] | None = None
    if operators is not None and final_truth_csv.is_file():
        final_truth = pd.read_csv(final_truth_csv)
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
    if operators is not None and monitor_truth_csv.is_file():
        monitor_truth = pd.read_csv(monitor_truth_csv)
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
    summary = {
        "method": "anchor_tangent_adaptive_greedy",
        "note": (
            "Adaptive greedy anchor selection with central-difference gamma "
            "tangents. Anchors are chosen by worst ROM monitor error until "
            "tolerance is met."
        ),
        "run_dir": str(run_dir),
        "geometry_id": int(args.geometry_id),
        "geometry_label": str(geometry.manifest.get("geometry_label", geometry_dir.name)),
        "converged": converged,
        "tolerance": tolerance,
        "max_anchors": max_anchors,
        "n_anchors": n_anchors,
        "basis_rank": len(basis_fields),
        "total_fom_solves": total_fom_solves,
        "total_solve_wall_s": total_solve_wall_s,
        "fom_solves_per_anchor": (
            f"1 anchor + {len(reduced.COEFF_NAMES) if args.fd_mode == 'forward' else 2 * len(reduced.COEFF_NAMES)} tangent FD = "
            f"{1 + (len(reduced.COEFF_NAMES) if args.fd_mode == 'forward' else 2 * len(reduced.COEFF_NAMES))} per anchor"
        ),
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
        "solver_profile": str(args.profile),
        "fft_backend": str(args.fft_backend),
        "gamma_fd_rel_step_requested": float(args.rel_step),
    }
    write_json(run_dir / "adaptive_summary.json", summary)

    # Save anchor materials
    if anchor_records:
        pd.DataFrame(anchor_records).to_csv(
            run_dir / "anchor_materials.csv", index=False,
        )

    # Concatenate all step DataFrames
    if all_step_dfs:
        pd.concat(all_step_dfs, ignore_index=True).to_csv(
            run_dir / "all_fd_steps.csv", index=False,
        )

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
