#!/usr/bin/env python3
"""Diagnostic: verify Ritz nesting, monotonicity, and anchor reproduction.

This script runs 3 anchors and checks the three critical properties:

  TEST 1 — Nesting:   ||(I - V_{k+1} V_{k+1}^T) V_k||_F ≈ 0
  TEST 2 — Monotonicity: H_FOM ≤ H_{k+1} ≤ H_k  (eigenvalue ordering)
  TEST 3 — Anchor reproduction: e(anchor) ≈ 0 after enrichment

Everything runs in float64 with FULL assembly (no incremental) to isolate
numerical precision from implementation correctness.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

# Reuse helpers from the adaptive script
from cmame_anchor_tangent_adaptive import (
    center_material,
    material_from_gamma,
    gamma_scales,
    _build_tangent_materials,
    solve_ordered_fields,
    write_json,
)

OUT_ROOT = ROOT / "results" / "cmame_method" / "interpretable_vf05_25_ar5_20"
BASE_RUN_STEM = "run_sobol_pod_20260815_010726"
MATERIAL_NAMES = tuple(sweep.MATERIAL_BOUNDS)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def basis_nesting_residual(
    V_old_flat: np.ndarray,
    V_new_flat: np.ndarray,
) -> float:
    """Compute ||(I - V_new V_new^T) V_old||_F / ||V_old||_F.

    If V_old ⊂ span(V_new), this should be ≈ 0.
    """
    # V_new is (r_new, d), V_old is (r_old, d)
    # Project V_old onto V_new: coeffs = V_old @ V_new^T, projected = coeffs @ V_new
    coeffs = V_old_flat @ V_new_flat.T  # (r_old, r_new)
    projected = coeffs @ V_new_flat  # (r_old, d)
    residual = V_old_flat - projected
    return float(np.linalg.norm(residual) / max(np.linalg.norm(V_old_flat), 1e-30))


def ritz_monotonicity_check(
    C_rom_sequence: list[np.ndarray],
    C_fom: np.ndarray,
    label: str,
) -> dict[str, Any]:
    """Check H_FOM ≤ H_{k+1} ≤ H_k for a material point.

    C_rom_sequence[0] = C_rom at rank r_0
    C_rom_sequence[1] = C_rom at rank r_1 > r_0
    ...
    C_fom = full-order model result

    Returns diagnostic dict with eigenvalue information.
    """
    result: dict[str, Any] = {"label": label, "passes": True}
    checks: list[dict[str, Any]] = []

    # Check C_rom_k - C_fom ≥ 0 (positive semidefinite) for all k
    for k, C_k in enumerate(C_rom_sequence):
        diff = C_k - C_fom
        diff_sym = 0.5 * (diff + diff.T)
        eigs = np.linalg.eigvalsh(diff_sym)
        check = {
            "rank_index": k,
            "min_eig_Crom_minus_Cfom": float(eigs[0]),
            "max_eig_Crom_minus_Cfom": float(eigs[-1]),
            "psd": bool(eigs[0] >= -1e-10),
        }
        if not check["psd"]:
            result["passes"] = False
        checks.append(check)

    # Check C_rom_k - C_rom_{k+1} ≥ 0 for consecutive pairs
    for k in range(len(C_rom_sequence) - 1):
        diff = C_rom_sequence[k] - C_rom_sequence[k + 1]
        diff_sym = 0.5 * (diff + diff.T)
        eigs = np.linalg.eigvalsh(diff_sym)
        check = {
            "transition": f"rank_{k}_minus_rank_{k+1}",
            "min_eig": float(eigs[0]),
            "max_eig": float(eigs[-1]),
            "monotone": bool(eigs[0] >= -1e-10),
        }
        if not check["monotone"]:
            result["passes"] = False
        checks.append(check)

    result["checks"] = checks
    return result


def anchor_reproduction_error(
    anchor_gamma: np.ndarray,
    anchor_ceff_fom: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> float:
    """After enriching with anchor's fields, the ROM should reproduce its Ceff exactly."""
    C_rom, _, _ = reduced._rom_ceff(anchor_gamma, Kq, Bq, Dq)
    C_fom = 0.5 * (anchor_ceff_fom + anchor_ceff_fom.T)
    return float(np.linalg.norm(C_rom - C_fom) / max(np.linalg.norm(C_fom), 1e-30))


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-id", type=int, default=3)
    parser.add_argument("--n-anchors", type=int, default=3,
                        help="Number of anchors for the diagnostic (keep small).")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--rel-step", type=float, default=2.0e-2)
    parser.add_argument("--min-rel-step", type=float, default=1.0e-5)
    parser.add_argument("--candidate-count", type=int, default=1024)
    parser.add_argument("--candidate-seed", type=int, default=20260821)
    parser.add_argument("--training-seed", type=int, default=20260821)
    parser.add_argument("--monitor-seed", type=int, default=20260822)
    parser.add_argument("--profile", choices=tuple(common.SOLVER_PROFILES), default="snapshot")
    parser.add_argument("--basis-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--fft-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--geometry-backend", choices=("numba", "cupy", "auto"), default="numba")
    parser.add_argument("--generator-cores", default="auto")
    parser.add_argument("--affine-q-block-size", type=int, default=7)
    parser.add_argument("--quiet-solver", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    geometry_dir = out_root / "geometries" / f"geometry_{int(args.geometry_id):02d}"
    geometry = common.load_fixed_geometry(geometry_dir)
    run_name = args.run_name or f"run_ritz_diagnostic_geometry_{int(args.geometry_id):02d}"
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

    # CRITICAL: Use float64 throughout for the diagnostic
    dtype = np.dtype(np.float64)
    basis_tolerance = float(args.basis_tolerance)

    voxel_order = reduced.phase_orientation_voxel_order(geometry.phase, geometry.ori)
    operator_phase = geometry.phase.reshape(-1)[voxel_order]
    operator_ori = geometry.ori.reshape(-1, 3)[voxel_order]
    nvox = int(voxel_order.size)

    scales = gamma_scales(int(args.candidate_count), int(args.candidate_seed))
    affine = reduced.affine_stress_batch_factory(operator_phase, operator_ori)

    # Load monitor FOM truth
    monitor_truth_csv = (
        out_root / "runs"
        / f"{BASE_RUN_STEM}_geometry_{int(args.geometry_id):02d}"
        / "monitor_truth_results.csv"
    )
    if not monitor_truth_csv.is_file():
        print(f"[DIAG] ERROR: monitor truth not found at {monitor_truth_csv}", flush=True)
        return 1
    monitor_truth = pd.read_csv(monitor_truth_csv)

    # Precompute FOM Ceff for all monitor points
    n_monitor = len(monitor_truth)
    C_fom_monitor = np.empty((n_monitor, 6, 6), dtype=np.float64)
    gamma_monitor = np.empty((n_monitor, len(reduced.COEFF_NAMES)), dtype=np.float64)
    for idx, (_, row) in enumerate(monitor_truth.iterrows()):
        C_fom_monitor[idx] = reduced._full_ceff_from_row(row)
        C_fom_monitor[idx] = 0.5 * (C_fom_monitor[idx] + C_fom_monitor[idx].T)
        gamma_monitor[idx] = reduced._material_coefficients(row.to_dict())

    print(
        f"[DIAG] geometry={int(args.geometry_id):02d} n_anchors={int(args.n_anchors)} "
        f"dtype=float64 monitor_points={n_monitor}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Collect all anchor fields first, then build bases incrementally
    # -----------------------------------------------------------------------
    n_anchors = int(args.n_anchors)
    anchor_gammas: list[np.ndarray] = []
    anchor_ceffs: list[np.ndarray] = []
    # tangent_blocks[a] = array (n_fields, 6, nvox) for anchor a
    all_tangent_blocks: list[np.ndarray] = []

    # First anchor: center
    gamma_center = reduced._material_coefficients(center_material())
    next_anchor_gamma = gamma_center.copy()
    next_anchor_label = "center"

    for a in range(n_anchors):
        if a == 0:
            anchor_mat = center_material()
            anchor_mat["material_id"] = a
            anchor_mat["material_label"] = f"anchor_{a}_center"
        else:
            try:
                anchor_mat = material_from_gamma(
                    next_anchor_gamma,
                    material_id=a,
                    label=f"anchor_{a}_{next_anchor_label}",
                )
            except ValueError as exc:
                print(f"[DIAG] Cannot build anchor {a}: {exc}", flush=True)
                break

        anchor_gamma = reduced._material_coefficients(anchor_mat)
        anchor_gammas.append(anchor_gamma)

        # Solve anchor FOM
        print(f"[DIAG] solving anchor {a}: {anchor_mat['material_label']}", flush=True)
        anchor_fields, anchor_record = solve_ordered_fields(
            run_dir=run_dir,
            geometry=geometry,
            runtime=runtime,
            material=anchor_mat,
            seed=int(args.training_seed),
            profile=str(args.profile),
            basis_dtype=dtype,
            voxel_order=voxel_order,
            quiet_solver=bool(args.quiet_solver),
        )

        # Extract Ceff from the solve record
        ceff = np.load(Path(anchor_record["Ceff_path"]))
        anchor_ceffs.append(0.5 * (ceff + ceff.T))

        # Solve tangent FOM snapshots
        perturbed_mats, step_df = _build_tangent_materials(
            gamma0=anchor_gamma,
            scales=scales,
            rel_step=float(args.rel_step),
            min_rel_step=float(args.min_rel_step),
            anchor_id=a,
        )

        perturbed_fields: dict[int, np.ndarray] = {}
        for mat in perturbed_mats:
            fields, _ = solve_ordered_fields(
                run_dir=run_dir,
                geometry=geometry,
                runtime=runtime,
                material=mat,
                seed=int(args.training_seed),
                profile=str(args.profile),
                basis_dtype=dtype,
                voxel_order=voxel_order,
                quiet_solver=bool(args.quiet_solver),
            )
            perturbed_fields[int(mat["material_id"])] = fields
        gc.collect()

        # Build central FD tangent block
        blocks = [anchor_fields]
        for row in step_df.to_dict(orient="records"):
            q = int(row["coefficient_index"])
            step = float(row["signed_plus_step"])
            plus_id = 1000 * a + 2 * q
            minus_id = 1000 * a + 2 * q + 1
            tangent = (perturbed_fields[plus_id] - perturbed_fields[minus_id]) / (2.0 * step)
            blocks.append(tangent)

        tangent_block = np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype=dtype)
        all_tangent_blocks.append(tangent_block)
        del perturbed_fields
        gc.collect()
        print(f"[DIAG] anchor {a}: {tangent_block.shape[0]} candidate fields", flush=True)

        # For greedy anchor selection: quick ROM evaluation to find worst point
        # (We'll do a proper evaluation below, but we need the next anchor gamma)
        if a < n_anchors - 1:
            # Build temporary basis and operators for greedy selection
            temp_fields = np.concatenate(all_tangent_blocks, axis=0)
            temp_flat = temp_fields.reshape(len(temp_fields), -1)
            # Simple QR orthogonalization in float64
            Q, R = np.linalg.qr(temp_flat.T, mode="reduced")
            r_diag = np.abs(np.diag(R))
            keep = r_diag > basis_tolerance
            temp_basis = Q[:, keep].T.reshape(-1, 6, nvox)

            Kq_t, Bq_t, Dq_t, _ = reduced._assemble_reduced_operators(
                phase=operator_phase,
                ori=operator_ori,
                basis=temp_basis,
                affine_stress_batch=affine,
            )
            rom_results = reduced._evaluate_rom(
                results_df=monitor_truth, Kq=Kq_t, Bq=Bq_t, Dq=Dq_t,
            )
            errors = rom_results["relative_frobenius_error"].to_numpy(dtype=float)
            worst_idx = int(np.argmax(errors))
            next_anchor_gamma = gamma_monitor[worst_idx].copy()
            next_anchor_label = f"worst_monitor_{worst_idx}"
            del temp_fields, temp_flat, Q, R, temp_basis, Kq_t, Bq_t, Dq_t
            gc.collect()

    actual_anchors = len(all_tangent_blocks)

    # -----------------------------------------------------------------------
    # Now build bases incrementally and run the three diagnostics
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70, flush=True)
    print("   DIAGNOSTIC TESTS", flush=True)
    print("=" * 70, flush=True)

    # We'll build V_0, V_1, V_2 by successively appending tangent blocks
    # using QR in float64 (NOT ContiguousBasis which uses the dtype).
    bases: list[np.ndarray] = []  # bases[k] = (r_k, d) flat basis

    C_rom_by_rank: list[list[np.ndarray]] = [[] for _ in range(n_monitor)]
    # C_rom_by_rank[j][k] = C_rom(monitor_j) at rank step k

    all_errors: list[dict[str, Any]] = []
    nesting_results: list[dict[str, Any]] = []
    anchor_repro_results: list[dict[str, Any]] = []
    monotonicity_results: list[dict[str, Any]] = []

    operators_per_step: list[dict[str, np.ndarray]] = []

    for k in range(actual_anchors):
        print(f"\n--- Anchor step k={k} ---", flush=True)

        # Accumulate all fields up to this anchor
        accumulated = np.concatenate(all_tangent_blocks[:k + 1], axis=0)
        flat = accumulated.reshape(len(accumulated), -1)

        # Orthogonalize via QR in float64
        Q, R = np.linalg.qr(flat.T, mode="reduced")
        r_diag = np.abs(np.diag(R))
        keep = r_diag > basis_tolerance
        V_flat = Q[:, keep].T  # (r_k, d) in float64
        rank_k = int(V_flat.shape[0])
        V_fields = V_flat.reshape(rank_k, 6, nvox)
        bases.append(V_flat.copy())

        print(f"[DIAG] rank at step {k}: {rank_k}", flush=True)

        # ----- TEST 1: Nesting check -----
        if k > 0:
            V_old = bases[k - 1]
            eta_nested = basis_nesting_residual(V_old, V_flat)
            nesting_ok = eta_nested < 1e-10
            nesting_results.append({
                "step": k,
                "eta_nested": eta_nested,
                "passes": nesting_ok,
            })
            status = "✓ PASS" if nesting_ok else "✗ FAIL"
            print(
                f"  TEST 1 (nesting): ||(I-V_new V_new^T) V_old||_F / ||V_old||_F "
                f"= {eta_nested:.3e}  [{status}]",
                flush=True,
            )

        # ----- Full assembly (NOT incremental) -----
        t0 = time.perf_counter()
        Kq, Bq, Dq, meta = reduced._assemble_reduced_operators(
            phase=operator_phase,
            ori=operator_ori,
            basis=V_fields,
            affine_stress_batch=affine,
        )
        assembly_s = time.perf_counter() - t0
        print(f"  Full assembly: {assembly_s:.2f}s (rank={rank_k})", flush=True)
        operators_per_step.append({"Kq": Kq.copy(), "Bq": Bq.copy(), "Dq": Dq.copy()})

        # ----- Evaluate ROM on all monitor points -----
        rom_results = reduced._evaluate_rom(
            results_df=monitor_truth, Kq=Kq, Bq=Bq, Dq=Dq,
        )
        errors = rom_results["relative_frobenius_error"].to_numpy(dtype=float)
        print(
            f"  Monitor errors: mean={np.mean(errors):.3e} "
            f"p95={np.quantile(errors, 0.95):.3e} max={np.max(errors):.3e}",
            flush=True,
        )

        # Collect C_rom for each monitor point
        C_rom_batch, _, _ = reduced._rom_ceff_batch(gamma_monitor, Kq, Bq, Dq)
        for j in range(n_monitor):
            C_rom_by_rank[j].append(C_rom_batch[j].copy())

        all_errors.append({
            "step": k,
            "rank": rank_k,
            "error_mean": float(np.mean(errors)),
            "error_p95": float(np.quantile(errors, 0.95)),
            "error_max": float(np.max(errors)),
            "worst_monitor_idx": int(np.argmax(errors)),
        })

        # ----- TEST 3: Anchor reproduction -----
        # Check that the anchor material whose fields are IN the basis
        # is reproduced to near machine precision
        for a_idx in range(k + 1):
            e_anchor = anchor_reproduction_error(
                anchor_gammas[a_idx],
                anchor_ceffs[a_idx],
                Kq, Bq, Dq,
            )
            anchor_repro_results.append({
                "step": k,
                "anchor": a_idx,
                "reproduction_error": e_anchor,
                "passes": e_anchor < 1e-6,
            })
            status = "✓ PASS" if e_anchor < 1e-6 else "✗ FAIL"
            print(
                f"  TEST 3 (anchor {a_idx} reproduction): "
                f"e = {e_anchor:.3e}  [{status}]",
                flush=True,
            )

        del accumulated, flat, Q, R, V_fields
        gc.collect()

    # ----- TEST 2: Monotonicity check -----
    print(f"\n--- TEST 2: Ritz monotonicity (per-material eigenvalue check) ---", flush=True)
    n_fail_mono = 0
    n_total_mono = 0
    worst_mono_violation = 0.0
    for j in range(n_monitor):
        result = ritz_monotonicity_check(
            C_rom_by_rank[j], C_fom_monitor[j],
            label=f"monitor_{j}",
        )
        monotonicity_results.append(result)
        if not result["passes"]:
            n_fail_mono += 1
            for check in result["checks"]:
                if "min_eig" in check and not check.get("monotone", True):
                    worst_mono_violation = min(worst_mono_violation, check["min_eig"])
                if "min_eig_Crom_minus_Cfom" in check and not check.get("psd", True):
                    worst_mono_violation = min(
                        worst_mono_violation, check["min_eig_Crom_minus_Cfom"]
                    )
        n_total_mono += 1

    mono_status = "✓ PASS" if n_fail_mono == 0 else "✗ FAIL"
    print(
        f"  Monotonicity: {n_total_mono - n_fail_mono}/{n_total_mono} pass  "
        f"[{mono_status}]",
        flush=True,
    )
    if n_fail_mono > 0:
        print(
            f"  Worst eigenvalue violation: {worst_mono_violation:.3e}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Also compare: full assembly vs what ContiguousBasis + update_reduced_operators would give
    # -----------------------------------------------------------------------
    print(f"\n--- BONUS: Compare full assembly vs ContiguousBasis pipeline ---", flush=True)

    # Rebuild using ContiguousBasis in float32 (like the original adaptive script)
    basis_f32 = common.ContiguousBasis(
        capacity=500,
        field_shape=(6, nvox),
        dtype=np.float32,
        projection_row_block_size=6,
    )
    ops_incr = None
    for k in range(actual_anchors):
        block_f32 = np.ascontiguousarray(all_tangent_blocks[k], dtype=np.float32)
        old_rank = int(basis_f32.rank)
        basis_f32.append_preordered(block_f32, tolerance=basis_tolerance)
        new_rank = int(basis_f32.rank)
        print(
            f"  ContiguousBasis step {k}: {old_rank} → {new_rank} (+{new_rank - old_rank})",
            flush=True,
        )

        if ops_incr is None:
            ops_incr, _ = common.update_reduced_operators(
                phase=operator_phase,
                ori=operator_ori,
                basis=basis_f32.active_fields,
                existing=None,
                new_fields=basis_f32.active_fields,
                affine_stress_batch=affine,
                affine_q_block_size=int(args.affine_q_block_size),
            )
        else:
            new_fields = basis_f32.active_fields[old_rank:new_rank]
            ops_incr, _ = common.update_reduced_operators(
                phase=operator_phase,
                ori=operator_ori,
                basis=basis_f32.active_fields,
                existing=ops_incr,
                new_fields=new_fields,
                affine_stress_batch=affine,
                affine_q_block_size=int(args.affine_q_block_size),
            )

        # Evaluate with incremental operators
        rom_incr = reduced._evaluate_rom(
            results_df=monitor_truth,
            Kq=ops_incr["Kq"],
            Bq=ops_incr["Bq"],
            Dq=ops_incr["Dq"],
        )
        errors_incr = rom_incr["relative_frobenius_error"].to_numpy(dtype=float)

        # Compare with full-assembly float64 result
        ref_errors = all_errors[k]
        print(
            f"    float64 full-assembly:   mean={ref_errors['error_mean']:.3e} "
            f"max={ref_errors['error_max']:.3e}",
            flush=True,
        )
        print(
            f"    float32 incremental:     mean={np.mean(errors_incr):.3e} "
            f"max={np.max(errors_incr):.3e}",
            flush=True,
        )

        # Also do full assembly on the float32 basis for comparison
        Kq_full32, Bq_full32, Dq_full32, _ = reduced._assemble_reduced_operators(
            phase=operator_phase,
            ori=operator_ori,
            basis=basis_f32.active_fields,
            affine_stress_batch=affine,
        )
        rom_full32 = reduced._evaluate_rom(
            results_df=monitor_truth,
            Kq=Kq_full32,
            Bq=Bq_full32,
            Dq=Dq_full32,
        )
        errors_full32 = rom_full32["relative_frobenius_error"].to_numpy(dtype=float)
        print(
            f"    float32 full-assembly:   mean={np.mean(errors_full32):.3e} "
            f"max={np.max(errors_full32):.3e}",
            flush=True,
        )

        # Nesting check for float32 basis
        if k > 0:
            V_f32 = basis_f32.active_flat[:old_rank].astype(np.float64)
            V_f32_new = basis_f32.active_flat[:new_rank].astype(np.float64)
            eta = basis_nesting_residual(V_f32, V_f32_new)
            print(f"    float32 nesting residual: {eta:.3e}", flush=True)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("   FINAL SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print("\nError evolution (float64 full assembly):", flush=True)
    for rec in all_errors:
        print(
            f"  step={rec['step']} r={rec['rank']} "
            f"mean={rec['error_mean']:.3e} max={rec['error_max']:.3e} "
            f"worst_idx={rec['worst_monitor_idx']}",
            flush=True,
        )

    n_nest_fail = sum(1 for r in nesting_results if not r["passes"])
    n_repro_fail = sum(1 for r in anchor_repro_results if not r["passes"])
    print(f"\nTEST 1 (nesting):      {len(nesting_results) - n_nest_fail}/{len(nesting_results)} pass", flush=True)
    print(f"TEST 2 (monotonicity): {n_total_mono - n_fail_mono}/{n_total_mono} pass", flush=True)
    print(f"TEST 3 (anchor repro): {len(anchor_repro_results) - n_repro_fail}/{len(anchor_repro_results)} pass", flush=True)

    # Save results
    summary = {
        "geometry_id": int(args.geometry_id),
        "n_anchors": actual_anchors,
        "dtype": "float64",
        "assembly_mode": "full (not incremental)",
        "error_evolution": all_errors,
        "nesting_results": nesting_results,
        "anchor_reproduction_results": anchor_repro_results,
        "monotonicity_failures": n_fail_mono,
        "monotonicity_total": n_total_mono,
        "worst_monotonicity_violation": worst_mono_violation,
        "all_tests_pass": n_nest_fail == 0 and n_fail_mono == 0 and n_repro_fail == 0,
    }
    write_json(run_dir / "ritz_diagnostic_summary.json", summary)
    print(f"\nResults saved to: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
