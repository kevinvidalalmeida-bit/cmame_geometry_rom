#!/usr/bin/env python3
"""Pilot anchor+tangent basis test for one CMAME geometry.

This is intentionally an experiment, not the production Sobol-POD path.
It approximates affine-gamma response tangents with one-sided finite
differences because the current FFTHomPy wrapper does not expose arbitrary
right-hand-side solves with a frozen K(gamma0).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
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


OUT_ROOT = ROOT / "results" / "cmame_method" / "interpretable_vf05_25_ar5_20"
BASE_RUN_STEM = "run_sobol_pod_20260815_010726"
MATERIAL_NAMES = tuple(sweep.MATERIAL_BOUNDS)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(common.jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def center_material() -> dict[str, Any]:
    sampled = {
        name: 0.5 * (bounds[0] + bounds[1])
        for name, bounds in sweep.MATERIAL_BOUNDS.items()
    }
    material = sweep._material_derived(sampled)
    sweep._validate_material(material)
    return {"material_id": 0, "material_label": "gamma_anchor_center", **material}


def local_fiber_stiffness_from_gamma(gamma: np.ndarray) -> np.ndarray:
    c_tt, c_tt_cross, c_lt, c_ll, g_lt = np.asarray(gamma, dtype=float)[2:]
    C = np.zeros((6, 6), dtype=np.float64)
    C[0, 0] = c_ll
    C[0, 1] = C[1, 0] = c_lt
    C[0, 2] = C[2, 0] = c_lt
    C[1, 1] = C[2, 2] = c_tt
    C[1, 2] = C[2, 1] = c_tt_cross
    C[3, 3] = c_tt - c_tt_cross
    C[4, 4] = C[5, 5] = 2.0 * g_lt
    return C


def gamma_to_material(gamma: np.ndarray, *, material_id: int, label: str) -> dict[str, Any]:
    gamma = np.asarray(gamma, dtype=np.float64)
    lam_m, mu_m = gamma[:2]
    if mu_m <= 0.0 or lam_m + mu_m <= 0.0:
        raise ValueError("invalid isotropic matrix affine coefficients")
    Em = mu_m * (3.0 * lam_m + 2.0 * mu_m) / (lam_m + mu_m)
    nu_m = lam_m / (2.0 * (lam_m + mu_m))

    C_f = local_fiber_stiffness_from_gamma(gamma)
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


def material_parameters_frame(count: int, seed: int) -> pd.DataFrame:
    frame = validate._build_independent_materials(int(count), int(seed)).copy()
    if "material_id" not in frame:
        frame.insert(0, "material_id", np.arange(len(frame), dtype=int))
    return frame


def gamma_scales(candidate_count: int, seed: int) -> np.ndarray:
    candidates = material_parameters_frame(candidate_count, seed)
    params = candidates.loc[:, MATERIAL_NAMES].to_numpy(dtype=np.float64)
    gammas = reduced._material_coefficients_batch(params)
    span = np.ptp(gammas, axis=0)
    center = np.abs(reduced._material_coefficients(center_material()))
    return np.maximum(span, np.maximum(center, 1.0))


def tangent_materials(
    *,
    gamma0: np.ndarray,
    candidate_count: int,
    candidate_seed: int,
    rel_step: float,
    min_rel_step: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scales = gamma_scales(candidate_count, candidate_seed)
    rows = [center_material()]
    step_rows: list[dict[str, Any]] = []
    for q, name in enumerate(reduced.COEFF_NAMES):
        accepted: dict[str, Any] | None = None
        used_step = float(rel_step) * float(scales[q])
        used_sign = 1
        while abs(used_step) >= float(min_rel_step) * float(scales[q]):
            for sign in (1, -1):
                trial = gamma0.copy()
                trial[q] += sign * used_step
                try:
                    accepted = gamma_to_material(
                        trial,
                        material_id=q + 1,
                        label=f"gamma_fd_{name}_{'plus' if sign > 0 else 'minus'}",
                    )
                except ValueError:
                    continue
                used_sign = sign
                break
            if accepted is not None:
                break
            used_step *= 0.5
        if accepted is None:
            raise RuntimeError(f"Could not build a valid gamma perturbation for {name}.")
        rows.append(accepted)
        step_rows.append(
            {
                "coefficient_index": q,
                "coefficient_name": name,
                "gamma0": float(gamma0[q]),
                "scale": float(scales[q]),
                "signed_step": float(used_sign * used_step),
                "relative_step_used": float(abs(used_step) / float(scales[q])),
                "material_id": int(accepted["material_id"]),
                "material_label": str(accepted["material_label"]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(step_rows)


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
) -> tuple[np.ndarray, dict[str, Any]]:
    nvox = int(voxel_order.size)
    fields = np.empty((6, 6, nvox), dtype=basis_dtype)

    def consume(load_id: int, field: np.ndarray) -> None:
        np.take(
            np.asarray(field).reshape(6, nvox),
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
            )
    return fields, record


def append_ceff_columns(frame: pd.DataFrame, records: list[dict[str, Any]]) -> pd.DataFrame:
    records_by_id = {int(record["material_id"]): record for record in records}
    out = frame.copy()
    for col in ("solve_wall_s", "solver_max_relative_residual", "solver_max_iterations"):
        out[col] = [records_by_id[int(mid)].get(col, np.nan) for mid in out["material_id"]]
    return out


def relative_stats(frame: pd.DataFrame) -> dict[str, float | int]:
    errors = frame["relative_frobenius_error"].to_numpy(dtype=float)
    return {
        "count": int(len(errors)),
        "error_mean": float(np.mean(errors)),
        "error_median": float(np.median(errors)),
        "error_p95": float(np.quantile(errors, 0.95)),
        "error_max": float(np.max(errors)),
        "worst_material_id": int(frame.iloc[int(np.argmax(errors))]["material_id"]),
        "rom_online_mean_s": float(frame["rom_online_s"].mean()),
    }


def sobol_reference_row(geometry_id: int, rank: int) -> dict[str, Any]:
    curve_path = (
        OUT_ROOT
        / "runs"
        / f"{BASE_RUN_STEM}_geometry_{int(geometry_id):02d}"
        / "sobol_pod_error_curve.csv"
    )
    curve = pd.read_csv(curve_path).sort_values("pod_rank")
    exact = curve.loc[curve["pod_rank"] == int(rank)]
    if not exact.empty:
        row = exact.iloc[0]
    else:
        row = curve.iloc[int(np.argmin(np.abs(curve["pod_rank"].to_numpy() - rank)))]
    return {
        "reference_curve": str(curve_path),
        "reference_training_materials": int(row["training_materials"]),
        "reference_rank": int(row["pod_rank"]),
        "reference_monitor_error_max": float(row["monitor_error_max"]),
        "reference_snapshot_step_wall_s": float(row["snapshot_step_wall_s"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-id", type=int, default=3)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rel-step", type=float, default=2.0e-2)
    parser.add_argument("--min-rel-step", type=float, default=1.0e-5)
    parser.add_argument("--candidate-count", type=int, default=1024)
    parser.add_argument("--candidate-seed", type=int, default=20260821)
    parser.add_argument("--training-seed", type=int, default=20260821)
    parser.add_argument("--monitor-seed", type=int, default=20260822)
    parser.add_argument("--final-validation-seed", type=int, default=20260824)
    parser.add_argument("--profile", choices=tuple(common.SOLVER_PROFILES), default="snapshot")
    parser.add_argument("--truth-profile", choices=tuple(common.SOLVER_PROFILES), default="snapshot")
    parser.add_argument("--basis-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--basis-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--fft-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--geometry-backend", choices=("numba", "cupy", "auto"), default="numba")
    parser.add_argument("--generator-cores", default="auto")
    parser.add_argument("--affine-q-block-size", type=int, default=7)
    parser.add_argument("--cleanup-training-truth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet-solver", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= int(args.geometry_id) <= 9:
        raise ValueError("geometry-id must be in [0, 9].")
    if float(args.rel_step) <= 0.0:
        raise ValueError("rel-step must be positive.")

    out_root = Path(args.out_root).resolve()
    geometry_dir = out_root / "geometries" / f"geometry_{int(args.geometry_id):02d}"
    geometry = common.load_fixed_geometry(geometry_dir)
    run_name = args.run_name or f"run_anchor_tangent_fd_geometry_{int(args.geometry_id):02d}"
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

    dtype = np.dtype(args.basis_dtype)
    gamma0 = reduced._material_coefficients(center_material())
    materials, steps = tangent_materials(
        gamma0=gamma0,
        candidate_count=int(args.candidate_count),
        candidate_seed=int(args.candidate_seed),
        rel_step=float(args.rel_step),
        min_rel_step=float(args.min_rel_step),
    )
    materials.to_csv(run_dir / "anchor_tangent_materials.csv", index=False)
    steps.to_csv(run_dir / "gamma_fd_steps.csv", index=False)

    voxel_order = reduced.phase_orientation_voxel_order(geometry.phase, geometry.ori)
    operator_phase = geometry.phase.reshape(-1)[voxel_order]
    operator_ori = geometry.ori.reshape(-1, 3)[voxel_order]
    nvox = int(voxel_order.size)

    solve_records: list[dict[str, Any]] = []
    solved_fields: dict[int, np.ndarray] = {}
    solve_started = time.perf_counter()
    for material in materials.to_dict(orient="records"):
        print(
            "[ANCHOR-TANGENT] FOM snapshot | "
            f"geometry=g{int(args.geometry_id):02d} | "
            f"material={int(material['material_id'])} {material['material_label']}",
            flush=True,
        )
        fields, record = solve_ordered_fields(
            run_dir=run_dir,
            geometry=geometry,
            runtime=runtime,
            material=material,
            seed=int(args.training_seed),
            profile=str(args.profile),
            basis_dtype=dtype,
            voxel_order=voxel_order,
            quiet_solver=bool(args.quiet_solver),
        )
        solved_fields[int(material["material_id"])] = fields
        solve_records.append(record)
    solve_wall_s = float(time.perf_counter() - solve_started)

    anchor = solved_fields[0]
    blocks = [anchor]
    for row in steps.to_dict(orient="records"):
        material_id = int(row["material_id"])
        step = float(row["signed_step"])
        blocks.append((solved_fields[material_id] - anchor) / step)
    tangent_block = np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype=dtype)

    basis_started = time.perf_counter()
    basis = common.ContiguousBasis(
        capacity=int(tangent_block.shape[0]),
        field_shape=(6, nvox),
        dtype=dtype,
        projection_row_block_size=6,
    )
    appended = basis.append_preordered(tangent_block, tolerance=float(args.basis_tolerance))
    basis_wall_s = float(time.perf_counter() - basis_started)
    if len(appended) < 1:
        raise RuntimeError("Anchor+tangent block produced an empty reduced basis.")

    affine = reduced.affine_stress_batch_factory(operator_phase, operator_ori)
    assembly_started = time.perf_counter()
    operators, assembly_meta = common.update_reduced_operators(
        phase=operator_phase,
        ori=operator_ori,
        basis=basis.active_fields,
        existing=None,
        new_fields=basis.active_fields,
        affine_stress_batch=affine,
        affine_q_block_size=int(args.affine_q_block_size),
    )
    assembly_wall_s = float(time.perf_counter() - assembly_started)
    np.savez_compressed(
        run_dir / "reduced_operators.npz",
        Kq=operators["Kq"],
        Bq=operators["Bq"],
        Dq=operators["Dq"],
        coefficient_names=np.asarray(reduced.COEFF_NAMES),
    )

    monitor_truth_csv = (
        out_root
        / "runs"
        / f"{BASE_RUN_STEM}_geometry_{int(args.geometry_id):02d}"
        / "monitor_truth_results.csv"
    )
    final_truth_csv = (
        out_root
        / "runs"
        / f"{BASE_RUN_STEM}_geometry_{int(args.geometry_id):02d}"
        / "final_validation_truth_results.csv"
    )
    monitor_truth = pd.read_csv(monitor_truth_csv)
    monitor_rom = reduced._evaluate_rom(
        results_df=monitor_truth,
        Kq=operators["Kq"],
        Bq=operators["Bq"],
        Dq=operators["Dq"],
    )
    monitor_rom.to_csv(run_dir / "monitor_rom_results.csv", index=False)
    final_summary: dict[str, Any] | None = None
    if final_truth_csv.is_file():
        final_truth = pd.read_csv(final_truth_csv)
        final_rom = reduced._evaluate_rom(
            results_df=final_truth,
            Kq=operators["Kq"],
            Bq=operators["Bq"],
            Dq=operators["Dq"],
        )
        final_rom.to_csv(run_dir / "final_validation_rom_results.csv", index=False)
        final_summary = relative_stats(final_rom)

    reference = sobol_reference_row(int(args.geometry_id), int(basis.rank))
    summary = {
        "method": "anchor_tangent_gamma_fd_probe",
        "note": (
            "Finite-difference gamma tangents; this tests basis quality but is not "
            "the exact cheap K0 tangent solve proposed for production."
        ),
        "run_dir": str(run_dir),
        "geometry_id": int(args.geometry_id),
        "geometry_label": str(geometry.manifest.get("geometry_label", geometry_dir.name)),
        "basis_rank": int(basis.rank),
        "maximum_possible_rank": int(tangent_block.shape[0]),
        "training_fom_materials": int(len(materials)),
        "training_solve_wall_s": solve_wall_s,
        "basis_update_wall_s": basis_wall_s,
        "operator_assembly_wall_s": assembly_wall_s,
        "assembly": assembly_meta,
        "monitor_truth_csv": str(monitor_truth_csv),
        "monitor_summary": relative_stats(monitor_rom),
        "final_validation_truth_csv": str(final_truth_csv) if final_truth_csv.is_file() else "",
        "final_validation_summary": final_summary,
        "sobol_reference_near_same_rank": reference,
        "basis_dtype": str(dtype),
        "solver_profile": str(args.profile),
        "fft_backend": str(args.fft_backend),
        "gamma_fd_rel_step_requested": float(args.rel_step),
        "gamma_fd_steps_csv": str(run_dir / "gamma_fd_steps.csv"),
    }
    write_json(run_dir / "anchor_tangent_fd_summary.json", summary)
    append_ceff_columns(materials, solve_records).to_csv(
        run_dir / "anchor_tangent_training_records.csv",
        index=False,
    )
    if bool(args.cleanup_training_truth):
        shutil.rmtree(run_dir / "training_truth", ignore_errors=True)

    print(
        "[ANCHOR-TANGENT] done | "
        f"out={run_dir} | r={int(basis.rank)} | "
        f"monitor_max={summary['monitor_summary']['error_max']:.3e} | "
        f"sobol_r{reference['reference_rank']}_max={reference['reference_monitor_error_max']:.3e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
