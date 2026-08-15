#!/usr/bin/env python3
"""Run the primal-dual CRE pilot on one fixed geometry."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DEFAULT = SCRIPT_DIR / "campaign_config.json"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from env_bootstrap import ensure_configured_venv


ensure_configured_venv(CONFIG_DEFAULT)

import argparse
import gc
import json
import os
import shutil
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = SCRIPT_DIR.parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cmame_campaign_common as common
import primal_dual_cre as cre
import rom_reduced_operator as reduced
import rom_validation_utils as validation


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(payload), indent=2, sort_keys=True), encoding="utf-8")


@contextmanager
def quiet_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def update_operators(
    *,
    phase: np.ndarray,
    ori: np.ndarray,
    basis: common.ContiguousBasis,
    new_fields: np.ndarray,
    existing: dict[str, np.ndarray] | None,
    affine_action: Any,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    if not len(new_fields):
        return existing, {
            "assembly_wall_s": 0.0,
            "affine_stress_wall_s": 0.0,
            "ritz_contraction_wall_s": 0.0,
        }
    return common.update_reduced_operators(
        phase=phase,
        ori=ori,
        basis=basis.active_fields,
        existing=existing,
        new_fields=new_fields,
        affine_stress_batch=affine_action,
        affine_q_block_size=7,
    )


def plot_results(run_dir: Path, curve: pd.DataFrame, results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axis = axes[0]
    axis.semilogy(
        curve["training_materials"],
        curve["candidate_eta_energy_max"],
        marker="o",
        linewidth=1.5,
        label=r"$\max_{\Xi_{1024}}\eta_E$",
    )
    axis.axhline(float(curve["target_error"].iloc[0]), color="black", linestyle="--", label="target")
    axis.set_xlabel("Training materials")
    axis.set_ylabel("CRE energy indicator")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)

    axis = axes[1]
    actual = results["true_energy_error"].to_numpy(dtype=float)
    bound = results["eta_energy"].to_numpy(dtype=float)
    low = max(min(np.min(actual), np.min(bound)) * 0.7, 1e-10)
    high = max(np.max(actual), np.max(bound)) * 1.4
    axis.loglog(actual, bound, "o", markersize=5)
    axis.loglog([low, high], [low, high], color="black", linestyle="--")
    axis.set_xlabel("True energy error")
    axis.set_ylabel("CRE energy bound")
    axis.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "cre_pilot_curve.png", dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    config = read_config(args.config)
    settings = config["cre_pilot"]
    generation = config["geometry_generation"]
    out_root = project_path(config["paths"]["out_root"])
    geometry_id = int(settings["geometry_id"])
    geometry_dir = out_root / "geometries" / f"geometry_{geometry_id:02d}"

    common.prepare_runtime(
        project_path(config["paths"]["venv_path"]),
        Path(__file__),
        marker_name="CMAME_CRE_PILOT_READY",
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_cre_pilot_{stamp}_geometry_{geometry_id:02d}"
    run_dir = out_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=bool(args.overwrite))
    blas_controller, blas_info = common.configure_blas_threads(int(settings["blas_threads"]))
    pipeline_started = time.perf_counter()
    try:
        geometry = common.load_fixed_geometry(geometry_dir)
        candidates = validation._build_independent_materials(
            int(settings["candidate_count"]), int(settings["candidate_seed"])
        ).copy()
        candidates.insert(0, "candidate_id", np.arange(len(candidates), dtype=int))
        candidates["material_id"] = candidates["candidate_id"]
        candidates["material_label"] = [
            f"candidate_sobol_{index:05d}" for index in candidates["candidate_id"]
        ]
        candidates.to_csv(run_dir / "candidate_pool_used.csv", index=False)

        training_limit = min(int(settings["training_limit"]), len(candidates))
        capacity = 6 * training_limit
        nvox = int(geometry.phase.size)
        voxel_order = reduced.phase_orientation_voxel_order(geometry.phase, geometry.ori)
        ordered_phase = geometry.phase.reshape(-1)[voxel_order]
        ordered_orientation = geometry.ori.reshape(-1, 3)[voxel_order]

        primal_basis = common.ContiguousBasis(
            capacity=capacity,
            field_shape=(6, nvox),
            dtype=str(settings["basis_dtype"]),
            projection_row_block_size=6,
        )
        dual_basis = common.ContiguousBasis(
            capacity=capacity,
            field_shape=(6, nvox),
            dtype=str(settings["basis_dtype"]),
            projection_row_block_size=6,
        )
        primal_action = reduced.affine_stress_batch_factory(
            ordered_phase, ordered_orientation
        )
        dual_action = reduced.affine_compliance_batch_factory(
            ordered_phase, ordered_orientation
        )
        physical_stiffness = reduced.affine_stress_batch_factory(
            geometry.phase, geometry.ori
        )
        projectors = cre.ElasticityProjectors(tuple(int(v) for v in geometry.phase.shape))

        runtime = common.configure_runtime(
            geometry_backend=str(generation["geometry_backend"]),
            generator_cores=int(generation["generator_cores"]),
            solver_tol=float(
                common.SOLVER_PROFILES[str(settings["basis_profile"])]["solver_rtol"]
            ),
            fft_backend=str(settings["fft_backend"]),
        )
        parameter_values = candidates[list(reduced.MATERIAL_PARAMETER_COLUMNS)].to_numpy(
            dtype=np.float64
        )
        primal_operators: dict[str, np.ndarray] | None = None
        dual_operators: dict[str, np.ndarray] | None = None
        curve_rows: list[dict[str, Any]] = []
        snapshot_rows: list[dict[str, Any]] = []
        stopping_streak = 0
        stopped_at: int | None = None

        print(
            f"[CRE] geometry={geometry_id:02d} N={geometry.phase.shape[0]} "
            f"pool={len(candidates)} target={float(settings['target_error']):.1e}",
            flush=True,
        )
        for training_materials, material in enumerate(
            candidates.iloc[:training_limit].to_dict(orient="records"), start=1
        ):
            candidate_id = int(material["candidate_id"])
            material_started = time.perf_counter()
            with quiet_output(bool(settings["quiet_solver"])):
                record = common.ensure_snapshot(
                    candidate_id=candidate_id,
                    candidates=candidates,
                    out_dir=run_dir,
                    geometry=geometry,
                    runtime=runtime,
                    seed=int(settings["candidate_seed"]),
                    profile=str(settings["basis_profile"]),
                    persistent_gpu_cache=False,
                )
            fields = np.stack(
                common.load_snapshot_fields(
                    common.snapshot_dir(run_dir, candidate_id), dtype=np.float64
                )
            )
            primal_snapshots, dual_snapshots, diagnostics = cre.stress_and_dual_snapshots(
                primal_fluctuations=fields,
                material=material,
                stiffness_action=physical_stiffness,
                projectors=projectors,
            )
            recovered = np.asarray(diagnostics.pop("recovered_effective"))
            truth = np.load(record["Ceff_path"])
            recovered_error = float(np.linalg.norm(recovered - truth) / np.linalg.norm(truth))

            ordered_primal = np.empty((6, 6, nvox), dtype=primal_basis.dtype)
            ordered_dual = np.empty((6, 6, nvox), dtype=dual_basis.dtype)
            for load_id in range(6):
                np.take(
                    primal_snapshots[load_id].reshape(6, nvox),
                    voxel_order,
                    axis=1,
                    out=ordered_primal[load_id],
                )
                np.take(
                    dual_snapshots[load_id].reshape(6, nvox),
                    voxel_order,
                    axis=1,
                    out=ordered_dual[load_id],
                )

            basis_started = time.perf_counter()
            new_primal = primal_basis.append_preordered(
                ordered_primal, tolerance=float(settings["basis_tolerance"])
            )
            new_dual = dual_basis.append_preordered(
                ordered_dual, tolerance=float(settings["basis_tolerance"])
            )
            basis_wall_s = float(time.perf_counter() - basis_started)
            primal_operators, primal_assembly = update_operators(
                phase=ordered_phase,
                ori=ordered_orientation,
                basis=primal_basis,
                new_fields=new_primal,
                existing=primal_operators,
                affine_action=primal_action,
            )
            dual_operators, dual_assembly = update_operators(
                phase=ordered_phase,
                ori=ordered_orientation,
                basis=dual_basis,
                new_fields=new_dual,
                existing=dual_operators,
                affine_action=dual_action,
            )
            if primal_operators is None or dual_operators is None:
                raise RuntimeError("Both primal and dual operators are required.")

            row = {
                "training_materials": training_materials,
                "candidate_id": candidate_id,
                "solve_wall_s": float(record["solve_wall_s"]),
                "material_step_wall_s": float(time.perf_counter() - material_started),
                "basis_update_wall_s": basis_wall_s,
                "primal_rank": len(primal_basis),
                "dual_rank": len(dual_basis),
                "new_primal_directions": len(new_primal),
                "new_dual_directions": len(new_dual),
                "primal_operator_wall_s": float(primal_assembly["assembly_wall_s"]),
                "dual_operator_wall_s": float(dual_assembly["assembly_wall_s"]),
                "recovered_ceff_relative_error": recovered_error,
                **diagnostics,
            }
            snapshot_rows.append(row)
            pd.DataFrame(snapshot_rows).to_csv(run_dir / "cre_snapshot_timing.csv", index=False)

            if training_materials < int(settings["start_materials"]):
                continue
            evaluation_started = time.perf_counter()
            bounds = cre.evaluate_bounds(
                parameters=parameter_values,
                primal_operators=primal_operators,
                dual_operators=dual_operators,
                backend=str(settings["rom_backend"]),
            )
            eta_max = float(np.max(bounds["eta_energy"]))
            eta_f_max = float(np.max(bounds["eta_frobenius"]))
            gap_min = float(np.min(bounds["gap_min_eigenvalue"]))
            worst_candidate = int(np.argmax(bounds["eta_energy"]))
            passes = eta_max <= float(settings["target_error"]) and gap_min >= -1.0e-9
            stopping_streak = stopping_streak + 1 if passes else 0
            stop_triggered = stopping_streak >= int(settings["stopping_consecutive"])
            curve_rows.append(
                {
                    "training_materials": training_materials,
                    "primal_rank": len(primal_basis),
                    "dual_rank": len(dual_basis),
                    "candidate_eta_energy_max": eta_max,
                    "candidate_eta_frobenius_max": eta_f_max,
                    "candidate_gap_min_eigenvalue": gap_min,
                    "worst_candidate_id": worst_candidate,
                    "target_error": float(settings["target_error"]),
                    "passes_target": passes,
                    "consecutive_passes": stopping_streak,
                    "stop_triggered": stop_triggered,
                    "finite_pool_evaluation_wall_s": float(
                        time.perf_counter() - evaluation_started
                    ),
                    "primal_online_wall_s": float(bounds["primal_online_wall_s"]),
                    "dual_online_wall_s": float(bounds["dual_online_wall_s"]),
                }
            )
            pd.DataFrame(curve_rows).to_csv(run_dir / "cre_adaptive_curve.csv", index=False)
            print(
                f"[CRE] m={training_materials:02d} r={len(primal_basis):03d} "
                f"s={len(dual_basis):03d} eta_max={eta_max:.3e} "
                f"gap_min={gap_min:.3e} worst={worst_candidate}",
                flush=True,
            )
            if stop_triggered:
                stopped_at = training_materials
                break

        trained_materials = stopped_at or len(snapshot_rows)
        selected = candidates.iloc[:trained_materials].copy()
        selected.to_csv(run_dir / "selected_sobol_sequence.csv", index=False)
        curve = pd.DataFrame(curve_rows)
        snapshots = pd.DataFrame(snapshot_rows)

        np.savez_compressed(
            run_dir / "cre_reduced_operators.npz",
            primal_Kq=primal_operators["Kq"],
            primal_Bq=primal_operators["Bq"],
            primal_Dq=primal_operators["Dq"],
            dual_Kq=dual_operators["Kq"],
            dual_Bq=dual_operators["Bq"],
            dual_Dq=dual_operators["Dq"],
            primal_coefficient_names=np.asarray(reduced.COEFF_NAMES),
            dual_coefficient_names=np.asarray(reduced.DUAL_COEFF_NAMES),
        )

        final_pool = validation._build_independent_materials(
            int(settings["final_validation_count"]),
            int(settings["final_validation_seed"]),
        ).copy()
        final_pool.insert(0, "validation_id", np.arange(len(final_pool), dtype=int))
        final_pool["material_id"] = final_pool["validation_id"]
        final_pool["material_label"] = [
            f"cre_validation_{index:04d}" for index in final_pool["validation_id"]
        ]
        final_pool.to_csv(run_dir / "cre_validation_pool.csv", index=False)
        truth_matrices: list[np.ndarray] = []
        truth_records: list[dict[str, Any]] = []
        validation_started = time.perf_counter()
        for material in final_pool.to_dict(orient="records"):
            validation_id = int(material["validation_id"])
            with quiet_output(bool(settings["quiet_solver"])):
                record = common.solve_material(
                    material_row=material,
                    material_dir=run_dir / "validation_truth" / f"material_{validation_id:04d}",
                    geometry=geometry,
                    runtime=runtime,
                    profile=str(settings["truth_profile"]),
                    seed=int(settings["final_validation_seed"]) + validation_id,
                    save_solution_fields=False,
                )
            truth_records.append(record)
            truth_matrices.append(np.load(record["Ceff_path"]))

        validation_parameters = final_pool[
            list(reduced.MATERIAL_PARAMETER_COLUMNS)
        ].to_numpy(dtype=np.float64)
        validation_bounds = cre.evaluate_bounds(
            parameters=validation_parameters,
            primal_operators=primal_operators,
            dual_operators=dual_operators,
            backend=str(settings["rom_backend"]),
        )
        comparison_rows: list[dict[str, Any]] = []
        for index, truth in enumerate(truth_matrices):
            comparison = cre.compare_with_truth(
                upper=validation_bounds["upper"][index],
                lower=validation_bounds["lower"][index],
                truth=truth,
            )
            comparison_rows.append(
                {
                    "validation_id": index,
                    **comparison.__dict__,
                    "gap_min_eigenvalue": float(
                        validation_bounds["gap_min_eigenvalue"][index]
                    ),
                    "solve_wall_s": float(truth_records[index]["solve_wall_s"]),
                }
            )
        validation_results = pd.DataFrame(comparison_rows)
        validation_results.to_csv(run_dir / "cre_validation_results.csv", index=False)
        validation_wall_s = float(time.perf_counter() - validation_started)
        plot_results(run_dir, curve, validation_results)

        lower_violations = int((validation_results["lower_truth_min_eig"] < -1e-9).sum())
        upper_violations = int((validation_results["truth_upper_min_eig"] < -1e-9).sum())
        underestimates = int(
            (validation_results["eta_energy"] + 1e-12 < validation_results["true_energy_error"]).sum()
        )
        if stopped_at is None:
            status = "pilot_limit_reached"
        elif lower_violations or upper_violations or underestimates:
            status = "independent_validation_failed"
        else:
            status = "finite_candidate_pool_passed"
        summary = {
            "status": status,
            "scientific_status": "experimental_not_approved_for_certification_claims",
            "geometry_id": geometry_id,
            "grid_shape": list(geometry.phase.shape),
            "candidate_count": len(candidates),
            "training_materials": trained_materials,
            "training_limit": training_limit,
            "primal_rank": len(primal_basis),
            "dual_rank": len(dual_basis),
            "target_error": float(settings["target_error"]),
            "candidate_eta_energy_max": float(curve.iloc[-1]["candidate_eta_energy_max"]),
            "candidate_eta_frobenius_max": float(curve.iloc[-1]["candidate_eta_frobenius_max"]),
            "validation_true_energy_error_max": float(validation_results["true_energy_error"].max()),
            "validation_eta_energy_max": float(validation_results["eta_energy"].max()),
            "validation_energy_effectivity_median": float(validation_results["energy_effectivity"].median()),
            "validation_energy_effectivity_max": float(validation_results["energy_effectivity"].max()),
            "validation_lower_bracket_violations": lower_violations,
            "validation_upper_bracket_violations": upper_violations,
            "validation_bound_underestimates": underestimates,
            "snapshot_fft_wall_s": float(snapshots["solve_wall_s"].sum()),
            "primal_operator_wall_s": float(snapshots["primal_operator_wall_s"].sum()),
            "dual_operator_wall_s": float(snapshots["dual_operator_wall_s"].sum()),
            "basis_update_wall_s": float(snapshots["basis_update_wall_s"].sum()),
            "validation_fft_stage_wall_s": validation_wall_s,
            "pipeline_wall_s": float(time.perf_counter() - pipeline_started),
            "basis_dtype": str(settings["basis_dtype"]),
            "basis_profile": str(settings["basis_profile"]),
            "truth_profile": str(settings["truth_profile"]),
            "finite_pool_scope": "configured Sobol candidates only",
            "discrete_bracket_scope": "GaNi discrete FOM",
            "odd_grid": bool(all(int(value) % 2 == 1 for value in geometry.phase.shape)),
            "continuous_domain_certified": False,
            "global_material_domain_certified": False,
            "blas_info": blas_info,
        }
        write_json(run_dir / "cre_pilot_summary.json", summary)
        write_json(run_dir / "campaign_config_snapshot.json", config)

        if bool(settings["cleanup_snapshot_fields"]):
            shutil.rmtree(run_dir / "snapshot_cache", ignore_errors=True)
        shutil.rmtree(run_dir / "validation_truth", ignore_errors=True)
        print(
            f"[CRE] completed status={summary['status']} "
            f"validation_violations={lower_violations + upper_violations} "
            f"output={run_dir}",
            flush=True,
        )
        return 0
    finally:
        del blas_controller
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
