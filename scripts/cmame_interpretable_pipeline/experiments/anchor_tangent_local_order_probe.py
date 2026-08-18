#!/usr/bin/env python3
"""Local order probe for anchor-only and anchor+tangent Ritz bases."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = EXPERIMENT_DIR.parents[0]
CONFIG_DEFAULT = SCRIPT_DIR / "campaign_config.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from env_bootstrap import ensure_configured_venv


ensure_configured_venv(CONFIG_DEFAULT)

ROOT = SCRIPT_DIR.parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cmame_campaign_common as common
import rom_reduced_operator as reduced

import anchor_tangent_fd_probe as fd


OUT_ROOT = fd.OUT_ROOT


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(common.jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def solve_material_record(
    *,
    run_dir: Path,
    geometry: common.GeometryData,
    runtime: dict[str, Any],
    material: dict[str, Any],
    seed: int,
    profile: str,
    quiet_solver: bool,
    subdir: str,
) -> dict[str, Any]:
    material_dir = run_dir / subdir / f"material_{int(material['material_id']):04d}"
    if quiet_solver:
        with open("/dev/null", "w", encoding="utf-8") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                return common.solve_material(
                    material_row=material,
                    material_dir=material_dir,
                    geometry=geometry,
                    runtime=runtime,
                    profile=profile,
                    seed=int(seed) + int(material["material_id"]),
                    save_solution_fields=False,
                )
    return common.solve_material(
        material_row=material,
        material_dir=material_dir,
        geometry=geometry,
        runtime=runtime,
        profile=profile,
        seed=int(seed) + int(material["material_id"]),
        save_solution_fields=False,
    )


def build_basis(
    fields: np.ndarray,
    *,
    nvox: int,
    dtype: np.dtype,
    tolerance: float,
) -> common.ContiguousBasis:
    basis = common.ContiguousBasis(
        capacity=int(fields.shape[0]),
        field_shape=(6, int(nvox)),
        dtype=dtype,
        projection_row_block_size=6,
    )
    appended = basis.append_preordered(
        np.ascontiguousarray(fields, dtype=dtype),
        tolerance=float(tolerance),
    )
    if len(appended) < 1:
        raise RuntimeError("Empty local reduced basis.")
    return basis


def assemble(
    *,
    operator_phase: np.ndarray,
    operator_ori: np.ndarray,
    basis: common.ContiguousBasis,
    affine: Any,
    q_block_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return common.update_reduced_operators(
        phase=operator_phase,
        ori=operator_ori,
        basis=basis.active_fields,
        existing=None,
        new_fields=basis.active_fields,
        affine_stress_batch=affine,
        affine_q_block_size=int(q_block_size),
    )


def random_directions(count: int, seed: int, q_count: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(size=(int(count), int(q_count)))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw


def valid_material_on_ray(
    gamma0: np.ndarray,
    scales: np.ndarray,
    direction: np.ndarray,
    h: float,
    *,
    material_id: int,
    label: str,
) -> dict[str, Any]:
    gamma = gamma0 + float(h) * scales * direction
    return fd.gamma_to_material(gamma, material_id=material_id, label=label)


def ray_levels(
    gamma0: np.ndarray,
    scales: np.ndarray,
    direction: np.ndarray,
    *,
    base_h: float,
    levels: int,
) -> np.ndarray:
    h0 = float(base_h)
    while h0 > 1.0e-8:
        values = h0 * 0.5 ** np.arange(int(levels), dtype=np.float64)
        try:
            for idx, h in enumerate(values):
                valid_material_on_ray(
                    gamma0,
                    scales,
                    direction,
                    float(h),
                    material_id=900000 + idx,
                    label="validity_probe",
                )
        except ValueError:
            h0 *= 0.5
            continue
        return values
    raise RuntimeError("Could not find a valid local ray inside the material domain.")


def fit_slope(group: pd.DataFrame, method: str, *, fit_tail: int) -> dict[str, Any]:
    values = group.loc[
        group["method"] == method,
        ["h", "relative_frobenius_error"],
    ].sort_values("h")
    if len(values) < 3:
        return {"method": method, "slope": float("nan"), "fit_count": int(len(values))}
    tail = values.tail(min(int(fit_tail), len(values)))
    x = np.log(tail["h"].to_numpy(dtype=float))
    y = np.log(np.maximum(tail["relative_frobenius_error"].to_numpy(dtype=float), 1e-300))
    slope, intercept = np.polyfit(x, y, deg=1)
    return {
        "method": method,
        "slope": float(slope),
        "intercept": float(intercept),
        "fit_count": int(len(tail)),
        "h_min": float(tail["h"].min()),
        "h_max": float(tail["h"].max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-id", type=int, default=3)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--central-rel-step", type=float, default=2.0e-2)
    parser.add_argument("--min-rel-step", type=float, default=1.0e-5)
    parser.add_argument("--candidate-count", type=int, default=1024)
    parser.add_argument("--candidate-seed", type=int, default=20260821)
    parser.add_argument("--training-seed", type=int, default=20260821)
    parser.add_argument("--validation-seed", type=int, default=20260825)
    parser.add_argument("--direction-seed", type=int, default=20260826)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--base-h", type=float, default=0.25)
    parser.add_argument("--fit-tail", type=int, default=4)
    parser.add_argument("--profile", choices=tuple(common.SOLVER_PROFILES), default="snapshot")
    parser.add_argument("--basis-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--basis-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--fft-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--geometry-backend", choices=("numba", "cupy", "auto"), default="numba")
    parser.add_argument("--generator-cores", default="auto")
    parser.add_argument("--affine-q-block-size", type=int, default=7)
    parser.add_argument("--quiet-solver", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup-training-truth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup-local-truth", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = OUT_ROOT
    geometry_dir = out_root / "geometries" / f"geometry_{int(args.geometry_id):02d}"
    geometry = common.load_fixed_geometry(geometry_dir)
    run_name = args.run_name or f"run_anchor_tangent_local_order_geometry_{int(args.geometry_id):02d}"
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
    gamma0 = reduced._material_coefficients(fd.center_material())
    scales = fd.gamma_scales(int(args.candidate_count), int(args.candidate_seed))
    voxel_order = reduced.phase_orientation_voxel_order(geometry.phase, geometry.ori)
    operator_phase = geometry.phase.reshape(-1)[voxel_order]
    operator_ori = geometry.ori.reshape(-1, 3)[voxel_order]
    nvox = int(voxel_order.size)

    materials, steps = fd.tangent_materials(
        gamma0=gamma0,
        candidate_count=int(args.candidate_count),
        candidate_seed=int(args.candidate_seed),
        rel_step=float(args.central_rel_step),
        min_rel_step=float(args.min_rel_step),
    )
    # Upgrade the one-sided list from tangent_materials to a central stencil
    # for this local-order check.
    central_rows = [fd.center_material()]
    central_step_rows = []
    next_id = 1
    for q, name in enumerate(reduced.COEFF_NAMES):
        step = abs(float(steps.iloc[q]["signed_step"]))
        plus = fd.gamma_to_material(
            gamma0 + step * np.eye(len(gamma0))[q],
            material_id=next_id,
            label=f"gamma_central_{name}_plus",
        )
        minus = fd.gamma_to_material(
            gamma0 - step * np.eye(len(gamma0))[q],
            material_id=next_id + 1,
            label=f"gamma_central_{name}_minus",
        )
        central_rows.extend([plus, minus])
        central_step_rows.append(
            {
                "coefficient_index": q,
                "coefficient_name": name,
                "gamma0": float(gamma0[q]),
                "signed_plus_step": float(step),
                "signed_minus_step": float(-step),
                "relative_step_used": float(step / scales[q]),
                "plus_material_id": int(plus["material_id"]),
                "minus_material_id": int(minus["material_id"]),
            }
        )
        next_id += 2
    central_materials = pd.DataFrame(central_rows)
    central_steps = pd.DataFrame(central_step_rows)
    central_materials.to_csv(run_dir / "central_tangent_materials.csv", index=False)
    central_steps.to_csv(run_dir / "central_gamma_fd_steps.csv", index=False)

    solved: dict[int, np.ndarray] = {}
    solve_records: list[dict[str, Any]] = []
    solve_started = time.perf_counter()
    for material in central_materials.to_dict(orient="records"):
        print(
            "[LOCAL-ORDER] training FOM | "
            f"geometry=g{int(args.geometry_id):02d} | "
            f"material={int(material['material_id'])} {material['material_label']}",
            flush=True,
        )
        fields, record = fd.solve_ordered_fields(
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
        solved[int(material["material_id"])] = fields
        solve_records.append(record)
    training_solve_wall_s = float(time.perf_counter() - solve_started)

    anchor = solved[0]
    tangent_blocks = [anchor]
    for row in central_step_rows:
        plus = solved[int(row["plus_material_id"])]
        minus = solved[int(row["minus_material_id"])]
        tangent_blocks.append((plus - minus) / (2.0 * float(row["signed_plus_step"])))
    tangent_fields = np.ascontiguousarray(np.concatenate(tangent_blocks, axis=0), dtype=dtype)

    basis0_started = time.perf_counter()
    basis0 = build_basis(anchor, nvox=nvox, dtype=dtype, tolerance=float(args.basis_tolerance))
    basis0_wall_s = float(time.perf_counter() - basis0_started)

    basis1_started = time.perf_counter()
    basis1 = build_basis(
        tangent_fields,
        nvox=nvox,
        dtype=dtype,
        tolerance=float(args.basis_tolerance),
    )
    basis1_wall_s = float(time.perf_counter() - basis1_started)

    affine = reduced.affine_stress_batch_factory(operator_phase, operator_ori)
    op0_started = time.perf_counter()
    ops0, meta0 = assemble(
        operator_phase=operator_phase,
        operator_ori=operator_ori,
        basis=basis0,
        affine=affine,
        q_block_size=int(args.affine_q_block_size),
    )
    op0_wall_s = float(time.perf_counter() - op0_started)
    op1_started = time.perf_counter()
    ops1, meta1 = assemble(
        operator_phase=operator_phase,
        operator_ori=operator_ori,
        basis=basis1,
        affine=affine,
        q_block_size=int(args.affine_q_block_size),
    )
    op1_wall_s = float(time.perf_counter() - op1_started)
    np.savez_compressed(run_dir / "anchor_only_reduced_operators.npz", **ops0)
    np.savez_compressed(run_dir / "anchor_tangent_reduced_operators.npz", **ops1)

    directions = random_directions(int(args.directions), int(args.direction_seed), len(gamma0))
    local_materials = []
    material_id = 1000
    for direction_id, direction in enumerate(directions):
        levels = ray_levels(
            gamma0,
            scales,
            direction,
            base_h=float(args.base_h),
            levels=int(args.levels),
        )
        for level_id, h in enumerate(levels):
            material = valid_material_on_ray(
                gamma0,
                scales,
                direction,
                float(h),
                material_id=material_id,
                label=f"local_d{direction_id:02d}_h{level_id:02d}",
            )
            material["direction_id"] = int(direction_id)
            material["level_id"] = int(level_id)
            material["h"] = float(h)
            local_materials.append(material)
            material_id += 1
    local_materials_df = pd.DataFrame(local_materials)
    local_materials_df.to_csv(run_dir / "local_order_materials.csv", index=False)

    truth_rows = []
    truth_started = time.perf_counter()
    for material in local_materials:
        print(
            "[LOCAL-ORDER] validation FOM | "
            f"material={int(material['material_id'])} | "
            f"d={material['direction_id']} | h={float(material['h']):.3e}",
            flush=True,
        )
        truth_rows.append(
            solve_material_record(
                run_dir=run_dir,
                geometry=geometry,
                runtime=runtime,
                material=material,
                seed=int(args.validation_seed),
                profile=str(args.profile),
                quiet_solver=bool(args.quiet_solver),
                subdir="local_truth",
            )
        )
    truth_wall_s = float(time.perf_counter() - truth_started)
    truth = pd.DataFrame(truth_rows)
    truth.to_csv(run_dir / "local_order_truth_results.csv", index=False)

    evaluated = []
    for method, ops in (
        ("anchor_only", ops0),
        ("anchor_tangent", ops1),
    ):
        frame = reduced._evaluate_rom(
            results_df=truth,
            Kq=ops["Kq"],
            Bq=ops["Bq"],
            Dq=ops["Dq"],
        )
        frame.insert(0, "method", method)
        evaluated.append(frame)
    errors = pd.concat(evaluated, ignore_index=True)
    metadata_cols = ["material_id", "direction_id", "level_id", "h"]
    errors = errors.merge(
        local_materials_df[metadata_cols],
        on="material_id",
        how="left",
    )
    errors.to_csv(run_dir / "local_order_errors.csv", index=False)

    slope_rows = []
    for direction_id, group in errors.groupby("direction_id"):
        for method in ("anchor_only", "anchor_tangent"):
            row = fit_slope(group, method, fit_tail=int(args.fit_tail))
            row["direction_id"] = int(direction_id)
            slope_rows.append(row)
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(run_dir / "local_order_slopes.csv", index=False)

    summary = {
        "method": "anchor_tangent_local_order_probe",
        "note": (
            "Tangents are central finite-difference approximations in affine gamma. "
            "This verifies the local hierarchy numerically but is not the exact K0 "
            "tangent solve."
        ),
        "run_dir": str(run_dir),
        "geometry_id": int(args.geometry_id),
        "basis_dtype": str(dtype),
        "solver_profile": str(args.profile),
        "training_materials": int(len(central_materials)),
        "basis0_rank": int(basis0.rank),
        "basis1_rank": int(basis1.rank),
        "training_solve_wall_s": training_solve_wall_s,
        "validation_truth_wall_s": truth_wall_s,
        "basis0_wall_s": basis0_wall_s,
        "basis1_wall_s": basis1_wall_s,
        "operator0_wall_s": op0_wall_s,
        "operator1_wall_s": op1_wall_s,
        "operator0_metadata": meta0,
        "operator1_metadata": meta1,
        "central_gamma_fd_steps_csv": str(run_dir / "central_gamma_fd_steps.csv"),
        "local_order_errors_csv": str(run_dir / "local_order_errors.csv"),
        "local_order_slopes_csv": str(run_dir / "local_order_slopes.csv"),
        "slope_summary": slopes.to_dict(orient="records"),
    }
    write_json(run_dir / "local_order_summary.json", summary)

    if bool(args.cleanup_training_truth):
        shutil.rmtree(run_dir / "training_truth", ignore_errors=True)
    if bool(args.cleanup_local_truth):
        shutil.rmtree(run_dir / "local_truth", ignore_errors=True)

    print(
        "[LOCAL-ORDER] done | "
        f"out={run_dir} | r0={basis0.rank} | r1={basis1.rank} | "
        f"slopes={slopes[['method','direction_id','slope']].to_dict(orient='records')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
