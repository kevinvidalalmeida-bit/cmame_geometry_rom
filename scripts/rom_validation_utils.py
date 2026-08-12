#!/usr/bin/env python3
"""Independent material validation for a fixed-geometry tangential ROM."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
RUN_DEFAULT = (
    PROJECT_ROOT
    / "results"
    / "fixed_geometry_ffthompy"
    / "fixed_geometry_ar15_vf20_sobol8_center_fields"
)
ROM_DEFAULT = RUN_DEFAULT / "rom_tangential_r36_center_m1_m2_m3_m5_m7"
SOLVER_PROFILES = {
    "truth": {"solver_real_dtype": "float64", "solver_rtol": 1.0e-10, "solver_atol": 0.0},
    "snapshot": {"solver_real_dtype": "float64", "solver_rtol": 1.0e-8, "solver_atol": 0.0},
    "timing": {"solver_real_dtype": "float32", "solver_rtol": 1.0e-5, "solver_atol": 0.0},
    # Economical pilot mode for the declared 1e-4 ROM floor.  It is useful
    # for feasibility and timing checks; truth validation remains float64.
    "rom_floor": {"solver_real_dtype": "float32", "solver_rtol": 1.0e-4, "solver_atol": 0.0},
}
DEFAULT_VENV_PATH = (
    Path.home() / "Documentos/ANDRES/COMPUTATIONAL_WORKSPACEV4/.venv"
)

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import fft_homogenization_solver as sweep
import rom_reduced_operator as reduced


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


def _prepare_runtime(venv_path: Path, *, no_reexec: bool) -> None:
    if no_reexec:
        return
    venv_python = venv_path.expanduser().resolve() / "bin/python"
    if not venv_python.is_file():
        raise FileNotFoundError(f"No existe el Python del entorno: {venv_python}")

    env = os.environ.copy()
    cuda_dirs = sorted(
        str(path)
        for path in (venv_path / "lib").glob("python*/site-packages/nvidia/*/lib")
        if path.is_dir()
    )
    if cuda_dirs:
        current = env.get("LD_LIBRARY_PATH", "")
        current_dirs = [value for value in current.split(":") if value]
        env["LD_LIBRARY_PATH"] = ":".join(
            [*cuda_dirs, *[value for value in current_dirs if value not in cuda_dirs]]
        )

    marker = ":".join(cuda_dirs)
    if env.get("FIXED_GEOMETRY_VALIDATE_CUDA_READY") != marker:
        env["FIXED_GEOMETRY_VALIDATE_CUDA_READY"] = marker
        os.execve(
            str(venv_python),
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            env,
        )
    os.environ.update(env)

    try:
        ctypes.CDLL("libcublas.so.12")
        ctypes.CDLL("libcudart.so.12")
    except OSError as exc:
        raise RuntimeError(
            "No se pudieron cargar las librerias CUDA del entorno virtual. "
            f"Detalle: {exc}"
        ) from exc


def _load_design_row(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        design = payload.get("geometry", {}).get("design")
        if isinstance(design, dict):
            return design
    geometry_manifest = json.loads(
        (run_dir / "_fixed_geometry" / "geometry_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    design = geometry_manifest.get("design")
    if not isinstance(design, dict):
        raise RuntimeError("No se pudo recuperar design_row de la geometria fija.")
    return design


def _configure_runtime(args: argparse.Namespace) -> dict[str, Any]:
    profile = SOLVER_PROFILES[str(args.solver_profile)]
    effective_rtol = (
        float(args.solver_rtol)
        if args.solver_rtol is not None
        else float(args.solver_tol)
        if args.solver_tol is not None
        else float(profile["solver_rtol"])
    )
    runtime_args = argparse.Namespace(
        geometry_backend=args.geometry_backend,
        generator_cores=args.generator_cores,
        compute_rve_metrics=False,
        solver_tol=effective_rtol,
    )
    _prepare_runtime(args.venv_path, no_reexec=bool(args.no_runtime_reexec))
    runtime = sweep._configure_fft_runtime(runtime_args)
    sobol_gpu = runtime["sobol_gpu"]
    sobol_gpu.check_cupy_gpu()
    sobol_gpu.warmup_gpu_once()
    return runtime


def _round(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def _build_independent_materials(n_points: int, seed: int) -> pd.DataFrame:
    if n_points < 1:
        raise ValueError("--material-points debe ser >= 1.")
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
                "material_label": f"independent_sobol_{material_id:04d}",
                **{key: _round(value) for key, value in material.items()},
            }
        )
    return pd.DataFrame(rows)[sweep.MATERIAL_COLUMNS].copy()


def _make_out_dir(
    run_dir: Path,
    out_name: str | None,
    seed: int,
    n_points: int,
    *,
    out_dir: Path | None,
) -> Path:
    if out_dir is not None:
        out_dir = out_dir.resolve()
    elif out_name:
        out_dir = run_dir / out_name
    else:
        out_dir = run_dir / f"independent_validation_sobol{n_points}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _full_ceff_from_row(row: pd.Series) -> np.ndarray:
    matrix = np.zeros((6, 6), dtype=float)
    for ii in range(6):
        for jj in range(6):
            matrix[ii, jj] = float(row[f"Ceff_{ii + 1}{jj + 1}"])
    return matrix


def _solve_full_order_materials(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    out_dir: Path,
    materials_df: pd.DataFrame,
    design_row: dict[str, Any],
    runtime: dict[str, Any],
) -> pd.DataFrame:
    sobol_gpu = runtime["sobol_gpu"]
    config = runtime["config"]
    geometry_dir = run_dir / "_fixed_geometry"
    geometry_manifest = json.loads(
        (geometry_dir / "geometry_manifest.json").read_text(encoding="utf-8")
    )
    phase = np.load(geometry_dir / "phase.npy").astype(np.uint8)
    ori = np.load(geometry_dir / "ori.npy").astype(np.float32)

    rows: list[dict[str, Any]] = []
    for material_row in materials_df.to_dict(orient="records"):
        material_id = int(material_row["material_id"])
        material_dir = out_dir / f"material_{material_id:04d}"
        material_dir.mkdir(parents=True, exist_ok=False)
        _write_json(material_dir / "material.json", material_row)

        params = sweep._material_solver_params(
            sobol_gpu=sobol_gpu,
            config=config,
            material_row=material_row,
            design_row=design_row,
            material_dir=material_dir,
            seed=int(args.material_seed),
            save_solution_fields=False,
        )
        profile = SOLVER_PROFILES[str(args.solver_profile)]
        params["solver_profile"] = str(args.solver_profile)
        params["solver_real_dtype"] = str(profile["solver_real_dtype"])
        params["solver_rtol"] = float(
            args.solver_rtol
            if args.solver_rtol is not None
            else args.solver_tol
            if args.solver_tol is not None
            else profile["solver_rtol"]
        )
        params["solver_atol"] = float(
            profile["solver_atol"] if args.solver_atol is None else args.solver_atol
        )
        params["cfield_storage"] = "sym21"
        params["cfield_indexed"] = str(profile["solver_real_dtype"]) == "float32"
        params["projection_storage"] = (
            "direct" if str(profile["solver_real_dtype"]) == "float32" else "full"
        )
        params["projection_backend"] = (
            "numpy" if str(profile["solver_real_dtype"]) == "float32" else "cupy"
        )
        params["phase_array"] = phase
        params["ori_array"] = ori
        params["preloaded_geometry"] = True
        params["free_gpu_memory_after_solve"] = True

        print(
            "[VALIDATION] FFTHomPy/CuPy | "
            f"material={material_id} | same fixed geometry",
            flush=True,
        )
        t0 = time.perf_counter()
        ceff = np.asarray(sobol_gpu.solve_homogenization(params), dtype=float)
        solve_wall_s = time.perf_counter() - t0
        np.save(material_dir / "Ceff.npy", ceff)

        sym = 0.5 * (ceff + ceff.T)
        eigvals = np.linalg.eigvalsh(sym)
        sym_rel = float(
            np.linalg.norm(ceff - ceff.T)
            / max(np.linalg.norm(sym), np.finfo(float).eps)
        )
        props = sobol_gpu.engineering_constants_from_Cmandel(sym)

        timing_path = material_dir / "solver_timing.json"
        timing_payload: dict[str, Any] = {}
        if timing_path.is_file():
            timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
        load_summary = timing_payload.get("load_solver_summary", {})

        row = {
            **material_row,
            "validation_dir": str(out_dir),
            "material_dir": str(material_dir),
            "geometry_dir": str(geometry_dir),
            "phase_sha256": str(geometry_manifest["phase_sha256"]),
            "ori_sha256": str(geometry_manifest["ori_sha256"]),
            "Ceff_path": str(material_dir / "Ceff.npy"),
            "solver_timing_path": str(timing_path),
            "solve_wall_s": float(solve_wall_s),
            "Ceff_symmetry_rel": sym_rel,
            "Ceff_min_eig": float(np.min(eigvals)),
            "Ceff_max_eig": float(np.max(eigvals)),
            "solver_total_wall_s": float(
                timing_payload.get("solver_total_wall_s", np.nan)
            ),
            "problem_calculate_s": float(
                timing_payload.get("problem_calculate_s", np.nan)
            ),
            "problem_postprocessing_s": float(
                timing_payload.get("problem_postprocessing_s", np.nan)
            ),
            "n_fiber_voxels": int(timing_payload.get("n_fiber_voxels", -1)),
            "n_unique_orientations": int(
                timing_payload.get("n_unique_orientations", -1)
            ),
            "solver_profile": str(timing_payload.get("solver_profile", args.solver_profile)),
            "solver_real_dtype": str(
                timing_payload.get("solver_real_dtype", profile["solver_real_dtype"])
            ),
            "solver_rtol": float(timing_payload.get("solver_rtol", params["solver_rtol"])),
            "solver_atol": float(timing_payload.get("solver_atol", params["solver_atol"])),
            "solver_all_converged": bool(load_summary.get("all_converged", False)),
            "solver_max_relative_residual": float(
                load_summary.get("final_norm_res_rel_max", np.nan)
            ),
            "solver_max_iterations": int(load_summary.get("cg_iterations_max", -1)),
        }
        for key in sweep.ENGINEERING_COLUMNS:
            row[key] = float(props.get(key, np.nan))
        for ii in range(6):
            for jj in range(6):
                row[f"Ceff_{ii + 1}{jj + 1}"] = float(ceff[ii, jj])
        rows.append(row)
    return pd.DataFrame(rows)


def _load_full_order_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No existe full-order-results-csv: {path}")
    return pd.read_csv(path)


def _evaluate_rom(rom_dir: Path, full_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    operators = np.load(rom_dir / "reduced_operators.npz")
    validation_df = reduced._evaluate_rom(
        results_df=full_df,
        Kq=np.asarray(operators["Kq"], dtype=float),
        Bq=np.asarray(operators["Bq"], dtype=float),
        Dq=np.asarray(operators["Dq"], dtype=float),
    )
    errors = validation_df["relative_frobenius_error"].to_numpy(dtype=float)
    summary = {
        "rom_dir": str(rom_dir),
        "rom_manifest": str(rom_dir / "rom_manifest.json"),
        "basis_rank": int(json.loads((rom_dir / "rom_manifest.json").read_text())["basis_rank"])
        if (rom_dir / "rom_manifest.json").is_file()
        else int(operators["Kq"].shape[1]),
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
    }
    return validation_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida un ROM tangencial con materiales independientes, usando "
            "la misma geometria fija y FFTHomPy/CuPy para full-order."
        )
    )
    parser.add_argument("run_dir", type=Path, nargs="?", default=RUN_DEFAULT)
    parser.add_argument("--rom-dir", type=Path, default=ROM_DEFAULT)
    parser.add_argument("--material-points", type=int, default=16)
    parser.add_argument("--material-seed", type=int, default=20260923)
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--full-order-results-csv", type=Path, default=None)
    parser.add_argument("--geometry-backend", choices=("numba", "cupy", "auto"), default="numba")
    parser.add_argument("--generator-cores", type=int, default=2)
    parser.add_argument(
        "--solver-profile",
        choices=tuple(SOLVER_PROFILES),
        default="truth",
    )
    parser.add_argument(
        "--solver-tol",
        type=float,
        default=None,
        help="Legacy alias for --solver-rtol.",
    )
    parser.add_argument("--solver-rtol", type=float, default=None)
    parser.add_argument("--solver-atol", type=float, default=None)
    parser.add_argument("--venv-path", type=Path, default=DEFAULT_VENV_PATH)
    parser.add_argument("--no-runtime-reexec", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rom_dir = args.rom_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No existe run_dir: {run_dir}")
    if not rom_dir.is_dir():
        raise FileNotFoundError(f"No existe rom_dir: {rom_dir}")

    if args.full_order_results_csv is None:
        _prepare_runtime(args.venv_path, no_reexec=bool(args.no_runtime_reexec))
    out_dir = _make_out_dir(
        run_dir,
        args.out_name,
        args.material_seed,
        args.material_points,
        out_dir=args.out_dir,
    )
    design_row = _load_design_row(run_dir)

    if args.full_order_results_csv is None:
        materials_df = _build_independent_materials(
            n_points=int(args.material_points),
            seed=int(args.material_seed),
        )
        materials_df.to_csv(out_dir / "validation_material_points.csv", index=False)
        materials_df.to_excel(out_dir / "validation_material_points.xlsx", index=False)
        runtime = _configure_runtime(args)
        full_df = _solve_full_order_materials(
            args=args,
            run_dir=run_dir,
            out_dir=out_dir,
            materials_df=materials_df,
            design_row=design_row,
            runtime=runtime,
        )
        full_df.to_csv(out_dir / "validation_full_order_results.csv", index=False)
        full_df.to_excel(out_dir / "validation_full_order_results.xlsx", index=False)
    else:
        full_df = _load_full_order_results(args.full_order_results_csv)
        full_df.to_csv(out_dir / "validation_full_order_results.csv", index=False)
        full_df.to_excel(out_dir / "validation_full_order_results.xlsx", index=False)
        material_cols = [
            column
            for column in sweep.MATERIAL_COLUMNS
            if column in full_df.columns
        ]
        full_df[material_cols].to_csv(
            out_dir / "validation_material_points.csv",
            index=False,
        )

    rom_df, rom_summary = _evaluate_rom(rom_dir, full_df)
    rom_df.to_csv(out_dir / "rom_validation_results.csv", index=False)
    rom_df.to_excel(out_dir / "rom_validation_results.xlsx", index=False)

    unique_phase = int(full_df["phase_sha256"].nunique()) if "phase_sha256" in full_df else -1
    unique_ori = int(full_df["ori_sha256"].nunique()) if "ori_sha256" in full_df else -1
    manifest = {
        "run_dir": str(run_dir),
        "validation_dir": str(out_dir),
        "rom_dir": str(rom_dir),
        "material_points": int(len(full_df)),
        "material_seed": int(args.material_seed),
        "solver_profile": str(args.solver_profile),
        "full_order_results_csv": str(args.full_order_results_csv)
        if args.full_order_results_csv is not None
        else str(out_dir / "validation_full_order_results.csv"),
        "same_fixed_geometry": bool(unique_phase == 1 and unique_ori == 1),
        "unique_phase_hashes": unique_phase,
        "unique_ori_hashes": unique_ori,
        "rom_summary": rom_summary,
    }
    _write_json(out_dir / "validation_manifest.json", manifest)

    text = f"""# Independent ROM Validation

- Source fixed-geometry run: `{run_dir}`
- ROM: `{rom_dir}`
- Material points: `{len(full_df)}`
- Same fixed geometry: `{manifest['same_fixed_geometry']}`
- Mean relative tensor error: `{rom_summary['mean_relative_error']:.6e}`
- P95 relative tensor error: `{rom_summary['p95_relative_error']:.6e}`
- Max relative tensor error: `{rom_summary['max_relative_error']:.6e}`
- Worst material ID: `{rom_summary['worst_material_id']}`
- Median ROM online time: `{rom_summary['median_online_s']:.6e} s`
- Min ROM eigenvalue: `{rom_summary['min_rom_eig']:.6e}`
- Min eig(Crom - Cfom): `{rom_summary['min_eig_Crom_minus_Cfom']:.6e}`
"""
    (out_dir / "validation_summary.md").write_text(text, encoding="utf-8")

    print(
        "[VALIDATION] listo | "
        f"out={out_dir} | mean={rom_summary['mean_relative_error']:.3e} | "
        f"max={rom_summary['max_relative_error']:.3e} | "
        f"worst={rom_summary['worst_material_id']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
