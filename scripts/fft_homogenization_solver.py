#!/usr/bin/env python3
"""Fixed-geometry SAM + FFTHomPy material sweep for the new paper.

This script intentionally lives outside ``FFT/`` so that ``FFT/`` remains a
clean SAM + FFTHomPy runtime core.  It creates one short-fiber RVE, stores its
``phase.npy`` and ``ori.npy`` once, and reuses those same arrays for every
material point in the constituent design space.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
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
FFT_ROOT = PROJECT_ROOT / "FFT"
DEFAULT_VENV_PATH = (
    Path.home() / "Documentos/ANDRES/COMPUTATIONAL_WORKSPACEV4/.venv"
)

MATERIAL_BOUNDS: dict[str, tuple[float, float]] = {
    "Em": (1.1, 4.4),
    "nu_m": (0.35, 0.42),
    "Ef_L": (72.0, 395.0),
    "Ef_T": (6.0, 23.0),
    "G_LT": (8.0, 30.0),
    "nu_LT": (0.20, 0.26),
    "nu_TT": (0.35, 0.40),
}

MATERIAL_COLUMNS = [
    "material_id",
    "material_label",
    "Em",
    "nu_m",
    "Gm",
    "Ef_L",
    "Ef_T",
    "G_LT",
    "nu_LT",
    "nu_TT",
    "G_TT",
    "Ef_L_over_Em",
    "Ef_T_over_Em",
    "G_LT_over_Gm",
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


def _prepare_runtime(venv_path: Path, *, no_reexec: bool) -> None:
    """Relaunch inside the configured CUDA virtualenv when needed."""
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

    runtime_marker = ":".join(cuda_dirs)
    if env.get("FIXED_GEOMETRY_CUDA_RUNTIME_READY") != runtime_marker:
        env["FIXED_GEOMETRY_CUDA_RUNTIME_READY"] = runtime_marker
        os.execve(
            str(venv_python),
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            env,
        )

    os.environ.update(env)

    import ctypes

    try:
        ctypes.CDLL("libcublas.so.12")
        ctypes.CDLL("libcudart.so.12")
    except OSError as exc:
        raise RuntimeError(
            "No se pudieron cargar las librerias CUDA del entorno virtual. "
            f"Detalle: {exc}"
        ) from exc


def _add_fft_paths() -> None:
    ffthompy_path = FFT_ROOT / "ffthompy_core" / "ffthompy"
    for path in (FFT_ROOT, ffthompy_path):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


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
        numeric = float(value)
        return None if not math.isfinite(numeric) else numeric
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=True)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def _round_up_to_multiple(value: float, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return int(math.ceil(float(value) / float(multiple)) * multiple)


def _orientation_a2(name: str) -> tuple[float, float, float]:
    key = str(name).strip().lower().replace("_", "-")
    if key == "quasi-isotropic":
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    if key == "aligned-x":
        return 0.90, 0.05, 0.05
    if key == "planar-xy":
        return 0.50, 0.50, 0.0
    raise ValueError(
        "orientation-state debe ser quasi-isotropic, aligned-x o planar-xy."
    )


def _derive_geometry_design(args: argparse.Namespace) -> dict[str, Any]:
    d_um = float(args.fiber_diameter_um)
    ar = float(args.aspect_ratio)
    vf = float(args.volume_fraction)
    target_fibers = int(args.target_fibers)
    if ar <= 0.0:
        raise ValueError("--aspect-ratio debe ser > 0.")
    if not 0.0 < vf < 1.0:
        raise ValueError("--volume-fraction debe estar entre 0 y 1.")
    if d_um <= 0.0:
        raise ValueError("--fiber-diameter-um debe ser > 0.")
    if target_fibers < 1:
        raise ValueError("--target-fibers debe ser >= 1.")

    L_um = ar * d_um
    single_fiber_volume = math.pi * d_um * d_um * L_um / 4.0
    caja_um = (target_fibers * single_fiber_volume / vf) ** (1.0 / 3.0)
    nvox_raw = float(args.voxels_per_diameter) * caja_um / d_um
    nvox = _round_up_to_multiple(nvox_raw, int(args.nvox_multiple))
    res = float(nvox) / caja_um
    voxel_um = caja_um / float(nvox)
    df_voxel = d_um * res
    a11, a22, a33 = _orientation_a2(args.orientation_state)

    return {
        "config_id": "fixed_AR15_Vf20_single_geometry",
        "label": "fft_homogenization_solver",
        "sobol_index": 0,
        "design_id": 0,
        "is_operable": True,
        "reject_reason": "",
        "BOX_FACTOR": _round(caja_um / L_um),
        "DF_VOXEL_TARGET": _round(args.voxels_per_diameter),
        "NVOX_MULTIPLE": int(args.nvox_multiple),
        "caja_um": _round(caja_um),
        "nvox": int(nvox),
        "nvox_ref": int(nvox),
        "res": _round(res),
        "res_ref": _round(res),
        "voxel_um": _round(voxel_um),
        "df_voxel": _round(df_voxel),
        "d_um": _round(d_um),
        "L_um": _round(L_um),
        "fiber_length_lf": _round(L_um),
        "AR": _round(ar),
        "Lf_Ldom": _round(L_um / caja_um),
        "Vf_target": _round(vf),
        "a11": float(a11),
        "a22": float(a22),
        "a33": float(a33),
        "target_fibers_nominal": int(target_fibers),
        "single_fiber_volume_um3": float(single_fiber_volume),
    }


def _material_derived(sampled: dict[str, float]) -> dict[str, float]:
    Em = float(sampled["Em"])
    nu_m = float(sampled["nu_m"])
    Gm = Em / (2.0 * (1.0 + nu_m))
    Ef_L = float(sampled["Ef_L"])
    Ef_T = float(sampled["Ef_T"])
    G_LT = float(sampled["G_LT"])
    nu_LT = float(sampled["nu_LT"])
    nu_TT = float(sampled["nu_TT"])
    G_TT = Ef_T / (2.0 * (1.0 + nu_TT))
    return {
        "Em": Em,
        "nu_m": nu_m,
        "Gm": Gm,
        "Ef_L": Ef_L,
        "Ef_T": Ef_T,
        "G_LT": G_LT,
        "nu_LT": nu_LT,
        "nu_TT": nu_TT,
        "G_TT": G_TT,
        "Ef_L_over_Em": Ef_L / Em,
        "Ef_T_over_Em": Ef_T / Em,
        "G_LT_over_Gm": G_LT / Gm,
    }


def _validate_material(row: dict[str, float]) -> None:
    if row["Em"] <= 0.0 or row["Ef_L"] <= 0.0 or row["Ef_T"] <= 0.0:
        raise ValueError(f"Material no positivo: {row}")
    if row["G_LT"] <= 0.0 or row["G_TT"] <= 0.0:
        raise ValueError(f"Modulo de corte no positivo: {row}")
    if not -1.0 < row["nu_m"] < 0.5:
        raise ValueError(f"nu_m fuera de dominio elastico: {row['nu_m']}")
    if not -1.0 < row["nu_TT"] < 0.5:
        raise ValueError(f"nu_TT fuera de dominio elastico: {row['nu_TT']}")
    if not -1.0 < row["nu_LT"] < 0.5:
        raise ValueError(f"nu_LT fuera de dominio elastico: {row['nu_LT']}")


def _build_material_points(n_points: int, seed: int) -> pd.DataFrame:
    if n_points < 1:
        raise ValueError("--material-points debe ser >= 1.")

    rows: list[dict[str, Any]] = []
    center = {
        name: 0.5 * (bounds[0] + bounds[1])
        for name, bounds in MATERIAL_BOUNDS.items()
    }
    center_row = _material_derived(center)
    _validate_material(center_row)
    rows.append(
        {
            "material_id": 0,
            "material_label": "center",
            **{key: _round(value) for key, value in center_row.items()},
        }
    )

    sobol_count = int(n_points) - 1
    if sobol_count > 0:
        names = list(MATERIAL_BOUNDS.keys())
        sampler = qmc.Sobol(d=len(names), scramble=True, seed=int(seed))
        power = int(math.ceil(math.log2(sobol_count)))
        unit = sampler.random_base2(m=power)[:sobol_count]
        lower = [MATERIAL_BOUNDS[name][0] for name in names]
        upper = [MATERIAL_BOUNDS[name][1] for name in names]
        scaled = qmc.scale(unit, lower, upper)
        for offset, values in enumerate(scaled, start=1):
            sampled = {name: float(value) for name, value in zip(names, values)}
            material = _material_derived(sampled)
            _validate_material(material)
            rows.append(
                {
                    "material_id": int(offset),
                    "material_label": f"sobol_{offset:04d}",
                    **{key: _round(value) for key, value in material.items()},
                }
            )

    df = pd.DataFrame(rows)
    return df[MATERIAL_COLUMNS].copy()


def _make_run_dir(out_root: Path, out_name: str | None) -> Path:
    if out_name:
        run_dir = out_root / out_name
    else:
        run_dir = out_root / time.strftime("fixed_geometry_ar15_vf20_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _configure_fft_runtime(args: argparse.Namespace) -> dict[str, Any]:
    _add_fft_paths()
    main_mod = importlib.import_module("main")
    workflow = importlib.import_module("pipeline.workflow")

    settings = main_mod.user_settings(
        smoke=False,
        sobol_points=1,
        seeds_per_design=1,
        geometry_backend=args.geometry_backend,
        pipeline_overlap=False,
    )
    settings["campaign_name"] = "fixed_geometry_ffthompy"
    settings["generator_num_cores"] = int(args.generator_cores)
    settings["cpu_budget"] = int(args.generator_cores)
    settings["parallel_geometries"] = 1
    settings["geometry_prefetch"] = 1
    settings["geometry_batch_size"] = 1
    settings["persistent_geometry_pool"] = False
    settings["pipeline_overlap"] = False
    workflow._set_environment(settings)

    # Import sobol_gpu after _set_environment; that module reads env at import.
    sobol_gpu = importlib.import_module("pipeline.sobol_gpu")
    config = sobol_gpu.make_run_config()
    config["generator_num_cores"] = int(args.generator_cores)
    config["geometry_cpu_budget"] = int(args.generator_cores)
    config["sam_geometry_backend"] = str(args.geometry_backend)
    config["compute_rve_metrics"] = bool(args.compute_rve_metrics)
    config["solver_tol"] = float(args.solver_tol)
    config["store_solution_fields"] = False
    config["delete_geometry_npy_after_solve"] = False
    config["delete_geometry_npy_after_failed_solve"] = False
    config["preload_geometries_to_ram"] = True
    return {"settings": settings, "sobol_gpu": sobol_gpu, "config": config}


def _generate_geometry(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    design_row: dict[str, Any],
    sobol_gpu: Any,
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    geometry_dir = run_dir / "_fixed_geometry"
    geometry_dir.mkdir(parents=True, exist_ok=False)
    generation_params = sobol_gpu.build_generation_params(
        mc_id=0,
        seed=int(args.geometry_seed),
        design_row=design_row,
        seed_dir=geometry_dir,
        config=config,
    )
    generation_params["export_fiber_table"] = True
    generation_params["save_continuous_geometry"] = True
    generation_params["sam_verbose"] = bool(args.verbose_geometry)

    print(
        "[FIXED-GEOMETRY] Generando G una sola vez | "
        f"AR={design_row['AR']} | Vf={design_row['Vf_target']} | "
        f"N={design_row['nvox']} | target_fibers={design_row['target_fibers_nominal']} "
        f"| backend={args.geometry_backend}",
        flush=True,
    )
    t0 = time.perf_counter()
    gen_info = sobol_gpu.generate_rve_main_sam(generation_params)
    gen_wall_s = time.perf_counter() - t0

    phase_path = geometry_dir / "phase.npy"
    ori_path = geometry_dir / "ori.npy"
    if not phase_path.is_file() or not ori_path.is_file():
        raise FileNotFoundError("La geometria fija no produjo phase.npy y ori.npy.")

    phase = np.load(phase_path)
    ori = np.load(ori_path)
    geometry_info = {
        "geometry_dir": str(geometry_dir),
        "phase_path": str(phase_path),
        "ori_path": str(ori_path),
        "phase_sha256": _sha256_file(phase_path),
        "ori_sha256": _sha256_file(ori_path),
        "phase_shape": list(phase.shape),
        "ori_shape": list(ori.shape),
        "phase_dtype": str(phase.dtype),
        "ori_dtype": str(ori.dtype),
        "actual_voxel_vf": float(np.mean(phase != 0)),
        "generation_wall_s": float(gen_wall_s),
        "design": design_row,
        "generation": gen_info,
    }
    _write_json(geometry_dir / "geometry_manifest.json", geometry_info)
    return geometry_dir, geometry_info


def _material_solver_params(
    *,
    sobol_gpu: Any,
    config: dict[str, Any],
    material_row: dict[str, Any],
    design_row: dict[str, Any],
    material_dir: Path,
    seed: int,
    save_solution_fields: bool,
) -> dict[str, Any]:
    solver_design_row = dict(design_row)
    solver_design_row.update(material_row)
    params = sobol_gpu.build_solver_params(
        seed=int(seed),
        design_row=solver_design_row,
        seed_dir=material_dir,
        config=config,
    )
    params["solver_timing_path"] = str(material_dir / "solver_timing.json")
    params["solver_verbose"] = False
    params["free_gpu_memory_after_solve"] = True
    if save_solution_fields:
        params["store_solution_fields"] = True
        params["solution_field_out_path"] = str(material_dir / "solution_fields.npz")
        params["solution_field_load_ids"] = list(range(6))
        params["solution_field_format"] = "npy_dir"
    return params


def _solve_materials(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    design_row: dict[str, Any],
    geometry_dir: Path,
    geometry_info: dict[str, Any],
    materials_df: pd.DataFrame,
    sobol_gpu: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
    phase = np.load(geometry_dir / "phase.npy").astype(np.uint8)
    ori = np.load(geometry_dir / "ori.npy").astype(np.float32)
    save_field_ids = {int(value) for value in args.solution_field_material_ids}
    if args.all_solution_fields:
        save_field_ids = {int(value) for value in materials_df["material_id"].tolist()}

    rows: list[dict[str, Any]] = []
    for material_row in materials_df.to_dict(orient="records"):
        material_id = int(material_row["material_id"])
        material_dir = run_dir / f"material_{material_id:04d}"
        material_dir.mkdir(parents=True, exist_ok=False)
        _write_json(material_dir / "material.json", material_row)

        params = _material_solver_params(
            sobol_gpu=sobol_gpu,
            config=config,
            material_row=material_row,
            design_row=design_row,
            material_dir=material_dir,
            seed=int(args.geometry_seed),
            save_solution_fields=material_id in save_field_ids,
        )
        params["phase_array"] = phase
        params["ori_array"] = ori
        params["preloaded_geometry"] = True

        print(
            "[FIXED-GEOMETRY] FFTHomPy/CuPy | "
            f"material={material_id} | Em={material_row['Em']:.3g} | "
            f"Ef_L={material_row['Ef_L']:.3g}",
            flush=True,
        )
        t0 = time.perf_counter()
        ceff = np.asarray(sobol_gpu.solve_homogenization(params), dtype=float)
        solve_wall_s = time.perf_counter() - t0
        np.save(material_dir / "Ceff.npy", ceff)

        sym = 0.5 * (ceff + ceff.T)
        sym_rel = float(
            np.linalg.norm(ceff - ceff.T)
            / max(np.linalg.norm(sym), np.finfo(float).eps)
        )
        eigvals = np.linalg.eigvalsh(sym)
        props = sobol_gpu.engineering_constants_from_Cmandel(sym)

        timing_payload: dict[str, Any] = {}
        timing_path = material_dir / "solver_timing.json"
        if timing_path.is_file():
            with timing_path.open("r", encoding="utf-8") as handle:
                timing_payload = json.load(handle)

        row = {
            **material_row,
            "run_dir": str(run_dir),
            "material_dir": str(material_dir),
            "geometry_dir": str(geometry_dir),
            "phase_sha256": geometry_info["phase_sha256"],
            "ori_sha256": geometry_info["ori_sha256"],
            "Ceff_path": str(material_dir / "Ceff.npy"),
            "solver_timing_path": str(timing_path),
            "solution_fields_path": str(material_dir / "solution_fields.npz")
            if (material_dir / "solution_fields.npz").is_file()
            else "",
            "solve_wall_s": float(solve_wall_s),
            "Ceff_symmetry_rel": sym_rel,
            "Ceff_min_eig": float(np.min(eigvals)),
            "Ceff_max_eig": float(np.max(eigvals)),
            "Ceff_condition": float(np.max(eigvals) / max(np.min(eigvals), 1e-30)),
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
        }
        for key in ENGINEERING_COLUMNS:
            row[key] = float(props.get(key, np.nan))
        for ii in range(6):
            for jj in range(6):
                row[f"Ceff_{ii + 1}{jj + 1}"] = float(ceff[ii, jj])
        rows.append(row)

    return pd.DataFrame(rows)


def _write_run_summary(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    design_row: dict[str, Any],
    geometry_info: dict[str, Any],
    results_df: pd.DataFrame,
) -> None:
    ok = bool(
        float(geometry_info["generation"].get("sam_vf_ok", False))
        and bool(geometry_info["generation"].get("sam_A2_ok", False))
        and bool(geometry_info["generation"].get("sam_overlap_ok", False))
    )
    max_sym = float(results_df["Ceff_symmetry_rel"].max())
    min_eig = float(results_df["Ceff_min_eig"].min())
    text = f"""# Fixed-Geometry FFTHomPy Sweep

- Geometry fixed once: `AR={design_row['AR']}`, `Vf_target={design_row['Vf_target']}`.
- Nominal short fibers: `{design_row['target_fibers_nominal']}`.
- Grid: `{design_row['nvox']}^3`.
- Actual voxel `Vf`: `{geometry_info['actual_voxel_vf']:.8f}`.
- SAM checks passed: `{ok}`.
- Material points solved on same `phase.npy`/`ori.npy`: `{len(results_df)}`.
- Max `Ceff` symmetry relative error: `{max_sym:.6e}`.
- Minimum eigenvalue of symmetrized `Ceff`: `{min_eig:.6e}`.
- Geometry hash: `{geometry_info['phase_sha256'][:12]}_{geometry_info['ori_sha256'][:12]}`.

Command pattern:

```bash
./scripts/fft_homogenization_solver.py --material-points {args.material_points}
```
"""
    (run_dir / "run_summary.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera una sola geometria SAM AR=15/Vf=20% y resuelve un barrido "
            "material con FFTHomPy/CuPy reutilizando el mismo phase.npy/ori.npy."
        )
    )
    parser.add_argument("--material-points", type=int, default=8)
    parser.add_argument("--material-seed", type=int, default=20260621)
    parser.add_argument("--geometry-seed", type=int, default=20260811)
    parser.add_argument("--aspect-ratio", type=float, default=15.0)
    parser.add_argument("--volume-fraction", type=float, default=0.20)
    parser.add_argument("--target-fibers", type=int, default=100)
    parser.add_argument("--fiber-diameter-um", type=float, default=1.0)
    parser.add_argument("--voxels-per-diameter", type=float, default=6.0)
    parser.add_argument("--nvox-multiple", type=int, default=1)
    parser.add_argument(
        "--orientation-state",
        choices=("quasi-isotropic", "aligned-x", "planar-xy"),
        default="quasi-isotropic",
    )
    parser.add_argument(
        "--geometry-backend",
        choices=("numba", "cupy", "auto"),
        default="numba",
    )
    parser.add_argument("--generator-cores", type=int, default=2)
    parser.add_argument("--solver-tol", type=float, default=1e-4)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "fixed_geometry_ffthompy",
    )
    parser.add_argument("--out-name", default=None)
    parser.add_argument(
        "--solution-field-material-ids",
        type=int,
        nargs="*",
        default=[],
        help="Material IDs whose six fluctuation fields should be saved.",
    )
    parser.add_argument(
        "--all-solution-fields",
        action="store_true",
        help="Save six fluctuation fields for every material point.",
    )
    parser.add_argument("--compute-rve-metrics", action="store_true")
    parser.add_argument("--verbose-geometry", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fixed geometry and material design without solving.",
    )
    parser.add_argument(
        "--no-runtime-reexec",
        action="store_true",
        help="Developer option: do not relaunch through FFT/main.py VENV_PATH.",
    )
    parser.add_argument(
        "--venv-path",
        type=Path,
        default=DEFAULT_VENV_PATH,
        help="Python environment with CuPy/CUDA runtime.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    design_row = _derive_geometry_design(args)
    materials_df = _build_material_points(args.material_points, args.material_seed)

    if args.dry_run:
        print(json.dumps(_jsonable({"geometry": design_row}), indent=2))
        print(materials_df.to_string(index=False))
        return 0

    _prepare_runtime(args.venv_path, no_reexec=bool(args.no_runtime_reexec))
    runtime = _configure_fft_runtime(args)
    sobol_gpu = runtime["sobol_gpu"]
    config = runtime["config"]

    sobol_gpu.check_cupy_gpu()
    sobol_gpu.warmup_gpu_once()
    if str(args.geometry_backend) == "numba":
        sobol_gpu.warmup_numba_geometry_kernels()

    run_dir = _make_run_dir(args.out_root, args.out_name)
    material_path = run_dir / "material_points.csv"
    materials_df.to_csv(material_path, index=False)
    materials_df.to_excel(run_dir / "material_points.xlsx", index=False)

    geometry_dir, geometry_info = _generate_geometry(
        args=args,
        run_dir=run_dir,
        design_row=design_row,
        sobol_gpu=sobol_gpu,
        config=config,
    )

    results_df = _solve_materials(
        args=args,
        run_dir=run_dir,
        design_row=design_row,
        geometry_dir=geometry_dir,
        geometry_info=geometry_info,
        materials_df=materials_df,
        sobol_gpu=sobol_gpu,
        config=config,
    )
    results_df.to_csv(run_dir / "fixed_geometry_ffthompy_results.csv", index=False)
    results_df.to_excel(run_dir / "fixed_geometry_ffthompy_results.xlsx", index=False)

    manifest = {
        "run_dir": str(run_dir),
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "objective": (
            "single fixed SAM geometry, AR=15 and Vf=20%, reused across "
            "constituent material points solved by FFTHomPy/CuPy"
        ),
        "no_geometry_realizations": True,
        "same_phase_ori_for_all_materials": True,
        "script": str(Path(__file__).resolve()),
        "arguments": vars(args),
        "material_bounds": MATERIAL_BOUNDS,
        "geometry": geometry_info,
        "fft_runtime_settings": runtime["settings"],
        "fft_run_config": config,
        "outputs": {
            "material_points_csv": str(material_path),
            "results_csv": str(run_dir / "fixed_geometry_ffthompy_results.csv"),
            "geometry_manifest": str(geometry_dir / "geometry_manifest.json"),
        },
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_run_summary(
        run_dir=run_dir,
        args=args,
        design_row=design_row,
        geometry_info=geometry_info,
        results_df=results_df,
    )

    print(
        "[FIXED-GEOMETRY] listo | "
        f"run_dir={run_dir} | materials={len(results_df)} | "
        f"max_sym={results_df['Ceff_symmetry_rel'].max():.3e} | "
        f"min_eig={results_df['Ceff_min_eig'].min():.3e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
