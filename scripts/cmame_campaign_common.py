#!/usr/bin/env python3
"""Shared, resumable FFTHomPy campaign utilities for the CMAME study."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fft_homogenization_solver as sweep
import rom_validation_utils as validate


DEFAULT_VENV_PATH = validate.DEFAULT_VENV_PATH
SOLVER_PROFILES = validate.SOLVER_PROFILES


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def prepare_runtime(venv_path: Path, script_path: Path, *, marker_name: str) -> None:
    """Relaunch the calling campaign inside the configured CUDA environment."""
    venv_path = Path(venv_path).expanduser().resolve()
    venv_python = venv_path / "bin/python"
    if not venv_python.is_file():
        raise FileNotFoundError(f"No existe el Python CUDA: {venv_python}")
    env = os.environ.copy()
    cuda_dirs = sorted(
        str(path)
        for path in (venv_path / "lib").glob("python*/site-packages/nvidia/*/lib")
        if path.is_dir()
    )
    current_dirs = [value for value in env.get("LD_LIBRARY_PATH", "").split(":") if value]
    env["LD_LIBRARY_PATH"] = ":".join(
        [*cuda_dirs, *[value for value in current_dirs if value not in cuda_dirs]]
    )
    marker = ":".join(cuda_dirs)
    if env.get(marker_name) != marker:
        env[marker_name] = marker
        os.execve(
            str(venv_python),
            [str(venv_python), str(Path(script_path).resolve()), *sys.argv[1:]],
            env,
        )
    os.environ.update(env)
    ctypes.CDLL("libcublas.so.12")
    ctypes.CDLL("libcudart.so.12")


def configure_runtime(
    *,
    geometry_backend: str = "numba",
    generator_cores: int = 2,
) -> dict[str, Any]:
    args = argparse.Namespace(
        geometry_backend=str(geometry_backend),
        generator_cores=int(generator_cores),
        compute_rve_metrics=False,
        solver_tol=SOLVER_PROFILES["snapshot"]["solver_rtol"],
    )
    runtime = sweep._configure_fft_runtime(args)
    sobol_gpu = runtime["sobol_gpu"]
    sobol_gpu.check_cupy_gpu()
    sobol_gpu.warmup_gpu_once()
    return runtime


def configure_blas_threads(count: int = 8) -> tuple[Any, list[dict[str, Any]]]:
    """Limit every loaded BLAS runtime to a reproducible thread count."""
    threads = int(count)
    if threads < 1:
        raise ValueError("BLAS thread count must be positive.")
    from threadpoolctl import threadpool_info, threadpool_limits

    controller = threadpool_limits(limits=threads, user_api="blas")
    info = [
        {
            "internal_api": item.get("internal_api"),
            "num_threads": int(item.get("num_threads", -1)),
            "version": item.get("version"),
            "architecture": item.get("architecture"),
        }
        for item in threadpool_info()
        if item.get("user_api") == "blas"
    ]
    return controller, info


@dataclass
class GeometryData:
    source_run_dir: Path
    geometry_dir: Path
    design_row: dict[str, Any]
    manifest: dict[str, Any]
    phase: np.ndarray
    ori: np.ndarray


def load_fixed_geometry(source_run_dir: Path) -> GeometryData:
    source_run_dir = Path(source_run_dir).resolve()
    geometry_dir = source_run_dir / "_fixed_geometry"
    manifest_path = geometry_dir / "geometry_manifest.json"
    
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        design = manifest.get("design")
        if not isinstance(design, dict):
            raise RuntimeError(f"{manifest_path} no contiene design.")
        phase = np.load(geometry_dir / "phase.npy").astype(np.uint8)
        ori = np.load(geometry_dir / "ori.npy").astype(np.float32)
    else:
        # Support new geometry format
        geometry_dir = source_run_dir
        generation_path = geometry_dir / "generation_result.json"
        if not generation_path.is_file():
            raise FileNotFoundError(f"No existe geometry manifest ni generation_result en: {source_run_dir}")
        manifest = json.loads(generation_path.read_text(encoding="utf-8"))
        design = manifest # Fallback using generation parameters
        phase = np.load(geometry_dir / "phase.npy").astype(np.uint8)
        ori = np.load(geometry_dir / "ori.npy").astype(np.float32)
    if tuple(ori.shape[:3]) != tuple(phase.shape) or ori.shape[-1] != 3:
        raise ValueError("phase.npy y ori.npy tienen formas incompatibles.")
    return GeometryData(source_run_dir, geometry_dir, design, manifest, phase, ori)


def _cached_record_is_valid(
    material_dir: Path,
    *,
    profile: str,
    require_fields: bool,
) -> bool:
    record_path = material_dir / "solve_record.json"
    if not record_path.is_file() or not (material_dir / "Ceff.npy").is_file():
        return False
    if require_fields and not _snapshot_fields_available(material_dir):
        return False
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return bool(record.get("solver_all_converged", False)) and record.get("solver_profile") == profile


def _snapshot_fields_available(material_dir: Path) -> bool:
    material_dir = Path(material_dir)
    field_dir = material_dir / "solution_fields"
    if field_dir.is_dir() and all(
        (field_dir / f"fluctuation_load{load_id}.npy").is_file()
        for load_id in range(6)
    ):
        return True
    return (material_dir / "solution_fields.npz").is_file()


def solve_material(
    *,
    material_row: dict[str, Any],
    material_dir: Path,
    geometry: GeometryData,
    runtime: dict[str, Any],
    profile: str,
    seed: int,
    save_solution_fields: bool,
    persistent_gpu_cache: bool = False,
) -> dict[str, Any]:
    """Solve one material or return a validated campaign-owned cache entry."""
    if profile not in SOLVER_PROFILES:
        raise ValueError(f"Perfil desconocido: {profile}")
    material_dir = Path(material_dir)
    material_dir.mkdir(parents=True, exist_ok=True)
    if _cached_record_is_valid(
        material_dir, profile=profile, require_fields=save_solution_fields
    ):
        return json.loads((material_dir / "solve_record.json").read_text(encoding="utf-8"))

    write_json(material_dir / "material.json", material_row)
    sobol_gpu = runtime["sobol_gpu"]
    params = sweep._material_solver_params(
        sobol_gpu=sobol_gpu,
        config=runtime["config"],
        material_row=material_row,
        design_row=geometry.design_row,
        material_dir=material_dir,
        seed=int(seed),
        save_solution_fields=bool(save_solution_fields),
    )
    settings = SOLVER_PROFILES[profile]
    params.update(
        {
            "solver_profile": profile,
            "solver_real_dtype": settings["solver_real_dtype"],
            "solver_rtol": float(settings["solver_rtol"]),
            "solver_atol": float(settings["solver_atol"]),
            "cfield_storage": "sym21",
            "cfield_indexed": settings["solver_real_dtype"] == "float32",
            "projection_storage": "direct" if settings["solver_real_dtype"] == "float32" else "full",
            "projection_backend": "numpy" if settings["solver_real_dtype"] == "float32" else "cupy",
            "phase_array": geometry.phase,
            "ori_array": geometry.ori,
            "preloaded_geometry": True,
            "cache_projection": bool(persistent_gpu_cache),
            "free_gpu_memory_after_solve": not bool(persistent_gpu_cache),
        }
    )
    started = time.perf_counter()
    ceff = np.asarray(sobol_gpu.solve_homogenization(params), dtype=np.float64)
    solve_wall_s = float(time.perf_counter() - started)
    np.save(material_dir / "Ceff.npy", ceff)

    timing_path = material_dir / "solver_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else {}
    load_summary = timing.get("load_solver_summary", {})
    all_converged = bool(load_summary.get("all_converged", False))
    max_residual = float(load_summary.get("final_norm_res_rel_max", np.nan))
    maximum_allowed = 1.05 * float(settings["solver_rtol"])
    sym = 0.5 * (ceff + ceff.T)
    symmetry = float(np.linalg.norm(ceff - ceff.T) / max(np.linalg.norm(sym), np.finfo(float).eps))
    eigenvalues = np.linalg.eigvalsh(sym)
    if not all_converged or not np.isfinite(max_residual) or max_residual > maximum_allowed:
        raise RuntimeError(
            f"Material no convergente en {material_dir}: converged={all_converged}, "
            f"residual={max_residual:.3e}, allowed={maximum_allowed:.3e}."
        )
    if ceff.shape != (6, 6) or not np.all(np.isfinite(ceff)):
        raise RuntimeError(f"Tensor incompleto o no finito en {material_dir}.")
    if float(eigenvalues.min()) <= 0.0:
        raise RuntimeError(f"Tensor efectivo no SPD en {material_dir}.")
    if save_solution_fields and not _snapshot_fields_available(material_dir):
        raise RuntimeError(f"Faltan campos snapshot en {material_dir}.")

    record: dict[str, Any] = {
        **material_row,
        "material_dir": str(material_dir),
        "Ceff_path": str(material_dir / "Ceff.npy"),
        "solution_fields_path": (
            str(material_dir / "solution_fields")
            if (material_dir / "solution_fields").is_dir()
            else str(material_dir / "solution_fields.npz")
        )
        if save_solution_fields
        else "",
        "solver_timing_path": str(timing_path),
        "solver_profile": profile,
        "solver_real_dtype": settings["solver_real_dtype"],
        "solver_rtol": float(settings["solver_rtol"]),
        "solver_atol": float(settings["solver_atol"]),
        "persistent_gpu_cache": bool(persistent_gpu_cache),
        "solver_all_converged": all_converged,
        "solver_max_relative_residual": max_residual,
        "solver_max_iterations": int(load_summary.get("cg_iterations_max", -1)),
        "solve_wall_s": solve_wall_s,
        "Ceff_symmetry_rel": symmetry,
        "Ceff_min_eig": float(eigenvalues.min()),
        "Ceff_max_eig": float(eigenvalues.max()),
        "phase_sha256": geometry.manifest.get("phase_sha256", ""),
        "ori_sha256": geometry.manifest.get("ori_sha256", ""),
    }
    properties = sobol_gpu.engineering_constants_from_Cmandel(sym)
    for name in sweep.ENGINEERING_COLUMNS:
        record[name] = float(properties.get(name, np.nan))
    for ii in range(6):
        for jj in range(6):
            record[f"Ceff_{ii + 1}{jj + 1}"] = float(ceff[ii, jj])
    write_json(material_dir / "solve_record.json", record)
    return record


def load_snapshot_fields(material_dir: Path) -> list[np.ndarray]:
    material_dir = Path(material_dir)
    field_dir = material_dir / "solution_fields"
    if field_dir.is_dir():
        return [
            np.asarray(
                np.load(field_dir / f"fluctuation_load{load_id}.npy", mmap_mode="r"),
                dtype=np.float64,
            )
            for load_id in range(6)
        ]
    path = material_dir / "solution_fields.npz"
    if not path.is_file():
        raise FileNotFoundError(f"No existe snapshot: {path} ni {field_dir}.")
    with np.load(path) as payload:
        return [np.asarray(payload[f"fluctuation_load{load_id}"], dtype=np.float64) for load_id in range(6)]


def snapshot_dir(out_dir: Path, candidate_id: int) -> Path:
    return Path(out_dir) / "snapshot_cache" / f"candidate_{int(candidate_id):04d}"


_snapshot_dir = snapshot_dir


def candidate_table(campaign_dir: Path, pool_size: int | None = None) -> pd.DataFrame:
    path = Path(campaign_dir) / "candidate_design.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No existe candidate_design: {path}")
    df = pd.read_csv(path)
    if pool_size is not None:
        df = df.iloc[: int(pool_size)].copy()
    return df


_candidate_table = candidate_table


def verify_frozen_design(campaign_dir: Path) -> None:
    manifest_path = Path(campaign_dir) / "campaign_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("frozen_candidate_design", False):
        raise RuntimeError(f"Campaign design is not frozen in {campaign_dir}")


_verify_frozen_design = verify_frozen_design


def ensure_snapshot(
    *,
    candidate_id: int,
    candidates: pd.DataFrame,
    out_dir: Path,
    geometry: GeometryData,
    runtime: dict[str, Any],
    seed: int,
    profile: str = "snapshot",
    persistent_gpu_cache: bool = False,
) -> dict[str, Any]:
    selected = candidates.loc[candidates["candidate_id"] == int(candidate_id)]
    if len(selected) != 1:
        raise KeyError(f"candidate_id={candidate_id} no es unico en el pool activo.")
    row = selected.iloc[0]
    material = {name: float(row[name]) for name in sweep.MATERIAL_BOUNDS if name in row}
    material["candidate_id"] = int(candidate_id)
    return solve_material(
        material_row=material,
        material_dir=snapshot_dir(out_dir, candidate_id),
        geometry=geometry,
        runtime=runtime,
        profile=str(profile),
        seed=int(seed) + int(candidate_id),
        save_solution_fields=True,
        persistent_gpu_cache=bool(persistent_gpu_cache),
    )


_ensure_snapshot = ensure_snapshot


def append_orthonormal(
    basis: list[np.ndarray],
    fields: Any,
    *,
    tolerance: float,
    basis_block_size: int = 12,
) -> list[np.ndarray]:
    appended: list[np.ndarray] = []
    for field in fields:
        vector = np.asarray(field, dtype=np.float64).copy()
        for block in (basis, appended, basis, appended):
            for base in block:
                proj = float(np.mean(base * vector))
                vector -= proj * base
        norm = float(np.sqrt(max(np.mean(vector * vector), 0.0)))
        if norm > float(tolerance):
            vec_norm = vector / norm
            appended.append(vec_norm)
            basis.append(vec_norm)
    return appended


_append_orthonormal = append_orthonormal



def error_stats(frame: pd.DataFrame) -> dict[str, Any]:
    errors = frame["relative_frobenius_error"].to_numpy(dtype=float)
    out: dict[str, Any] = {
        "error_mean": float(np.mean(errors)),
        "error_p95": float(np.quantile(errors, 0.95)),
        "error_max": float(np.max(errors)),
        "error_median": float(np.median(errors)),
    }
    if "validation_id" in frame.columns:
        out["worst_validation_id"] = int(frame.iloc[int(np.argmax(errors))]["validation_id"])
    if "rom_online_s" in frame.columns:
        out["rom_online_median_s"] = float(frame["rom_online_s"].median())
        out["rom_online_p95_s"] = float(frame["rom_online_s"].quantile(0.95))
    return out


_error_stats = error_stats



