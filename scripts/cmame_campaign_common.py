#!/usr/bin/env python3
"""Shared, resumable FFTHomPy campaign utilities for the CMAME study."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.linalg import blas as scipy_blas


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
    solver_tol: float | None = None,
    fft_backend: str = "gpu",
) -> dict[str, Any]:
    runtime_tol = (
        float(SOLVER_PROFILES["snapshot"]["solver_rtol"])
        if solver_tol is None
        else float(solver_tol)
    )
    args = argparse.Namespace(
        geometry_backend=str(geometry_backend),
        generator_cores=int(generator_cores),
        compute_rve_metrics=False,
        solver_tol=runtime_tol,
    )
    runtime = sweep._configure_fft_runtime(args)
    sobol_gpu = runtime["sobol_gpu"]
    sobol_gpu.SOLVER_TOL = runtime_tol
    backend = str(fft_backend).lower()
    if backend not in {"cpu", "gpu"}:
        raise ValueError("fft_backend must be cpu or gpu.")
    solver_backend = "cupy" if backend == "gpu" else "scipy"
    sobol_gpu.FFT_BACKEND = solver_backend
    runtime["config"]["fft_backend"] = solver_backend
    if backend == "gpu":
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


def available_memory_bytes() -> int:
    """Return currently available host memory without an optional dependency."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    return page_size * available_pages


def cpu_resource_info() -> dict[str, Any]:
    """Report CPU capacity visible to this process, including physical cores."""
    try:
        affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except AttributeError:
        affinity = list(range(int(os.cpu_count() or 1)))

    physical_ids: set[tuple[int, int]] = set()
    for cpu in affinity:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package_id = int(
                (topology / "physical_package_id").read_text(encoding="ascii")
            )
            core_id = int((topology / "core_id").read_text(encoding="ascii"))
        except (OSError, ValueError):
            physical_ids.clear()
            break
        physical_ids.add((package_id, core_id))

    logical_count = max(1, len(affinity))
    physical_count = len(physical_ids) if physical_ids else logical_count
    scheduler_limits = []
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        try:
            value = int(os.environ.get(name, ""))
        except ValueError:
            continue
        if value > 0:
            scheduler_limits.append(value)
    scheduler_limit = min(scheduler_limits) if scheduler_limits else logical_count
    auto_workers = max(1, min(physical_count, scheduler_limit))
    return {
        "logical_cpus_available": logical_count,
        "physical_cores_available": physical_count,
        "scheduler_cpu_limit": scheduler_limit,
        "auto_workers": auto_workers,
        "affinity": affinity,
    }


def resolve_cpu_workers(value: str | int, *, resource_info: dict[str, Any] | None = None) -> int:
    """Resolve an explicit worker count or a portable physical-core policy."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        info = cpu_resource_info() if resource_info is None else resource_info
        return int(info["auto_workers"])
    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CPU worker count must be 'auto' or a positive integer.") from exc
    if workers < 1:
        raise ValueError("CPU worker count must be positive.")
    return workers


def full_rank_memory_plan(
    *,
    nvox: int,
    max_rank: int,
    basis_dtype: str | np.dtype,
    pod_batch_max_gib: float,
    affine_stress_max_gib: float,
    memory_safety_fraction: float,
    max_material_batch: int | None = None,
    available_bytes: int | None = None,
) -> dict[str, int | float]:
    """Plan bounded workspaces and estimate the exact full-rank peak memory."""
    voxel_count = int(nvox)
    rank = int(max_rank)
    dtype = np.dtype(basis_dtype)
    fraction = float(memory_safety_fraction)
    if voxel_count < 1 or rank < 1:
        raise ValueError("nvox and max_rank must be positive.")
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("basis_dtype must be float32 or float64.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("memory_safety_fraction must be in (0, 1].")

    gib = 1024**3
    pod_cap = int(float(pod_batch_max_gib) * gib)
    stress_cap = int(float(affine_stress_max_gib) * gib)
    if pod_cap < 1 or stress_cap < 1:
        raise ValueError("POD and affine stress memory limits must be positive.")

    field_bytes = 6 * voxel_count * dtype.itemsize
    material_snapshot_bytes = 6 * field_bytes
    stress_coefficient_bytes = 6 * field_bytes
    basis_bytes = rank * field_bytes
    geometry_bytes = voxel_count * (np.dtype(np.uint8).itemsize + 3 * np.dtype(np.float32).itemsize)
    available = int(available_bytes or available_memory_bytes())
    safe_bytes = int(available * fraction)
    workspace_budget = max(0, safe_bytes - basis_bytes - geometry_bytes)

    memory_limited_materials = max(1, pod_cap // material_snapshot_bytes)
    requested_materials = (
        memory_limited_materials
        if max_material_batch is None
        else max(1, int(max_material_batch))
    )
    rank_limited_materials = max(
        1, workspace_budget // material_snapshot_bytes - 1
    )
    pod_material_limit = min(
        memory_limited_materials,
        requested_materials,
        rank_limited_materials,
    )
    configured_q_block_size = min(
        7, max(1, stress_cap // stress_coefficient_bytes)
    )
    rank_limited_q_block_size = max(
        1, workspace_budget // stress_coefficient_bytes
    )
    affine_q_block_size = min(
        configured_q_block_size, rank_limited_q_block_size
    )
    pod_workspace_bytes = pod_material_limit * material_snapshot_bytes
    projection_row_block_size = min(6, 6 * pod_material_limit)
    projection_workspace_bytes = projection_row_block_size * field_bytes
    stress_workspace_bytes = affine_q_block_size * stress_coefficient_bytes
    nonbasis_peak_bytes = geometry_bytes + max(
        pod_workspace_bytes
        + max(material_snapshot_bytes, projection_workspace_bytes),
        stress_workspace_bytes,
    )
    estimated_peak = basis_bytes + nonbasis_peak_bytes
    minimum_workspace_bytes = geometry_bytes + max(
        2 * material_snapshot_bytes,
        stress_coefficient_bytes,
    )
    max_safe_rank = max(
        0, (safe_bytes - minimum_workspace_bytes) // field_bytes
    )
    return {
        "available_memory_bytes": available,
        "safe_memory_bytes": safe_bytes,
        "estimated_peak_bytes": estimated_peak,
        "nonbasis_peak_bytes": nonbasis_peak_bytes,
        "max_safe_rank": max_safe_rank,
        "basis_bytes": basis_bytes,
        "field_bytes": field_bytes,
        "material_snapshot_bytes": material_snapshot_bytes,
        "pod_workspace_bytes": pod_workspace_bytes,
        "pod_projection_workspace_bytes": projection_workspace_bytes,
        "stress_workspace_bytes": stress_workspace_bytes,
        "pod_batch_material_limit": pod_material_limit,
        "pod_requested_material_batch": requested_materials,
        "pod_memory_limited_material_batch": memory_limited_materials,
        "pod_rank_limited_material_batch": rank_limited_materials,
        "pod_projection_row_block_size": projection_row_block_size,
        "affine_q_block_size": affine_q_block_size,
        "affine_configured_q_block_size": configured_q_block_size,
        "affine_rank_limited_q_block_size": rank_limited_q_block_size,
        "workspace_budget_after_basis_bytes": workspace_budget,
        "memory_safety_fraction": fraction,
    }


def runtime_affine_q_block_size(
    *,
    appended_fields_bytes: int,
    coefficient_count: int,
    memory_max_gib: float,
    memory_safety_fraction: float,
    available_bytes: int | None = None,
    coefficient_supports: tuple[tuple[int, float], ...] | None = None,
) -> int:
    """Choose the widest safe affine block for the current basis update."""
    bytes_per_coefficient = int(appended_fields_bytes)
    count = int(coefficient_count)
    fraction = float(memory_safety_fraction)
    if bytes_per_coefficient < 1 or count < 1:
        raise ValueError("Appended-field bytes and coefficient count must be positive.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("memory_safety_fraction must be in (0, 1].")

    configured_budget = int(float(memory_max_gib) * 1024**3)
    current_budget = int(
        (available_memory_bytes() if available_bytes is None else available_bytes)
        * fraction
    )
    workspace_budget = min(configured_budget, current_budget)
    supports = coefficient_supports or ((count, 1.0),)

    def workspace_bytes(block_size: int) -> int:
        return int(
            max(
                min(int(block_size), int(support_count)) * float(fraction_of_domain)
                for support_count, fraction_of_domain in supports
            )
            * bytes_per_coefficient
        )

    if workspace_budget < workspace_bytes(1):
        raise MemoryError(
            "Insufficient available memory for one affine stress coefficient block."
        )
    for block_size in range(count, 0, -1):
        if workspace_bytes(block_size) <= workspace_budget:
            return block_size
    raise RuntimeError("Could not derive an affine coefficient block size.")


def rom_chunk_memory_plan(
    *,
    rank: int,
    requested_chunk_size: int,
    memory_max_gib: float,
) -> dict[str, int | float]:
    """Bound batched dense ROM workspaces, whose memory grows as O(chunk*r^2)."""
    reduced_rank = int(rank)
    requested = int(requested_chunk_size)
    max_bytes = int(float(memory_max_gib) * 1024**3)
    if reduced_rank < 1 or requested < 1 or max_bytes < 1:
        raise ValueError("rank, chunk size, and ROM memory limit must be positive.")
    bytes_per_query = 8 * (
        4 * reduced_rank * reduced_rank + 12 * reduced_rank + 72
    )
    memory_limited_chunk = max(1, max_bytes // bytes_per_query)
    effective_chunk = min(requested, memory_limited_chunk)
    return {
        "requested_chunk_size": requested,
        "effective_chunk_size": effective_chunk,
        "memory_limited_chunk_size": memory_limited_chunk,
        "bytes_per_query_estimate": bytes_per_query,
        "workspace_bytes_estimate": effective_chunk * bytes_per_query,
        "memory_max_gib": float(memory_max_gib),
    }


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
    return_solution_fields: bool = False,
    solution_field_dtype: str | np.dtype | None = None,
    solution_field_consumer: Callable[[int, np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Solve one material or return a validated campaign-owned cache entry."""
    if profile not in SOLVER_PROFILES:
        raise ValueError(f"Perfil desconocido: {profile}")
    in_memory_fields = return_solution_fields or solution_field_consumer is not None
    if in_memory_fields and not save_solution_fields:
        raise ValueError("In-memory solution fields require save_solution_fields=True.")
    material_dir = Path(material_dir)
    material_dir.mkdir(parents=True, exist_ok=True)
    if not in_memory_fields and _cached_record_is_valid(
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
    if return_solution_fields:
        params.pop("solution_field_out_path", None)
        params["solution_field_return_in_memory"] = True
    if solution_field_consumer is not None:
        params.pop("solution_field_out_path", None)
        params["solution_field_consumer"] = solution_field_consumer
    if solution_field_dtype is not None:
        params["solution_field_dtype"] = str(np.dtype(solution_field_dtype))
    started = time.perf_counter()
    try:
        ceff = np.asarray(sobol_gpu.solve_homogenization(params), dtype=np.float64)
    finally:
        if not persistent_gpu_cache:
            # The local FFTHomPy Problem must be unreachable before CuPy can
            # release its large FFT workspaces.
            gc.collect()
            sobol_gpu.free_gpu_memory_pool(clear_fft_cache=True)
    solve_wall_s = float(time.perf_counter() - started)
    solution_fields = params.pop("_solution_fields_result", None)
    consumed_fields = params.pop("_solution_fields_consumed", None)
    params.pop("solution_field_consumer", None)
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
    if save_solution_fields:
        if solution_field_consumer is not None:
            if consumed_fields != tuple(range(6)):
                raise RuntimeError(
                    f"El consumidor no recibio los seis campos snapshot en {material_dir}."
                )
        elif return_solution_fields:
            if solution_fields is None or len(solution_fields) != 6:
                raise RuntimeError(
                    f"El solver no devolvio los seis campos snapshot en {material_dir}."
                )
        elif not _snapshot_fields_available(material_dir):
            raise RuntimeError(f"Faltan campos snapshot en {material_dir}.")

    solution_field_transport = "none"
    if save_solution_fields:
        solution_field_transport = "memory" if in_memory_fields else "disk"

    record: dict[str, Any] = {
        **material_row,
        "material_dir": str(material_dir),
        "Ceff_path": str(material_dir / "Ceff.npy"),
        "solution_fields_path": (
            str(material_dir / "solution_fields")
            if (material_dir / "solution_fields").is_dir()
            else str(material_dir / "solution_fields.npz")
        )
        if save_solution_fields and not in_memory_fields
        else "",
        "solution_field_transport": solution_field_transport,
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
    if solution_fields is not None:
        record["_solution_fields_result"] = solution_fields
    return record


def load_snapshot_fields(
    material_dir: Path,
    *,
    dtype: str | np.dtype = np.float64,
) -> list[np.ndarray]:
    material_dir = Path(material_dir)
    field_dtype = np.dtype(dtype)
    if field_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("snapshot dtype must be float32 or float64.")
    field_dir = material_dir / "solution_fields"
    if field_dir.is_dir():
        return [
            np.asarray(
                np.load(field_dir / f"fluctuation_load{load_id}.npy", mmap_mode="r"),
                dtype=field_dtype,
            )
            for load_id in range(6)
        ]
    path = material_dir / "solution_fields.npz"
    if not path.is_file():
        raise FileNotFoundError(f"No existe snapshot: {path} ni {field_dir}.")
    with np.load(path) as payload:
        return [
            np.asarray(payload[f"fluctuation_load{load_id}"], dtype=field_dtype)
            for load_id in range(6)
        ]


class ContiguousBasis:
    """Preallocated full-rank POD storage without repeated basis copies."""

    def __init__(
        self,
        capacity: int,
        field_shape: tuple[int, ...],
        *,
        dtype: str | np.dtype = np.float32,
        projection_row_block_size: int = 6,
    ) -> None:
        if int(capacity) < 1:
            raise ValueError("basis capacity must be positive.")
        self.field_shape = tuple(int(value) for value in field_shape)
        self.dimension = int(np.prod(self.field_shape))
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("basis dtype must be float32 or float64.")
        if int(projection_row_block_size) < 1:
            raise ValueError("projection_row_block_size must be positive.")
        self.projection_row_block_size = int(projection_row_block_size)
        self.last_projection_backend = "not_used"
        self._values = np.empty((int(capacity), self.dimension), dtype=self.dtype)
        self.rank = 0

    def __len__(self) -> int:
        return int(self.rank)

    @property
    def capacity(self) -> int:
        return int(self._values.shape[0])

    @property
    def active_flat(self) -> np.ndarray:
        return self._values[: self.rank]

    @property
    def active_fields(self) -> np.ndarray:
        return self.active_flat.reshape((self.rank,) + self.field_shape)

    def _project(self, values: np.ndarray, count: float) -> None:
        basis = self.active_flat
        coefficients = (values @ basis.T) / count
        gemm = scipy_blas.sgemm if self.dtype == np.dtype(np.float32) else scipy_blas.dgemm
        updated = gemm(
            -1.0,
            basis.T,
            coefficients.T,
            beta=1.0,
            c=values.T,
            overwrite_c=1,
        )
        if np.shares_memory(updated, values):
            self.last_projection_backend = "scipy_blas_gemm_in_place"
            return

        # Unusual BLAS wrappers may copy the Fortran-contiguous transpose.
        # Fall back to a bounded row workspace instead of retaining that copy.
        del updated
        self.last_projection_backend = "blocked_numpy_fallback"
        block_size = min(self.projection_row_block_size, len(values))
        workspace = np.empty((block_size, self.dimension), dtype=self.dtype)
        for start in range(0, len(values), block_size):
            end = min(start + block_size, len(values))
            active_workspace = workspace[: end - start]
            np.matmul(coefficients[start:end], basis, out=active_workspace)
            values[start:end] -= active_workspace

    def _append_values(self, values: np.ndarray, *, tolerance: float) -> np.ndarray:
        values = np.asarray(values, dtype=self.dtype).reshape(-1, self.dimension)
        count = float(self.dimension)
        if self.rank:
            self._project(values, count)
        else:
            self.last_projection_backend = "initial_block_no_projection"

        gram = np.asarray((values @ values.T) / count, dtype=np.float64)
        gram = 0.5 * (gram + gram.T)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        keep = eigenvalues > float(tolerance) ** 2
        if not np.any(keep):
            return self.active_fields[0:0]

        indices = np.flatnonzero(keep)[::-1]
        appended_count = int(len(indices))
        if self.rank + appended_count > self.capacity:
            raise MemoryError(
                f"basis capacity {self.capacity} is smaller than requested rank "
                f"{self.rank + appended_count}."
            )
        transform = (
            eigenvectors[:, indices] / np.sqrt(eigenvalues[indices])
        ).T.astype(self.dtype, copy=False)
        start = self.rank
        np.matmul(transform, values, out=self._values[start : start + appended_count])
        self.rank += appended_count
        return self.active_fields[start : self.rank]

    def append_preordered(
        self,
        fields: np.ndarray,
        *,
        tolerance: float,
    ) -> np.ndarray:
        """Consume a contiguous block that already uses the basis voxel order."""
        values = np.asarray(fields)
        expected = (len(values),) + self.field_shape
        if values.dtype != self.dtype or values.shape != expected or not values.flags.c_contiguous:
            raise ValueError(
                "preordered fields must be C-contiguous with the basis dtype and shape."
            )
        return self._append_values(values, tolerance=float(tolerance))

    def append(
        self,
        fields: Any,
        *,
        tolerance: float,
        voxel_order: np.ndarray | None = None,
    ) -> np.ndarray:
        incoming = [np.asarray(field, dtype=self.dtype) for field in fields]
        if not incoming:
            return self.active_fields[0:0]
        if any(int(field.size) != self.dimension for field in incoming):
            raise ValueError("incoming snapshot fields have incompatible sizes.")

        if voxel_order is None:
            if any(tuple(field.shape) != self.field_shape for field in incoming):
                raise ValueError("incoming snapshot fields have incompatible shapes.")
            values = np.stack(incoming, axis=0).reshape(len(incoming), self.dimension)
        else:
            order = np.asarray(voxel_order)
            nvox = int(order.size)
            if self.field_shape != (6, nvox):
                raise ValueError("ordered basis storage must have shape (6, nvox).")
            values = np.empty((len(incoming), 6, nvox), dtype=self.dtype)
            for index, field in enumerate(incoming):
                np.take(field.reshape(6, nvox), order, axis=1, out=values[index])
        return self._append_values(values, tolerance=float(tolerance))


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
    return_solution_fields: bool = False,
    solution_field_dtype: str | np.dtype | None = None,
    solution_field_consumer: Callable[[int, np.ndarray], None] | None = None,
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
        return_solution_fields=bool(return_solution_fields),
        solution_field_dtype=solution_field_dtype,
        solution_field_consumer=solution_field_consumer,
    )


_ensure_snapshot = ensure_snapshot


def _project_against_basis_blocks(
    vector: np.ndarray,
    basis: list[np.ndarray],
    *,
    basis_block_size: int,
) -> None:
    if not basis:
        return
    flat = vector.reshape(-1)
    count = float(flat.size)
    # A wider block turns many memory-bound level-2 products into one BLAS-3
    # projection. Campaign machines have ample host RAM; cap the temporary so
    # very high ranks still stream in bounded chunks.
    max_block_bytes = 2 * 1024 * 1024 * 1024
    bytes_per_field = max(int(flat.nbytes), 1)
    block_size = max(
        1,
        min(int(basis_block_size), max(1, max_block_bytes // bytes_per_field)),
    )
    for start in range(0, len(basis), block_size):
        block = np.stack(
            [
                np.asarray(base, dtype=np.float64).reshape(-1)
                for base in basis[start : start + block_size]
            ],
            axis=0,
        )
        coefficients = block @ flat / count
        flat -= coefficients @ block


def _project_matrix_against_basis_blocks(
    values: np.ndarray,
    basis: list[np.ndarray],
    *,
    basis_block_size: int,
) -> None:
    """Project several snapshot fields in one pass over each basis block."""
    if not basis:
        return
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    count = float(flat.shape[1])
    max_block_bytes = 2 * 1024 * 1024 * 1024
    bytes_per_field = max(int(flat.shape[1] * flat.dtype.itemsize), 1)
    block_size = max(
        1,
        min(int(basis_block_size), max(1, max_block_bytes // bytes_per_field)),
    )
    for start in range(0, len(basis), block_size):
        block = np.stack(
            [
                np.asarray(base, dtype=np.float64).reshape(-1)
                for base in basis[start : start + block_size]
            ],
            axis=0,
        )
        coefficients = flat @ block.T / count
        flat -= coefficients @ block


def append_orthonormal(
    basis: list[np.ndarray],
    fields: Any,
    *,
    tolerance: float,
    basis_block_size: int = 12,
) -> list[np.ndarray]:
    incoming = [np.asarray(field, dtype=np.float64) for field in fields]
    if not incoming:
        return []

    # Block modified Gram-Schmidt: project all six load snapshots together,
    # then orthonormalize their tiny Gram matrix. This preserves the incoming
    # snapshot subspace while avoiding a full basis sweep for every load.
    values = np.stack(incoming, axis=0)
    for _ in range(2):
        _project_matrix_against_basis_blocks(
            values,
            basis,
            basis_block_size=int(basis_block_size),
        )
    flat = values.reshape(len(values), -1)
    count = float(flat.shape[1])
    gram = (flat @ flat.T) / count
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > float(tolerance) ** 2
    if not np.any(keep):
        return []

    # Eigenvectors are ordered from the smallest eigenvalue upward. Reversing
    # them keeps the best-conditioned directions first and makes the retained
    # block deterministic for a fixed snapshot set.
    indices = np.flatnonzero(keep)[::-1]
    transform = (eigenvectors[:, indices] / np.sqrt(eigenvalues[indices])).T
    orthonormal = transform @ flat
    appended = [
        row.reshape(incoming[0].shape)
        for row in orthonormal
    ]
    basis.extend(appended)
    return appended


_append_orthonormal = append_orthonormal


def save_basis_block(method_dir: Path, start_index: int, fields: list[np.ndarray]) -> None:
    basis_dir = Path(method_dir) / "basis_fields"
    basis_dir.mkdir(parents=True, exist_ok=True)
    for offset, field in enumerate(fields):
        np.save(basis_dir / f"basis_{int(start_index) + offset:04d}.npy", np.asarray(field))


_save_basis_block = save_basis_block



def load_operators(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as payload:
        return {
            "Kq": np.asarray(payload["Kq"], dtype=np.float64),
            "Bq": np.asarray(payload["Bq"], dtype=np.float64),
            "Dq": np.asarray(payload["Dq"], dtype=np.float64),
        }


_load_operators = load_operators


def update_reduced_operators(
    *,
    phase: np.ndarray,
    ori: np.ndarray,
    basis: np.ndarray | list[np.ndarray],
    existing: dict[str, np.ndarray] | None,
    new_fields: np.ndarray | list[np.ndarray],
    affine_stress_batch: Any,
    affine_q_block_size: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import rom_reduced_operator as reduced

    if existing is None:
        Kq, Bq, Dq, metadata = reduced._assemble_reduced_operators(
            phase=phase,
            ori=ori,
            basis=basis,
            affine_stress_batch=affine_stress_batch,
            affine_q_block_size=affine_q_block_size,
        )
    else:
        old_basis = basis[: -len(new_fields)]
        Kq, Bq, Dq, metadata = reduced._extend_reduced_operators(
            existing=existing,
            old_basis=old_basis,
            new_basis=new_fields,
            affine_stress_batch=affine_stress_batch,
            affine_q_block_size=affine_q_block_size,
        )
    return {"Kq": Kq, "Bq": Bq, "Dq": Dq}, metadata


_update_reduced_operators = update_reduced_operators


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


RANK_THRESHOLDS = (1.0e-2, 1.0e-3, 1.0e-4)
RANK_DESCRIPTORS = (
    "Vf_realized",
    "aspect_ratio",
    "A2_anisotropy",
    "cluster_fraction_target",
    "interface_density",
    "Ripley_peak",
    "D_star",
    "n_fibers",
)


def threshold_tag(threshold: float) -> str:
    exponent = int(round(-math.log10(float(threshold))))
    return f"1e-{exponent}"


def required_rank(curve: pd.DataFrame, threshold: float) -> dict[str, Any]:
    ordered = curve.sort_values("rank").reset_index(drop=True)
    errors = ordered["error_max"].to_numpy(dtype=float)
    ranks = ordered["rank"].to_numpy(dtype=int)
    suffix_max = np.maximum.accumulate(errors[::-1])[::-1]
    first = np.flatnonzero(errors <= float(threshold))
    stable = np.flatnonzero(suffix_max <= float(threshold))
    return {
        "first_rank": int(ranks[first[0]]) if len(first) else None,
        "stable_rank": int(ranks[stable[0]]) if len(stable) else None,
        "achieved": bool(len(stable)),
        "max_tested_rank": int(ranks[-1]),
        "error_at_max_rank": float(errors[-1]),
    }


def descriptor_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import pearsonr, spearmanr

    geometry = summary.loc[summary["case_kind"] == "geometry"].copy()
    rows: list[dict[str, Any]] = []
    for threshold in RANK_THRESHOLDS:
        rank_column = f"r_{threshold_tag(threshold)}"
        if rank_column not in geometry:
            continue
        for descriptor in RANK_DESCRIPTORS:
            if descriptor not in geometry:
                continue
            subset = geometry[[descriptor, rank_column]].dropna()
            x = subset[descriptor].to_numpy(dtype=float)
            y = subset[rank_column].to_numpy(dtype=float)
            if len(subset) >= 3 and np.ptp(x) > 0.0 and np.ptp(y) > 0.0:
                pearson = pearsonr(x, y)
                spearman = spearmanr(x, y)
                pearson_r = float(pearson.statistic)
                pearson_p = float(pearson.pvalue)
                spearman_rho = float(spearman.statistic)
                spearman_p = float(spearman.pvalue)
            else:
                pearson_r = pearson_p = spearman_rho = spearman_p = np.nan
            rows.append(
                {
                    "threshold": threshold,
                    "rank_column": rank_column,
                    "descriptor": descriptor,
                    "geometry_count": int(len(subset)),
                    "censored_count": int(len(geometry) - len(subset)),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                    "interpretation": "exploratory_n10_not_universal_law",
                }
            )
    return pd.DataFrame(rows)


def isotropic_moduli(E: float, nu: float) -> tuple[float, float]:
    return E / (3.0 * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def hashin_shtrikman_bounds(
    K1: float,
    G1: float,
    K2: float,
    G2: float,
    volume_fraction_2: float,
) -> tuple[float, float, float, float]:
    if not (K2 > K1 > 0.0 and G2 > G1 > 0.0):
        raise ValueError("Phase 2 must be strictly stiffer than phase 1.")
    vf = float(volume_fraction_2)
    vm = 1.0 - vf
    zeta1 = G1 * (9.0 * K1 + 8.0 * G1) / (6.0 * (K1 + 2.0 * G1))
    zeta2 = G2 * (9.0 * K2 + 8.0 * G2) / (6.0 * (K2 + 2.0 * G2))
    K_lower = K1 + vf / (1.0 / (K2 - K1) + vm / (K1 + 4.0 * G1 / 3.0))
    K_upper = K2 + vm / (1.0 / (K1 - K2) + vf / (K2 + 4.0 * G2 / 3.0))
    G_lower = G1 + vf / (1.0 / (G2 - G1) + vm / (G1 + zeta1))
    G_upper = G2 + vm / (1.0 / (G1 - G2) + vf / (G2 + zeta2))
    return K_lower, K_upper, G_lower, G_upper
