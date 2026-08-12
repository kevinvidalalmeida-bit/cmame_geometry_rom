"""Etapa 2: genera RVEs, resuelve FFT/CuPy y para por convergencia Kanit."""

from __future__ import annotations

import multiprocessing
import json
import math
import os
import platform
import signal
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# RUTAS
# ============================================================================

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
FFTHOMPY_DIR = WORKSPACE_DIR / "ffthompy_core" / "ffthompy"

for directory in (WORKSPACE_DIR, FFTHOMPY_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


from pipeline.rve_generator import generate_rve_main as generate_rve_main_sam
from pipeline.fft_solver import solve_homogenization
from pipeline.sam_generator import warmup_numba_geometry_kernels
from ffthompy_core.ffthompy.RESU import engineering_constants_from_Cmandel


# ============================================================================
# DEFINICIONES LOCALES
# ============================================================================

ENGINEERING_PROPERTY_COLUMNS = [
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


def _ignore_sigint_in_worker() -> None:
    """Deja que el proceso principal gestione Ctrl+C y cierre el pool."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def derived_mc_seeds(
    design_id: int,
    n_seeds: int,
    base_seed: int,
) -> List[int]:
    if n_seeds < 1:
        raise ValueError("n_seeds debe ser >= 1.")

    rng = np.random.default_rng(
        np.random.SeedSequence([int(base_seed), int(design_id)])
    )

    seeds: List[int] = []
    used = set()

    while len(seeds) < int(n_seeds):
        candidate = int(rng.integers(1, 1_000_000_000))
        if candidate in used:
            continue

        used.add(candidate)
        seeds.append(candidate)

    return seeds


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return bool(default)

    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "sí", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return int(default)
    return int(value)


def env_optional_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return float(default)
    return float(value)


# ============================================================================
# USER HEADER
# ============================================================================

STUDY_NAME = os.environ.get("STUDY_NAME", "Estudio_Sobol_Continuo")
PHASE_LABEL = os.environ.get(
    "PHASE_LABEL",
    "sobol_stage2_gpu_cupy_two_phase_online_stop",
)
RESULTS_ROOT = WORKSPACE_DIR / "results"
TEMP_GEOMETRY_ROOT_NAME = "_tmp_seed_geometries"

VALID_EXCEL_PATH: Optional[str] = os.environ.get("VALID_EXCEL_PATH") or None
RUN_DIR: Optional[str] = os.environ.get("RUN_DIR") or None

MAX_DESIGNS_TO_RUN: Optional[int] = env_optional_int("MAX_DESIGNS_TO_RUN", None)


# ============================================================================
# MONTE CARLO Y CONVERGENCIA
# ============================================================================

MC_BASE_SEED = env_int("MC_BASE_SEED", 20260621)
MAX_SEEDS = env_int("MAX_SEEDS", 400)

# Criterio de Kanit para el error relativo del estimador de la media:
#   err_Kanit = z * std / (sqrt(n) * |mean|)
# Con z=2 se aproxima un intervalo de confianza del 95%.
KANIT_REL_TOL = env_float("KANIT_REL_TOL", 0.02)
KANIT_CONFIDENCE_FACTOR = env_float("KANIT_CONFIDENCE_FACTOR", 1.96)
KANIT_CONFIDENCE_MODE = os.environ.get(
    "KANIT_CONFIDENCE_MODE",
    "student_t",
).strip().lower()
KANIT_PERCENTILE = env_float("KANIT_PERCENTILE", 100.0)
N_MIN = env_int("N_MIN", 10)
STABLE_STEPS = env_int("STABLE_STEPS", 1)


# ============================================================================
# FASE A: GEOMETRÍA CPU/SAM
# ============================================================================

# El generador escala por geometrías independientes y usa dos hilos Numba por
# SAM; 15 productores aprovechan el presupuesto total de 30 CPU.
MAX_PARALLEL_GEOMETRIES = env_int("MAX_PARALLEL_GEOMETRIES", 15)
GENERATOR_NUM_CORES = env_int("GENERATOR_NUM_CORES", 2)
GEOMETRY_CPU_BUDGET = env_int("GEOMETRY_CPU_BUDGET", 30)
GENERATOR_VERBOSE = env_bool("GENERATOR_VERBOSE", False)
COMPUTE_RVE_METRICS = False
SAM_BATCH_VF = env_float("SAM_BATCH_VF", 0.025)
SAM_USE_VOXEL_STOP_CALLBACK = env_bool("SAM_USE_VOXEL_STOP_CALLBACK", False)
SAM_VOXEL_STOP_START = env_float("SAM_VOXEL_STOP_START", 0.60)
SAM_INFLATION_SAMPLES = env_int("SAM_INFLATION_SAMPLES", 2)
SAM_TARGET_SAFETY = env_float("SAM_TARGET_SAFETY", 1.03)
SAM_MAX_TOPUPS = env_int("SAM_MAX_TOPUPS", 6)
SAM_TOPUP_GAIN = env_float("SAM_TOPUP_GAIN", 1.35)
SAM_CONTINUOUS_VF_CAP_FACTOR = env_float("SAM_CONTINUOUS_VF_CAP_FACTOR", 1.25)
SAM_VF_TOLERANCE = env_float("SAM_VF_TOLERANCE", 0.005)
SAM_REJECT_VF_MISS = env_bool("SAM_REJECT_VF_MISS", True)
SAM_A2_TOLERANCE = env_float("SAM_A2_TOLERANCE", 0.01)
SAM_VOXEL_A2_TOLERANCE = env_float("SAM_VOXEL_A2_TOLERANCE", 0.01)
SAM_REJECT_A2_MISS = env_bool("SAM_REJECT_A2_MISS", True)
SAM_ORIENTATION_MAX_ITER = env_int("SAM_ORIENTATION_MAX_ITER", 160)
SAM_GEOMETRY_BACKEND = os.environ.get(
    "SAM_GEOMETRY_BACKEND",
    "numba",
).strip().lower()
SAM_CUPY_MIN_CANDIDATE_PAIRS = env_int(
    "SAM_CUPY_MIN_CANDIDATE_PAIRS",
    50_000,
)
SAM_COMPACTION = env_bool("SAM_COMPACTION", True)
SAM_COMPACTION_MIN_VF = env_float("SAM_COMPACTION_MIN_VF", 0.25)
SAM_COMPACTION_MIN_PACKING_LOAD = env_float(
    "SAM_COMPACTION_MIN_PACKING_LOAD",
    3.0,
)
SAM_COMPACTION_START_SCALE = env_float("SAM_COMPACTION_START_SCALE", 0.65)
SAM_COMPACTION_MID_START_SCALE = env_float("SAM_COMPACTION_MID_START_SCALE", 0.75)
SAM_COMPACTION_TOPUP_SCALE = env_float("SAM_COMPACTION_TOPUP_SCALE", 0.65)
SAM_COMPACTION_STAGES = env_int("SAM_COMPACTION_STAGES", 6)
SAM_COMPACTION_MAX_ITER = env_int("SAM_COMPACTION_MAX_ITER", 120)
SAM_COMPACTION_PASSES = env_int("SAM_COMPACTION_PASSES", 2)
SAM_COLLECTIVE_FIRE = env_bool("SAM_COLLECTIVE_FIRE", True)
SAM_COLLECTIVE_FIRE_MIN_PACKING_LOAD = env_float(
    "SAM_COLLECTIVE_FIRE_MIN_PACKING_LOAD",
    3.0,
)
SAM_COLLECTIVE_FIRE_MAX_ITER = env_int(
    "SAM_COLLECTIVE_FIRE_MAX_ITER",
    2500,
)
SAM_COLLECTIVE_FIRE_MAX_RESTARTS = env_int(
    "SAM_COLLECTIVE_FIRE_MAX_RESTARTS",
    3,
)
SAM_COLLECTIVE_FIRE_RESTART_PATIENCE = env_int(
    "SAM_COLLECTIVE_FIRE_RESTART_PATIENCE",
    250,
)
SAM_OVERLAP_TOLERANCE = env_float("SAM_OVERLAP_TOLERANCE", 0.05)
SAM_REJECT_OVERLAP_MISS = env_bool("SAM_REJECT_OVERLAP_MISS", True)
SAM_OVERLAP_RESCUE = env_bool("SAM_OVERLAP_RESCUE", True)
SAM_OVERLAP_RESCUE_PASSES = env_int("SAM_OVERLAP_RESCUE_PASSES", 2)
SAM_OVERLAP_RESCUE_MAX_ITER = env_int("SAM_OVERLAP_RESCUE_MAX_ITER", 160)
SAM_OVERLAP_RESCUE_MAX_FIBERS = env_int(
    "SAM_OVERLAP_RESCUE_MAX_FIBERS",
    48,
)
MAX_CONSECUTIVE_INVALID_GEOMETRIES = env_int(
    "MAX_CONSECUTIVE_INVALID_GEOMETRIES",
    5,
)

# Fallback sin pipeline solapado.
GEOMETRY_BATCH_SIZE = env_int("GEOMETRY_BATCH_SIZE", 15)

# Productor CPU persistente: mantiene una ventana de geometrías futuras mientras
# el proceso principal consume la GPU.
PIPELINE_OVERLAP_GENERATION = env_bool("PIPELINE_OVERLAP_GENERATION", True)
PERSISTENT_GEOMETRY_POOL = env_bool("PERSISTENT_GEOMETRY_POOL", True)
GEOMETRY_PREFETCH_COUNT = env_int("GEOMETRY_PREFETCH_COUNT", 30)
GEOMETRY_REFILL_LOW_WATERMARK = env_int(
    "GEOMETRY_REFILL_LOW_WATERMARK",
    0,
)
HARD_GEOMETRY_PREFETCH_PER_DESIGN = env_int(
    "HARD_GEOMETRY_PREFETCH_PER_DESIGN",
    10,
)
GEOMETRY_PREFETCH_DISK_BUDGET_GIB = env_float(
    "GEOMETRY_PREFETCH_DISK_BUDGET_GIB",
    4.0,
)
GEOMETRY_MIN_FREE_DISK_GIB = env_float("GEOMETRY_MIN_FREE_DISK_GIB", 8.0)
GEOMETRY_PROCESS_START_METHOD = (
    os.environ.get("GEOMETRY_PROCESS_START_METHOD", "spawn").strip().lower()
    or "spawn"
)


# ============================================================================
# FASE B: SOLVER GPU/CUPY
# ============================================================================

SOLVER_TOL = env_float("SOLVER_TOL", 1e-4)

FFT_BACKEND = "cupy"
SOLVER_FFT_FORM = (os.environ.get("SOLVER_FFT_FORM", "r").strip().lower() or "r")
FFT_WORKERS = 1

CUPY_PLAN_MODE = "manual"

# Con una sola GPU normalmente conviene False/1.
# Si quieres probar, usa True/2.
CUPY_LOADCASE_PARALLEL = env_bool("CUPY_LOADCASE_PARALLEL", False)
CUPY_LOADCASE_WORKERS = env_int("CUPY_LOADCASE_WORKERS", 1)

SOLVER_CALLBACK = (os.environ.get("SOLVER_CALLBACK", "none").strip().lower() or "none")
SOLVER_REAL_DTYPE = (os.environ.get("SOLVER_REAL_DTYPE", "float32").strip().lower() or "float32")
USE_CFIELD_MATERIAL_FAST_PATH = env_bool("USE_CFIELD_MATERIAL_FAST_PATH", True)
CFIELD_ORIGIN = (os.environ.get("CFIELD_ORIGIN", "zero").strip().lower() or "zero")
CFIELD_STORAGE = (os.environ.get("CFIELD_STORAGE", "sym21").strip().lower() or "sym21")
CFIELD_ROTATION_BATCH_SIZE = env_int("CFIELD_ROTATION_BATCH_SIZE", 0)
CFIELD_ASSIGN_CHUNK_VOXELS = env_int("CFIELD_ASSIGN_CHUNK_VOXELS", 2_000_000)
CFIELD_INDEXED = env_bool("CFIELD_INDEXED", True)
PROJECTION_STORAGE = (os.environ.get("PROJECTION_STORAGE", "direct").strip().lower() or "direct")
PROJECTION_BACKEND = (os.environ.get("PROJECTION_BACKEND", "numpy").strip().lower() or "numpy")
KEEP_SOLUTIONS_ON_DEVICE = env_bool("KEEP_SOLUTIONS_ON_DEVICE", True)
CUPY_FUSED_MATVEC = env_bool("CUPY_FUSED_MATVEC", True)
CUPY_UNSCALED_FFT_PAIR = env_bool("CUPY_UNSCALED_FFT_PAIR", False)
CUPY_LAZY_SCALARS = env_bool("CUPY_LAZY_SCALARS", False)
CUPY_FUSED_CG_UPDATES = env_bool("CUPY_FUSED_CG_UPDATES", True)
CUPY_FUSED_XR_RR = env_bool("CUPY_FUSED_XR_RR", True)
CUPY_FUSED_DOT = env_bool("CUPY_FUSED_DOT", True)
CUPY_RESIDUAL_CHECK_EVERY = env_int("CUPY_RESIDUAL_CHECK_EVERY", 1)
FAST_MACRO_ADD = env_bool("FAST_MACRO_ADD", False)
CHECK_MACRO_MEAN = env_bool("CHECK_MACRO_MEAN", True)
STORE_SOLUTION_FIELDS = env_bool("STORE_SOLUTION_FIELDS", False)
SOLVER_LOAD_BATCH_SIZE = env_int("SOLVER_LOAD_BATCH_SIZE", 1)
POSTPROCESS_BATCH_SIZE = env_int("POSTPROCESS_BATCH_SIZE", 1)
POSTPROCESS_ASSEMBLY = (os.environ.get("POSTPROCESS_ASSEMBLY", "scalar").strip().lower() or "scalar")

PRELOAD_GEOMETRIES_TO_RAM = env_bool("PRELOAD_GEOMETRIES_TO_RAM", True)

WARMUP_GPU_ONCE = env_bool("WARMUP_GPU_ONCE", True)
FREE_GPU_MEMORY_EACH_BATCH = env_bool("FREE_GPU_MEMORY_EACH_BATCH", False)

# Ahorra disco/RAM de cache: cada seed conserva phase.npy y ori.npy solo hasta
# terminar su solve. Ceff.npy, solver_timing.json y los Excel se mantienen.
DELETE_GEOMETRY_NPY_AFTER_SOLVE = env_bool("DELETE_GEOMETRY_NPY_AFTER_SOLVE", True)

# False = si falla el solver, deja phase/ori para depurar esa seed.
DELETE_GEOMETRY_NPY_AFTER_FAILED_SOLVE = env_bool(
    "DELETE_GEOMETRY_NPY_AFTER_FAILED_SOLVE",
    False,
)


class SolverEnvironmentError(RuntimeError):
    """Se levanta cuando faltan librerias o el runtime CUDA no funciona."""


def is_fatal_solver_environment_error(error: Any) -> bool:
    text = str(error).lower()
    fatal_fragments = (
        "libcublas.so",
        "libcudart.so",
        "libcusolver.so",
        "libcusparse.so",
        "cuda driver",
        "cuda runtime",
        "cudaerror",
        "cupy/cuda no está funcionando",
    )
    return any(fragment in text for fragment in fatal_fragments)


# ============================================================================
# HELPERS
# ============================================================================

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def flatten_mapping(
    payload: Dict[str, Any],
    *,
    prefix: str = "",
) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_mapping(value, prefix=name))
        elif isinstance(value, Path):
            flat[name] = str(value)
        elif isinstance(value, (list, tuple, set)):
            flat[name] = json.dumps(list(value), ensure_ascii=True, default=str)
        elif isinstance(value, np.generic):
            flat[name] = value.item()
        else:
            flat[name] = value
    return flat


def config_table(payload: Dict[str, Any]) -> pd.DataFrame:
    flat = flatten_mapping(payload)
    return pd.DataFrame(
        [
            {
                "parameter": key,
                "value": (
                    ""
                    if value is None
                    else "true"
                    if value is True
                    else "false"
                    if value is False
                    else repr(value)
                    if isinstance(value, float)
                    else str(value)
                ),
                "value_type": type(value).__name__,
            }
            for key, value in flat.items()
        ]
    )


def write_excel_atomic(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
            for sheet_name, dataframe in sheets.items():
                dataframe.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def check_cupy_gpu() -> None:
    if FFT_BACKEND.lower() != "cupy":
        print(f"[WARN] FFT_BACKEND={FFT_BACKEND}. El solver no usará GPU.", flush=True)
        return

    try:
        import cupy as cp

        ndev = cp.cuda.runtime.getDeviceCount()
        if ndev < 1:
            raise RuntimeError("CuPy no detecta ninguna GPU CUDA.")

        dev_id = cp.cuda.runtime.getDevice()
        props = cp.cuda.runtime.getDeviceProperties(dev_id)

        name = props.get("name", "GPU CUDA")
        if isinstance(name, bytes):
            name = name.decode(errors="ignore")

        print(
            f"[GPU] CuPy OK | device={dev_id}/{ndev} | gpu={name} | "
            f"plan={CUPY_PLAN_MODE} | tol={SOLVER_TOL:.0e}",
            flush=True,
        )

    except Exception as exc:
        raise RuntimeError(
            "FFT_BACKEND='cupy', pero CuPy/CUDA no está funcionando. "
            f"Detalle: {exc}"
        ) from exc


def warmup_gpu_once() -> None:
    if FFT_BACKEND.lower() != "cupy" or not WARMUP_GPU_ONCE:
        return

    try:
        import cupy as cp

        a = cp.random.rand(256, 256, dtype=cp.float32)
        _ = a @ a

        b = cp.random.rand(32, 32, 32, dtype=cp.float32).astype(cp.complex64)
        _ = cp.fft.fftn(b)

        cp.cuda.Stream.null.synchronize()

        print("[GPU] Warm-up listo.", flush=True)

    except Exception as exc:
        raise SolverEnvironmentError(
            "El warm-up GPU/cuBLAS/cuFFT falló antes de iniciar la campaña: "
            f"{exc}"
        ) from exc


def free_gpu_memory_pool(*, clear_fft_cache: bool = True) -> None:
    """Libera el pool de CuPy y los planes FFT para evitar OOM."""
    try:
        import cupy
        pool = cupy.get_default_memory_pool()
        pool.free_all_blocks()
        if clear_fft_cache:
            cupy.fft.config.get_plan_cache().clear()
    except Exception:
        pass


def estimate_design_nvox(design_row: Dict[str, Any]) -> int:
    for key in ("nvox", "NVOX", "nvox_ref"):
        if key in design_row and pd.notna(design_row[key]):
            return max(1, int(round(float(design_row[key]))))

    if "res" in design_row and "caja_um" in design_row:
        return max(1, int(round(float(design_row["res"]) * float(design_row["caja_um"]))))

    raise ValueError("No puedo estimar nvox: falta nvox o res*caja_um.")


def estimate_design_fiber_count(design_row: Dict[str, Any]) -> int:
    """Estimate the integer fiber count required by a design."""
    cell_volume = float(design_row["caja_um"]) ** 3
    fiber_volume = (
        math.pi
        * float(design_row["d_um"]) ** 2
        * float(design_row["L_um"])
        / 4.0
    )
    if fiber_volume <= 0.0:
        raise ValueError("No puedo estimar fibras: L_um y d_um deben ser positivos.")
    return max(
        1,
        int(round(float(design_row["Vf_target"]) * cell_volume / fiber_volume)),
    )


def order_designs_for_execution(design_df: pd.DataFrame) -> pd.DataFrame:
    """Run low-cost geometries first without changing design identifiers."""
    if design_df.empty:
        return design_df.copy()

    ordered = design_df.copy()
    ordered["_estimated_fibers"] = [
        estimate_design_fiber_count(row.to_dict())
        for _, row in ordered.iterrows()
    ]
    ordered["_estimated_nvox"] = [
        estimate_design_nvox(row.to_dict())
        for _, row in ordered.iterrows()
    ]
    ordered = ordered.sort_values(
        ["_estimated_fibers", "_estimated_nvox", "design_id"],
        kind="stable",
    )
    return ordered.drop(
        columns=["_estimated_fibers", "_estimated_nvox"],
    ).reset_index(drop=True)


def balance_design_ids_for_prefetch(
    design_rows: List[Dict[str, Any]],
) -> List[int]:
    """Alternate cheap and expensive designs to reduce the geometry tail."""
    ordered = sorted(
        design_rows,
        key=lambda row: (
            estimate_design_fiber_count(row),
            estimate_design_nvox(row),
            int(row["design_id"]),
        ),
    )
    balanced: List[int] = []
    low = 0
    high = len(ordered) - 1
    while low <= high:
        balanced.append(int(ordered[low]["design_id"]))
        low += 1
        if low <= high:
            balanced.append(int(ordered[high]["design_id"]))
            high -= 1
    return balanced


def completed_design_matches_config(
    completed_row: Dict[str, Any],
    config: Dict[str, Any],
) -> bool:
    """Accept resume data only when statistical and geometry criteria match."""
    integer_fields = (
        ("n_min", "n_min"),
        ("n_seeds_max", "max_seeds"),
        ("mc_base_seed", "mc_base_seed"),
        ("generator_num_cores", "generator_num_cores"),
    )
    for row_key, config_key in integer_fields:
        try:
            if int(completed_row.get(row_key, -1)) != int(config[config_key]):
                return False
        except (TypeError, ValueError):
            return False

    if (
        str(completed_row.get("kanit_confidence_mode", "")).strip().lower()
        != str(config["kanit_confidence_mode"]).strip().lower()
    ):
        return False

    float_fields = (
        ("kanit_rel_tol", "kanit_rel_tol"),
        ("sam_vf_tolerance", "sam_vf_tolerance"),
        ("sam_A2_tolerance", "sam_A2_tolerance"),
        ("sam_voxel_A2_tolerance", "sam_voxel_A2_tolerance"),
        ("sam_overlap_tolerance", "sam_overlap_tolerance"),
    )
    for row_key, config_key in float_fields:
        try:
            if not math.isclose(
                float(completed_row.get(row_key, np.nan)),
                float(config[config_key]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                return False
        except (TypeError, ValueError):
            return False

    bool_fields = (
        ("sam_reject_vf_miss", "sam_reject_vf_miss"),
        ("sam_reject_A2_miss", "sam_reject_A2_miss"),
        ("sam_reject_overlap_miss", "sam_reject_overlap_miss"),
    )
    for row_key, config_key in bool_fields:
        value = completed_row.get(row_key)
        if value is None or pd.isna(value):
            return False
        if isinstance(value, str):
            value = value.strip().lower() in {"1", "true", "yes", "on"}
        if bool(value) != bool(config[config_key]):
            return False

    return True


def resolve_valid_excel() -> Path:
    if VALID_EXCEL_PATH:
        path = Path(VALID_EXCEL_PATH)

        if not path.exists():
            raise FileNotFoundError(f"No existe VALID_EXCEL_PATH: {path}")

        return path

    fallback_excels = sorted(
        RESULTS_ROOT.rglob("sobol_points_valid.xlsx"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    if fallback_excels:
        return fallback_excels[0]

    raise FileNotFoundError(
        f"No se encontró sobol_points_valid.xlsx en: {RESULTS_ROOT}"
    )


def resolve_run_dir() -> Path:
    if RUN_DIR:
        run_dir = Path(RUN_DIR)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    study_id = f"{STUDY_NAME}_{PHASE_LABEL}_{now_tag()}"
    run_dir = RESULTS_ROOT / study_id
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def make_run_config() -> Dict[str, Any]:
    return {
        "generator_num_cores": int(GENERATOR_NUM_CORES),
        "geometry_cpu_budget": int(GEOMETRY_CPU_BUDGET),
        "generator_verbose": bool(GENERATOR_VERBOSE),
        "compute_rve_metrics": bool(COMPUTE_RVE_METRICS),
        "sam_batch_vf": float(SAM_BATCH_VF),
        "sam_use_voxel_stop_callback": bool(SAM_USE_VOXEL_STOP_CALLBACK),
        "sam_voxel_stop_start": float(SAM_VOXEL_STOP_START),
        "sam_inflation_samples": int(SAM_INFLATION_SAMPLES),
        "sam_target_safety": float(SAM_TARGET_SAFETY),
        "sam_max_topups": int(SAM_MAX_TOPUPS),
        "sam_topup_gain": float(SAM_TOPUP_GAIN),
        "sam_continuous_vf_cap_factor": float(SAM_CONTINUOUS_VF_CAP_FACTOR),
        "sam_vf_tolerance": float(SAM_VF_TOLERANCE),
        "sam_reject_vf_miss": bool(SAM_REJECT_VF_MISS),
        "sam_A2_tolerance": float(SAM_A2_TOLERANCE),
        "sam_voxel_A2_tolerance": float(SAM_VOXEL_A2_TOLERANCE),
        "sam_reject_A2_miss": bool(SAM_REJECT_A2_MISS),
        "sam_orientation_max_iter": int(SAM_ORIENTATION_MAX_ITER),
        "sam_geometry_backend": str(SAM_GEOMETRY_BACKEND),
        "sam_cupy_min_candidate_pairs": int(
            SAM_CUPY_MIN_CANDIDATE_PAIRS
        ),
        "sam_compaction": bool(SAM_COMPACTION),
        "sam_compaction_min_vf": float(SAM_COMPACTION_MIN_VF),
        "sam_compaction_min_packing_load": float(
            SAM_COMPACTION_MIN_PACKING_LOAD
        ),
        "sam_compaction_start_scale": float(SAM_COMPACTION_START_SCALE),
        "sam_compaction_mid_start_scale": float(SAM_COMPACTION_MID_START_SCALE),
        "sam_compaction_topup_scale": float(SAM_COMPACTION_TOPUP_SCALE),
        "sam_compaction_stages": int(SAM_COMPACTION_STAGES),
        "sam_compaction_max_iter": int(SAM_COMPACTION_MAX_ITER),
        "sam_compaction_passes": int(SAM_COMPACTION_PASSES),
        "sam_collective_fire": bool(SAM_COLLECTIVE_FIRE),
        "sam_collective_fire_min_packing_load": float(
            SAM_COLLECTIVE_FIRE_MIN_PACKING_LOAD
        ),
        "sam_collective_fire_max_iter": int(
            SAM_COLLECTIVE_FIRE_MAX_ITER
        ),
        "sam_collective_fire_max_restarts": int(
            SAM_COLLECTIVE_FIRE_MAX_RESTARTS
        ),
        "sam_collective_fire_restart_patience": int(
            SAM_COLLECTIVE_FIRE_RESTART_PATIENCE
        ),
        "sam_overlap_tolerance": float(SAM_OVERLAP_TOLERANCE),
        "sam_reject_overlap_miss": bool(SAM_REJECT_OVERLAP_MISS),
        "sam_overlap_rescue": bool(SAM_OVERLAP_RESCUE),
        "sam_overlap_rescue_passes": int(SAM_OVERLAP_RESCUE_PASSES),
        "sam_overlap_rescue_max_iter": int(SAM_OVERLAP_RESCUE_MAX_ITER),
        "sam_overlap_rescue_max_fibers": int(
            SAM_OVERLAP_RESCUE_MAX_FIBERS
        ),
        "max_consecutive_invalid_geometries": int(
            MAX_CONSECUTIVE_INVALID_GEOMETRIES
        ),
        "max_parallel_geometries": int(MAX_PARALLEL_GEOMETRIES),
        "geometry_batch_size": int(GEOMETRY_BATCH_SIZE),
        "pipeline_overlap_generation": bool(PIPELINE_OVERLAP_GENERATION),
        "persistent_geometry_pool": bool(PERSISTENT_GEOMETRY_POOL),
        "geometry_prefetch_count": int(GEOMETRY_PREFETCH_COUNT),
        "geometry_refill_low_watermark": int(
            GEOMETRY_REFILL_LOW_WATERMARK
        ),
        "hard_geometry_prefetch_per_design": int(
            HARD_GEOMETRY_PREFETCH_PER_DESIGN
        ),
        "geometry_prefetch_disk_budget_gib": float(
            GEOMETRY_PREFETCH_DISK_BUDGET_GIB
        ),
        "geometry_min_free_disk_gib": float(GEOMETRY_MIN_FREE_DISK_GIB),
        "geometry_process_start_method": str(GEOMETRY_PROCESS_START_METHOD),

        "solver_tol": float(SOLVER_TOL),

        "fft_backend": str(FFT_BACKEND),
        "solver_fft_form": str(SOLVER_FFT_FORM),
        "fft_workers": int(FFT_WORKERS),
        "cupy_plan_mode": str(CUPY_PLAN_MODE),
        "cupy_loadcase_parallel": bool(CUPY_LOADCASE_PARALLEL),
        "cupy_loadcase_workers": int(CUPY_LOADCASE_WORKERS),
        "solver_callback": str(SOLVER_CALLBACK),
        "solver_real_dtype": str(SOLVER_REAL_DTYPE),
        "use_cfield_material_fast_path": bool(USE_CFIELD_MATERIAL_FAST_PATH),
        "cfield_origin": str(CFIELD_ORIGIN),
        "cfield_storage": str(CFIELD_STORAGE),
        "cfield_rotation_batch_size": int(CFIELD_ROTATION_BATCH_SIZE),
        "cfield_assign_chunk_voxels": int(CFIELD_ASSIGN_CHUNK_VOXELS),
        "cfield_indexed": bool(CFIELD_INDEXED),
        "projection_storage": str(PROJECTION_STORAGE),
        "projection_backend": str(PROJECTION_BACKEND),
        "keep_solutions_on_device": bool(KEEP_SOLUTIONS_ON_DEVICE),
        "cupy_fused_matvec": bool(CUPY_FUSED_MATVEC),
        "cupy_unscaled_fft_pair": bool(CUPY_UNSCALED_FFT_PAIR),
        "cupy_lazy_scalars": bool(CUPY_LAZY_SCALARS),
        "cupy_fused_cg_updates": bool(CUPY_FUSED_CG_UPDATES),
        "cupy_fused_xr_rr": bool(CUPY_FUSED_XR_RR),
        "cupy_fused_dot": bool(CUPY_FUSED_DOT),
        "cupy_residual_check_every": int(CUPY_RESIDUAL_CHECK_EVERY),
        "fast_macro_add": bool(FAST_MACRO_ADD),
        "check_macro_mean": bool(CHECK_MACRO_MEAN),
        "store_solution_fields": bool(STORE_SOLUTION_FIELDS),
        "solver_load_batch_size": int(SOLVER_LOAD_BATCH_SIZE),
        "postprocess_batch_size": int(POSTPROCESS_BATCH_SIZE),
        "postprocess_assembly": str(POSTPROCESS_ASSEMBLY),

        "preload_geometries_to_ram": bool(PRELOAD_GEOMETRIES_TO_RAM),
        "free_gpu_memory_each_batch": bool(FREE_GPU_MEMORY_EACH_BATCH),
        "delete_geometry_npy_after_solve": bool(DELETE_GEOMETRY_NPY_AFTER_SOLVE),
        "delete_geometry_npy_after_failed_solve": bool(
            DELETE_GEOMETRY_NPY_AFTER_FAILED_SOLVE
        ),

        "mc_base_seed": int(MC_BASE_SEED),
        "max_seeds": int(MAX_SEEDS),
        "kanit_rel_tol": float(KANIT_REL_TOL),
        "kanit_confidence_factor": float(KANIT_CONFIDENCE_FACTOR),
        "kanit_confidence_mode": str(KANIT_CONFIDENCE_MODE),
        "kanit_percentile": float(KANIT_PERCENTILE),
        "n_min": int(N_MIN),
        "stable_steps": int(STABLE_STEPS),
    }


def design_output_paths(run_dir: Path, design_id: int) -> Tuple[Path, Path, Path]:
    design_id = int(design_id)
    return (
        run_dir / f"geometry_generation_design_{design_id}.xlsx",
        run_dir / f"convergencia_design_{design_id}.xlsx",
        run_dir / f"summary_design_{design_id}.xlsx",
    )


def seed_work_root(run_dir: Path) -> Path:
    return run_dir / TEMP_GEOMETRY_ROOT_NAME


def seed_directory(work_root: Path, design_id: int, mc_id: int, seed: int) -> Path:
    return work_root / f"design_{int(design_id):04d}_seed_{mc_id:04d}_{seed}"


def remove_empty_seed_work_root(run_dir: Path) -> bool:
    root = seed_work_root(run_dir)
    try:
        root.rmdir()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


# ============================================================================
# PARAMETROS
# ============================================================================

def build_generation_params(
    mc_id: int,
    seed: int,
    design_row: Dict[str, Any],
    seed_dir: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    vf_target = float(design_row["Vf_target"])
    aspect_ratio = float(design_row["L_um"]) / max(
        float(design_row["d_um"]),
        1e-12,
    )
    packing_load = aspect_ratio * vf_target
    use_compaction = bool(config.get("sam_compaction", True)) and (
        vf_target >= float(config.get("sam_compaction_min_vf", 0.30))
        or packing_load
        >= float(config.get("sam_compaction_min_packing_load", 3.0))
    )
    if aspect_ratio >= 24.0:
        compaction_start_scale = 0.45
    elif aspect_ratio >= 18.0:
        compaction_start_scale = 0.50
    elif aspect_ratio >= 12.0:
        compaction_start_scale = float(
            config.get("sam_compaction_start_scale", 0.64)
        )
    else:
        compaction_start_scale = float(
            config.get("sam_compaction_mid_start_scale", 0.72)
        )
    return {
        "gap_um": 0,
        "resol": float(design_row["res"]),
        "seed": int(seed),
        "output_dir": str(seed_dir),

        "num_cores": int(config["generator_num_cores"]),
        "sam_num_threads": int(config["generator_num_cores"]),

        "caja_um": float(design_row["caja_um"]),
        "L_um": float(design_row["L_um"]),
        "d_um": float(design_row["d_um"]),
        "Vf_target": vf_target,

        "a11": float(design_row["a11"]),
        "a22": float(design_row["a22"]),

        "compute_metrics": bool(config["compute_rve_metrics"]),
        "sam_batch_vf": float(config["sam_batch_vf"]),
        "sam_use_voxel_stop_callback": bool(
            config["sam_use_voxel_stop_callback"]
        ),
        "sam_inflation_samples": int(config["sam_inflation_samples"]),
        "sam_target_safety": float(config["sam_target_safety"]),
        "sam_max_topups": int(config["sam_max_topups"]),
        "sam_topup_gain": float(config["sam_topup_gain"]),
        "sam_continuous_vf_cap_factor": float(
            config["sam_continuous_vf_cap_factor"]
        ),
        "sam_vf_tolerance": float(config["sam_vf_tolerance"]),
        "sam_A2_tolerance": float(config.get("sam_A2_tolerance", 0.01)),
        "sam_voxel_A2_tolerance": float(
            config.get("sam_voxel_A2_tolerance", 0.01)
        ),
        "sam_orientation_max_iter": int(
            config.get("sam_orientation_max_iter", 160)
        ),
        "sam_geometry_backend": str(
            config.get("sam_geometry_backend", "numba")
        ),
        "sam_cupy_min_candidate_pairs": int(
            config.get("sam_cupy_min_candidate_pairs", 50_000)
        ),
        "sam_compaction": use_compaction,
        "sam_packing_load": float(packing_load),
        "sam_compaction_start_scale": compaction_start_scale,
        "sam_compaction_topup_scale": float(compaction_start_scale),
        "sam_compaction_stages": int(config["sam_compaction_stages"]),
        "sam_compaction_max_iter": int(config["sam_compaction_max_iter"]),
        "sam_compaction_passes": int(config["sam_compaction_passes"]),
        "sam_collective_fire": bool(
            config.get("sam_collective_fire", True)
        ),
        "sam_collective_fire_min_packing_load": float(
            config.get("sam_collective_fire_min_packing_load", 3.0)
        ),
        "sam_collective_fire_max_iter": int(
            config.get("sam_collective_fire_max_iter", 2500)
        ),
        "sam_collective_fire_max_restarts": int(
            config.get("sam_collective_fire_max_restarts", 3)
        ),
        "sam_collective_fire_restart_patience": int(
            config.get("sam_collective_fire_restart_patience", 250)
        ),
        "sam_overlap_tolerance": float(config["sam_overlap_tolerance"]),
        "sam_overlap_rescue": bool(config["sam_overlap_rescue"]),
        "sam_overlap_rescue_passes": int(config["sam_overlap_rescue_passes"]),
        "sam_overlap_rescue_max_iter": int(
            config["sam_overlap_rescue_max_iter"]
        ),
        "sam_overlap_rescue_max_fibers": int(
            config["sam_overlap_rescue_max_fibers"]
        ),
        "sam_verbose": bool(config.get("generator_verbose", False)),
    }


def build_solver_params(
    seed: int,
    design_row: Dict[str, Any],
    seed_dir: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    tol = float(config["solver_tol"])

    return {
        "Em": float(design_row["Em"]),
        "nu_m": float(design_row["nu_m"]),

        "Ef_L": float(design_row["Ef_L"]),
        "Ef_T": float(design_row["Ef_T"]),
        "nu_LT": float(design_row["nu_LT"]),
        "nu_TT": float(design_row["nu_TT"]),
        "G_LT": float(design_row["G_LT"]),

        "input_dir": str(seed_dir),
        "Ceff_out_path": str(seed_dir / "Ceff.npy"),
        "seed": int(seed),

        "solver_tol": float(tol),

        "fft_backend": str(config["fft_backend"]),
        "solver_fft_form": str(config["solver_fft_form"]),
        "fft_workers": int(config["fft_workers"]),
        "cupy_plan_mode": str(config["cupy_plan_mode"]),

        "internal_load_parallel": bool(config["cupy_loadcase_parallel"]),
        "internal_load_workers": int(config["cupy_loadcase_workers"]),
        "solver_callback": str(config["solver_callback"]),
        "solver_real_dtype": str(config["solver_real_dtype"]),
        "use_cfield_material_fast_path": bool(config["use_cfield_material_fast_path"]),
        "cfield_origin": str(config["cfield_origin"]),
        "cfield_storage": str(config["cfield_storage"]),
        "cfield_rotation_batch_size": int(config["cfield_rotation_batch_size"]),
        "cfield_assign_chunk_voxels": int(config["cfield_assign_chunk_voxels"]),
        "cfield_indexed": bool(config["cfield_indexed"]),
        "projection_storage": str(config["projection_storage"]),
        "projection_backend": str(config["projection_backend"]),
        "keep_solution_on_device": bool(config["keep_solutions_on_device"]),
        "cupy_fused_matvec": bool(config["cupy_fused_matvec"]),
        "cupy_unscaled_fft_pair": bool(config["cupy_unscaled_fft_pair"]),
        "cupy_lazy_scalars": bool(config["cupy_lazy_scalars"]),
        "cupy_fused_cg_updates": bool(config["cupy_fused_cg_updates"]),
        "cupy_fused_xr_rr": bool(config["cupy_fused_xr_rr"]),
        "cupy_fused_dot": bool(config["cupy_fused_dot"]),
        "cupy_residual_check_every": int(config["cupy_residual_check_every"]),
        "fast_macro_add": bool(config["fast_macro_add"]),
        "check_macro_mean": bool(config["check_macro_mean"]),
        "store_solution_fields": bool(config["store_solution_fields"]),
        "load_batch_size": int(config["solver_load_batch_size"]),
        "postprocess_batch_size": int(config["postprocess_batch_size"]),
        "postprocess_assembly": str(config["postprocess_assembly"]),

        "solver_timing_path": str(seed_dir / "solver_timing.json"),
        "free_gpu_memory_after_solve": False,
    }


# ============================================================================
# FASE A
# ============================================================================

def generate_single_seed_geometry(
    mc_id: int,
    seed: int,
    design_row: Dict[str, Any],
    work_root: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    design_id = int(design_row.get("design_id", -1))
    seed_dir = seed_directory(work_root, design_id, mc_id, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    started_epoch_s = time.time()

    try:
        p_gen = build_generation_params(
            mc_id=mc_id,
            seed=seed,
            design_row=design_row,
            seed_dir=seed_dir,
            config=config,
        )

        gen_info = generate_rve_main_sam(p_gen)
        t_gen_s = time.perf_counter() - t0

        phase_path = seed_dir / "phase.npy"
        ori_path = seed_dir / "ori.npy"

        if not phase_path.exists() or not ori_path.exists():
            raise FileNotFoundError(
                f"No se generaron archivos esperados: {phase_path.name}, {ori_path.name}"
            )

        result: Dict[str, Any] = {
            "config_id": str(design_row.get("config_id", "")),
            "design_id": int(design_row.get("design_id", -1)),
            "sobol_index": int(design_row.get("sobol_index", -1)),
            "mc_id": int(mc_id),
            "seed": int(seed),
            "status": "geometry_ok",
            "error": "",
            "AR": float(design_row.get("AR", np.nan)),
            "Vf_target": float(design_row.get("Vf_target", np.nan)),
            "a11": float(design_row.get("a11", np.nan)),
            "a22": float(design_row.get("a22", np.nan)),
            "a33": float(design_row.get("a33", np.nan)),
            "nvox": int(design_row.get("nvox", -1)),
            "seed_dir": str(seed_dir),
            "phase_path": str(phase_path),
            "ori_path": str(ori_path),
            "t_gen_s": float(t_gen_s),
            "t_gen_started_epoch_s": float(started_epoch_s),
            "t_gen_completed_epoch_s": float(time.time()),
            "generator_num_cores": int(config["generator_num_cores"]),
            "sam_packing_load": float(
                float(design_row.get("AR", np.nan))
                * float(design_row.get("Vf_target", np.nan))
            ),
        }

        if isinstance(gen_info, dict):
            result.update(
                {
                    "sam_t_gen_reported_s": float(gen_info.get("t_gen", np.nan)),
                    "sam_t_construct_s": float(gen_info.get("t_construct", np.nan)),
                    "sam_t_raster_s": float(gen_info.get("t_raster", np.nan)),
                    "sam_t_raster_total_s": float(
                        gen_info.get("sam_raster_total_s", np.nan)
                    ),
                    "sam_t_calibration_s": float(
                        gen_info.get("sam_calibration_s", np.nan)
                    ),
                    "sam_t_run_s": float(gen_info.get("sam_run_s", np.nan)),
                    "sam_t_voxel_callback_s": float(
                        gen_info.get("sam_voxel_callback_s", np.nan)
                    ),
                    "sam_t_save_s": float(gen_info.get("sam_save_s", np.nan)),
                    "sam_t_metrics_s": float(gen_info.get("t_metrics", np.nan)),
                    "sam_n_fibers": int(gen_info.get("n_fibers", -1)),
                    "sam_vf": float(gen_info.get("Vf", np.nan)),
                    "sam_vf_target": float(
                        gen_info.get("sam_vf_target", design_row.get("Vf_target", np.nan))
                    ),
                    "sam_vf_error": float(gen_info.get("sam_vf_error", np.nan)),
                    "sam_vf_error_percent_points": float(
                        gen_info.get("sam_vf_error_percent_points", np.nan)
                    ),
                    "sam_vf_deficit": float(gen_info.get("sam_vf_deficit", np.nan)),
                    "sam_vf_tolerance": float(gen_info.get("sam_vf_tolerance", np.nan)),
                    "sam_vf_tolerance_configured": float(
                        gen_info.get("sam_vf_tolerance_configured", np.nan)
                    ),
                    "sam_single_fiber_vf_quantum": float(
                        gen_info.get("sam_single_fiber_vf_quantum", np.nan)
                    ),
                    "sam_vf_ok": bool(gen_info.get("sam_vf_ok", True)),
                    "sam_overlap_tolerance": float(
                        gen_info.get("sam_overlap_tolerance", np.nan)
                    ),
                    "sam_overlap_ok": bool(gen_info.get("sam_overlap_ok", True)),
                    "sam_A2_tolerance": float(
                        gen_info.get("sam_A2_tolerance", np.nan)
                    ),
                    "sam_voxel_A2_tolerance": float(
                        gen_info.get("sam_voxel_A2_tolerance", np.nan)
                    ),
                    "sam_A2_ok": bool(gen_info.get("sam_A2_ok", True)),
                    "sam_inflation_factor": float(
                        gen_info.get("sam_inflation_factor", np.nan)
                    ),
                    "sam_effective_vf_target": float(
                        gen_info.get("sam_effective_vf_target", np.nan)
                    ),
                    "sam_generator_target_vf": float(
                        gen_info.get("sam_generator_target_vf", np.nan)
                    ),
                    "sam_generator_target_n_fibers": int(
                        gen_info.get("sam_generator_target_n_fibers", -1)
                    ),
                    "sam_geometry_backend_requested": str(
                        gen_info.get(
                            "sam_geometry_backend_requested",
                            "numba",
                        )
                    ),
                    "sam_geometry_backend_effective": str(
                        gen_info.get(
                            "sam_geometry_backend_effective",
                            "numba",
                        )
                    ),
                    "sam_cupy_min_candidate_pairs": int(
                        gen_info.get("sam_cupy_min_candidate_pairs", 0)
                    ),
                    "sam_cupy_candidate_calls": int(
                        gen_info.get("sam_cupy_candidate_calls", 0)
                    ),
                    "sam_cupy_candidate_pairs": int(
                        gen_info.get("sam_cupy_candidate_pairs", 0)
                    ),
                    "sam_cupy_candidate_s": float(
                        gen_info.get("sam_cupy_candidate_s", 0.0)
                    ),
                    "sam_cupy_candidate_fallbacks": int(
                        gen_info.get("sam_cupy_candidate_fallbacks", 0)
                    ),
                    "sam_cupy_force_calls": int(
                        gen_info.get("sam_cupy_force_calls", 0)
                    ),
                    "sam_cupy_force_pairs": int(
                        gen_info.get("sam_cupy_force_pairs", 0)
                    ),
                    "sam_cupy_force_s": float(
                        gen_info.get("sam_cupy_force_s", 0.0)
                    ),
                    "sam_cupy_force_fallbacks": int(
                        gen_info.get("sam_cupy_force_fallbacks", 0)
                    ),
                    "sam_final_continuous_vf": float(
                        gen_info.get("sam_final_continuous_vf", np.nan)
                    ),
                    "sam_used_continuous_vf": float(
                        gen_info.get("sam_used_continuous_vf", np.nan)
                    ),
                    "sam_periodic_boundary_mode": str(
                        gen_info.get("sam_periodic_boundary_mode", "")
                    ),
                    "sam_periodic_crossing_fibers": int(
                        gen_info.get("sam_periodic_crossing_fibers", 0)
                    ),
                    "sam_periodic_crossing_fraction": float(
                        gen_info.get("sam_periodic_crossing_fraction", np.nan)
                    ),
                    "sam_soft_insert": bool(gen_info.get("sam_soft_insert", False)),
                    "sam_soft_insert_start": float(
                        gen_info.get("sam_soft_insert_start", np.nan)
                    ),
                    "sam_soft_inserted_batches": int(
                        gen_info.get("sam_soft_inserted_batches", 0)
                    ),
                    "sam_use_local_relax": bool(
                        gen_info.get("sam_use_local_relax", True)
                    ),
                    "sam_compaction": bool(gen_info.get("sam_compaction", False)),
                    "sam_collective_fire_configured": bool(
                        gen_info.get(
                            "sam_collective_fire_configured",
                            False,
                        )
                    ),
                    "sam_collective_fire_used": bool(
                        gen_info.get("sam_collective_fire_used", False)
                    ),
                    "sam_collective_fire_mode": str(
                        gen_info.get(
                            "sam_collective_fire_mode",
                            "disabled",
                        )
                    ),
                    "sam_collective_fire_min_packing_load": float(
                        gen_info.get(
                            "sam_collective_fire_min_packing_load",
                            np.nan,
                        )
                    ),
                    "sam_collective_fire_calls": int(
                        gen_info.get("sam_collective_fire_calls", 0)
                    ),
                    "sam_collective_fire_mode": str(
                        gen_info.get("sam_collective_fire_mode", "disabled")
                    ),
                    "sam_collective_fire_s": float(
                        gen_info.get("sam_collective_fire_s", 0.0)
                    ),
                    "sam_collective_fire_iters": int(
                        gen_info.get("sam_collective_fire_iters", 0)
                    ),
                    "sam_collective_fire_restarts": int(
                        gen_info.get("sam_collective_fire_restarts", 0)
                    ),
                    "sam_collective_fire_force_evaluations": int(
                        gen_info.get(
                            "sam_collective_fire_force_evaluations",
                            0,
                        )
                    ),
                    "sam_collective_fire_pair_build_s": float(
                        gen_info.get(
                            "sam_collective_fire_pair_build_s",
                            0.0,
                        )
                    ),
                    "sam_collective_fire_pair_forces_s": float(
                        gen_info.get(
                            "sam_collective_fire_pair_forces_s",
                            0.0,
                        )
                    ),
                    "sam_collective_fire_apply_updates_s": float(
                        gen_info.get(
                            "sam_collective_fire_apply_updates_s",
                            0.0,
                        )
                    ),
                    "sam_collective_fire_successful_batches": int(
                        gen_info.get(
                            "sam_collective_fire_successful_batches",
                            0,
                        )
                    ),
                    "sam_collective_fire_failed_batches": int(
                        gen_info.get(
                            "sam_collective_fire_failed_batches",
                            0,
                        )
                    ),
                    "sam_collective_fire_smallest_batch": int(
                        gen_info.get(
                            "sam_collective_fire_smallest_batch",
                            0,
                        )
                    ),
                    "sam_compaction_start_scale": float(
                        gen_info.get("sam_compaction_start_scale", np.nan)
                    ),
                    "sam_compaction_topup_scale": float(
                        gen_info.get("sam_compaction_topup_scale", np.nan)
                    ),
                    "sam_compaction_stages": int(
                        gen_info.get("sam_compaction_stages", 0)
                    ),
                    "sam_compaction_subdivisions": int(
                        gen_info.get("sam_compaction_subdivisions", 0)
                    ),
                    "sam_compaction_relax_iters": int(
                        gen_info.get("sam_compaction_relax_iters", 0)
                    ),
                    "sam_compaction_relax_s": float(
                        gen_info.get("sam_compaction_relax_s", np.nan)
                    ),
                    "sam_compaction_pair_build_s": float(
                        gen_info.get("sam_compaction_pair_build_s", np.nan)
                    ),
                    "sam_compaction_pair_forces_s": float(
                        gen_info.get("sam_compaction_pair_forces_s", np.nan)
                    ),
                    "sam_compaction_converged": bool(
                        gen_info.get("sam_compaction_converged", True)
                    ),
                    "sam_compaction_removed_fibers": int(
                        gen_info.get("sam_compaction_removed_fibers", 0)
                    ),
                    "sam_compaction_max_fiber_removals": int(
                        gen_info.get("sam_compaction_max_fiber_removals", 0)
                    ),
                    "sam_post_raster_compaction_retries": int(
                        gen_info.get("sam_post_raster_compaction_retries", 0)
                    ),
                    "sam_post_raster_orientation_retries": int(
                        gen_info.get("sam_post_raster_orientation_retries", 0)
                    ),
                    "sam_orientation_optimizer_accepted_steps": int(
                        gen_info.get(
                            "sam_orientation_optimizer_accepted_steps",
                            0,
                        )
                    ),
                    "sam_orientation_optimizer_line_search_evaluations": int(
                        gen_info.get(
                            "sam_orientation_optimizer_line_search_evaluations",
                            0,
                        )
                    ),
                    "sam_orientation_optimizer_contact_projections": int(
                        gen_info.get(
                            "sam_orientation_optimizer_contact_projections",
                            0,
                        )
                    ),
                    "sam_orientation_optimizer_s": float(
                        gen_info.get("sam_orientation_optimizer_s", 0.0)
                    ),
                    "sam_long_fiber_vf_fallback": bool(
                        gen_info.get("sam_long_fiber_vf_fallback", False)
                    ),
                    "sam_generation_strategy": str(
                        gen_info.get("sam_generation_strategy", "sam_lite")
                    ),
                    "sam_t_compaction_s": float(
                        gen_info.get("sam_compaction_s", np.nan)
                    ),
                    "sam_overlap_rescue": bool(
                        gen_info.get("sam_overlap_rescue", False)
                    ),
                    "sam_overlap_rescue_s": float(
                        gen_info.get("sam_overlap_rescue_s", np.nan)
                    ),
                    "sam_overlap_rescue_attempts": int(
                        gen_info.get("sam_overlap_rescue_attempts", 0)
                    ),
                    "sam_overlap_rescue_reinserted": int(
                        gen_info.get("sam_overlap_rescue_reinserted", 0)
                    ),
                    "sam_overlap_rescue_shaken": int(
                        gen_info.get("sam_overlap_rescue_shaken", 0)
                    ),
                    "sam_overlap_rescue_iters": int(
                        gen_info.get("sam_overlap_rescue_iters", 0)
                    ),
                    "sam_overlap_rescue_removed": int(
                        gen_info.get("sam_overlap_rescue_removed", 0)
                    ),
                    "sam_final_A_err": float(gen_info.get("sam_final_A_err", np.nan)),
                    "sam_A2_error_rel": float(gen_info.get("sam_A2_error_rel", np.nan)),
                    "sam_voxel_A2_error_rel": float(
                        gen_info.get("sam_voxel_A2_error_rel", np.nan)
                    ),
                    "sam_batches": int(gen_info.get("sam_batches", 0)),
                    "sam_attempts": int(gen_info.get("sam_attempts", 0)),
                    "sam_relax_iters": int(gen_info.get("sam_relax_iters", 0)),
                    "sam_topup_count": int(gen_info.get("sam_topup_count", 0)),
                    "sam_voxel_stop_checks": int(
                        gen_info.get("sam_voxel_stop_checks", 0)
                    ),
                    "sam_final_overlap": float(
                        gen_info.get("sam_final_overlap", np.nan)
                    ),
                    "sam_relax_s": float(
                        gen_info.get("sam_relax_s_time", np.nan)
                    ),
                    "sam_pair_build_s": float(
                        gen_info.get("sam_pair_build_s", np.nan)
                    ),
                    "sam_pair_forces_s": float(
                        gen_info.get("sam_pair_forces_s", np.nan)
                    ),
                }
            )
            vf_miss = (
                bool(config.get("sam_reject_vf_miss", True))
                and not bool(gen_info.get("sam_vf_ok", True))
            )
            overlap_miss = (
                bool(config.get("sam_reject_overlap_miss", True))
                and not bool(gen_info.get("sam_overlap_ok", True))
            )
            orientation_miss = (
                bool(config.get("sam_reject_A2_miss", True))
                and not bool(gen_info.get("sam_A2_ok", True))
            )
            if vf_miss:
                result["status"] = "geometry_vf_miss"
                result["error"] = (
                    "SAM no alcanzo Vf_target: "
                    f"target={float(gen_info.get('sam_vf_target', np.nan)):.6f}, "
                    f"actual={float(gen_info.get('Vf', np.nan)):.6f}, "
                    f"deficit={float(gen_info.get('sam_vf_deficit', np.nan)):.6f}, "
                    f"tol={float(gen_info.get('sam_vf_tolerance', np.nan)):.6f}"
                )
            elif overlap_miss:
                result["status"] = "geometry_overlap_miss"
                result["error"] = (
                    "SAM no elimino los solapes: "
                    f"max_overlap={float(gen_info.get('sam_final_overlap', np.nan)):.6f}, "
                    f"tol={float(gen_info.get('sam_overlap_tolerance', np.nan)):.6f}"
                )
            elif orientation_miss:
                result["status"] = "geometry_orientation_miss"
                result["error"] = (
                    "SAM no alcanzo A2_target: "
                    f"continuous_error={float(gen_info.get('sam_A2_error_rel', np.nan)):.6f}, "
                    f"voxel_error={float(gen_info.get('sam_voxel_A2_error_rel', np.nan)):.6f}, "
                    f"continuous_tol={float(gen_info.get('sam_A2_tolerance', np.nan)):.6f}, "
                    f"voxel_tol={float(gen_info.get('sam_voxel_A2_tolerance', np.nan)):.6f}"
                )

        return result

    except Exception as exc:
        t_gen_s = time.perf_counter() - t0

        print(
            f"[ERROR][FASE A] design={design_row.get('design_id')} "
            f"seed={seed}: {exc}",
            flush=True,
        )

        return {
            "config_id": str(design_row.get("config_id", "")),
            "design_id": int(design_row.get("design_id", -1)),
            "sobol_index": int(design_row.get("sobol_index", -1)),
            "mc_id": int(mc_id),
            "seed": int(seed),
            "status": "geometry_error",
            "error": str(exc),
            "AR": float(design_row.get("AR", np.nan)),
            "Vf_target": float(design_row.get("Vf_target", np.nan)),
            "a11": float(design_row.get("a11", np.nan)),
            "a22": float(design_row.get("a22", np.nan)),
            "a33": float(design_row.get("a33", np.nan)),
            "nvox": int(design_row.get("nvox", -1)),
            "seed_dir": str(seed_dir),
            "phase_path": str(seed_dir / "phase.npy"),
            "ori_path": str(seed_dir / "ori.npy"),
            "t_gen_s": float(t_gen_s),
            "t_gen_started_epoch_s": float(started_epoch_s),
            "t_gen_completed_epoch_s": float(time.time()),
            "generator_num_cores": int(config["generator_num_cores"]),
        }


def generate_geometry_batch(
    design_row: Dict[str, Any],
    work_root: Path,
    seed_items: List[Tuple[int, int]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    n_workers = max(
        1,
        min(len(seed_items), int(config["max_parallel_geometries"])),
    )

    results: List[Dict[str, Any]] = []

    if n_workers == 1:
        for mc_id, seed in seed_items:
            results.append(
                generate_single_seed_geometry(
                    mc_id=mc_id,
                    seed=seed,
                    design_row=design_row,
                    work_root=work_root,
                    config=config,
                )
            )

        results.sort(key=lambda item: int(item["mc_id"]))
        return results

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_ignore_sigint_in_worker,
    ) as executor:
        future_to_meta = {
            executor.submit(
                generate_single_seed_geometry,
                mc_id,
                seed,
                design_row,
                work_root,
                config,
            ): (mc_id, seed)
            for mc_id, seed in seed_items
        }

        for future in as_completed(future_to_meta):
            mc_id, seed = future_to_meta[future]

            try:
                result = future.result()
            except Exception as exc:
                design_id = int(design_row.get("design_id", -1))
                seed_dir = seed_directory(work_root, design_id, mc_id, seed)
                result = {
                    "config_id": str(design_row.get("config_id", "")),
                    "design_id": int(design_row.get("design_id", -1)),
                    "sobol_index": int(design_row.get("sobol_index", -1)),
                    "mc_id": int(mc_id),
                    "seed": int(seed),
                    "status": "geometry_error",
                    "error": str(exc),
                    "AR": float(design_row.get("AR", np.nan)),
                    "Vf_target": float(design_row.get("Vf_target", np.nan)),
                    "a11": float(design_row.get("a11", np.nan)),
                    "a22": float(design_row.get("a22", np.nan)),
                    "a33": float(design_row.get("a33", np.nan)),
                    "nvox": int(design_row.get("nvox", -1)),
                    "seed_dir": str(seed_dir),
                    "phase_path": str(seed_dir / "phase.npy"),
                    "ori_path": str(seed_dir / "ori.npy"),
                    "t_gen_s": np.nan,
                    "generator_num_cores": int(config["generator_num_cores"]),
                }

            results.append(result)

    results.sort(key=lambda item: int(item["mc_id"]))
    return results


def resolve_geometry_worker_count(
    seed_count: int,
    config: Dict[str, Any],
) -> int:
    threads_per_geometry = max(1, int(config["generator_num_cores"]))
    cpu_budget = max(1, int(config.get("geometry_cpu_budget", 1)))
    workers_by_budget = max(1, cpu_budget // threads_per_geometry)
    return max(
        1,
        min(
            int(seed_count),
            int(config["max_parallel_geometries"]),
            workers_by_budget,
        ),
    )



class GlobalGeometryManager:
    """Mantiene el prefetch de geometrías vivo mientras se resuelven diseños."""

    def __init__(
        self,
        design_rows: list[dict],
        run_dir: Path,
        config: dict,
        executor: ProcessPoolExecutor,
    ) -> None:
        self.config = dict(config)
        self.run_dir = Path(run_dir)
        self.work_root = seed_work_root(self.run_dir)
        self.executor = executor

        self.design_rows = {}
        self.seed_items = {}

        self.next_submit = {}
        self.next_take = {}
        self.futures = {}
        self.converged_designs = set()

        nvox_est = max(
            (estimate_design_nvox(row) for row in design_rows),
            default=290,
        )
        self.worker_count = executor._max_workers
        self.geometry_disk_est_bytes = int(nvox_est ** 3 * 13 + 4096)

        requested_prefetch = max(
            1,
            int(self.config.get("geometry_prefetch_count", 30)),
        )
        disk_budget_bytes = max(
            1,
            int(float(self.config.get("geometry_prefetch_disk_budget_gib", 4.0)) * 1024 ** 3),
        )
        free_disk_bytes = shutil.disk_usage(self.run_dir).free
        reserved_free_bytes = max(
            0,
            int(float(self.config.get("geometry_min_free_disk_gib", 2.0)) * 1024 ** 3),
        )
        usable_free_bytes = max(0, free_disk_bytes - reserved_free_bytes)
        disk_limited_prefetch = max(
            1,
            min(disk_budget_bytes, usable_free_bytes) // self.geometry_disk_est_bytes,
        )
        self.global_prefetch_count = max(
            1,
            min(requested_prefetch, int(disk_limited_prefetch)),
        )
        self.requested_prefetch_count = int(requested_prefetch)

        # Preparar estructuras por diseno
        for row in design_rows:
            did = int(row["design_id"])

            seeds = derived_mc_seeds(
                design_id=did,
                n_seeds=int(self.config["max_seeds"]),
                base_seed=int(self.config["mc_base_seed"]),
            )
            self.design_rows[did] = dict(row)
            self.seed_items[did] = list(enumerate(seeds))

            self.next_submit[did] = 0
            self.next_take[did] = 0
            self.futures[did] = {}

        self.lock = threading.Lock()
        self.closed = False
        self.cancelled_count = 0
        self.active_design_id = next(iter(self.design_rows), None)
        self.prefetch_design_ids = balance_design_ids_for_prefetch(
            list(self.design_rows.values())
        )

        print(
            f"[GLOBAL PREFETCH] Iniciado | solicitado={requested_prefetch} | "
            f"efectivo={self.global_prefetch_count} | workers={self.worker_count} | "
            f"geom_max={self.geometry_disk_est_bytes / (1024 ** 2):.1f} MiB | "
            f"orden=barato/costoso alternado",
            flush=True
        )
        self._fill_pipeline()

    def _count_in_flight(self) -> int:
        count = 0
        for did in self.design_rows.keys():
            if did in self.converged_designs:
                continue
            count += (self.next_submit[did] - self.next_take[did])
        return count

    def _prefetch_window(self) -> int:
        return max(
            1,
            min(
                int(self.config.get("n_min", 10)),
                self.global_prefetch_count // 2,
            ),
        )

    def _outstanding_limit(self, did: int, window: int) -> int:
        if did == self.active_design_id:
            return int(window)

        row = self.design_rows[did]
        packing_load = float(row["AR"]) * float(row["Vf_target"])
        hard_threshold = float(
            self.config.get("sam_collective_fire_min_packing_load", 3.0)
        )
        if packing_load >= hard_threshold:
            return max(
                1,
                min(
                    int(window),
                    int(
                        self.config.get(
                            "hard_geometry_prefetch_per_design",
                            10,
                        )
                    ),
                ),
            )
        return int(window)

    def _submit_next_locked(self, did: int) -> bool:
        ns = self.next_submit[did]
        if ns >= len(self.seed_items[did]):
            return False

        mc_id, seed = self.seed_items[did][ns]
        self.futures[did][ns] = self.executor.submit(
            generate_single_seed_geometry,
            mc_id,
            seed,
            self.design_rows[did],
            self.work_root,
            self.config,
        )
        self.next_submit[did] += 1
        return True

    def _activate_design(self, did: int) -> None:
        with self.lock:
            if self.closed or did in self.converged_designs:
                return

            self.active_design_id = did
            window = self._prefetch_window()
            while (
                self.next_submit[did] - self.next_take[did] < window
                and self._submit_next_locked(did)
            ):
                pass

    def _fill_pipeline(self) -> None:
        with self.lock:
            if self.closed:
                return
            in_flight = self._count_in_flight()
            window = self._prefetch_window()

            while in_flight < self.global_prefetch_count:
                submitted_any = False
                for did in self.prefetch_design_ids:
                    if did in self.converged_designs:
                        continue

                    ns = self.next_submit[did]
                    nt = self.next_take[did]
                    outstanding_limit = self._outstanding_limit(did, window)

                    if (
                        (ns - nt) < outstanding_limit
                        and self._submit_next_locked(did)
                    ):
                        in_flight += 1
                        submitted_any = True
                        break

                if not submitted_any:
                    break

    def _failure_result(self, did: int, index: int, exc: BaseException) -> dict:
        mc_id, seed = self.seed_items[did][index]
        seed_dir = seed_directory(self.work_root, did, mc_id, seed)
        design_row = self.design_rows.get(did, {})
        return {
            "config_id": str(design_row.get("config_id", "")),
            "design_id": int(design_row.get("design_id", did)),
            "sobol_index": int(design_row.get("sobol_index", -1)),
            "mc_id": int(mc_id),
            "seed": int(seed),
            "status": "geometry_error",
            "error": str(exc),
            "AR": float(design_row.get("AR", np.nan)),
            "Vf_target": float(design_row.get("Vf_target", np.nan)),
            "a11": float(design_row.get("a11", np.nan)),
            "a22": float(design_row.get("a22", np.nan)),
            "a33": float(design_row.get("a33", np.nan)),
            "nvox": int(design_row.get("nvox", -1)),
            "seed_dir": str(seed_dir),
            "phase_path": str(seed_dir / "phase.npy"),
            "ori_path": str(seed_dir / "ori.npy"),
            "t_gen_s": float('nan'),
            "t_geometry_queue_wait_s": float('nan'),
            "generator_num_cores": int(self.config["generator_num_cores"]),
        }

    def has_next(self, did: int) -> bool:
        return self.next_take[did] < len(self.seed_items[did]) and did not in self.converged_designs

    def take_next(self, did: int) -> dict:
        if self.closed or not self.has_next(did):
            raise StopIteration

        self._activate_design(did)
        index = self.next_take[did]
        future = self.futures[did].pop(index)
        wait_t0 = time.perf_counter()

        try:
            result = dict(future.result())
        except Exception as exc:
            result = self._failure_result(did, index, exc)

        result["t_geometry_queue_wait_s"] = float(time.perf_counter() - wait_t0)
        result["geometry_prefetch_count"] = int(self.global_prefetch_count)
        result["geometry_prefetch_requested"] = int(
            self.requested_prefetch_count
        )
        result["geometry_refill_low_watermark_effective"] = 0
        design_nvox = estimate_design_nvox(self.design_rows[did])
        result["geometry_disk_est_mib"] = float(
            (design_nvox ** 3 * 13 + 4096) / (1024 ** 2)
        )
        result["geometry_workers_effective"] = int(self.worker_count)

        print(
            f"[GLOBAL][D{did}][S{result.get('seed')}] "
            f"gen={result.get('t_gen_s', float('nan')):.2f}s | "
            f"wait={result['t_geometry_queue_wait_s']:.3f}s",
            flush=True
        )

        self.next_take[did] += 1
        self._fill_pipeline()
        return result

    def take_batch(self, did: int, batch_size: int) -> list[dict]:
        out = []
        for _ in range(max(1, int(batch_size))):
            if not self.has_next(did):
                break
            out.append(self.take_next(did))
        return out

    @staticmethod
    def _cleanup_orphan_future(future: Any) -> None:
        if future.cancelled():
            return
        try:
            result = future.result()
            seed_dir = result.get("seed_dir") if isinstance(result, dict) else None
            if seed_dir:
                shutil.rmtree(seed_dir, ignore_errors=True)
        except Exception:
            pass

    def mark_converged(self, did: int) -> None:
        with self.lock:
            if did in self.converged_designs:
                return
            self.converged_designs.add(did)
            if self.active_design_id == did:
                self.active_design_id = None

            for idx, future in list(self.futures[did].items()):
                if idx < self.next_take[did]:
                    continue
                if future.cancel():
                    self.cancelled_count += 1
                elif future.done():
                    self._cleanup_orphan_future(future)
                else:
                    future.add_done_callback(self._cleanup_orphan_future)
                self.futures[did].pop(idx, None)
        self._fill_pipeline()

    def close(self) -> None:
        self.closed = True
        for did in self.design_rows.keys():
            self.mark_converged(did)


def preload_geometry_to_ram(geometry_result: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(geometry_result)

    if out.get("status") != "geometry_ok":
        return out

    seed_dir = Path(str(out["seed_dir"]))

    t0 = time.perf_counter()

    out["phase_array"] = np.load(seed_dir / "phase.npy").astype(np.uint8, copy=False)
    out["ori_array"] = np.load(seed_dir / "ori.npy").astype(np.float32, copy=False)
    out["t_preload_s"] = float(time.perf_counter() - t0)
    out["preloaded_geometry"] = True

    return out

def drop_large_arrays(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    out.pop("phase_array", None)
    out.pop("ori_array", None)
    return out


def cleanup_seed_geometry_npy(
    seed_dir: Path | str,
    *,
    delete_phase_ori: bool,
) -> Dict[str, Any]:
    out = {
        "geometry_npy_deleted": False,
        "geometry_npy_deleted_bytes": 0,
        "geometry_npy_cleanup_error": "",
        "seed_dir_removed": False,
    }

    seed_text = str(seed_dir).strip()
    if not delete_phase_ori or not seed_text:
        return out

    seed_path = Path(seed_text)
    if "seed" not in seed_path.name.lower():
        out["geometry_npy_cleanup_error"] = (
            f"ruta no parece carpeta temporal de semilla: {seed_path}"
        )
        return out

    if not seed_path.exists():
        return out
    if not seed_path.is_dir():
        out["geometry_npy_cleanup_error"] = f"ruta no es carpeta: {seed_path}"
        return out

    try:
        deleted_bytes = 0
        for path in seed_path.rglob("*"):
            if path.is_file() or path.is_symlink():
                deleted_bytes += int(path.stat().st_size)
        shutil.rmtree(seed_path)
        out["geometry_npy_deleted"] = True
        out["geometry_npy_deleted_bytes"] = int(deleted_bytes)
        out["seed_dir_removed"] = True
    except Exception as exc:
        out["geometry_npy_cleanup_error"] = str(exc)
    return out


def cleanup_seed_geometry_after_result(
    solve_result: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    status = str(solve_result.get("status", ""))
    should_delete = (
        bool(config.get("delete_geometry_npy_after_solve", True))
        if status == "ok"
        else bool(config.get("delete_geometry_npy_after_failed_solve", False))
    )

    cleanup = cleanup_seed_geometry_npy(
        solve_result.get("seed_dir", ""),
        delete_phase_ori=should_delete,
    )
    solve_result.update(cleanup)

    if cleanup["geometry_npy_deleted"]:
        tqdm.write(
            f"[CLEANUP][seed {solve_result.get('seed')}] "
            f"geometría temporal borrada | "
            f"{cleanup['geometry_npy_deleted_bytes'] / (1024 ** 2):.1f} MiB"
        )
    elif cleanup["geometry_npy_cleanup_error"]:
        tqdm.write(
            f"[CLEANUP][seed {solve_result.get('seed')}] "
            f"error: {cleanup['geometry_npy_cleanup_error']}"
        )

    return cleanup


def cleanup_unsolved_geometry_results(
    geometry_results: List[Dict[str, Any]],
    solved_seed_dirs: set[str],
    config: Dict[str, Any],
    *,
    reason: str,
) -> Dict[str, Any]:
    out = {
        "n_unsolved_geometry_cleaned": 0,
        "unsolved_geometry_deleted_bytes": 0,
    }

    if not bool(config.get("delete_geometry_npy_after_solve", False)):
        return out

    for geometry_result in geometry_results:
        seed_dir = str(geometry_result.get("seed_dir", ""))
        if not seed_dir or seed_dir in solved_seed_dirs:
            continue

        cleanup = cleanup_seed_geometry_npy(seed_dir, delete_phase_ori=True)
        if cleanup["geometry_npy_deleted"]:
            out["n_unsolved_geometry_cleaned"] += 1
            out["unsolved_geometry_deleted_bytes"] += int(
                cleanup["geometry_npy_deleted_bytes"]
            )
            tqdm.write(
                f"[CLEANUP][seed {geometry_result.get('seed')}] "
                f"geometría no simulada borrada por {reason} | "
                f"{cleanup['geometry_npy_deleted_bytes'] / (1024 ** 2):.1f} MiB"
            )
        elif cleanup["geometry_npy_cleanup_error"]:
            tqdm.write(
                f"[CLEANUP][seed {geometry_result.get('seed')}] "
                f"error borrando geometría no simulada: "
                f"{cleanup['geometry_npy_cleanup_error']}"
            )

    return out


def solve_single_seed_from_geometry(
    geometry_result: Dict[str, Any],
    design_row: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    seed = int(geometry_result["seed"])
    seed_dir = Path(str(geometry_result["seed_dir"]))

    base_result = dict(geometry_result)
    base_result["props"] = None
    completed_epoch_s = geometry_result.get("t_gen_completed_epoch_s", np.nan)
    if pd.notna(completed_epoch_s):
        base_result["geometry_ready_age_s"] = max(
            0.0,
            float(time.time() - float(completed_epoch_s)),
        )
    else:
        base_result["geometry_ready_age_s"] = np.nan

    if geometry_result.get("status") != "geometry_ok":
        base_result.update(
            {
                "status": "solve_skipped_geometry_error",
                "solve_error": (
                    geometry_result.get("error")
                    or "Se omite solver porque fallo la geometria."
                ),
                "t_solver_s": np.nan,
                "t_extract_s": np.nan,
                "t_total_after_geometry_s": np.nan,
                "fft_backend": str(config["fft_backend"]),
                "tol_used": np.nan,
            }
        )
        return drop_large_arrays(base_result)

    if not (seed_dir / "phase.npy").exists() or not (seed_dir / "ori.npy").exists():
        base_result.update(
            {
                "status": "solve_error",
                "solve_error": f"No existen phase.npy/ori.npy en {seed_dir}",
                "t_solver_s": np.nan,
                "t_extract_s": np.nan,
                "t_total_after_geometry_s": np.nan,
                "fft_backend": str(config["fft_backend"]),
                "tol_used": np.nan,
            }
        )
        return drop_large_arrays(base_result)

    t0 = time.perf_counter()

    try:
        p_solv = build_solver_params(
            seed=seed,
            design_row=design_row,
            seed_dir=seed_dir,
            config=config,
        )

        if "phase_array" in geometry_result and "ori_array" in geometry_result:
            p_solv["phase_array"] = geometry_result["phase_array"]
            p_solv["ori_array"] = geometry_result["ori_array"]
            p_solv["preloaded_geometry"] = True
        else:
            p_solv["preloaded_geometry"] = False

        print(
            f"[FASE B][design {design_row['design_id']}][seed {seed}] "
            f"FFT | backend={p_solv['fft_backend']} | "
            f"tol={p_solv['solver_tol']:.0e}",
            flush=True,
        )

        t_solver0 = time.perf_counter()
        ceff = np.asarray(solve_homogenization(p_solv), dtype=float)
        t_solver_s = time.perf_counter() - t_solver0

        t_extract0 = time.perf_counter()
        props = engineering_constants_from_Cmandel(ceff)
        t_extract_s = time.perf_counter() - t_extract0

        t_total_after_geometry_s = time.perf_counter() - t0

        if not props:
            base_result.update(
                {
                    "status": "solve_error",
                    "solve_error": "No se pudieron extraer propiedades.",
                    "props": None,
                    "t_solver_s": float(t_solver_s),
                    "t_extract_s": float(t_extract_s),
                    "t_total_after_geometry_s": float(t_total_after_geometry_s),
                    "fft_backend": p_solv["fft_backend"],
                    "cupy_plan_mode": p_solv["cupy_plan_mode"],
                    "tol_used": float(p_solv["solver_tol"]),
                }
            )
            return drop_large_arrays(base_result)

        props_out = {
            col: float(props[col])
            for col in ENGINEERING_PROPERTY_COLUMNS
        }

        ceff_out = {
            f"Ceff_{row + 1}{column + 1}": float(ceff[row, column])
            for row in range(ceff.shape[0])
            for column in range(ceff.shape[1])
        }
        timing_out: Dict[str, Any] = {}
        timing_path = Path(str(p_solv["solver_timing_path"]))
        if timing_path.exists():
            try:
                with timing_path.open("r", encoding="utf-8") as handle:
                    timing_payload = json.load(handle)
                timing_out = {
                    f"solver_timing_{key}": value
                    for key, value in flatten_mapping(timing_payload).items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                }
            except Exception as exc:
                timing_out = {"solver_timing_read_error": str(exc)}

        base_result.update(
            {
                "status": "ok",
                "solve_error": "",
                "props": props_out,
                "t_solver_s": float(t_solver_s),
                "t_extract_s": float(t_extract_s),
                "t_total_after_geometry_s": float(t_total_after_geometry_s),
                "fft_backend": p_solv["fft_backend"],
                "cupy_plan_mode": p_solv["cupy_plan_mode"],
                "cupy_loadcase_parallel": bool(config["cupy_loadcase_parallel"]),
                "cupy_loadcase_workers": int(config["cupy_loadcase_workers"]),
                "tol_used": float(p_solv["solver_tol"]),
                **ceff_out,
                **timing_out,
            }
        )

        return drop_large_arrays(base_result)

    except Exception as exc:
        t_total_after_geometry_s = time.perf_counter() - t0

        print(
            f"[ERROR][FASE B] design={design_row.get('design_id')} "
            f"seed={seed}: {exc}",
            flush=True,
        )

        base_result.update(
            {
                "status": "solve_error",
                "solve_error": str(exc),
                "props": None,
                "t_solver_s": np.nan,
                "t_extract_s": np.nan,
                "t_total_after_geometry_s": float(t_total_after_geometry_s),
                "fft_backend": str(config["fft_backend"]),
                "tol_used": np.nan,
            }
        )

        return drop_large_arrays(base_result)


def solve_geometry_batch_gpu(
    solve_batch: List[Dict[str, Any]],
    design_row: Dict[str, Any],
    config: Dict[str, Any],
) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for geometry_result in solve_batch:
        result = solve_single_seed_from_geometry(
            geometry_result=geometry_result,
            design_row=design_row,
            config=config,
        )
        yield result, geometry_result


# ============================================================================
# CONVERGENCIA ONLINE
# ============================================================================

class OnlineConvergence:
    def __init__(
        self,
        property_columns: List[str],
        kanit_rel_tol: float,
        kanit_confidence_factor: float,
        kanit_percentile: float,
        n_min: int,
        stable_steps: int,
        kanit_confidence_mode: str = "fixed",
    ) -> None:
        self.property_columns = list(property_columns)
        self.kanit_rel_tol = float(kanit_rel_tol)
        self.kanit_confidence_factor = float(kanit_confidence_factor)
        self.kanit_confidence_mode = str(kanit_confidence_mode).strip().lower()
        if self.kanit_confidence_mode not in {"fixed", "student_t"}:
            raise ValueError(
                "kanit_confidence_mode must be 'fixed' or 'student_t'."
            )
        self.kanit_percentile = float(np.clip(kanit_percentile, 0.0, 100.0))
        self.n_min = int(n_min)
        self.stable_steps = int(stable_steps)

        self.n = 0
        self.means = {col: 0.0 for col in self.property_columns}
        self.m2 = {col: 0.0 for col in self.property_columns}

        self.stable_count = 0
        self.converged = False
        self.last_kanit_error = np.nan
        self.last_kanit_n_required = np.nan
        self.last_kanit_errors = {
            col: np.nan for col in self.property_columns
        }
        self.last_kanit_n_required_by_property = {
            col: np.nan for col in self.property_columns
        }

    def current_confidence_factor(self) -> float:
        if self.kanit_confidence_mode == "student_t" and self.n >= 2:
            from scipy.stats import t as student_t

            return float(student_t.ppf(0.975, self.n - 1))
        return float(self.kanit_confidence_factor)

    def current_std(self) -> Dict[str, float]:
        if self.n <= 1:
            return {col: 0.0 for col in self.property_columns}

        return {
            col: float(np.sqrt(max(self.m2[col], 0.0) / (self.n - 1)))
            for col in self.property_columns
        }

    def current_cv(self) -> Dict[str, float]:
        stds = self.current_std()
        return {
            col: (
                float(stds[col] / abs(self.means[col]))
                if abs(self.means[col]) > 1e-12
                else np.nan
            )
            for col in self.property_columns
        }

    def snapshot(self) -> Dict[str, Any]:
        curr_mean = dict(self.means)
        curr_std = self.current_std()
        curr_cv = self.current_cv()
        finite_cv = [
            float(value)
            for value in curr_cv.values()
            if math.isfinite(float(value))
        ]
        q = self.kanit_percentile / 100.0
        cv_pctl = float(np.quantile(finite_cv, q)) if finite_cv else np.nan
        return {
            "n_samples": int(self.n),
            "means": curr_mean,
            "stds": curr_std,
            "cvs": curr_cv,
            "cv_pctl": cv_pctl,
            "cv_pctl_percent": 100.0 * cv_pctl,
            "kanit_errors": dict(self.last_kanit_errors),
            "kanit_n_required": dict(
                self.last_kanit_n_required_by_property
            ),
            "kanit_error": float(self.last_kanit_error),
            "kanit_error_percent": 100.0 * float(self.last_kanit_error),
            "kanit_n_required_pctl": float(self.last_kanit_n_required),
            "kanit_rel_tol": float(self.kanit_rel_tol),
            "kanit_rel_tol_percent": 100.0 * float(self.kanit_rel_tol),
            "kanit_confidence_factor": self.current_confidence_factor(),
            "kanit_confidence_mode": self.kanit_confidence_mode,
            "kanit_percentile": float(self.kanit_percentile),
            "stable_count": int(self.stable_count),
            "converged": bool(self.converged),
        }

    def update(self, props: Dict[str, float]) -> Dict[str, Any]:
        self.n += 1

        for col in self.property_columns:
            value = float(props[col])

            delta = value - self.means[col]
            self.means[col] += delta / self.n

            delta2 = value - self.means[col]
            self.m2[col] += delta * delta2

        curr_mean = dict(self.means)
        curr_std = self.current_std()

        kanit_errors = {col: np.nan for col in self.property_columns}
        kanit_n_required = {col: np.nan for col in self.property_columns}

        if self.n >= self.n_min:
            confidence_factor = self.current_confidence_factor()
            for col in self.property_columns:
                mean_abs = abs(float(curr_mean[col]))
                std = float(curr_std[col])
                if mean_abs > 1e-12 and math.isfinite(std):
                    kanit_error = confidence_factor * std / (
                        math.sqrt(float(self.n)) * mean_abs
                    )
                    kanit_errors[col] = float(kanit_error)
                    if self.kanit_rel_tol > 0.0:
                        kanit_n_required[col] = float(
                            math.ceil(
                                (
                                    confidence_factor
                                    * std
                                    / (self.kanit_rel_tol * mean_abs)
                                )
                                ** 2
                            )
                        )

            finite_errors = [
                float(value)
                for value in kanit_errors.values()
                if math.isfinite(float(value))
            ]
            finite_n_required = [
                float(value)
                for value in kanit_n_required.values()
                if math.isfinite(float(value))
            ]
            q = self.kanit_percentile / 100.0
            self.last_kanit_error = (
                float(np.quantile(finite_errors, q)) if finite_errors else np.nan
            )
            self.last_kanit_n_required = (
                float(np.quantile(finite_n_required, q)) if finite_n_required else np.nan
            )
            self.last_kanit_errors = dict(kanit_errors)
            self.last_kanit_n_required_by_property = dict(kanit_n_required)

            if self.last_kanit_error <= self.kanit_rel_tol:
                self.stable_count += 1
            else:
                self.stable_count = 0

            if self.stable_count >= self.stable_steps:
                self.converged = True

        return self.snapshot()


def vf_meets_convergence_metric(result: Dict[str, Any]) -> bool:
    if result.get("status") != "ok" or not result.get("props"):
        return False
    if result.get("sam_vf_ok") is not True:
        return False
    try:
        vf = float(result["sam_vf"])
        target = float(result["sam_vf_target"])
        tolerance = float(result["sam_vf_tolerance"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (vf, target, tolerance)):
        return False
    return abs(vf - target) <= tolerance + 1e-12


# ============================================================================
# RESULTADOS
# ============================================================================

def build_seed_row(
    result: Dict[str, Any],
    snapshot: Dict[str, Any],
    design_row: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    props = result.get("props")

    t_gen = result.get("t_gen_s", np.nan)
    t_after = result.get("t_total_after_geometry_s", np.nan)

    if np.isfinite(t_gen) and np.isfinite(t_after):
        t_total_seed = float(t_gen + t_after)
    else:
        t_total_seed = np.nan

    row: Dict[str, Any] = dict(design_row)
    row.update({
        "config_id": result.get("config_id", ""),
        "design_id": result.get("design_id", np.nan),
        "sobol_index": result.get("sobol_index", np.nan),
        "mc_id": int(result["mc_id"]),
        "seed": int(result["seed"]),
        "mc_base_seed": int(config["mc_base_seed"]),
        "seed_derivation": "SeedSequence([mc_base_seed, design_id])",
        "n_samples": int(snapshot["n_samples"]),
        "accepted_for_convergence": bool(
            result.get("accepted_for_convergence", False)
        ),
        "convergence_exclusion_reason": result.get(
            "convergence_exclusion_reason",
            "",
        ),

        "status": result.get("status", ""),
        "error": result.get("error", ""),
        "solve_error": result.get("solve_error", ""),
        "seed_dir": result.get("seed_dir", ""),

        "fft_backend": result.get("fft_backend", ""),
        "cupy_plan_mode": result.get("cupy_plan_mode", ""),
        "tol_used": result.get("tol_used", np.nan),

        "generator_num_cores": result.get("generator_num_cores", np.nan),
        "geometry_workers_effective": result.get("geometry_workers_effective", np.nan),
        "geometry_prefetch_count": result.get("geometry_prefetch_count", np.nan),
        "geometry_prefetch_requested": result.get("geometry_prefetch_requested", np.nan),
        "geometry_refill_low_watermark_effective": result.get(
            "geometry_refill_low_watermark_effective",
            np.nan,
        ),
        "geometry_disk_est_mib": result.get("geometry_disk_est_mib", np.nan),
        "cupy_loadcase_parallel": result.get("cupy_loadcase_parallel", np.nan),
        "cupy_loadcase_workers": result.get("cupy_loadcase_workers", np.nan),
        "geometry_npy_deleted": result.get("geometry_npy_deleted", False),
        "geometry_npy_deleted_bytes": result.get("geometry_npy_deleted_bytes", 0),
        "geometry_npy_cleanup_error": result.get("geometry_npy_cleanup_error", ""),

        "t_gen_s": result.get("t_gen_s", np.nan),
        "t_geometry_queue_wait_s": result.get("t_geometry_queue_wait_s", np.nan),
        "geometry_ready_age_s": result.get("geometry_ready_age_s", np.nan),
        "t_preload_s": result.get("t_preload_s", np.nan),
        "preloaded_geometry": result.get("preloaded_geometry", False),
        "t_solver_s": result.get("t_solver_s", np.nan),
        "t_extract_s": result.get("t_extract_s", np.nan),
        "t_total_after_geometry_s": result.get("t_total_after_geometry_s", np.nan),
        "t_total_seed_s": t_total_seed,

        "sam_t_gen_reported_s": result.get("sam_t_gen_reported_s", np.nan),
        "sam_t_construct_s": result.get("sam_t_construct_s", np.nan),
        "sam_t_raster_s": result.get("sam_t_raster_s", np.nan),
        "sam_t_metrics_s": result.get("sam_t_metrics_s", np.nan),
        "sam_n_fibers": result.get("sam_n_fibers", np.nan),
        "sam_vf": result.get("sam_vf", np.nan),
        "Vf_target": result.get("Vf_target", np.nan),
        "sam_vf_target": result.get("sam_vf_target", np.nan),
        "sam_vf_error": result.get("sam_vf_error", np.nan),
        "sam_vf_error_percent_points": result.get(
            "sam_vf_error_percent_points",
            np.nan,
        ),
        "sam_vf_deficit": result.get("sam_vf_deficit", np.nan),
        "sam_vf_tolerance": result.get("sam_vf_tolerance", np.nan),
        "sam_vf_ok": result.get("sam_vf_ok", np.nan),
        "sam_overlap_tolerance": result.get("sam_overlap_tolerance", np.nan),
        "sam_overlap_ok": result.get("sam_overlap_ok", np.nan),
        "sam_A2_tolerance": result.get("sam_A2_tolerance", np.nan),
        "sam_voxel_A2_tolerance": result.get(
            "sam_voxel_A2_tolerance",
            np.nan,
        ),
        "sam_A2_ok": result.get("sam_A2_ok", np.nan),
        "sam_inflation_factor": result.get("sam_inflation_factor", np.nan),
        "sam_effective_vf_target": result.get("sam_effective_vf_target", np.nan),
        "sam_generator_target_vf": result.get("sam_generator_target_vf", np.nan),
        "sam_generator_target_n_fibers": result.get(
            "sam_generator_target_n_fibers",
            np.nan,
        ),
        "sam_geometry_backend_requested": result.get(
            "sam_geometry_backend_requested",
            "",
        ),
        "sam_geometry_backend_effective": result.get(
            "sam_geometry_backend_effective",
            "",
        ),
        "sam_cupy_min_candidate_pairs": result.get(
            "sam_cupy_min_candidate_pairs",
            np.nan,
        ),
        "sam_cupy_candidate_calls": result.get(
            "sam_cupy_candidate_calls",
            np.nan,
        ),
        "sam_cupy_candidate_pairs": result.get(
            "sam_cupy_candidate_pairs",
            np.nan,
        ),
        "sam_cupy_candidate_s": result.get(
            "sam_cupy_candidate_s",
            np.nan,
        ),
        "sam_cupy_candidate_fallbacks": result.get(
            "sam_cupy_candidate_fallbacks",
            np.nan,
        ),
        "sam_cupy_force_calls": result.get(
            "sam_cupy_force_calls",
            np.nan,
        ),
        "sam_cupy_force_pairs": result.get(
            "sam_cupy_force_pairs",
            np.nan,
        ),
        "sam_cupy_force_s": result.get(
            "sam_cupy_force_s",
            np.nan,
        ),
        "sam_cupy_force_fallbacks": result.get(
            "sam_cupy_force_fallbacks",
            np.nan,
        ),
        "sam_final_continuous_vf": result.get("sam_final_continuous_vf", np.nan),
        "sam_used_continuous_vf": result.get("sam_used_continuous_vf", np.nan),
        "sam_soft_insert": result.get("sam_soft_insert", np.nan),
        "sam_soft_insert_start": result.get("sam_soft_insert_start", np.nan),
        "sam_soft_inserted_batches": result.get("sam_soft_inserted_batches", np.nan),
        "sam_compaction": result.get("sam_compaction", np.nan),
        "sam_collective_fire_configured": result.get(
            "sam_collective_fire_configured",
            np.nan,
        ),
        "sam_collective_fire_used": result.get(
            "sam_collective_fire_used",
            np.nan,
        ),
        "sam_collective_fire_min_packing_load": result.get(
            "sam_collective_fire_min_packing_load",
            np.nan,
        ),
        "sam_collective_fire_calls": result.get(
            "sam_collective_fire_calls",
            np.nan,
        ),
        "sam_collective_fire_mode": result.get(
            "sam_collective_fire_mode",
            "",
        ),
        "sam_collective_fire_successful_batches": result.get(
            "sam_collective_fire_successful_batches",
            np.nan,
        ),
        "sam_collective_fire_failed_batches": result.get(
            "sam_collective_fire_failed_batches",
            np.nan,
        ),
        "sam_collective_fire_smallest_batch": result.get(
            "sam_collective_fire_smallest_batch",
            np.nan,
        ),
        "sam_collective_fire_s": result.get(
            "sam_collective_fire_s",
            np.nan,
        ),
        "sam_collective_fire_iters": result.get(
            "sam_collective_fire_iters",
            np.nan,
        ),
        "sam_collective_fire_restarts": result.get(
            "sam_collective_fire_restarts",
            np.nan,
        ),
        "sam_collective_fire_force_evaluations": result.get(
            "sam_collective_fire_force_evaluations",
            np.nan,
        ),
        "sam_collective_fire_pair_build_s": result.get(
            "sam_collective_fire_pair_build_s",
            np.nan,
        ),
        "sam_collective_fire_pair_forces_s": result.get(
            "sam_collective_fire_pair_forces_s",
            np.nan,
        ),
        "sam_collective_fire_apply_updates_s": result.get(
            "sam_collective_fire_apply_updates_s",
            np.nan,
        ),
        "sam_compaction_start_scale": result.get(
            "sam_compaction_start_scale",
            np.nan,
        ),
        "sam_compaction_topup_scale": result.get(
            "sam_compaction_topup_scale",
            np.nan,
        ),
        "sam_compaction_stages": result.get("sam_compaction_stages", np.nan),
        "sam_compaction_subdivisions": result.get(
            "sam_compaction_subdivisions",
            np.nan,
        ),
        "sam_compaction_relax_iters": result.get(
            "sam_compaction_relax_iters",
            np.nan,
        ),
        "sam_compaction_relax_s": result.get(
            "sam_compaction_relax_s",
            np.nan,
        ),
        "sam_compaction_pair_build_s": result.get(
            "sam_compaction_pair_build_s",
            np.nan,
        ),
        "sam_compaction_pair_forces_s": result.get(
            "sam_compaction_pair_forces_s",
            np.nan,
        ),
        "sam_compaction_converged": result.get(
            "sam_compaction_converged",
            np.nan,
        ),
        "sam_compaction_removed_fibers": result.get(
            "sam_compaction_removed_fibers",
            np.nan,
        ),
        "sam_compaction_max_fiber_removals": result.get(
            "sam_compaction_max_fiber_removals",
            np.nan,
        ),
        "sam_post_raster_compaction_retries": result.get(
            "sam_post_raster_compaction_retries",
            np.nan,
        ),
        "sam_post_raster_orientation_retries": result.get(
            "sam_post_raster_orientation_retries",
            np.nan,
        ),
        "sam_orientation_optimizer_accepted_steps": result.get(
            "sam_orientation_optimizer_accepted_steps",
            np.nan,
        ),
        "sam_orientation_optimizer_line_search_evaluations": result.get(
            "sam_orientation_optimizer_line_search_evaluations",
            np.nan,
        ),
        "sam_orientation_optimizer_contact_projections": result.get(
            "sam_orientation_optimizer_contact_projections",
            np.nan,
        ),
        "sam_orientation_optimizer_s": result.get(
            "sam_orientation_optimizer_s",
            np.nan,
        ),
        "sam_long_fiber_vf_fallback": result.get(
            "sam_long_fiber_vf_fallback",
            np.nan,
        ),
        "sam_generation_strategy": result.get(
            "sam_generation_strategy",
            "",
        ),
        "sam_t_compaction_s": result.get("sam_t_compaction_s", np.nan),
        "sam_topup_count": result.get("sam_topup_count", np.nan),
        "sam_final_overlap": result.get("sam_final_overlap", np.nan),
        "sam_final_A_err": result.get("sam_final_A_err", np.nan),
        "sam_A2_error_rel": result.get("sam_A2_error_rel", np.nan),
        "sam_voxel_A2_error_rel": result.get(
            "sam_voxel_A2_error_rel",
            np.nan,
        ),

        "cv_pctl": snapshot["cv_pctl"],
        "cv_pctl_percent": snapshot["cv_pctl_percent"],
        "kanit_error": snapshot["kanit_error"],
        "kanit_error_percent": snapshot["kanit_error_percent"],
        "kanit_n_required_pctl": snapshot["kanit_n_required_pctl"],
        "kanit_rel_tol": snapshot["kanit_rel_tol"],
        "kanit_rel_tol_percent": snapshot["kanit_rel_tol_percent"],
        "kanit_confidence_factor": snapshot["kanit_confidence_factor"],
        "kanit_percentile": snapshot["kanit_percentile"],
        "stable_count": snapshot["stable_count"],
        "converged": snapshot["converged"],
    })

    for key, value in result.items():
        if key in {"props", "phase_array", "ori_array"} or key in row:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value

    for col in ENGINEERING_PROPERTY_COLUMNS:
        row[f"{col}_val"] = float(props[col]) if props else np.nan

        row[f"{col}_mean"] = float(snapshot["means"][col])
        row[f"{col}_std"] = float(snapshot["stds"][col])
        row[f"{col}_cv"] = float(snapshot["cvs"][col])
        row[f"{col}_cv_percent"] = 100.0 * float(snapshot["cvs"][col])
        row[f"{col}_kanit_error"] = float(snapshot["kanit_errors"][col])
        row[f"{col}_kanit_error_percent"] = 100.0 * float(snapshot["kanit_errors"][col])
        row[f"{col}_kanit_n_required"] = float(snapshot["kanit_n_required"][col])

    return row


def write_design_audit_workbooks(
    *,
    geometry_path: Path,
    convergence_path: Path,
    geometry_rows: List[Dict[str, Any]],
    convergence_rows: List[Dict[str, Any]],
    design_row: Dict[str, Any],
    config: Dict[str, Any],
    seed_items: List[Tuple[int, int]],
) -> None:
    design_input = pd.DataFrame([flatten_mapping(design_row)])
    run_configuration = config_table(config)
    seed_schedule = pd.DataFrame(
        [
            {
                "design_id": int(design_row["design_id"]),
                "mc_id": int(mc_id),
                "seed": int(seed),
                "mc_base_seed": int(config["mc_base_seed"]),
                "seed_derivation": "SeedSequence([mc_base_seed, design_id])",
            }
            for mc_id, seed in seed_items
        ]
    )
    write_excel_atomic(
        geometry_path,
        {
            "geometrias": pd.DataFrame(geometry_rows),
            "input_diseno": design_input,
            "configuracion": run_configuration,
        },
    )
    write_excel_atomic(
        convergence_path,
        {
            "realizaciones": pd.DataFrame(convergence_rows),
            "input_diseno": design_input,
            "configuracion": run_configuration,
            "semillas_planeadas": seed_schedule,
        },
    )


def collect_campaign_audit_rows(
    run_dir: Path,
    patterns: str | Iterable[str],
    sheet_name: str,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    seen: set[Path] = set()
    if isinstance(patterns, str):
        patterns = (patterns,)

    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            try:
                frame = pd.read_excel(path, sheet_name=sheet_name)
            except ValueError:
                frame = pd.read_excel(path)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["audit_source_xlsx"] = str(path)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def write_campaign_audit_workbook(
    *,
    summary_path: Path,
    summary: pd.DataFrame,
    design_df: pd.DataFrame,
    run_config: Dict[str, Any],
    run_dir: Path,
) -> None:
    realizations = collect_campaign_audit_rows(
        run_dir,
        (
            "convergencia_design_*.xlsx",
            "*/convergencia_design_*.xlsx",
        ),
        "realizaciones",
    )
    geometries = collect_campaign_audit_rows(
        run_dir,
        (
            "geometry_generation_design_*.xlsx",
            "*/geometry_generation_design_*.xlsx",
        ),
        "geometrias",
    )
    write_excel_atomic(
        summary_path,
        {
            "resumen_disenos": summary,
            "inputs_sobol": design_df,
            "configuracion": config_table(run_config),
            "realizaciones": realizations,
            "geometrias": geometries,
        },
    )


def combine_design_metadata_with_summary(
    design_row: Dict[str, Any],
    summary_row: Dict[str, Any],
) -> Dict[str, Any]:
    """Mantiene la metadata Sobol aunque el resumen venga de convergencia."""
    out = dict(design_row)
    out.update(summary_row)
    out["config_id"] = design_row.get("config_id", out.get("config_id"))
    out["design_id"] = design_row.get("design_id", out.get("design_id"))
    return out


def complete_summary_metadata(
    summary_rows: List[Dict[str, Any]],
    design_df: pd.DataFrame,
) -> pd.DataFrame:
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary

    design_metadata = design_df.copy()
    design_metadata["config_id"] = design_metadata["config_id"].astype(str)
    design_metadata["design_id"] = design_metadata["design_id"].astype(int)

    summary = summary.copy()
    summary["config_id"] = summary["config_id"].astype(str)
    summary["design_id"] = summary["design_id"].astype(int)

    metadata_columns = [
        column
        for column in design_metadata.columns
        if column not in {"config_id", "design_id"}
    ]
    overlap_columns = [
        column
        for column in metadata_columns
        if column in summary.columns
    ]
    metadata_to_join = design_metadata[
        ["config_id", "design_id", *metadata_columns]
    ]
    merged = summary.merge(
        metadata_to_join,
        on=["config_id", "design_id"],
        how="left",
        suffixes=("", "_sobol"),
        validate="one_to_one",
    )

    for column in overlap_columns:
        sobol_column = f"{column}_sobol"
        merged[column] = merged[column].combine_first(merged[sobol_column])
        merged.drop(columns=[sobol_column], inplace=True)

    sobol_only_columns = [
        f"{column}_sobol"
        for column in metadata_columns
        if f"{column}_sobol" in merged.columns
    ]
    if sobol_only_columns:
        merged.drop(columns=sobol_only_columns, inplace=True)

    ordered_columns = [
        column
        for column in design_metadata.columns
        if column in merged.columns
    ]
    ordered_columns.extend(
        column for column in merged.columns if column not in ordered_columns
    )
    return merged[ordered_columns]


# ============================================================================
# DISEÑO COMPLETO
# ============================================================================

def run_single_design_two_phase_online_stop(
    design_row: Dict[str, Any],
    run_dir: Path,
    config: Dict[str, Any],
    global_manager: Any = None,
) -> Dict[str, Any]:
    config = dict(config)

    design_id = int(design_row["design_id"])
    geometry_path, convergence_path, summary_design_path = design_output_paths(
        run_dir,
        design_id,
    )
    work_root = seed_work_root(run_dir)

    seeds = derived_mc_seeds(
        design_id=design_id,
        n_seeds=int(config["max_seeds"]),
        base_seed=int(config["mc_base_seed"]),
    )

    seed_items = list(enumerate(seeds))

    online = OnlineConvergence(
        property_columns=ENGINEERING_PROPERTY_COLUMNS,
        kanit_rel_tol=float(config["kanit_rel_tol"]),
        kanit_confidence_factor=float(config["kanit_confidence_factor"]),
        kanit_confidence_mode=str(config["kanit_confidence_mode"]),
        kanit_percentile=float(config["kanit_percentile"]),
        n_min=int(config["n_min"]),
        stable_steps=int(config["stable_steps"]),
    )

    all_geometry_rows: List[Dict[str, Any]] = []
    convergence_rows: List[Dict[str, Any]] = []
    ok_results: List[Dict[str, Any]] = []

    selected_seed_dir: Optional[str] = None
    selected_seed: Optional[int] = None
    convergence_mc_id: Optional[int] = None
    abort_design = False
    abort_reason = ""
    consecutive_invalid_geometries = 0

    n_generated = 0
    n_solve_attempts = 0
    n_unsolved_geometry_cleaned = 0
    unsolved_geometry_deleted_bytes = 0
    geometry_wait_sum_s = 0.0
    registered_geometry_ids: set[int] = set()
    all_solved_seed_dirs: set[str] = set()

    t_design0 = time.perf_counter()

    geometry_batch_size = max(1, int(config["geometry_batch_size"]))
    geometry_workers_effective = resolve_geometry_worker_count(
        len(seed_items),
        config,
    )
    gpu_solve_batch_size = 1

    print(
        f"[DESIGN {design_id}] seeds={len(seeds)} | "
        f"batch={geometry_batch_size} | prefetch={config['geometry_prefetch_count']} | "
        f"geom_workers={geometry_workers_effective}/{config['geometry_cpu_budget']} | "
        f"nvox={estimate_design_nvox(design_row)}",
        flush=True,
    )

    def register_geometry_results(items: List[Dict[str, Any]]) -> None:
        nonlocal n_generated, geometry_wait_sum_s
        for item in items:
            mc_id = int(item.get("mc_id", -1))
            if mc_id in registered_geometry_ids:
                continue
            registered_geometry_ids.add(mc_id)
            n_generated += 1
            wait_s = item.get("t_geometry_queue_wait_s", 0.0)
            if pd.notna(wait_s):
                geometry_wait_sum_s += float(wait_s)
            all_geometry_rows.append(drop_large_arrays(item))

    def geometry_batch_iterator() -> Iterator[List[Dict[str, Any]]]:
        if global_manager is not None:
            while (
                global_manager.has_next(design_id)
                and not online.converged
                and not abort_design
            ):
                yield global_manager.take_batch(design_id, gpu_solve_batch_size)
            return

        # Fallback sin global manager
        for batch_start in range(0, len(seed_items), geometry_batch_size):
            if online.converged or abort_design:
                return
            current_seed_items = seed_items[batch_start:batch_start + geometry_batch_size]
            yield generate_geometry_batch(
                design_row=design_row,
                work_root=work_root,
                seed_items=current_seed_items,
                config=config,
            )

    with tqdm(
        total=len(seed_items),
        desc=f"Design {design_id} seeds",
        unit="seed",
        leave=True,
    ) as seed_pbar:
        for geometry_results in geometry_batch_iterator():
            if online.converged or abort_design:
                break

            register_geometry_results(geometry_results)

            solved_seed_dirs: set[str] = set()

            for solve_start in range(0, len(geometry_results), gpu_solve_batch_size):
                if online.converged or abort_design:
                    break

                raw_solve_batch = geometry_results[
                    solve_start:solve_start + gpu_solve_batch_size
                ]

                if bool(config["preload_geometries_to_ram"]):
                    solve_batch = [
                        preload_geometry_to_ram(item)
                        for item in raw_solve_batch
                    ]
                else:
                    solve_batch = [dict(item) for item in raw_solve_batch]

                solved_pairs = list(
                    solve_geometry_batch_gpu(
                        solve_batch=solve_batch,
                        design_row=design_row,
                        config=config,
                    )
                )
                solved_pairs.sort(
                    key=lambda pair: int(pair[0].get("mc_id", -1))
                )

                for solve_result, geometry_result in solved_pairs:
                    n_solve_attempts += 1

                    snapshot = online.snapshot()
                    abort_after_row = False
                    fatal_environment_error = False

                    solver_ok = bool(
                        solve_result.get("status") == "ok"
                        and solve_result.get("props")
                    )
                    accepted_for_convergence = bool(
                        solver_ok and vf_meets_convergence_metric(solve_result)
                    )
                    solve_result["accepted_for_convergence"] = (
                        accepted_for_convergence
                    )

                    if accepted_for_convergence:
                        consecutive_invalid_geometries = 0
                        solve_result["convergence_exclusion_reason"] = ""
                        snapshot = online.update(solve_result["props"])
                        ok_results.append(solve_result)

                        selected_seed_dir = solve_result.get("seed_dir")
                        selected_seed = int(solve_result["seed"])
                        if math.isfinite(snapshot["kanit_error_percent"]):
                            kanit_text = (
                                f"Kanit_p{snapshot['kanit_percentile']:.0f}="
                                f"{snapshot['kanit_error_percent']:.3f}% | "
                                f"n_req≈{snapshot['kanit_n_required_pctl']:.0f}"
                            )
                        else:
                            kanit_text = (
                                "Kanit=pendiente | "
                                f"minimo={config['n_min']} seeds"
                            )
                        tqdm.write(
                            f"[KANIT][design {design_id}][seed {selected_seed}] "
                            f"n={snapshot['n_samples']} | "
                            f"{kanit_text} | "
                            f"tol={snapshot['kanit_rel_tol_percent']:.3f}% | "
                            f"stable={snapshot['stable_count']} | "
                            f"converged={snapshot['converged']}"
                        )

                        if snapshot["converged"] and convergence_mc_id is None:
                            convergence_mc_id = int(solve_result["mc_id"])

                    else:
                        consecutive_invalid_geometries += 1
                        if solver_ok:
                            exclusion_reason = "vf_metric_not_satisfied"
                        else:
                            exclusion_reason = str(
                                solve_result.get("status", "solver_not_ok")
                            )
                        solve_result["convergence_exclusion_reason"] = (
                            exclusion_reason
                        )
                        snapshot = online.snapshot()
                        fatal_environment_error = (
                            solve_result.get("status") == "solve_error"
                            and is_fatal_solver_environment_error(
                                solve_result.get("solve_error", "")
                            )
                        )
                        tqdm.write(
                            f"[ONLINE][design {design_id}][seed {solve_result['seed']}] "
                            f"no entra a estadística | "
                            f"status={solve_result.get('status')} | "
                            f"razon={exclusion_reason} | "
                            f"error={solve_result.get('solve_error', '')}"
                        )
                        if fatal_environment_error:
                            abort_after_row = True

                    cleanup_seed_geometry_after_result(solve_result, config)
                    if solve_result.get("seed_dir"):
                        seed_dir_value = str(solve_result["seed_dir"])
                        solved_seed_dirs.add(seed_dir_value)
                        all_solved_seed_dirs.add(seed_dir_value)

                    seed_row = build_seed_row(
                        solve_result,
                        snapshot,
                        design_row,
                        config,
                    )
                    convergence_rows.append(seed_row)

                    seed_pbar.update(1)
                    postfix = {
                        "samples": online.n,
                        "stable": online.stable_count,
                        "conv": online.converged,
                    }
                    if (
                        snapshot
                        and math.isfinite(snapshot["kanit_error_percent"])
                    ):
                        postfix["Kanit%"] = f"{snapshot['kanit_error_percent']:.3f}"
                        postfix["tol%"] = f"{snapshot['kanit_rel_tol_percent']:.3f}"
                        postfix["n_req"] = f"{snapshot['kanit_n_required_pctl']:.0f}"
                    elif snapshot:
                        postfix["Kanit%"] = "pendiente"
                    elif online.n > 0 and np.isfinite(online.last_kanit_error):
                        postfix["Kanit%"] = f"{100.0 * online.last_kanit_error:.3f}"
                    else:
                        postfix["Kanit%"] = "-"

                    seed_pbar.set_postfix(postfix)

                    if (
                        online.n == 0
                        and consecutive_invalid_geometries
                        >= int(config["max_consecutive_invalid_geometries"])
                    ):
                        abort_design = True
                        abort_reason = (
                            f"{consecutive_invalid_geometries} geometrías "
                            "consecutivas sin una muestra válida; "
                            f"última razón={exclusion_reason}"
                        )
                        tqdm.write(
                            f"[STOP][design {design_id}] {abort_reason}"
                        )

                    if abort_after_row:
                        write_design_audit_workbooks(
                            geometry_path=geometry_path,
                            convergence_path=convergence_path,
                            geometry_rows=all_geometry_rows,
                            convergence_rows=convergence_rows,
                            design_row=design_row,
                            config=config,
                            seed_items=seed_items,
                        )
                        if fatal_environment_error:
                            cleanup_seed_geometry_npy(
                                solve_result.get("seed_dir", ""),
                                delete_phase_ori=True,
                            )
                            raise SolverEnvironmentError(
                                "La GPU no puede cargar sus librerias CUDA. "
                                f"Detalle: {solve_result.get('solve_error', '')}"
                            )

                    if abort_design:
                        break

                del solve_batch

                if bool(config["free_gpu_memory_each_batch"]):
                    free_gpu_memory_pool()

                if abort_design:
                    break

            if online.converged or abort_design:
                cleanup_summary = cleanup_unsolved_geometry_results(
                    geometry_results,
                    solved_seed_dirs,
                    config,
                    reason=(
                        "convergencia Kanit"
                        if online.converged
                        else "rechazos geométricos consecutivos"
                    ),
                )
                n_unsolved_geometry_cleaned += int(
                    cleanup_summary["n_unsolved_geometry_cleaned"]
                )
                unsolved_geometry_deleted_bytes += int(
                    cleanup_summary["unsolved_geometry_deleted_bytes"]
                )

            if online.converged:
                print(
                    f"[STOP] Design {design_id} convergió con "
                    f"{online.n} muestras válidas. "
                    f"No se generan ni resuelven más seeds.",
                    flush=True,
                )
                break
            if abort_design:
                break

    write_design_audit_workbooks(
        geometry_path=geometry_path,
        convergence_path=convergence_path,
        geometry_rows=all_geometry_rows,
        convergence_rows=convergence_rows,
        design_row=design_row,
        config=config,
        seed_items=seed_items,
    )

    t_design_s = time.perf_counter() - t_design0

    if online.n > 0:
        final_mean = dict(online.means)
        final_std = online.current_std()
    else:
        final_mean = {col: np.nan for col in ENGINEERING_PROPERTY_COLUMNS}
        final_std = {col: np.nan for col in ENGINEERING_PROPERTY_COLUMNS}

    final_row = dict(design_row)

    final_row["n_seeds_max"] = int(config["max_seeds"])
    final_row["mc_base_seed"] = int(config["mc_base_seed"])
    final_row["n_seeds_generated"] = int(n_generated)
    final_row["n_solve_attempts"] = int(n_solve_attempts)
    final_row["n_samples_stats"] = int(online.n)
    final_row["n_rejected_from_convergence"] = int(
        sum(
            not bool(row.get("accepted_for_convergence", False))
            for row in convergence_rows
        )
    )
    final_row["vf_acceptance_rule"] = (
        "sam_vf_ok and abs(sam_vf-sam_vf_target)<=sam_vf_tolerance"
    )
    vf_miss_rows = [
        row for row in all_geometry_rows
        if str(row.get("status", "")) == "geometry_vf_miss"
    ]
    overlap_miss_rows = [
        row for row in all_geometry_rows
        if (
            pd.notna(row.get("sam_overlap_ok", np.nan))
            and not bool(row.get("sam_overlap_ok"))
        )
        or str(row.get("status", "")) == "geometry_overlap_miss"
    ]
    long_fiber_fallback_rows = [
        row for row in all_geometry_rows
        if bool(row.get("sam_long_fiber_vf_fallback", False))
    ]
    sam_vf_values = [
        float(row.get("sam_vf"))
        for row in all_geometry_rows
        if pd.notna(row.get("sam_vf", np.nan))
    ]
    sam_vf_errors = [
        float(row.get("sam_vf_error"))
        for row in all_geometry_rows
        if pd.notna(row.get("sam_vf_error", np.nan))
    ]
    final_row["n_geometry_vf_miss"] = int(len(vf_miss_rows))
    final_row["n_geometry_overlap_miss"] = int(len(overlap_miss_rows))
    final_row["n_long_fiber_vf_fallback"] = int(len(long_fiber_fallback_rows))
    final_row["sam_vf_min_generated"] = (
        float(np.nanmin(sam_vf_values)) if sam_vf_values else np.nan
    )
    final_row["sam_vf_mean_generated"] = (
        float(np.nanmean(sam_vf_values)) if sam_vf_values else np.nan
    )
    final_row["sam_vf_error_min_generated"] = (
        float(np.nanmin(sam_vf_errors)) if sam_vf_errors else np.nan
    )
    final_row["sam_vf_error_mean_generated"] = (
        float(np.nanmean(sam_vf_errors)) if sam_vf_errors else np.nan
    )
    final_row["n_unsolved_geometry_cleaned"] = int(n_unsolved_geometry_cleaned)
    final_row["unsolved_geometry_deleted_bytes"] = int(unsolved_geometry_deleted_bytes)
    final_row["unsolved_geometry_deleted_mib"] = float(
        unsolved_geometry_deleted_bytes / (1024 ** 2)
    )

    final_row["converged"] = bool(online.converged)
    if online.converged:
        final_row["run_status"] = "completed_converged"
        final_row["run_error"] = ""
    elif abort_design:
        final_row["run_status"] = "geometry_rejected_no_valid_samples"
        final_row["run_error"] = abort_reason
    else:
        final_row["run_status"] = "completed_max_seeds_without_convergence"
        final_row["run_error"] = ""
    final_row["consecutive_invalid_geometries_final"] = int(
        consecutive_invalid_geometries
    )
    final_row["max_consecutive_invalid_geometries"] = int(
        config["max_consecutive_invalid_geometries"]
    )
    final_row["convergence_mc_id"] = convergence_mc_id
    final_row["selected_seed"] = selected_seed
    final_row["selected_seed_dir"] = selected_seed_dir

    final_snapshot = online.snapshot()
    final_row["convergence_criterion"] = (
        "student_t_mean_confidence_interval"
        if str(config["kanit_confidence_mode"]) == "student_t"
        else "kanit_mean_confidence_interval"
    )
    final_row["kanit_rel_tol"] = float(config["kanit_rel_tol"])
    final_row["kanit_rel_tol_percent"] = 100.0 * float(config["kanit_rel_tol"])
    final_row["kanit_confidence_factor"] = float(
        final_snapshot["kanit_confidence_factor"]
    )
    final_row["kanit_confidence_mode"] = str(config["kanit_confidence_mode"])
    final_row["kanit_percentile"] = float(config["kanit_percentile"])
    final_row["kanit_error_final"] = float(online.last_kanit_error)
    final_row["kanit_error_final_percent"] = 100.0 * float(online.last_kanit_error)
    final_row["kanit_n_required_final_pctl"] = float(online.last_kanit_n_required)
    final_row["cv_final_pctl"] = float(final_snapshot["cv_pctl"])
    final_row["cv_final_pctl_percent"] = float(
        final_snapshot["cv_pctl_percent"]
    )
    final_row["n_min"] = int(config["n_min"])
    final_row["stable_steps"] = int(config["stable_steps"])

    final_row["geometry_batch_size"] = int(config["geometry_batch_size"])
    final_row["max_parallel_geometries"] = int(config["max_parallel_geometries"])
    final_row["geometry_workers_effective"] = int(geometry_workers_effective)
    final_row["generator_num_cores"] = int(config["generator_num_cores"])
    final_row["geometry_cpu_budget"] = int(config["geometry_cpu_budget"])
    final_row["generator_verbose"] = bool(config["generator_verbose"])
    final_row["sam_vf_tolerance"] = float(config["sam_vf_tolerance"])
    final_row["sam_reject_vf_miss"] = bool(config["sam_reject_vf_miss"])
    final_row["sam_A2_tolerance"] = float(config["sam_A2_tolerance"])
    final_row["sam_voxel_A2_tolerance"] = float(
        config["sam_voxel_A2_tolerance"]
    )
    final_row["sam_reject_A2_miss"] = bool(config["sam_reject_A2_miss"])
    final_row["sam_overlap_tolerance"] = float(config["sam_overlap_tolerance"])
    final_row["sam_reject_overlap_miss"] = bool(
        config["sam_reject_overlap_miss"]
    )
    final_row["pipeline_overlap_generation"] = bool(
        config["pipeline_overlap_generation"]
    )
    final_row["persistent_geometry_pool"] = bool(
        config["persistent_geometry_pool"]
    )
    final_row["geometry_prefetch_count"] = int(config["geometry_prefetch_count"])
    final_row["geometry_prefetch_effective"] = int(
        global_manager.global_prefetch_count
        if global_manager is not None
        else min(len(seed_items), geometry_batch_size)
    )
    final_row["geometry_refill_low_watermark"] = int(
        config["geometry_refill_low_watermark"]
    )
    final_row["geometry_refill_low_watermark_effective"] = 0
    final_row["geometry_disk_est_mib"] = float(
        (estimate_design_nvox(design_row) ** 3 * 13 + 4096) / (1024 ** 2)
    )
    final_row["geometry_prefetch_disk_budget_gib"] = float(
        config["geometry_prefetch_disk_budget_gib"]
    )
    final_row["geometry_min_free_disk_gib"] = float(
        config["geometry_min_free_disk_gib"]
    )
    final_row["geometry_process_start_method"] = str(
        config["geometry_process_start_method"]
    )
    final_row["t_geometry_queue_wait_sum_s"] = float(geometry_wait_sum_s)
    final_row["t_geometry_queue_wait_mean_s"] = float(
        geometry_wait_sum_s / max(1, n_solve_attempts)
    )
    final_row["n_geometries_generated_unused"] = int(
        max(0, n_generated - n_solve_attempts)
    )
    final_row["n_geometries_consumed"] = int(n_generated)
    final_row["n_geometries_submitted"] = int(
        global_manager.next_submit.get(design_id, n_generated)
        if global_manager is not None
        else n_generated
    )
    final_row["n_geometries_prefetched_not_consumed"] = int(
        max(
            0,
            final_row["n_geometries_submitted"] - n_generated,
        )
    )
    final_row["delete_geometry_npy_after_solve"] = bool(
        config["delete_geometry_npy_after_solve"]
    )
    final_row["delete_geometry_npy_after_failed_solve"] = bool(
        config["delete_geometry_npy_after_failed_solve"]
    )

    final_row["fft_backend"] = str(config["fft_backend"])
    final_row["cupy_plan_mode"] = str(config["cupy_plan_mode"])
    final_row["solver_tol"] = float(config["solver_tol"])
    final_row["cupy_loadcase_parallel"] = bool(config["cupy_loadcase_parallel"])
    final_row["cupy_loadcase_workers"] = int(config["cupy_loadcase_workers"])
    final_row["solver_real_dtype"] = str(config["solver_real_dtype"])
    final_row["cfield_origin"] = str(config["cfield_origin"])
    final_row["cfield_storage"] = str(config["cfield_storage"])
    final_row["projection_storage"] = str(config["projection_storage"])
    final_row["projection_backend"] = str(config["projection_backend"])
    final_row["keep_solutions_on_device"] = bool(config["keep_solutions_on_device"])
    final_row["cupy_fused_matvec"] = bool(config["cupy_fused_matvec"])
    final_row["cupy_unscaled_fft_pair"] = bool(config["cupy_unscaled_fft_pair"])
    final_row["cupy_lazy_scalars"] = bool(config["cupy_lazy_scalars"])
    final_row["cupy_fused_cg_updates"] = bool(config["cupy_fused_cg_updates"])
    final_row["solver_load_batch_size"] = int(config["solver_load_batch_size"])
    final_row["postprocess_batch_size"] = int(config["postprocess_batch_size"])
    final_row["postprocess_assembly"] = str(config["postprocess_assembly"])

    final_row["t_design_total_s"] = float(t_design_s)
    final_row["geometry_xlsx"] = str(geometry_path)
    final_row["convergence_xlsx"] = str(convergence_path)

    if ok_results:
        t_gen_list = [r.get("t_gen_s", np.nan) for r in ok_results]
        t_solver_list = [r.get("t_solver_s", np.nan) for r in ok_results]
        t_extract_list = [r.get("t_extract_s", np.nan) for r in ok_results]

        t_total_seed_list = [
            r.get("t_gen_s", np.nan) + r.get("t_total_after_geometry_s", np.nan)
            for r in ok_results
        ]

        final_row["t_gen_mean_s"] = float(np.nanmean(t_gen_list))
        final_row["t_solver_mean_s"] = float(np.nanmean(t_solver_list))
        final_row["t_extract_mean_s"] = float(np.nanmean(t_extract_list))
        final_row["t_total_seed_mean_s"] = float(np.nanmean(t_total_seed_list))
        final_row["t_gen_sum_s"] = float(np.nansum(t_gen_list))
        final_row["t_solver_sum_s"] = float(np.nansum(t_solver_list))
        final_row["t_solver_min_s"] = float(np.nanmin(t_solver_list))
        final_row["t_solver_max_s"] = float(np.nanmax(t_solver_list))
    else:
        final_row["t_gen_mean_s"] = np.nan
        final_row["t_solver_mean_s"] = np.nan
        final_row["t_extract_mean_s"] = np.nan
        final_row["t_total_seed_mean_s"] = np.nan
        final_row["t_gen_sum_s"] = np.nan
        final_row["t_solver_sum_s"] = np.nan
        final_row["t_solver_min_s"] = np.nan
        final_row["t_solver_max_s"] = np.nan

    for col in ENGINEERING_PROPERTY_COLUMNS:
        final_row[f"{col}_mean"] = final_mean[col]
        final_row[f"{col}_std"] = final_std[col]
        final_row[f"{col}_cv"] = float(final_snapshot["cvs"][col])
        final_row[f"{col}_cv_percent"] = (
            100.0 * float(final_snapshot["cvs"][col])
        )

    write_excel_atomic(
        summary_design_path,
        {"resumen_diseno": pd.DataFrame([final_row])},
    )

    return final_row


# ============================================================================
# VALIDACIÓN EXCEL
# ============================================================================

def validate_input_excel(df: pd.DataFrame) -> None:
    required_columns = [
        "design_id",
        "config_id",
        "res",
        "caja_um",
        "L_um",
        "d_um",
        "Vf_target",
        "a11",
        "a22",
        "Em",
        "nu_m",
        "Ef_L",
        "Ef_T",
        "nu_LT",
        "nu_TT",
        "G_LT",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            "El Excel de entrada no tiene las columnas necesarias para fase 2: "
            f"{missing}"
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    check_cupy_gpu()
    warmup_gpu_once()

    valid_excel_path = resolve_valid_excel()
    run_dir = resolve_run_dir()

    print(
        "[SOBOL GPU] "
        f"valid_excel={valid_excel_path} | run_dir={run_dir} | "
        f"max_seeds={MAX_SEEDS} | n_min={N_MIN} | "
        f"kanit_tol={100.0 * KANIT_REL_TOL:.3f}% | "
        f"geom_batch={GEOMETRY_BATCH_SIZE} | geom_workers={MAX_PARALLEL_GEOMETRIES}",
        flush=True,
    )

    design_df = pd.read_excel(valid_excel_path)
    validate_input_excel(design_df)

    if "is_operable" in design_df.columns:
        design_df = design_df.loc[design_df["is_operable"].fillna(False)].copy()

    if MAX_DESIGNS_TO_RUN is not None:
        design_df = design_df.head(int(MAX_DESIGNS_TO_RUN)).copy()

    total_points = len(design_df)

    if total_points < 1:
        raise ValueError("No hay diseños válidos para ejecutar.")

    execution_df = order_designs_for_execution(design_df)
    execution_order = [
        int(value) for value in execution_df["design_id"].tolist()
    ]
    prefetch_order = balance_design_ids_for_prefetch(
        [row.to_dict() for _, row in execution_df.iterrows()]
    )
    print(
        "[PIPELINE] Orden de ejecucion por costo geometrico ascendente | "
        f"design_ids={execution_order}\n"
        "[PIPELINE] Prefetch geometrico balanceado barato/costoso | "
        f"design_ids={prefetch_order}",
        flush=True,
    )

    run_config = make_run_config()
    sam_numba_warmup_s = 0.0
    if str(run_config["sam_geometry_backend"]).lower() == "numba":
        try:
            sam_numba_warmup_s = warmup_numba_geometry_kernels()
            print(
                f"[SAM NUMBA] kernels listos/cacheados en "
                f"{sam_numba_warmup_s:.2f}s",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[WARN] No se pudo precalentar Numba; los workers "
                f"compilaran al iniciar: {exc}",
                flush=True,
            )
    run_config["sam_numba_warmup_s"] = float(sam_numba_warmup_s)
    try:
        campaign_settings = json.loads(
            os.environ.get("CAMPAIGN_SETTINGS_JSON", "{}")
        )
    except json.JSONDecodeError:
        campaign_settings = {
            "raw_campaign_settings": os.environ.get(
                "CAMPAIGN_SETTINGS_JSON",
                "",
            )
        }
    run_config.update(
        {
            "study_name": str(STUDY_NAME),
            "phase_label": str(PHASE_LABEL),
            "valid_excel_path": str(valid_excel_path),
            "run_dir": str(run_dir),
            "run_started_at": datetime.now().astimezone().isoformat(),
            "python_executable": str(sys.executable),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "seed_derivation": "SeedSequence([mc_base_seed, design_id])",
            "design_execution_order": execution_order,
            "design_execution_order_rule": (
                "estimated_fiber_count,nvox,design_id ascending"
            ),
            "geometry_prefetch_design_order": prefetch_order,
            "geometry_prefetch_design_order_rule": (
                "alternating lowest/highest estimated fiber count"
            ),
            "campaign_settings": campaign_settings,
        }
    )

    global_summary: List[Dict[str, Any]] = []
    design_error_count = 0
    geometry_executor: Optional[ProcessPoolExecutor] = None

    if (
        bool(run_config["pipeline_overlap_generation"])
        and bool(run_config["persistent_geometry_pool"])
    ):
        threads_per_geometry = max(1, int(run_config["generator_num_cores"]))
        geometry_pool_workers = max(
            1,
            min(
                int(run_config["max_parallel_geometries"]),
                max(
                    1,
                    int(run_config["geometry_cpu_budget"])
                    // threads_per_geometry,
                ),
            ),
        )
        geometry_mp_context = multiprocessing.get_context(
            str(run_config["geometry_process_start_method"])
        )
        geometry_executor = ProcessPoolExecutor(
            max_workers=geometry_pool_workers,
            mp_context=geometry_mp_context,
            initializer=_ignore_sigint_in_worker,
        )

        print(
            f"[PIPELINE] Pool SAM persistente | workers={geometry_pool_workers} | "
            f"threads/geom={threads_per_geometry}",
            flush=True,
        )

    # CREAR GESTOR GLOBAL
    global_manager = None
    if geometry_executor is not None:
        global_manager = GlobalGeometryManager(
            design_rows=[row.to_dict() for _, row in execution_df.iterrows()],
            run_dir=run_dir,
            config=run_config,
            executor=geometry_executor
        )

    try:
        for _, design_row in tqdm(
            execution_df.iterrows(),
            total=total_points,
            desc="Sobol designs",
            unit="design",
        ):
            payload = design_row.to_dict()

            try:
                design_id = payload.get('design_id')
                _, convergence_path, summary_design_path = design_output_paths(
                    run_dir,
                    int(design_id),
                )

                if summary_design_path.exists():
                    try:
                        completed_df = pd.read_excel(
                            summary_design_path,
                            sheet_name="resumen_diseno",
                        )
                        if len(completed_df) == 1:
                            completed_row = completed_df.iloc[0].to_dict()
                            same_criterion = completed_design_matches_config(
                                completed_row,
                                run_config,
                            )
                        else:
                            same_criterion = False
                        if same_criterion:
                            global_summary.append(
                                combine_design_metadata_with_summary(
                                    payload,
                                    completed_row,
                                )
                            )
                            if global_manager is not None:
                                global_manager.mark_converged(design_id)
                            continue
                    except Exception:
                        pass
                elif convergence_path.exists():
                    print(
                        f"[RESUME][design {design_id}] Excel parcial detectado sin "
                        "summary_design; se recalcula para no aceptar una corrida incompleta.",
                        flush=True,
                    )

                final_row = run_single_design_two_phase_online_stop(
                    design_row=payload,
                    run_dir=run_dir,
                    config=run_config,
                    global_manager=global_manager,
                )

                global_summary.append(final_row)

                # Limpiar cache FFT al cambiar de diseno (la malla cambia)
                free_gpu_memory_pool(clear_fft_cache=True)

            except SolverEnvironmentError as exc:
                print(f"[GPU][ABORT] {exc}", flush=True)
                return 3

            except Exception as exc:
                design_error_count += 1
                global_summary.append(
                    {
                        **payload,
                        "run_status": "design_error",
                        "run_error": str(exc),
                        "converged": False,
                        "n_samples_stats": 0,
                        "n_solve_attempts": 0,
                        "n_min": int(run_config["n_min"]),
                        "n_seeds_max": int(run_config["max_seeds"]),
                        "mc_base_seed": int(run_config["mc_base_seed"]),
                        "kanit_rel_tol": float(run_config["kanit_rel_tol"]),
                        "kanit_rel_tol_percent": (
                            100.0 * float(run_config["kanit_rel_tol"])
                        ),
                    }
                )
                print(
                    f"[ERROR] Falló design_id={payload.get('design_id')}: {exc}",
                    flush=True,
                )
            finally:
                if global_manager is not None:
                    global_manager.mark_converged(payload.get('design_id'))
    finally:
        if geometry_executor is not None:
            geometry_executor.shutdown(wait=True, cancel_futures=True)
        remove_empty_seed_work_root(run_dir)

    global_summary.sort(key=lambda row: int(row.get("design_id", -1)))

    summary_path = run_dir / "sobol_summary_final_gpu_cupy_two_phase_online_stop.xlsx"
    final_summary = complete_summary_metadata(global_summary, design_df)
    run_config["run_finished_at"] = datetime.now().astimezone().isoformat()
    run_config["n_designs_requested"] = int(total_points)
    run_config["n_designs_completed"] = int(len(final_summary))
    run_config["n_design_errors"] = int(design_error_count)
    n_designs_converged = int(
        final_summary.get("converged", pd.Series(dtype=bool))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    run_config["n_designs_converged"] = n_designs_converged
    run_config["n_designs_not_converged"] = int(
        total_points - n_designs_converged
    )
    write_campaign_audit_workbook(
        summary_path=summary_path,
        summary=final_summary,
        design_df=design_df,
        run_config=run_config,
        run_dir=run_dir,
    )

    print(f"[SOBOL GPU] completado | summary={summary_path}", flush=True)

    return 2 if design_error_count or n_designs_converged < total_points else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
