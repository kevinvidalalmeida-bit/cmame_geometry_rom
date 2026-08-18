"""Orquestacion del flujo principal definido desde ``main.py``."""

from __future__ import annotations

from datetime import datetime
import importlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


OFFICIAL_GPU_PROFILE = {
    "CUPY_ACCELERATORS": "cub",
    "SOLVER_REAL_DTYPE": "float32",
    "SOLVER_FFT_FORM": "r",
    "USE_CFIELD_MATERIAL_FAST_PATH": "1",
    "CFIELD_ORIGIN": "zero",
    "CFIELD_STORAGE": "sym21",
    "CFIELD_ROTATION_BATCH_SIZE": "0",
    "CFIELD_ASSIGN_CHUNK_VOXELS": "2000000",
    "CFIELD_INDEXED": "1",
    "PROJECTION_STORAGE": "direct",
    "PROJECTION_BACKEND": "numpy",
    "KEEP_SOLUTIONS_ON_DEVICE": "1",
    "CUPY_FUSED_MATVEC": "1",
    "CUPY_UNSCALED_FFT_PAIR": "1",
    "CUPY_LAZY_SCALARS": "1",
    "CUPY_FUSED_CG_UPDATES": "1",
    "CUPY_FUSED_XR_RR": "1",
    "CUPY_FUSED_DOT": "1",
    "CUPY_RESIDUAL_CHECK_EVERY": "1",
    "FAST_MACRO_ADD": "1",
    "CHECK_MACRO_MEAN": "0",
    "STORE_SOLUTION_FIELDS": "0",
    "SOLVER_LOAD_BATCH_SIZE": "1",
    "POSTPROCESS_BATCH_SIZE": "1",
    "GENERATOR_NUM_CORES": "2",
    "GENERATOR_VERBOSE": "0",
    "GEOMETRY_BATCH_SIZE": "15",
    "PIPELINE_OVERLAP_GENERATION": "1",
    "PERSISTENT_GEOMETRY_POOL": "1",
    "GEOMETRY_REFILL_LOW_WATERMARK": "0",
    "HARD_GEOMETRY_PREFETCH_PER_DESIGN": "10",
    "GEOMETRY_PREFETCH_DISK_BUDGET_GIB": "4.0",
    "GEOMETRY_MIN_FREE_DISK_GIB": "4.0",
    "GEOMETRY_PROCESS_START_METHOD": "spawn",
    "SAM_COMPACTION": "1",
    "SAM_COMPACTION_START_SCALE": "0.5",
    "SAM_COMPACTION_TOPUP_SCALE": "0.5",
    "SAM_COMPACTION_STAGES": "6",
    "SAM_COMPACTION_MAX_ITER": "120",
    "SAM_COMPACTION_PASSES": "2",
    "SAM_COLLECTIVE_FIRE": "1",
    "SAM_COLLECTIVE_FIRE_MAX_ITER": "2500",
    "SAM_COLLECTIVE_FIRE_MAX_RESTARTS": "3",
    "SAM_COLLECTIVE_FIRE_RESTART_PATIENCE": "250",
    "SAM_COLLECTIVE_FIRE_PAIR_REBUILD_INTERVAL": "4",
    "SAM_COLLECTIVE_FIRE_RESCUE_PASSES": "3",
    "SAM_COLLECTIVE_FIRE_REMOVAL_BATCH": "8",
    "SAM_A2_TOLERANCE": "0.01",
    "SAM_VOXEL_A2_TOLERANCE": "0.01",
    "SAM_REJECT_A2_MISS": "1",
    "SAM_ORIENTATION_MAX_ITER": "160",
    "SAM_GEOMETRY_BACKEND": "numba",
    "SAM_CUPY_MIN_CANDIDATE_PAIRS": "50000",
    "SAM_VF_TOLERANCE": "0.005",
    "SAM_OVERLAP_TOLERANCE_RELATIVE": "0.008333333333333333",
    "SAM_REJECT_OVERLAP_MISS": "1",
    "SAM_OVERLAP_RESCUE": "1",
    "KANIT_CONFIDENCE_MODE": "student_t",
    "CUPY_LOADCASE_PARALLEL": "0",
    "CUPY_LOADCASE_WORKERS": "1",
    "PRELOAD_GEOMETRIES_TO_RAM": "1",
    "FREE_GPU_MEMORY_EACH_BATCH": "0",
    "DELETE_GEOMETRY_NPY_AFTER_SOLVE": "1",
    "DELETE_GEOMETRY_NPY_AFTER_FAILED_SOLVE": "1",
    "SOLVER_TOL": "1e-4",

}


def _validate_range(name: str, value: object) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} debe ser una pareja (minimo, maximo).")
    low, high = float(value[0]), float(value[1])
    if not low < high:
        raise ValueError(f"{name}: el minimo debe ser menor que el maximo.")
    return low, high


def validate_settings(settings: dict[str, Any]) -> None:
    if int(settings["sobol_points"]) < 1:
        raise ValueError("SOBOL_POINTS debe ser >= 1.")
    if float(settings["fiber_diameter_um"]) <= 0:
        raise ValueError("FIBER_DIAMETER_UM debe ser > 0.")
    if float(settings["domain_length_factor"]) <= 0:
        raise ValueError("DOMAIN_LENGTH_FACTOR debe ser > 0.")
    if float(settings["voxels_per_fiber_diameter"]) <= 0:
        raise ValueError("VOXELS_PER_FIBER_DIAMETER debe ser > 0.")
    if int(settings["nvox_multiple"]) < 1:
        raise ValueError("NVOX_MULTIPLE debe ser >= 1.")
    if int(settings["max_seeds"]) < 1:
        raise ValueError("MAX_SEEDS_PER_DESIGN debe ser >= 1.")
    if int(settings["min_seeds"]) < 1:
        raise ValueError("MIN_SEEDS_BEFORE_STOP debe ser >= 1.")
    if int(settings["min_seeds"]) > int(settings["max_seeds"]):
        raise ValueError("MIN_SEEDS_BEFORE_STOP no puede superar MAX_SEEDS_PER_DESIGN.")
    if not 0 < float(settings["relative_error_target"]) < 1:
        raise ValueError("RELATIVE_ERROR_TARGET debe estar entre 0 y 1.")
    if str(settings.get("kanit_confidence_mode", "student_t")) not in {
        "fixed",
        "student_t",
    }:
        raise ValueError(
            "kanit_confidence_mode debe ser fixed o student_t."
        )
    if str(settings.get("geometry_backend", "numba")) not in {
        "numba",
        "cupy",
        "auto",
    }:
        raise ValueError("geometry_backend debe ser numba, cupy o auto.")

    ranges = settings["ranges"]
    if not isinstance(ranges, dict):
        raise ValueError("ranges debe ser un diccionario.")
    checked_ranges = {
        name: _validate_range(name, value)
        for name, value in ranges.items()
    }

    fixed_domain = settings.get("domain_length_um")
    if fixed_domain is not None:
        fixed_domain = float(fixed_domain)
        if fixed_domain <= 0:
            raise ValueError("DOMAIN_LENGTH_UM debe ser > 0 o None.")
        max_fiber_length = (
            checked_ranges["AR"][1] * float(settings["fiber_diameter_um"])
        )
        if fixed_domain <= max_fiber_length:
            raise ValueError(
                "DOMAIN_LENGTH_UM debe superar la longitud maxima de fibra "
                f"({max_fiber_length:g} um)."
            )


def _set_environment(settings: dict[str, Any]) -> None:
    for name, value in OFFICIAL_GPU_PROFILE.items():
        os.environ[name] = value

    ranges = settings["ranges"]
    values = {
        "STUDY_NAME": settings["campaign_name"],
        "N_SOBOL_POINTS": settings["sobol_points"],
        "SOBOL_SCRAMBLE": int(bool(settings["sobol_scramble"])),
        "SOBOL_SCRAMBLE_SEED": settings["sobol_seed"],
        "FIBER_DIAMETER_UM": settings["fiber_diameter_um"],
        "BOX_FACTOR": settings["domain_length_factor"],
        "DOMAIN_LENGTH_UM": settings.get("domain_length_um"),
        "DF_VOXEL_TARGET": settings["voxels_per_fiber_diameter"],
        "NVOX_MULTIPLE": settings["nvox_multiple"],
        "MAX_DESIGNS_TO_RUN": settings["sobol_points"],
        "MAX_SEEDS": settings["max_seeds"],
        "N_MIN": settings["min_seeds"],
        "KANIT_REL_TOL": settings["relative_error_target"],
        "KANIT_CONFIDENCE_MODE": settings.get(
            "kanit_confidence_mode",
            "student_t",
        ),
        "STABLE_STEPS": settings["stable_steps"],
        "MC_BASE_SEED": settings["mc_base_seed"],
        "MAX_PARALLEL_GEOMETRIES": settings["parallel_geometries"],
        "GEOMETRY_CPU_BUDGET": settings["cpu_budget"],
        "GEOMETRY_PREFETCH_COUNT": settings["geometry_prefetch"],
        "GEOMETRY_BATCH_SIZE": settings.get("geometry_batch_size", 15),
        "HARD_GEOMETRY_PREFETCH_PER_DESIGN": settings.get(
            "hard_geometry_prefetch_per_design",
            10,
        ),

        "GENERATOR_NUM_CORES": settings.get("generator_num_cores", 2),
        "SAM_GEOMETRY_BACKEND": settings.get("geometry_backend", "numba"),
        "GEOMETRY_REFILL_LOW_WATERMARK": settings.get("geometry_refill_low_watermark", 0),
        "GEOMETRY_PREFETCH_DISK_BUDGET_GIB": settings.get("geometry_prefetch_disk_budget_gib", 4.0),
        "GEOMETRY_MIN_FREE_DISK_GIB": settings.get("geometry_min_free_disk_gib", 2.0),



        "PIPELINE_OVERLAP_GENERATION": int(
            bool(settings.get("pipeline_overlap", True))
        ),
        "PERSISTENT_GEOMETRY_POOL": int(
            bool(settings.get("persistent_geometry_pool", True))
        ),
    }
    for name, bounds in ranges.items():
        values[f"SOBOL_{name.upper()}_MIN"] = bounds[0]
        values[f"SOBOL_{name.upper()}_MAX"] = bounds[1]

    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)

    os.environ["CAMPAIGN_SETTINGS_JSON"] = json.dumps(
        settings,
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    )


def _print_user_summary(settings: dict[str, Any]) -> None:
    domain = settings.get("domain_length_um")
    if domain is None:
        domain_text = f"{settings['domain_length_factor']} x longitud de fibra"
    else:
        domain_text = f"{domain} um fijos"

    print(
        "[MAIN] "
        f"sobol={settings['sobol_points']} | dominio={domain_text} | "
        f"d={settings['fiber_diameter_um']} um | "
        f"res={settings['voxels_per_fiber_diameter']} vox/diam | "
        f"max_seeds={settings['max_seeds']} | "
        f"kanit_tol={settings['relative_error_target']} | "
        f"geom_workers={settings['parallel_geometries']} | "
        f"geom_backend={settings.get('geometry_backend', 'numba')} | "
        f"overlap={bool(settings.get('pipeline_overlap', True))}"
    )


def run_campaign(settings: dict[str, Any]) -> int:
    validate_settings(settings)
    _set_environment(settings)
    _print_user_summary(settings)

    os.environ["PHASE_LABEL"] = "sobol_designs"
    designs = importlib.import_module("pipeline.generate_sobol_designs")
    valid_excel = designs.generate_designs()

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        PROJECT_ROOT
        / "results"
        / f"{settings['campaign_name']}_sobol_gpu_convergence_{run_tag}"
    )
    if "VALID_EXCEL_PATH" not in os.environ:
        os.environ["VALID_EXCEL_PATH"] = str(valid_excel)
    if "RUN_DIR" not in os.environ:
        os.environ["RUN_DIR"] = str(run_dir)
    os.environ["PHASE_LABEL"] = "sobol_gpu_convergence"

    solver = importlib.import_module("pipeline.sobol_gpu")
    return int(solver.main())
