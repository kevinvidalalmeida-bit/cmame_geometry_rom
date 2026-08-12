#!/usr/bin/env python3
"""Configuracion y ejecucion de la campana Sobol GPU."""

from __future__ import annotations

import os
from pathlib import Path
import sys


# ============================================================================
# CONFIGURACION DEL USUARIO
# Cambia solamente esta seccion.
# ============================================================================

# Entorno Python con CuPy y las dependencias del proyecto.
VENV_PATH = Path.home() / "Documentos/ANDRES/COMPUTATIONAL_WORKSPACEV4/.venv"

# Numero de puntos del espacio Sobol.
SOBOL_POINTS =2**10
SOBOL_SCRAMBLE = True
SOBOL_SEED = 20260621

# Geometria y dominio.
FIBER_DIAMETER_UM = 1.0

# Opcion recomendada: el dominio mide DOMAIN_LENGTH_FACTOR veces la longitud
# de la fibra de cada punto Sobol.
DOMAIN_LENGTH_FACTOR = 2

# Para usar una longitud de dominio fija, escribe un numero en micras.
# Ejemplo: DOMAIN_LENGTH_UM = 30.0
# Deja None para usar DOMAIN_LENGTH_FACTOR.
DOMAIN_LENGTH_UM = None

# Resolucion: numero objetivo de voxeles a traves del diametro de la fibra.
# Decision de produccion: voxelizacion binaria SAM/FFT con 6 voxeles por
# diametro; los composite voxels quedan solo como ablation de coarsening.
VOXELS_PER_FIBER_DIAMETER = 6
NVOX_MULTIPLE = 1

# Espacio de parametros Sobol: (minimo, maximo).
AR_RANGE = (5.0, 30.0)
VOLUME_FRACTION_RANGE = (0.05, 0.30)
MATRIX_MODULUS_RANGE = (1.1, 4.4)

# Rangos ampliados para incluir la fibra T395 y los mínimos originales
FIBER_LONGITUDINAL_MODULUS_RANGE = (72.0, 395.0)
FIBER_TRANSVERSE_MODULUS_RANGE = (6.0, 23.0)
FIBER_SHEAR_MODULUS_RANGE = (8.0, 30.0)
FIBER_NU_LT_RANGE = (0.20, 0.26)
FIBER_NU_TT_RANGE = (0.35, 0.40)

MATRIX_NU_RANGE = (0.35, 0.42)

# Monte Carlo y convergencia.
MAX_SEEDS_PER_DESIGN = 400
MIN_SEEDS_BEFORE_STOP = 10
RELATIVE_ERROR_TARGET = 0.02
STABLE_CONVERGENCE_STEPS = 1
MONTE_CARLO_BASE_SEED = 20260621
KANIT_CONFIDENCE_MODE = "student_t"

# Productores de geometria. El perfil de produccion recomendado usa Numba en
# CPU y deja la GPU libre para FFT.
PARALLEL_GEOMETRIES = 15
CPU_BUDGET = 30
GEOMETRY_PREFETCH = 30
GEOMETRY_BATCH_SIZE = 15
GENERATOR_NUM_CORES = 2
HARD_GEOMETRY_PREFETCH_PER_DESIGN = 10
GEOMETRY_BACKEND = "numba"
PIPELINE_OVERLAP = True

# Nombre usado para las carpetas dentro de results/.
CAMPAIGN_NAME = "Estudio_Sobol_Continuo"


# ============================================================================
# EJECUCION
# No es necesario editar debajo de esta linea.
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent


def _prepare_runtime() -> None:
    venv_python = VENV_PATH.expanduser().resolve() / "bin/python"
    if not venv_python.is_file():
        raise FileNotFoundError(f"No existe el Python del entorno: {venv_python}")

    env = os.environ.copy()
    cuda_dirs = sorted(
        str(path)
        for path in (VENV_PATH / "lib").glob("python*/site-packages/nvidia/*/lib")
        if path.is_dir()
    )
    if cuda_dirs:
        current = env.get("LD_LIBRARY_PATH", "")
        current_dirs = [value for value in current.split(":") if value]
        env["LD_LIBRARY_PATH"] = ":".join(
            [*cuda_dirs, *[value for value in current_dirs if value not in cuda_dirs]]
        )

    runtime_marker = ":".join(cuda_dirs)
    if env.get("ANDRES_CUDA_RUNTIME_READY") != runtime_marker:
        env["ANDRES_CUDA_RUNTIME_READY"] = runtime_marker
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


def user_settings(
    *,
    smoke: bool = False,
    sobol_points: int | None = None,
    seeds_per_design: int | None = None,
    geometry_backend: str | None = None,
    pipeline_overlap: bool | None = None,
) -> dict[str, object]:
    selected_sobol_points = (
        SOBOL_POINTS if sobol_points is None else int(sobol_points)
    )
    if selected_sobol_points < 1:
        raise ValueError("sobol_points debe ser >= 1.")
    selected_max_seeds = (
        MAX_SEEDS_PER_DESIGN
        if seeds_per_design is None
        else int(seeds_per_design)
    )
    if selected_max_seeds < 1:
        raise ValueError("seeds_per_design debe ser >= 1.")
    selected_min_seeds = min(
        MIN_SEEDS_BEFORE_STOP,
        selected_max_seeds,
    )
    selected_geometry_backend = (
        GEOMETRY_BACKEND
        if geometry_backend is None
        else str(geometry_backend).strip().lower()
    )
    if selected_geometry_backend not in {"numba", "cupy", "auto"}:
        raise ValueError(
            "geometry_backend debe ser 'numba', 'cupy' o 'auto'."
        )
    selected_pipeline_overlap = (
        PIPELINE_OVERLAP
        if pipeline_overlap is None
        else bool(pipeline_overlap)
    )

    settings: dict[str, object] = {
        "campaign_name": CAMPAIGN_NAME,
        "sobol_points": selected_sobol_points,
        "sobol_scramble": SOBOL_SCRAMBLE,
        "sobol_seed": SOBOL_SEED,
        "fiber_diameter_um": FIBER_DIAMETER_UM,
        "domain_length_factor": DOMAIN_LENGTH_FACTOR,
        "domain_length_um": DOMAIN_LENGTH_UM,
        "voxels_per_fiber_diameter": VOXELS_PER_FIBER_DIAMETER,
        "nvox_multiple": NVOX_MULTIPLE,
        "ranges": {
            "AR": AR_RANGE,
            "Vf_target": VOLUME_FRACTION_RANGE,
            "Em": MATRIX_MODULUS_RANGE,
            "Ef_L": FIBER_LONGITUDINAL_MODULUS_RANGE,
            "Ef_T": FIBER_TRANSVERSE_MODULUS_RANGE,
            "G_LT": FIBER_SHEAR_MODULUS_RANGE,
            "nu_LT": FIBER_NU_LT_RANGE,
            "nu_TT": FIBER_NU_TT_RANGE,
            "nu_m": MATRIX_NU_RANGE,
        },
        "max_seeds": selected_max_seeds,
        "min_seeds": selected_min_seeds,
        "relative_error_target": RELATIVE_ERROR_TARGET,
        "kanit_confidence_mode": KANIT_CONFIDENCE_MODE,
        "stable_steps": STABLE_CONVERGENCE_STEPS,
        "mc_base_seed": MONTE_CARLO_BASE_SEED,

        # geometría CPU — nombres que lee workflow.py
        "parallel_geometries": PARALLEL_GEOMETRIES,
        "cpu_budget": CPU_BUDGET,
        "geometry_prefetch": GEOMETRY_PREFETCH,
        "geometry_batch_size": GEOMETRY_BATCH_SIZE,
        "hard_geometry_prefetch_per_design": (
            HARD_GEOMETRY_PREFETCH_PER_DESIGN
        ),

        # pipeline overlap — nombre exacto que busca workflow._set_environment
        "pipeline_overlap": selected_pipeline_overlap,
        "persistent_geometry_pool": True,

        # nuevos: requieren el parche en workflow._set_environment
        "geometry_refill_low_watermark": 4,
        "geometry_prefetch_disk_budget_gib": 8.0,
        "geometry_min_free_disk_gib": 2.0,
        "generator_num_cores": GENERATOR_NUM_CORES,
        "geometry_backend": selected_geometry_backend,
    }
    if smoke:
        smoke_ranges = dict(settings["ranges"])
        smoke_ranges.update(
            {
                "AR": (7.9, 8.1),
                "Vf_target": (0.095, 0.105),
            }
        )
        settings.update(
            {
                "sobol_points": 1,
                "max_seeds": 1,
                "min_seeds": 1,
                "stable_steps": 1,
                "parallel_geometries": 1,
                "cpu_budget": 1,
                "generator_num_cores": 1,
                "geometry_prefetch": 1,
                "geometry_batch_size": 1,
                "pipeline_overlap": False,           # ← nombre correcto
                "persistent_geometry_pool": False,
                "ranges": smoke_ranges,
            }
        )
    return settings
def main() -> int:
    _prepare_runtime()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import argparse

    from pipeline.workflow import run_campaign

    parser = argparse.ArgumentParser(
        description="Ejecuta la campana Sobol GPU configurada al inicio de main.py."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Ejecuta un diseno y una seed para comprobar el flujo.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Sobrescribe temporalmente el numero de puntos Sobol "
            "sin editar la configuracion."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Limita las realizaciones por diseno. Usa 1 solo para "
            "medir rendimiento, no para resultados estadisticos."
        ),
    )
    parser.add_argument(
        "--geometry-backend",
        choices=("numba", "cupy", "auto"),
        default=None,
        help=(
            "Backend temporal para SAM. Produccion solapada: numba; "
            "generacion dedicada: cupy."
        ),
    )
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        help=(
            "Desactiva el solapamiento geometria/FFT. Recomendado al usar "
            "--geometry-backend cupy."
        ),
    )
    args = parser.parse_args()

    try:
        return run_campaign(
            user_settings(
                smoke=args.smoke,
                sobol_points=args.points,
                seeds_per_design=args.seeds,
                geometry_backend=args.geometry_backend,
                pipeline_overlap=False if args.no_overlap else None,
            )
        )
    except KeyboardInterrupt:
        print("\n[INTERRUMPIDO] Campana detenida de forma segura por el usuario.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
