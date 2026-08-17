#!/usr/bin/env python3
"""Run adaptive Sobol + full-rank POD with independent final FFT validation."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DEFAULT = SCRIPT_DIR / "campaign_config.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from env_bootstrap import ensure_configured_venv
from validation_reporting import empirical_coverage


ensure_configured_venv(CONFIG_DEFAULT)

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


ROOT = SCRIPT_DIR.parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cmame_campaign_common as common
import rom_reduced_operator as reduced
import rom_validation_utils as validate


DEFAULT_OUT_ROOT = ROOT / "results" / "cmame_method" / "interpretable_vf05_25_ar5_20"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


@contextmanager
def quiet_solver_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_identity(geometry_dir: Path) -> dict[str, Any]:
    phase_path = geometry_dir / "phase.npy"
    ori_path = geometry_dir / "ori.npy"
    generation_path = geometry_dir / "generation_result.json"
    if not phase_path.is_file() or not ori_path.is_file():
        raise FileNotFoundError(
            f"Geometry arrays are incomplete in {geometry_dir}; expected phase.npy and ori.npy."
        )
    if not generation_path.is_file():
        raise FileNotFoundError(f"Missing geometry manifest: {generation_path}")

    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    if not bool(generation.get("accepted_local", False)):
        raise RuntimeError(
            f"Geometry {geometry_dir.name} did not pass its generation tolerances."
        )

    identity = {
        "geometry_id": int(generation.get("geometry_id", -1)),
        "geometry_label": str(generation.get("geometry_label", geometry_dir.name)),
        "phase_sha256": sha256(phase_path),
        "ori_sha256": sha256(ori_path),
        "generation_manifest": str(generation_path),
    }
    for name in ("phase_sha256", "ori_sha256"):
        expected = str(generation.get(name, ""))
        if expected and expected != identity[name]:
            raise RuntimeError(
                f"{name} mismatch for {geometry_dir.name}: the arrays do not match "
                "generation_result.json. Regenerate the geometry before running the ROM."
            )
    return identity


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def default_run_name(geometry_id: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_geometry{int(geometry_id):02d}_adaptive_sobol_pod_{stamp}"


def material_sequence(candidates: pd.DataFrame, count: int) -> pd.DataFrame:
    sequence = candidates.iloc[: int(count)].copy().reset_index(drop=True)
    sequence.insert(0, "sobol_order", np.arange(1, len(sequence) + 1, dtype=int))
    return sequence


def independent_pool(
    count: int,
    seed: int,
    *,
    id_column: str,
    label_prefix: str,
) -> pd.DataFrame:
    if int(count) < 1:
        raise ValueError("Independent material-pool size must be positive.")
    frame = validate._build_independent_materials(int(count), int(seed)).copy()
    frame.insert(0, id_column, np.arange(len(frame), dtype=int))
    frame["material_id"] = frame[id_column].astype(int)
    frame["material_label"] = [
        f"{label_prefix}_{idx:04d}" for idx in frame[id_column].astype(int)
    ]
    return frame


def append_sobol_batch(
    *,
    run_dir: Path,
    geometry: common.GeometryData,
    runtime: dict[str, Any],
    candidates: pd.DataFrame,
    candidate_ids: list[int],
    seed: int,
    profile: str,
    quiet_solver: bool,
    basis: common.ContiguousBasis,
    voxel_order: np.ndarray,
    operator_phase: np.ndarray,
    operator_ori: np.ndarray,
    operators: dict[str, np.ndarray] | None,
    affine_stress_batch: Any,
    affine_stress_max_gib: float,
    memory_safety_fraction: float,
    basis_tolerance: float,
    cleanup_snapshot_fields: bool,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray] | None]:
    if not candidate_ids:
        return [], operators
    started = time.perf_counter()
    nvox = int(voxel_order.size)
    ordered_fields = np.empty(
        (6 * len(candidate_ids), 6, nvox),
        dtype=basis.dtype,
    )
    records: list[dict[str, Any]] = []
    for material_index, candidate_id in enumerate(candidate_ids):
        offset = 6 * material_index

        def consume_field(load_id: int, field: np.ndarray) -> None:
            np.take(
                field.reshape(6, nvox),
                voxel_order,
                axis=1,
                out=ordered_fields[offset + int(load_id)],
            )

        with quiet_solver_output(bool(quiet_solver)):
            record = common.ensure_snapshot(
                candidate_id=int(candidate_id),
                candidates=candidates,
                out_dir=run_dir,
                geometry=geometry,
                runtime=runtime,
                seed=int(seed),
                profile=str(profile),
                persistent_gpu_cache=False,
                solution_field_dtype=basis.dtype,
                solution_field_consumer=consume_field,
            )
        if cleanup_snapshot_fields:
            shutil.rmtree(common.snapshot_dir(run_dir, candidate_id), ignore_errors=True)
        records.append(record)

    rank_before = len(basis)
    basis_started = time.perf_counter()
    new_fields = basis.append_preordered(
        ordered_fields,
        tolerance=float(basis_tolerance),
    )
    basis_wall_s = float(time.perf_counter() - basis_started)
    del ordered_fields
    affine_q_block_size = 0
    if len(new_fields):
        fiber_fraction = float(np.count_nonzero(operator_phase)) / float(nvox)
        affine_q_block_size = common.runtime_affine_q_block_size(
            appended_fields_bytes=int(np.asarray(new_fields).nbytes),
            coefficient_count=len(
                getattr(
                    affine_stress_batch, "coefficient_names", reduced.COEFF_NAMES
                )
            ),
            memory_max_gib=float(affine_stress_max_gib),
            memory_safety_fraction=float(memory_safety_fraction),
            coefficient_supports=(
                (2, 1.0 - fiber_fraction),
                (5, fiber_fraction),
            ),
        )
    assembly_totals = {
        "assembly_wall_s": 0.0,
        "affine_stress_wall_s": 0.0,
        "ritz_contraction_wall_s": 0.0,
    }
    stress_workspace_peak_bytes = 0
    contraction_workspace_peak_bytes = 0
    full_volume_equivalent_passes = 0.0
    contraction_modes: set[str] = set()
    for start in range(0, len(new_fields), 6):
        end = min(start + 6, len(new_fields))
        absolute_end = rank_before + end
        operators, assembly = common._update_reduced_operators(
            phase=operator_phase,
            ori=operator_ori,
            basis=basis.active_fields[:absolute_end],
            existing=operators,
            new_fields=basis.active_fields[rank_before + start : absolute_end],
            affine_stress_batch=affine_stress_batch,
            affine_q_block_size=int(affine_q_block_size),
        )
        for name in assembly_totals:
            assembly_totals[name] += float(assembly.get(name, 0.0))
        stress_workspace_peak_bytes = max(
            stress_workspace_peak_bytes,
            int(assembly.get("stress_workspace_peak_bytes", 0)),
        )
        contraction_workspace_peak_bytes = max(
            contraction_workspace_peak_bytes,
            int(assembly.get("contraction_workspace_peak_bytes", 0)),
        )
        full_volume_equivalent_passes = max(
            full_volume_equivalent_passes,
            float(assembly.get("full_volume_equivalent_passes", 0.0)),
        )
        contraction_modes.add(str(assembly.get("contraction_mode", "unknown")))

    contraction_mode = "+".join(sorted(contraction_modes)) or "none"

    batch_wall_s = float(time.perf_counter() - started)
    material_count = len(records)
    solve_wall_s = sum(float(record.get("solve_wall_s", 0.0)) for record in records)
    non_solve_share = max(0.0, batch_wall_s - solve_wall_s) / material_count
    directions, remainder = divmod(int(len(new_fields)), material_count)
    cumulative_rank = rank_before
    for index, (candidate_id, record) in enumerate(zip(candidate_ids, records, strict=True)):
        added = directions + int(index < remainder)
        cumulative_rank += added
        record.update(
            {
                "candidate_id": int(candidate_id),
                "snapshot_profile": str(profile),
                "snapshot_step_wall_s": float(record.get("solve_wall_s", 0.0))
                + non_solve_share,
                "basis_update_wall_s": basis_wall_s / material_count,
                "new_directions": added,
                "basis_rank": cumulative_rank,
                "operator_assembly_wall_s": assembly_totals["assembly_wall_s"]
                / material_count,
                "affine_stress_wall_s": assembly_totals["affine_stress_wall_s"]
                / material_count,
                "ritz_contraction_wall_s": assembly_totals["ritz_contraction_wall_s"]
                / material_count,
                "operator_assembly_mode": "sequential_exact_prefix_incremental",
                "ritz_contraction_mode": contraction_mode,
                "basis_projection_backend": str(basis.last_projection_backend),
                "operator_stress_workspace_peak_bytes": stress_workspace_peak_bytes,
                "operator_contraction_workspace_peak_bytes": contraction_workspace_peak_bytes,
                "ritz_full_volume_equivalent_passes": full_volume_equivalent_passes,
                "affine_q_block_size": int(affine_q_block_size),
                "pod_batch_materials": material_count,
            }
        )
    return records, operators


def solve_truth_pool(
    *,
    run_dir: Path,
    geometry: common.GeometryData,
    runtime: dict[str, Any],
    materials: pd.DataFrame,
    seed: int,
    profile: str,
    quiet_solver: bool,
    pool_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    truth_dir = run_dir / f"{pool_name}_truth"
    for index, material in enumerate(materials.to_dict(orient="records"), start=1):
        material_id = int(material["material_id"])
        print(
            f"[SOBOL-POD] {pool_name} FFT {index}/{len(materials)} | "
            f"material={material_id}",
            flush=True,
        )
        with quiet_solver_output(bool(quiet_solver)):
            record = common.solve_material(
                material_row=material,
                material_dir=truth_dir / f"material_{material_id:04d}",
                geometry=geometry,
                runtime=runtime,
                profile=str(profile),
                seed=int(seed) + material_id,
                save_solution_fields=False,
                persistent_gpu_cache=False,
            )
        rows.append(record)
    return pd.DataFrame(rows)


def error_stats(
    frame: pd.DataFrame,
    *,
    id_column: str,
    threshold: float | None = None,
) -> dict[str, Any]:
    errors = frame["relative_frobenius_error"].to_numpy(dtype=float)
    worst = int(np.argmax(errors))
    stats = {
        "count": int(len(frame)),
        "error_mean": float(np.mean(errors)),
        "error_median": float(np.median(errors)),
        "error_p95": float(np.quantile(errors, 0.95)),
        "error_max": float(np.max(errors)),
        f"worst_{id_column}": int(frame.iloc[worst][id_column]),
        "rom_online_mean_s": float(frame["rom_online_s"].mean()),
        "rom_online_p95_s": float(frame["rom_online_s"].quantile(0.95)),
    }
    if threshold is not None:
        stats.update(empirical_coverage(errors, threshold))
    return stats


def candidate_rom_cloud(
    *,
    count: int,
    seed: int,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    chunk_size: int,
    memory_max_gib: float,
    backend: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = validate._build_independent_materials(int(count), int(seed)).copy()
    candidates.insert(0, "query_id", np.arange(len(candidates), dtype=int))
    parameters = candidates[list(reduced.MATERIAL_PARAMETER_COLUMNS)].to_numpy(
        dtype=np.float64
    )
    properties = np.empty(
        (len(candidates), len(reduced.ENGINEERING_COLUMNS)), dtype=np.float64
    )
    minimum_eigenvalues = np.empty(len(candidates), dtype=np.float64)
    per_query_times = np.empty(len(candidates), dtype=np.float64)
    online_wall_s = 0.0
    memory_plan = common.rom_chunk_memory_plan(
        rank=int(Kq.shape[1]),
        requested_chunk_size=int(chunk_size),
        memory_max_gib=float(memory_max_gib),
    )
    effective_chunk = int(memory_plan["effective_chunk_size"])
    requested_backend = str(backend).lower()
    if requested_backend not in {"auto", "cpu", "gpu"}:
        raise ValueError("ROM backend must be auto, cpu, or gpu.")
    actual_backend = "cpu"
    gpu_evaluator = None
    gpu_setup_error = None
    if requested_backend in {"auto", "gpu"}:
        try:
            gpu_evaluator = reduced.GpuAffineBatchEvaluator(Kq, Bq, Dq)
            actual_backend = "gpu"
        except Exception as exc:
            gpu_setup_error = f"{type(exc).__name__}: {exc}"
            if requested_backend == "gpu":
                raise RuntimeError(f"GPU ROM backend initialization failed: {exc}") from exc
    for start in range(0, len(candidates), effective_chunk):
        stop = min(start + effective_chunk, len(candidates))
        coefficients = reduced._material_coefficients_batch(
            parameters[start:stop]
        )
        if gpu_evaluator is not None:
            C_batch, batch_wall_s = gpu_evaluator.evaluate(coefficients)
        else:
            C_batch, _, batch_wall_s = reduced._rom_ceff_batch(
                coefficients, Kq, Bq, Dq
            )
        online_wall_s += float(batch_wall_s)
        eigenvalues = np.linalg.eigvalsh(C_batch)
        properties[start:stop] = reduced._engineering_constants_batch(C_batch)
        minimum_eigenvalues[start:stop] = eigenvalues[:, 0]
        per_query_times[start:stop] = float(batch_wall_s) / max(stop - start, 1)
    summary = {
        **memory_plan,
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "gpu_operator_transfer_s": float(
            gpu_evaluator.operator_transfer_wall_s if gpu_evaluator is not None else 0.0
        ),
        "gpu_setup_error": gpu_setup_error,
        "query_count": int(len(candidates)),
        "rom_online_total_s": float(online_wall_s),
        "rom_online_mean_s": float(online_wall_s) / max(len(candidates), 1),
        "rom_online_queries_per_s": float(len(candidates)) / max(
            float(online_wall_s), np.finfo(float).tiny
        ),
    }
    result = candidates.copy()
    result["rom_online_s"] = per_query_times
    result["rom_min_eig"] = minimum_eigenvalues
    for index, name in enumerate(reduced.ENGINEERING_COLUMNS):
        result[f"rom_{name}"] = properties[:, index]
    return result, summary


def write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path) as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name[:31], index=False)
    except Exception as exc:  # pragma: no cover
        print(f"[SOBOL-POD] Excel skipped: {exc}", flush=True)


def write_plot(curve: pd.DataFrame, path: Path, target_error: float) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        x = curve["training_materials"].to_numpy(dtype=int)
        ax.semilogy(x, curve["monitor_error_mean"], marker="o", label="mean")
        ax.semilogy(x, curve["monitor_error_p95"], marker="s", label="p95")
        ax.semilogy(x, curve["monitor_error_max"], marker="^", label="maximum")
        ax.axhline(float(target_error), color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Sobol training materials used for full-rank POD")
        ax.set_ylabel("relative Frobenius error on the FFT monitor set")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"[SOBOL-POD] Plot skipped: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    config = read_config(CONFIG_DEFAULT)
    pipeline = config["sobol_pod_pipeline"]
    generation = config["geometry_generation"]
    paths = config["paths"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--geometry-id", type=int, default=0)
    parser.add_argument("--geometry-dir", type=Path, default=None)
    parser.add_argument("--candidate-pool", type=Path, default=None)
    parser.add_argument("--candidate-count", type=int, default=int(pipeline.get("candidate_count", 1024)))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start-materials", type=int, default=int(pipeline.get("start_materials", 2)))
    parser.add_argument(
        "--training-limit",
        type=int,
        default=int(pipeline.get("training_limit", 0) or 0),
        help="Optional operational cap; zero uses the memory-safe candidate limit.",
    )
    parser.add_argument("--monitor-count", type=int, default=int(pipeline.get("monitor_count", 16)))
    parser.add_argument("--final-validation-count", type=int, default=int(pipeline.get("final_validation_count", 16)))
    parser.add_argument("--candidate-seed", type=int, default=int(pipeline.get("candidate_seed", 20260821)))
    parser.add_argument("--monitor-seed", type=int, default=int(pipeline.get("monitor_seed", 20260822)))
    parser.add_argument("--final-validation-seed", type=int, default=int(pipeline.get("final_validation_seed", 20260824)))
    parser.add_argument("--rom-timing-count", type=int, default=int(pipeline.get("rom_timing_count", 10000)))
    parser.add_argument("--rom-timing-seed", type=int, default=int(pipeline.get("rom_timing_seed", 20260823)))
    parser.add_argument("--rom-chunk-size", type=int, default=int(pipeline.get("rom_chunk_size", 2048)))
    parser.add_argument("--basis-profile", choices=tuple(common.SOLVER_PROFILES), default=str(pipeline.get("basis_profile", "snapshot")))
    parser.add_argument("--truth-profile", choices=tuple(common.SOLVER_PROFILES), default=str(pipeline.get("truth_profile", "snapshot")))
    parser.add_argument("--target-error", type=float, default=float(pipeline.get("target_error", 1.0e-4)))
    parser.add_argument("--basis-tolerance", type=float, default=float(pipeline.get("basis_tolerance", 1.0e-12)))
    parser.add_argument(
        "--basis-dtype",
        choices=("float32", "float64"),
        default=str(pipeline.get("basis_dtype", "float32")),
    )
    parser.add_argument(
        "--pod-batch-max-gib",
        type=float,
        default=float(pipeline.get("pod_batch_max_gib", 8.0)),
    )
    parser.add_argument(
        "--affine-stress-max-gib",
        type=float,
        default=float(pipeline.get("affine_stress_max_gib", 8.0)),
    )
    parser.add_argument(
        "--rom-batch-max-gib",
        type=float,
        default=float(pipeline.get("rom_batch_max_gib", 1.0)),
    )
    parser.add_argument(
        "--rom-backend",
        choices=("auto", "cpu", "gpu"),
        default=str(pipeline.get("rom_backend", "gpu")),
    )
    parser.add_argument(
        "--memory-safety-fraction",
        type=float,
        default=float(pipeline.get("memory_safety_fraction", 0.8)),
    )
    parser.add_argument(
        "--blas-threads",
        default=str(pipeline.get("blas_threads", "auto")),
        help="Use 'auto' for the physical cores available to this process.",
    )
    parser.add_argument(
        "--fft-backend",
        choices=("cpu", "gpu"),
        default=str(pipeline.get("fft_backend", "gpu")),
    )
    parser.add_argument("--geometry-backend", choices=("numba", "cupy", "auto"), default=str(generation["geometry_backend"]))
    parser.add_argument(
        "--generator-cores",
        default=str(generation.get("generator_cores", "auto")),
    )
    parser.add_argument("--venv-path", type=Path, default=project_path(paths["venv_path"]))
    parser.add_argument("--save-operators", action=argparse.BooleanOptionalAction, default=bool(pipeline.get("save_operators", True)))
    parser.add_argument("--cleanup-snapshot-fields", action=argparse.BooleanOptionalAction, default=bool(pipeline.get("cleanup_snapshot_fields", True)))
    parser.add_argument("--quiet-solver", action=argparse.BooleanOptionalAction, default=bool(pipeline.get("quiet_solver", True)))
    parser.add_argument("--write-plot", action=argparse.BooleanOptionalAction, default=bool(pipeline.get("write_per_geometry_plots", False)))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    common.prepare_runtime(
        Path(args.venv_path),
        Path(__file__),
        marker_name="CMAME_SOBOL_POD_CUDA_READY",
    )
    cpu_info = common.cpu_resource_info()
    blas_threads = common.resolve_cpu_workers(args.blas_threads, resource_info=cpu_info)
    generator_cores = common.resolve_cpu_workers(
        args.generator_cores,
        resource_info=cpu_info,
    )
    blas_controller, blas_info = common.configure_blas_threads(blas_threads)
    pipeline_started = time.perf_counter()
    try:
        if int(args.start_materials) < 1:
            raise ValueError("start_materials must be positive.")
        if not np.isfinite(float(args.target_error)) or float(args.target_error) <= 0.0:
            raise ValueError("target_error must be finite and positive.")
        if len({int(args.candidate_seed), int(args.monitor_seed), int(args.final_validation_seed)}) != 3:
            raise ValueError(
                "candidate_seed, monitor_seed, and final_validation_seed must be distinct."
            )

        out_root = Path(args.out_root).resolve()
        geometry_dir = (
            Path(args.geometry_dir).resolve()
            if args.geometry_dir is not None
            else out_root / "geometries" / f"geometry_{int(args.geometry_id):02d}"
        )
        if not geometry_dir.is_dir():
            raise FileNotFoundError(f"Geometry directory not found: {geometry_dir}")

        geometry_info = geometry_identity(geometry_dir)
        if int(geometry_info["geometry_id"]) != int(args.geometry_id):
            raise RuntimeError(
                f"Requested geometry {int(args.geometry_id):02d}, but {geometry_dir} "
                f"contains geometry {int(geometry_info['geometry_id']):02d}."
            )
        geometry_started = time.perf_counter()
        geometry = common.load_fixed_geometry(geometry_dir)
        geometry_load_wall_s = float(time.perf_counter() - geometry_started)

        run_dir = out_root / "runs" / str(args.run_name or default_run_name(args.geometry_id))
        run_dir.mkdir(parents=True, exist_ok=bool(args.overwrite))

        candidate_pool = Path(args.candidate_pool).resolve() if args.candidate_pool else None
        if candidate_pool is not None and candidate_pool.is_file():
            candidates = pd.read_csv(candidate_pool)
            source = str(candidate_pool)
            pool_hash = sha256(candidate_pool)
        else:
            candidates = validate._build_independent_materials(
                int(args.candidate_count), int(args.candidate_seed)
            ).copy()
            candidates.insert(0, "candidate_id", np.arange(len(candidates), dtype=int))
            source = "generated_from_candidate_seed"
            pool_hash = ""
        if "candidate_id" not in candidates:
            candidates.insert(0, "candidate_id", np.arange(len(candidates), dtype=int))
        if candidates["candidate_id"].duplicated().any():
            raise ValueError("candidate_id values must be unique.")
        candidates["material_id"] = candidates["candidate_id"].astype(int)
        candidates["material_label"] = [
            f"candidate_sobol_{idx:05d}" for idx in candidates["candidate_id"].astype(int)
        ]
        if len(candidates) < 1:
            raise ValueError("Candidate pool is empty.")

        provisional_plan = common.full_rank_memory_plan(
            nvox=int(geometry.phase.size),
            max_rank=6 * len(candidates),
            basis_dtype=str(args.basis_dtype),
            pod_batch_max_gib=float(args.pod_batch_max_gib),
            affine_stress_max_gib=float(args.affine_stress_max_gib),
            memory_safety_fraction=float(args.memory_safety_fraction),
            max_material_batch=1,
        )
        safe_material_limit = min(
            len(candidates), int(provisional_plan["max_safe_rank"]) // 6
        )
        requested_limit = int(args.training_limit)
        if requested_limit < 0:
            raise ValueError("training_limit cannot be negative.")
        if requested_limit and requested_limit > safe_material_limit:
            raise MemoryError(
                f"training_limit={requested_limit} exceeds the exact full-rank "
                f"memory-safe limit of {safe_material_limit} materials."
            )
        training_limit = requested_limit or safe_material_limit
        minimum_required = int(args.start_materials)
        if training_limit < minimum_required:
            raise MemoryError(
                f"Only {training_limit} materials fit, but at least {minimum_required} "
                "are required to evaluate the stopping rule."
            )
        memory_plan = common.full_rank_memory_plan(
            nvox=int(geometry.phase.size),
            max_rank=6 * training_limit,
            basis_dtype=str(args.basis_dtype),
            pod_batch_max_gib=float(args.pod_batch_max_gib),
            affine_stress_max_gib=float(args.affine_stress_max_gib),
            memory_safety_fraction=float(args.memory_safety_fraction),
            max_material_batch=1,
            available_bytes=int(provisional_plan["available_memory_bytes"]),
        )
        if int(memory_plan["estimated_peak_bytes"]) > int(memory_plan["safe_memory_bytes"]):
            raise MemoryError("The derived exact full-rank training limit is not memory-safe.")
        candidates.to_csv(run_dir / "candidate_pool_used.csv", index=False)
        if not pool_hash:
            pool_hash = sha256(run_dir / "candidate_pool_used.csv")

        sequence = material_sequence(candidates, training_limit)
        sequence.to_csv(run_dir / "planned_sobol_sequence.csv", index=False)
        monitor = independent_pool(
            int(args.monitor_count),
            int(args.monitor_seed),
            id_column="monitor_id",
            label_prefix="monitor_sobol",
        )
        monitor.to_csv(run_dir / "monitor_pool.csv", index=False)

        runtime = common.configure_runtime(
            geometry_backend=str(args.geometry_backend),
            generator_cores=generator_cores,
            solver_tol=float(common.SOLVER_PROFILES[str(args.basis_profile)]["solver_rtol"]),
            fft_backend=str(args.fft_backend),
        )
        write_json(
            run_dir / "run_manifest.json",
            {
                "method": "adaptive_sobol_pod_full_rank",
                "run_dir": str(run_dir),
                "geometry_dir": str(geometry_dir),
                **geometry_info,
                "candidate_pool": source,
                "candidate_pool_sha256": pool_hash,
                "selection_policy": "sequential_sobol_prefix",
                "training_limit_policy": "memory_safe_candidate_limit"
                if requested_limit == 0
                else "explicit_operational_limit",
                "training_limit": int(training_limit),
                "memory_safe_material_limit": int(safe_material_limit),
                "monitoring_policy": "fixed_independent_fft_pool_after_each_material",
                "monitor_count": int(len(monitor)),
                "monitor_seed": int(args.monitor_seed),
                "monitor_start_materials": int(args.start_materials),
                "final_validation_policy": "new_independent_fft_pool_after_freeze",
                "final_validation_count": int(args.final_validation_count),
                "final_validation_seed": int(args.final_validation_seed),
                "rom_timing_count": int(args.rom_timing_count),
                "basis_profile": str(args.basis_profile),
                "truth_profile": str(args.truth_profile),
                "fft_backend": str(args.fft_backend),
                "target_error": float(args.target_error),
                "basis_tolerance": float(args.basis_tolerance),
                "basis_dtype": str(args.basis_dtype),
                "basis_storage": "preallocated_contiguous",
                "basis_projection_passes": 1,
                "basis_projection_backend": "scipy_blas_gemm_in_place",
                "snapshot_field_transport": f"in_memory_{np.dtype(args.basis_dtype).name}",
                "pod_batching": "one_material_exact_prefix",
                "pod_batch_max_gib": float(args.pod_batch_max_gib),
                "pod_batch_material_limit": 1,
                "affine_stress_max_gib": float(args.affine_stress_max_gib),
                "affine_q_block_size_policy": "runtime_memory_bounded",
                "affine_q_block_size_at_max_rank": int(
                    memory_plan["affine_q_block_size"]
                ),
                "rom_batch_max_gib": float(args.rom_batch_max_gib),
                "rom_backend": str(args.rom_backend),
                "memory_plan": memory_plan,
                "basis_voxel_order": "matrix_then_fiber_by_orientation",
                "basis_component_layout": "mandel_component_then_voxel",
                "affine_orientation_kernel": "grouped_blocks_with_voxelwise_fallback",
                "ritz_contraction_kernel": "exact_phase_supported_component_batched",
                "blas_thread_policy": str(args.blas_threads),
                "blas_threads": blas_threads,
                "generator_core_policy": str(args.generator_cores),
                "generator_cores": generator_cores,
                "cpu_resources": cpu_info,
                "blas_info": blas_info,
            },
        )

        voxel_order = reduced.phase_orientation_voxel_order(geometry.phase, geometry.ori)
        operator_phase = geometry.phase.reshape(-1)[voxel_order]
        operator_ori = geometry.ori.reshape(-1, 3)[voxel_order]
        basis = common.ContiguousBasis(
            capacity=6 * training_limit,
            field_shape=(6, int(geometry.phase.size)),
            dtype=str(args.basis_dtype),
            projection_row_block_size=int(
                memory_plan["pod_projection_row_block_size"]
            ),
        )
        operators: dict[str, np.ndarray] | None = None
        affine_started = time.perf_counter()
        affine = reduced.affine_stress_batch_factory(
            operator_phase,
            operator_ori,
        )
        affine_setup_wall_s = float(time.perf_counter() - affine_started)

        monitor_fft_started = time.perf_counter()
        monitor_truth = solve_truth_pool(
            run_dir=run_dir,
            geometry=geometry,
            runtime=runtime,
            materials=monitor,
            seed=int(args.monitor_seed),
            profile=str(args.truth_profile),
            quiet_solver=bool(args.quiet_solver),
            pool_name="monitor",
        )
        monitor_truth.to_csv(run_dir / "monitor_truth_results.csv", index=False)
        monitor_fft_stage_wall_s = float(time.perf_counter() - monitor_fft_started)

        snapshot_rows: list[dict[str, Any]] = []
        curve_rows: list[dict[str, Any]] = []
        monitor_frames: list[pd.DataFrame] = []
        candidate_ids = sequence["candidate_id"].astype(int).tolist()
        stop_materials: int | None = None
        last_monitor_stats: dict[str, Any] | None = None
        monitor_rom_total_wall_s = 0.0
        cumulative = {
            "solve_wall_s": 0.0,
            "snapshot_step_wall_s": 0.0,
            "basis_update_wall_s": 0.0,
            "operator_assembly_wall_s": 0.0,
            "affine_stress_wall_s": 0.0,
            "ritz_contraction_wall_s": 0.0,
        }

        training_started = time.perf_counter()
        for training_materials, candidate_id in enumerate(candidate_ids, start=1):
            records, operators = append_sobol_batch(
                run_dir=run_dir,
                geometry=geometry,
                runtime=runtime,
                candidates=candidates,
                candidate_ids=[candidate_id],
                seed=int(args.candidate_seed),
                profile=str(args.basis_profile),
                quiet_solver=bool(args.quiet_solver),
                basis=basis,
                voxel_order=voxel_order,
                operator_phase=operator_phase,
                operator_ori=operator_ori,
                operators=operators,
                affine_stress_batch=affine,
                affine_stress_max_gib=float(args.affine_stress_max_gib),
                memory_safety_fraction=float(args.memory_safety_fraction),
                basis_tolerance=float(args.basis_tolerance),
                cleanup_snapshot_fields=bool(args.cleanup_snapshot_fields),
            )
            snapshot_rows.extend(records)
            for record in records:
                for name in cumulative:
                    cumulative[name] += float(record[name])
            pd.DataFrame(snapshot_rows).to_csv(run_dir / "snapshot_timing.csv", index=False)
            for record in records:
                print(
                    f"[SOBOL-POD] snapshot candidate={int(record['candidate_id'])} | "
                    f"rank={int(record['basis_rank'])} | "
                    f"step={float(record['snapshot_step_wall_s']):.2f}s",
                    flush=True,
                )
            if operators is None:
                raise RuntimeError("No reduced operators were assembled.")
            if training_materials < int(args.start_materials):
                continue

            monitor_rom_started = time.perf_counter()
            frame = reduced._evaluate_rom(
                results_df=monitor_truth,
                Kq=operators["Kq"],
                Bq=operators["Bq"],
                Dq=operators["Dq"],
            )
            monitor_rom_wall_s = float(time.perf_counter() - monitor_rom_started)
            monitor_rom_total_wall_s += monitor_rom_wall_s
            frame.insert(0, "monitor_id", monitor["monitor_id"].to_numpy(dtype=int))
            frame.insert(1, "training_materials", int(training_materials))
            frame.insert(2, "pod_rank", int(len(basis)))
            stats = error_stats(frame, id_column="monitor_id")
            passes = bool(stats["error_max"] <= float(args.target_error))
            last_monitor_stats = stats
            monitor_frames.append(frame)
            curve_rows.append(
                {
                    "method": "adaptive_sobol_pod_full_rank",
                    "training_materials": int(training_materials),
                    "pod_rank": int(len(basis)),
                    "snapshot_solve_wall_s": cumulative["solve_wall_s"],
                    "snapshot_step_wall_s": cumulative["snapshot_step_wall_s"],
                    "basis_update_wall_s": cumulative["basis_update_wall_s"],
                    "operator_assembly_wall_s": cumulative["operator_assembly_wall_s"],
                    "affine_stress_wall_s": cumulative["affine_stress_wall_s"],
                    "ritz_contraction_wall_s": cumulative["ritz_contraction_wall_s"],
                    "monitor_rom_cumulative_wall_s": float(monitor_rom_total_wall_s),
                    "monitor_error_mean": stats["error_mean"],
                    "monitor_error_median": stats["error_median"],
                    "monitor_error_p95": stats["error_p95"],
                    "monitor_error_max": stats["error_max"],
                    "passes_target_max": passes,
                    "stop_triggered": passes,
                    "rom_online_mean_s": stats["rom_online_mean_s"],
                    "worst_monitor_id": stats["worst_monitor_id"],
                }
            )
            pd.DataFrame(curve_rows).to_csv(
                run_dir / "sobol_pod_error_curve.csv", index=False
            )
            pd.concat(monitor_frames, ignore_index=True).to_csv(
                run_dir / "monitor_rom_results.csv", index=False
            )
            print(
                f"[SOBOL-POD] monitor materials={training_materials} | "
                f"rank={len(basis)} | error_max={stats['error_max']:.3e} | "
                f"target={float(args.target_error):.1e}",
                flush=True,
            )
            if passes:
                stop_materials = int(training_materials)
                break
        training_stage_wall_s = float(time.perf_counter() - training_started)
        final_basis_rank = int(len(basis))

        curve = pd.DataFrame(curve_rows)
        monitor_rom = pd.concat(monitor_frames, ignore_index=True)
        curve.to_csv(run_dir / "sobol_pod_error_curve.csv", index=False)
        monitor_rom.to_csv(run_dir / "monitor_rom_results.csv", index=False)
        if stop_materials is None:
            selected = sequence.iloc[: len(snapshot_rows)].copy()
            selected.to_csv(run_dir / "selected_sobol_sequence.csv", index=False)
            write_json(
                run_dir / "adaptive_incomplete_summary.json",
                {
                    "status": "stopping_not_achieved",
                    "training_materials": int(len(snapshot_rows)),
                    "training_limit": int(training_limit),
                    "target_error": float(args.target_error),
                    "last_monitor_summary": last_monitor_stats,
                },
            )
            del basis
            gc.collect()
            raise RuntimeError(
                "Adaptive stopping was not achieved before the memory-safe or "
                f"operational limit of {training_limit} materials."
            )

        selected = sequence.iloc[:stop_materials].copy()
        selected.to_csv(run_dir / "selected_sobol_sequence.csv", index=False)
        if bool(args.save_operators) and operators is not None:
            np.savez_compressed(
                run_dir / "reduced_operators.npz",
                Kq=operators["Kq"],
                Bq=operators["Bq"],
                Dq=operators["Dq"],
                coefficient_names=np.asarray(reduced.COEFF_NAMES),
                candidate_ids=selected["candidate_id"].to_numpy(dtype=np.int64),
            )

        del basis
        gc.collect()

        final_validation = independent_pool(
            int(args.final_validation_count),
            int(args.final_validation_seed),
            id_column="final_validation_id",
            label_prefix="final_validation_sobol",
        )
        final_validation.to_csv(run_dir / "final_validation_pool.csv", index=False)
        validation_started = time.perf_counter()
        final_validation_truth = solve_truth_pool(
            run_dir=run_dir,
            geometry=geometry,
            runtime=runtime,
            materials=final_validation,
            seed=int(args.final_validation_seed),
            profile=str(args.truth_profile),
            quiet_solver=bool(args.quiet_solver),
            pool_name="final_validation",
        )
        final_validation_truth.to_csv(
            run_dir / "final_validation_truth_results.csv", index=False
        )
        final_validation_rom = reduced._evaluate_rom(
            results_df=final_validation_truth,
            Kq=operators["Kq"],
            Bq=operators["Bq"],
            Dq=operators["Dq"],
        )
        final_validation_rom.insert(
            0,
            "final_validation_id",
            final_validation["final_validation_id"].to_numpy(dtype=int),
        )
        final_validation_rom.insert(1, "training_materials", int(stop_materials))
        final_validation_rom.insert(2, "pod_rank", int(final_basis_rank))
        final_validation_rom["relative_frobenius_error_percent"] = (
            100.0 * final_validation_rom["relative_frobenius_error"]
        )
        final_validation_rom["below_target"] = (
            final_validation_rom["relative_frobenius_error"]
            <= float(args.target_error)
        )
        final_validation_rom.to_csv(
            run_dir / "final_validation_rom_results.csv", index=False
        )
        final_stats = error_stats(
            final_validation_rom,
            id_column="final_validation_id",
            threshold=float(args.target_error),
        )
        validation_stage_wall_s = float(time.perf_counter() - validation_started)

        rom_timing_started = time.perf_counter()
        rom_cloud, rom_timing = candidate_rom_cloud(
            count=int(args.rom_timing_count),
            seed=int(args.rom_timing_seed),
            Kq=operators["Kq"],
            Bq=operators["Bq"],
            Dq=operators["Dq"],
            chunk_size=int(args.rom_chunk_size),
            memory_max_gib=float(args.rom_batch_max_gib),
            backend=str(args.rom_backend),
        )
        rom_cloud.to_csv(run_dir / "candidate_rom_cloud.csv", index=False)
        write_json(run_dir / "rom_timing_summary.json", rom_timing)
        rom_timing_stage_wall_s = float(time.perf_counter() - rom_timing_started)

        plot_path = run_dir / "sobol_pod_error_curve.png"
        if bool(args.write_plot):
            write_plot(curve, plot_path, float(args.target_error))
        timing = curve[
            [
                "training_materials",
                "pod_rank",
                "snapshot_solve_wall_s",
                "snapshot_step_wall_s",
                "basis_update_wall_s",
                "operator_assembly_wall_s",
                "affine_stress_wall_s",
                "ritz_contraction_wall_s",
                "monitor_rom_cumulative_wall_s",
                "rom_online_mean_s",
            ]
        ].copy()
        timing["monitor_fft_total_wall_s"] = float(monitor_truth["solve_wall_s"].sum())
        timing["final_validation_fft_total_wall_s"] = float(
            final_validation_truth["solve_wall_s"].sum()
        )
        snapshot_timing = pd.DataFrame(snapshot_rows)
        active_q_blocks = snapshot_timing.loc[
            snapshot_timing["affine_q_block_size"] > 0,
            "affine_q_block_size",
        ]
        stage_timing = pd.DataFrame(
            [
                {"stage": "geometry_load", "wall_s": geometry_load_wall_s},
                {"stage": "affine_setup", "wall_s": affine_setup_wall_s},
                {"stage": "monitor_fft_once", "wall_s": monitor_fft_stage_wall_s},
                {"stage": "adaptive_sobol_fft_pod_kbd", "wall_s": training_stage_wall_s},
                {"stage": "final_independent_fft_validation", "wall_s": validation_stage_wall_s},
                {"stage": "rom_timing_queries", "wall_s": rom_timing_stage_wall_s},
            ]
        )
        write_excel(
            run_dir / "sobol_pod_paper_tables.xlsx",
            {
                "monitor_curve": curve,
                "timing": timing,
                "stage_timing": stage_timing,
                "sequence": selected,
                "monitor_truth": monitor_truth,
                "monitor_rom": monitor_rom,
                "final_validation_truth": final_validation_truth,
                "final_validation_rom": final_validation_rom,
            },
        )

        summary = {
            "run_dir": str(run_dir),
            "status": "complete",
            "method": "adaptive_sobol_pod_full_rank",
            **geometry_info,
            "selection_policy": "sequential_sobol_prefix",
            "training_limit": int(training_limit),
            "memory_safe_material_limit": int(safe_material_limit),
            "final_selected_materials": int(stop_materials),
            "basis_rank": final_basis_rank,
            "basis_dtype": str(args.basis_dtype),
            "snapshot_field_transport": f"in_memory_{np.dtype(args.basis_dtype).name}",
            "pod_batch_max_gib": float(args.pod_batch_max_gib),
            "pod_batch_material_limit": 1,
            "affine_stress_max_gib": float(args.affine_stress_max_gib),
            "affine_q_block_size_min": int(active_q_blocks.min()),
            "affine_q_block_size_max": int(
                snapshot_timing["affine_q_block_size"].max()
            ),
            "rom_batch_max_gib": float(args.rom_batch_max_gib),
            "rom_backend": str(rom_timing["actual_backend"]),
            "memory_plan": memory_plan,
            "monitor_count": int(len(monitor)),
            "monitor_seed": int(args.monitor_seed),
            "monitor_start_materials": int(args.start_materials),
            "stop_materials": int(stop_materials),
            "monitor_summary_at_stop": last_monitor_stats,
            "final_validation_count": int(len(final_validation)),
            "final_validation_seed": int(args.final_validation_seed),
            "final_validation_passes_target": bool(
                final_stats["error_max"] <= float(args.target_error)
            ),
            "final_validation_below_target_count": int(
                final_stats["below_target_count"]
            ),
            "final_validation_above_target_count": int(
                final_stats["above_target_count"]
            ),
            "final_validation_below_target_percent": float(
                final_stats["below_target_percent"]
            ),
            "target_error": float(args.target_error),
            "snapshot_total_solve_wall_s": float(snapshot_timing["solve_wall_s"].sum()),
            "snapshot_total_step_wall_s": float(snapshot_timing["snapshot_step_wall_s"].sum()),
            "basis_update_total_wall_s": float(snapshot_timing["basis_update_wall_s"].sum()),
            "operator_assembly_total_wall_s": float(snapshot_timing["operator_assembly_wall_s"].sum()),
            "operator_stress_workspace_peak_bytes": int(
                snapshot_timing["operator_stress_workspace_peak_bytes"].max()
            ),
            "operator_contraction_workspace_peak_bytes": int(
                snapshot_timing[
                    "operator_contraction_workspace_peak_bytes"
                ].max()
            ),
            "ritz_full_volume_equivalent_passes": float(
                snapshot_timing["ritz_full_volume_equivalent_passes"].max()
            ),
            "affine_stress_total_wall_s": float(snapshot_timing["affine_stress_wall_s"].sum()),
            "ritz_contraction_total_wall_s": float(snapshot_timing["ritz_contraction_wall_s"].sum()),
            "geometry_load_wall_s": geometry_load_wall_s,
            "affine_setup_wall_s": affine_setup_wall_s,
            "monitor_fft_total_wall_s": float(monitor_truth["solve_wall_s"].sum()),
            "monitor_fft_stage_wall_s": monitor_fft_stage_wall_s,
            "monitor_rom_total_wall_s": monitor_rom_total_wall_s,
            "training_stage_wall_s": training_stage_wall_s,
            "final_validation_fft_total_wall_s": float(
                final_validation_truth["solve_wall_s"].sum()
            ),
            "final_validation_stage_wall_s": validation_stage_wall_s,
            "rom_timing_stage_wall_s": rom_timing_stage_wall_s,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "pipeline_wall_s_before_final_write": float(time.perf_counter() - pipeline_started),
            "final_validation_summary": final_stats,
            "fft_backend": str(args.fft_backend),
            "rom_timing": rom_timing,
            "curve_csv": str(run_dir / "sobol_pod_error_curve.csv"),
            "monitor_truth_csv": str(run_dir / "monitor_truth_results.csv"),
            "monitor_rom_csv": str(run_dir / "monitor_rom_results.csv"),
            "final_validation_truth_csv": str(
                run_dir / "final_validation_truth_results.csv"
            ),
            "final_validation_rom_csv": str(
                run_dir / "final_validation_rom_results.csv"
            ),
            "rom_cloud_csv": str(run_dir / "candidate_rom_cloud.csv"),
        }
        write_json(run_dir / "sobol_pod_summary.json", summary)
        if bool(args.cleanup_snapshot_fields):
            shutil.rmtree(run_dir / "snapshot_cache", ignore_errors=True)
        print(
            f"[SOBOL-POD] done | run={run_dir} | "
            f"materials={stop_materials} | final_error_max={final_stats['error_max']:.3e} | "
            f"ROM{int(args.rom_timing_count)}={rom_timing['rom_online_total_s']:.3f}s",
            flush=True,
        )
    finally:
        blas_controller.restore_original_limits()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
