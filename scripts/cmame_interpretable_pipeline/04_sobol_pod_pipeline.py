#!/usr/bin/env python3
"""Run fixed or explicitly adaptive Sobol + full-rank POD validation."""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DEFAULT = SCRIPT_DIR / "campaign_config.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from env_bootstrap import ensure_configured_venv
from validation_reporting import empirical_coverage


if __name__ == "__main__":
    ensure_configured_venv(CONFIG_DEFAULT)

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
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
    return f"run_geometry{int(geometry_id):02d}_sobol_pod_{stamp}"


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


def resolve_training_protocol(
    adaptive: bool, monitor_count: int, training_limit: int
) -> str:
    """Validate and resolve the explicit fixed/adaptive FOM protocol."""
    monitors = int(monitor_count)
    limit = int(training_limit)
    if monitors < 0:
        raise ValueError("monitor_count cannot be negative.")
    if limit < 0:
        raise ValueError("training_limit cannot be negative.")
    if not bool(adaptive):
        if limit == 0:
            raise ValueError(
                "Fixed training requires an explicit positive training_limit."
            )
        return "fixed"
    if monitors == 0:
        raise ValueError("Adaptive training requires a positive monitor_count.")
    return "adaptive"


def fixed_warm_start_route(sequence: pd.DataFrame) -> pd.DataFrame:
    """Order a fixed Sobol set by affine proximity without changing the set."""
    if len(sequence) < 2:
        routed = sequence.copy()
        routed.insert(0, "sobol_set_position", np.arange(len(routed), dtype=int))
        routed.insert(1, "solve_position", np.arange(len(routed), dtype=int))
        return routed

    records = sequence.to_dict(orient="records")
    gamma = np.stack([reduced._material_coefficients(row) for row in records])
    lower = np.min(gamma, axis=0)
    span = np.maximum(np.max(gamma, axis=0) - lower, np.finfo(float).eps)
    normalized = (gamma - lower) / span

    remaining = set(range(len(sequence)))
    current = min(remaining, key=lambda idx: float(np.linalg.norm(normalized[idx] - 0.5)))
    route = [current]
    remaining.remove(current)
    while remaining:
        current = min(
            remaining,
            key=lambda idx: float(np.linalg.norm(normalized[idx] - normalized[current])),
        )
        route.append(current)
        remaining.remove(current)

    routed = sequence.iloc[route].copy().reset_index(drop=True)
    routed.insert(0, "sobol_set_position", np.asarray(route, dtype=int))
    routed.insert(1, "solve_position", np.arange(len(routed), dtype=int))
    return routed


def affine_maximin_sequence(candidates: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select a deterministic space-filling subset in affine operator space."""
    requested = int(count)
    if requested < 1 or requested > len(candidates):
        raise ValueError("maximin count must be between one and the pool size")
    records = candidates.to_dict(orient="records")
    gamma = np.stack([reduced._material_coefficients(row) for row in records])
    lower = np.min(gamma, axis=0)
    span = np.maximum(np.max(gamma, axis=0) - lower, np.finfo(float).eps)
    normalized = (gamma - lower) / span

    center_distance = np.linalg.norm(normalized - 0.5, axis=1)
    selected = [int(np.argmin(center_distance))]
    minimum_distance = np.linalg.norm(normalized - normalized[selected[0]], axis=1)
    minimum_distance[selected[0]] = -np.inf
    while len(selected) < requested:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.linalg.norm(normalized - normalized[next_index], axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -np.inf

    result = candidates.iloc[selected].copy().reset_index(drop=True)
    result.insert(0, "design_pool_position", np.asarray(selected, dtype=int))
    return result


def maximin_validation_pool(
    *, pool_count: int, count: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an independent Sobol pool and select a deterministic maximin design."""
    pool = independent_pool(
        int(pool_count),
        int(seed),
        id_column="validation_pool_id",
        label_prefix="validation_pool_sobol",
    )
    selected = affine_maximin_sequence(pool, int(count))
    selected.insert(
        0, "final_validation_id", np.arange(len(selected), dtype=int)
    )
    selected["material_label"] = [
        f"final_validation_maximin_{idx:04d}"
        for idx in selected["final_validation_id"].astype(int)
    ]
    return pool, selected


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
    compile_operators: bool = True,
    initial_solution_fields: np.ndarray | None = None,
    full_rank_basis_mode: str = "orthonormal",
    ritz_contraction_dtype: str = "float32",
    ritz_gram_compute_dtype: str = "float64",
    ritz_gram_backend: str = "auto",
    ritz_gram_rank_rtol: float = 1.0e-6,
    overlap_cpu_gram_gpu: bool = False,
    preserve_raw_coordinates: bool = False,
    factorized_ritz: bool = False,
    async_ritz: bool = False,
    experimental_local_frame_ritz: bool = False,
) -> tuple[
    list[dict[str, Any]], dict[str, np.ndarray] | None, np.ndarray | None
]:
    if not candidate_ids:
        return [], operators, initial_solution_fields
    started = time.perf_counter()
    nvox = int(voxel_order.size)
    ordered_fields = np.empty(
        (6 * len(candidate_ids), 6, nvox),
        dtype=basis.dtype,
    )
    records: list[dict[str, Any]] = []
    next_initial_fields = np.empty((6, 6, nvox), dtype=basis.dtype)
    for material_index, candidate_id in enumerate(candidate_ids):
        offset = 6 * material_index

        def consume_field(load_id: int, field: np.ndarray) -> None:
            field_view = np.asarray(field).reshape(6, nvox)
            next_initial_fields[int(load_id)] = field_view
            np.take(
                field_view,
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
                initial_solution_fields=initial_solution_fields,
            )
        initial_solution_fields = next_initial_fields
        if cleanup_snapshot_fields:
            shutil.rmtree(common.snapshot_dir(run_dir, candidate_id), ignore_errors=True)
        records.append(record)

    local_frame_metadata = {
        "local_frame_backend": "disabled",
        "local_frame_transform_wall_s": 0.0,
        "local_frame_chunk_voxels": 0,
        "local_frame_workspace_peak_bytes": 0,
    }
    if bool(experimental_local_frame_ritz):
        local_frame_metadata = reduced._localize_snapshot_fields_inplace(
            ordered_fields,
            affine_stress_batch,
        )

    rank_before = len(basis)
    basis_started = time.perf_counter()
    if str(full_rank_basis_mode) == "raw-ritz":
        new_fields = basis.append_raw_preordered(ordered_fields)
    elif str(full_rank_basis_mode) == "orthonormal":
        new_fields = basis.append_preordered(
            ordered_fields,
            tolerance=float(basis_tolerance),
        )
    else:
        raise ValueError(f"Unknown full-rank basis mode: {full_rank_basis_mode}")
    basis_wall_s = float(time.perf_counter() - basis_started)
    del ordered_fields
    affine_q_block_size = 0
    if compile_operators and len(new_fields):
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
    basis_gpu_uploads = 0
    avoided_duplicate_basis_gpu_uploads = 0
    gram_condition = 1.0
    gram_relative_min = 1.0
    gram_transform_mode = "not_assembled"
    gram_discarded_rank = 0
    ritz_effective_rank = len(basis)
    affine_stress_backend = "not_assembled"
    gpu_affine_chunks = 0
    cpu_affine_chunks = 0
    gpu_affine_fallback = ""
    factorized_upload_wall_s = 0.0
    factorized_host_prepare_wall_s = 0.0
    factorized_kernel_enqueue_wall_s = 0.0
    async_pinned_double_buffer = False
    async_pinned_bytes = 0
    contraction_modes: set[str] = set()
    if compile_operators and len(new_fields):
        operators, assembly = common._update_reduced_operators(
            phase=operator_phase,
            ori=operator_ori,
            basis=basis.active_fields,
            existing=operators,
            new_fields=basis.active_fields[rank_before:],
            affine_stress_batch=affine_stress_batch,
            affine_q_block_size=int(affine_q_block_size),
            gram_rank_reveal=str(full_rank_basis_mode) == "raw-ritz",
            gram_rank_rtol=float(ritz_gram_rank_rtol),
            contraction_compute_dtype=str(ritz_contraction_dtype),
            gram_compute_dtype=str(ritz_gram_compute_dtype),
            gram_backend=str(ritz_gram_backend),
            overlap_cpu_gram_gpu=bool(overlap_cpu_gram_gpu),
            preserve_raw_coordinates=bool(preserve_raw_coordinates),
            factorized_ritz=bool(factorized_ritz),
            async_ritz=bool(async_ritz),
        )
        for name in assembly_totals:
            source_name = (
                "contraction_wall_s"
                if name == "ritz_contraction_wall_s"
                else name
            )
            assembly_totals[name] += float(assembly.get(source_name, 0.0))
        stress_workspace_peak_bytes = int(
            assembly.get("stress_workspace_peak_bytes", 0)
        )
        contraction_workspace_peak_bytes = int(
            assembly.get("contraction_workspace_peak_bytes", 0)
        )
        full_volume_equivalent_passes = float(
            assembly.get("full_volume_equivalent_passes", 0.0)
        )
        basis_gpu_uploads = int(assembly.get("basis_gpu_uploads", 0))
        avoided_duplicate_basis_gpu_uploads = int(
            assembly.get("avoided_duplicate_basis_gpu_uploads", 0)
        )
        gram_condition = float(assembly.get("gram_condition", 1.0))
        gram_relative_min = float(assembly.get("gram_relative_min", 1.0))
        gram_transform_mode = str(
            assembly.get("gram_transform_mode", "unknown")
        )
        gram_discarded_rank = int(assembly.get("discarded_rank", 0))
        ritz_effective_rank = int(assembly.get("effective_rank", len(basis)))
        affine_stress_backend = str(
            assembly.get("affine_stress_backend", "cpu")
        )
        gpu_affine_chunks = int(assembly.get("gpu_affine_chunks", 0))
        cpu_affine_chunks = int(assembly.get("cpu_affine_chunks", 0))
        gpu_affine_fallback = str(assembly.get("gpu_affine_fallback", ""))
        factorized_upload_wall_s = float(
            assembly.get("factorized_upload_wall_s", 0.0)
        )
        factorized_host_prepare_wall_s = float(
            assembly.get("factorized_host_prepare_wall_s", 0.0)
        )
        factorized_kernel_enqueue_wall_s = float(
            assembly.get("factorized_kernel_enqueue_wall_s", 0.0)
        )
        async_pinned_double_buffer = bool(
            assembly.get("async_pinned_double_buffer", False)
        )
        async_pinned_bytes = int(assembly.get("async_pinned_bytes", 0))
        contraction_modes.add(str(assembly.get("contraction_mode", "unknown")))
        gram_product_wall_s = float(assembly.get("gram_product_wall_s", 0.0))
        gram_overlap_wait_wall_s = float(
            assembly.get("gram_overlap_wait_wall_s", 0.0)
        )
        gram_overlap_hidden_wall_s = float(
            assembly.get("gram_overlap_hidden_wall_s", 0.0)
        )
        gram_overlap_enabled = bool(assembly.get("gram_overlap_enabled", False))
    else:
        gram_product_wall_s = 0.0
        gram_overlap_wait_wall_s = 0.0
        gram_overlap_hidden_wall_s = 0.0
        gram_overlap_enabled = False

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
                "local_frame_transform_wall_s": float(
                    local_frame_metadata["local_frame_transform_wall_s"]
                )
                / material_count,
                "local_frame_backend": str(
                    local_frame_metadata["local_frame_backend"]
                ),
                "local_frame_chunk_voxels": int(
                    local_frame_metadata["local_frame_chunk_voxels"]
                ),
                "local_frame_workspace_peak_bytes": int(
                    local_frame_metadata["local_frame_workspace_peak_bytes"]
                ),
                "new_directions": added,
                "basis_rank": cumulative_rank,
                "operator_assembly_wall_s": assembly_totals["assembly_wall_s"]
                / material_count,
                "affine_stress_wall_s": assembly_totals["affine_stress_wall_s"]
                / material_count,
                "ritz_contraction_wall_s": assembly_totals["ritz_contraction_wall_s"]
                / material_count,
                "operator_assembly_mode": (
                    "monitor_checkpoint_exact_prefix"
                    if compile_operators
                    else "deferred_until_monitor_checkpoint"
                ),
                "ritz_contraction_mode": contraction_mode,
                "basis_projection_backend": str(basis.last_projection_backend),
                "operator_stress_workspace_peak_bytes": stress_workspace_peak_bytes,
                "operator_contraction_workspace_peak_bytes": contraction_workspace_peak_bytes,
                "ritz_full_volume_equivalent_passes": full_volume_equivalent_passes,
                "basis_gpu_uploads": basis_gpu_uploads,
                "avoided_duplicate_basis_gpu_uploads": avoided_duplicate_basis_gpu_uploads,
                "ritz_gram_condition": gram_condition,
                "ritz_gram_relative_min": gram_relative_min,
                "ritz_gram_transform_mode": gram_transform_mode,
                "ritz_gram_discarded_rank": gram_discarded_rank,
                "ritz_effective_rank": ritz_effective_rank,
                "ritz_contraction_compute_dtype": str(
                    assembly.get("contraction_compute_dtype", ritz_contraction_dtype)
                    if compile_operators and len(new_fields)
                    else ritz_contraction_dtype
                ),
                "ritz_gram_compute_dtype": str(
                    assembly.get("gram_product_dtype", ritz_gram_compute_dtype)
                    if compile_operators and len(new_fields)
                    else ritz_gram_compute_dtype
                ),
                "ritz_gram_backend": str(
                    assembly.get("gram_product_backend", ritz_gram_backend)
                    if compile_operators and len(new_fields)
                    else ritz_gram_backend
                ),
                "ritz_gram_rank_rtol": float(ritz_gram_rank_rtol),
                "gram_overlap_requested": bool(overlap_cpu_gram_gpu),
                "gram_overlap_enabled": bool(gram_overlap_enabled),
                "gram_product_wall_s": gram_product_wall_s / material_count,
                "gram_overlap_wait_wall_s": gram_overlap_wait_wall_s
                / material_count,
                "gram_overlap_hidden_wall_s": gram_overlap_hidden_wall_s
                / material_count,
                "affine_stress_backend": affine_stress_backend,
                "gpu_affine_chunks": gpu_affine_chunks,
                "cpu_affine_chunks": cpu_affine_chunks,
                "gpu_affine_fallback": gpu_affine_fallback,
                "factorized_upload_wall_s": factorized_upload_wall_s
                / material_count,
                "factorized_host_prepare_wall_s": factorized_host_prepare_wall_s
                / material_count,
                "factorized_kernel_enqueue_wall_s": factorized_kernel_enqueue_wall_s
                / material_count,
                "async_pinned_double_buffer": async_pinned_double_buffer,
                "async_pinned_bytes": async_pinned_bytes,
                "affine_q_block_size": int(affine_q_block_size),
                "pod_batch_materials": material_count,
            }
        )
    return records, operators, next_initial_fields


def recompile_full_basis(
    *,
    basis: common.ContiguousBasis | np.ndarray,
    operator_phase: np.ndarray,
    operator_ori: np.ndarray,
    affine_stress_batch: Any,
    contraction_compute_dtype: str,
    gram_compute_dtype: str,
    gram_backend: str,
    gram_rank_rtol: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Rebuild all raw Ritz blocks in a requested precision without FOM solves."""
    basis_fields = (
        basis.active_fields
        if isinstance(basis, common.ContiguousBasis)
        else np.asarray(basis)
    )
    return common._update_reduced_operators(
        phase=operator_phase,
        ori=operator_ori,
        basis=basis_fields,
        existing=None,
        new_fields=basis_fields,
        affine_stress_batch=affine_stress_batch,
        affine_q_block_size=1,
        gram_rank_reveal=True,
        gram_rank_rtol=float(gram_rank_rtol),
        contraction_compute_dtype=str(contraction_compute_dtype),
        gram_compute_dtype=str(gram_compute_dtype),
        gram_backend=str(gram_backend),
    )


def build_float32_snapshot_audit_variants(
    *,
    run_dir: Path,
    geometry: common.GeometryData,
    runtime: dict[str, Any],
    candidates: pd.DataFrame,
    selected_candidate_ids: list[int],
    candidate_seed: int,
    quiet_solver: bool,
    voxel_order: np.ndarray,
    operator_phase: np.ndarray,
    operator_ori: np.ndarray,
    affine_stress_batch: Any,
    affine_stress_max_gib: float,
    memory_safety_fraction: float,
    basis_tolerance: float,
    gram_rank_rtol: float,
) -> tuple[common.ContiguousBasis, dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    """Re-solve one frozen prefix in float32 and compile two audit variants."""
    audit_basis = common.ContiguousBasis(
        capacity=6 * len(selected_candidate_ids),
        field_shape=(6, int(geometry.phase.size)),
        dtype="float32",
        storage_path=run_dir / "float32_snapshot_audit_basis.dat",
    )
    audit_run_dir = run_dir / "float32_snapshot_audit"
    warm_start: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    for candidate_id in selected_candidate_ids:
        records, _, warm_start = append_sobol_batch(
            run_dir=audit_run_dir,
            geometry=geometry,
            runtime=runtime,
            candidates=candidates,
            candidate_ids=[int(candidate_id)],
            seed=int(candidate_seed),
            profile="snapshot32",
            quiet_solver=bool(quiet_solver),
            basis=audit_basis,
            voxel_order=voxel_order,
            operator_phase=operator_phase,
            operator_ori=operator_ori,
            operators=None,
            affine_stress_batch=affine_stress_batch,
            affine_stress_max_gib=float(affine_stress_max_gib),
            memory_safety_fraction=float(memory_safety_fraction),
            basis_tolerance=float(basis_tolerance),
            cleanup_snapshot_fields=True,
            compile_operators=False,
            initial_solution_fields=warm_start,
            full_rank_basis_mode="raw-ritz",
            ritz_contraction_dtype="float32",
            ritz_gram_compute_dtype="float32",
            ritz_gram_backend="auto",
            ritz_gram_rank_rtol=float(gram_rank_rtol),
        )
        rows.extend(records)
    float32_operators, _ = recompile_full_basis(
        basis=audit_basis,
        operator_phase=operator_phase,
        operator_ori=operator_ori,
        affine_stress_batch=affine_stress_batch,
        contraction_compute_dtype="float32",
        gram_compute_dtype="float32",
        gram_backend="auto",
        gram_rank_rtol=float(gram_rank_rtol),
    )
    mixed_operators, _ = recompile_full_basis(
        basis=audit_basis,
        operator_phase=operator_phase,
        operator_ori=operator_ori,
        affine_stress_batch=affine_stress_batch,
        contraction_compute_dtype="float64",
        gram_compute_dtype="float64",
        gram_backend="auto",
        gram_rank_rtol=float(gram_rank_rtol),
    )
    float32_operators["raw_basis"] = audit_basis.active_fields
    mixed_operators["raw_basis"] = audit_basis.active_fields
    return (
        audit_basis,
        {
            "float32_snapshots_float32_contractions": float32_operators,
            "float32_snapshots_float64_contractions": mixed_operators,
        },
        pd.DataFrame(rows),
    )


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


def solve_timing_and_reference_pools(
    *,
    run_dir: Path,
    geometry: common.GeometryData,
    runtime: dict[str, Any],
    materials: pd.DataFrame,
    seed: int,
    timing_profile: str,
    validation_profile: str,
    quiet_solver: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time each FOM from its standard start, then refine it for validation."""
    timing_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    nvox = int(geometry.phase.size)
    for index, material in enumerate(materials.to_dict(orient="records"), start=1):
        material_id = int(material["material_id"])
        timing_fields = np.empty((6, 6, nvox), dtype=np.float64)

        def consume(load_id: int, field: np.ndarray) -> None:
            timing_fields[int(load_id)] = np.asarray(
                field, dtype=np.float64
            ).reshape(6, nvox)

        print(
            f"[SOBOL-POD] FOM timing {index}/{len(materials)} | "
            f"material={material_id}",
            flush=True,
        )
        with quiet_solver_output(bool(quiet_solver)):
            timing_record = common.solve_material(
                material_row=material,
                material_dir=run_dir / "fom_timing_truth" / f"material_{material_id:04d}",
                geometry=geometry,
                runtime=runtime,
                profile=str(timing_profile),
                seed=int(seed) + material_id,
                save_solution_fields=True,
                persistent_gpu_cache=False,
                solution_field_dtype=np.float64,
                solution_field_consumer=consume,
            )
        timing_rows.append(timing_record)
        print(
            f"[SOBOL-POD] validation refinement {index}/{len(materials)} | "
            f"material={material_id}",
            flush=True,
        )
        with quiet_solver_output(bool(quiet_solver)):
            validation_record = common.solve_material(
                material_row=material,
                material_dir=run_dir / "final_validation_truth" / f"material_{material_id:04d}",
                geometry=geometry,
                runtime=runtime,
                profile=str(validation_profile),
                seed=int(seed) + material_id,
                save_solution_fields=False,
                persistent_gpu_cache=False,
                initial_solution_fields=timing_fields,
            )
        validation_rows.append(validation_record)
        del timing_fields
        gc.collect()
    return pd.DataFrame(timing_rows), pd.DataFrame(validation_rows)


def error_stats(
    frame: pd.DataFrame,
    *,
    id_column: str,
    threshold: float | None = None,
) -> dict[str, Any]:
    errors = frame["relative_frobenius_error"].to_numpy(dtype=float)
    finite = np.isfinite(errors)
    finite_errors = errors[finite]
    worst = int(np.nanargmax(errors)) if np.any(finite) else None
    stats = {
        "count": int(len(frame)),
        "finite_error_count": int(np.count_nonzero(finite)),
        "numerical_failure_count": int(len(frame) - np.count_nonzero(finite)),
        "error_mean": float(np.mean(finite_errors)) if finite_errors.size else float("nan"),
        "error_median": float(np.median(finite_errors)) if finite_errors.size else float("nan"),
        "error_p95": float(np.quantile(finite_errors, 0.95)) if finite_errors.size else float("nan"),
        "error_max": float(np.max(finite_errors)) if finite_errors.size else float("nan"),
        f"worst_{id_column}": (
            int(frame.iloc[worst][id_column]) if worst is not None else None
        ),
        "rom_online_mean_s": float(frame["rom_online_s"].mean()),
        "rom_online_p95_s": float(frame["rom_online_s"].quantile(0.95)),
    }
    if threshold is not None:
        stats.update(empirical_coverage(errors, threshold))
    coverage_1e3 = empirical_coverage(errors, 1.0e-3)
    stats.update(
        {
            "coverage_1e3_count": int(coverage_1e3["below_target_count"]),
            "coverage_1e3_fraction": float(coverage_1e3["below_target_fraction"]),
            "coverage_1e3_percent": float(coverage_1e3["below_target_percent"]),
        }
    )
    for column, output in (
        ("rom_min_eig", "rom_min_eig_min"),
        ("reduced_K_min_eig", "reduced_K_min_eig_min"),
        ("reduced_K_spectral_spd_margin", "reduced_K_spectral_spd_margin_min"),
        ("min_eig_Crom_minus_Cfom", "schur_difference_min_eig_min"),
        ("schur_eta", "schur_eta_min"),
    ):
        if column in frame:
            stats[output] = float(frame[column].min())
    return stats


def rom_tensor_difference_stats(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarize relative differences between two ROM tensor predictions."""
    columns = [f"Crom_{ii}{jj}" for ii in range(1, 7) for jj in range(1, 7)]
    left = reference[columns].to_numpy(dtype=np.float64)
    right = comparison[columns].to_numpy(dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("ROM comparison frames have incompatible tensor counts")
    denominators = np.maximum(
        np.linalg.norm(left, axis=1), np.finfo(np.float64).tiny
    )
    differences = np.linalg.norm(right - left, axis=1) / denominators
    return {
        "count": int(len(differences)),
        "relative_difference_mean": float(np.mean(differences)),
        "relative_difference_p95": float(np.quantile(differences, 0.95)),
        "relative_difference_max": float(np.max(differences)),
    }


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
    repetitions: int = 10,
    warm_single_count: int = 1000,
    cold_start_processes: int = 10,
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
    if int(repetitions) < 1:
        raise ValueError("ROM timing repetitions must be positive.")
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
    def evaluate_chunk(coefficients: np.ndarray) -> tuple[np.ndarray, float]:
        if gpu_evaluator is not None:
            return gpu_evaluator.evaluate(coefficients)
        C_batch, _, batch_wall_s = reduced._rom_ceff_batch(
            coefficients, Kq, Bq, Dq
        )
        return C_batch, float(batch_wall_s)

    warmup_stop = min(1, len(candidates))
    warmup_coefficients = reduced._material_coefficients_batch(
        parameters[:warmup_stop]
    )
    _, warmup_wall_s = evaluate_chunk(warmup_coefficients)

    warm_single_times: list[float] = []
    for query_id in range(int(warm_single_count)):
        coefficients = reduced._material_coefficients_batch(
            parameters[query_id % len(parameters) : query_id % len(parameters) + 1]
        )
        _, query_wall_s = evaluate_chunk(coefficients)
        warm_single_times.append(float(query_wall_s))

    if effective_chunk < len(candidates):
        raise MemoryError(
            "The configured ROM workspace cannot execute the requested one-shot batch: "
            f"effective_chunk={effective_chunk}, query_count={len(candidates)}."
        )

    repetition_wall_s: list[float] = []
    for repetition in range(int(repetitions)):
        coefficients = reduced._material_coefficients_batch(parameters)
        C_batch, batch_wall_s = evaluate_chunk(coefficients)
        online_wall_s = float(batch_wall_s)
        if repetition == int(repetitions) - 1:
            eigenvalues = np.linalg.eigvalsh(C_batch)
            properties[:] = reduced._engineering_constants_batch(C_batch)
            minimum_eigenvalues[:] = eigenvalues[:, 0]
        repetition_wall_s.append(float(online_wall_s))
    representative_wall_s = float(np.median(repetition_wall_s))
    per_query_times.fill(representative_wall_s / max(len(candidates), 1))
    cold_start_wall_s: list[float] = []
    if actual_backend == "gpu":
        cold_command = (
            "import time; t=time.perf_counter(); import cupy as cp; "
            "x=cp.zeros(1,dtype=cp.float64); x+=1; "
            "cp.cuda.Stream.null.synchronize(); print(time.perf_counter()-t)"
        )
        for _ in range(int(cold_start_processes)):
            completed = subprocess.run(
                [sys.executable, "-c", cold_command],
                check=True,
                capture_output=True,
                text=True,
            )
            cold_start_wall_s.append(float(completed.stdout.strip().splitlines()[-1]))
    warm_single_median_s = float(np.median(warm_single_times))
    summary = {
        **memory_plan,
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "gpu_operator_transfer_s": float(
            gpu_evaluator.operator_transfer_wall_s if gpu_evaluator is not None else 0.0
        ),
        "gpu_setup_error": gpu_setup_error,
        "query_count": int(len(candidates)),
        "timing_repetitions": int(repetitions),
        "batch_mode": "one_shot",
        "warmup_query_count": int(warmup_stop),
        "warmup_wall_s": float(warmup_wall_s),
        "warm_single_query_count": int(len(warm_single_times)),
        "warm_single_query_median_s": warm_single_median_s,
        "warm_single_query_p95_s": float(np.quantile(warm_single_times, 0.95)),
        "cold_start_process_count": int(len(cold_start_wall_s)),
        "cold_start_wall_s": cold_start_wall_s,
        "cold_start_median_s": (
            float(np.median(cold_start_wall_s)) if cold_start_wall_s else None
        ),
        "repetition_wall_s": repetition_wall_s,
        "rom_online_total_s": representative_wall_s,
        "rom_online_mean_s": representative_wall_s / max(len(candidates), 1),
        "rom_online_queries_per_s": float(len(candidates)) / max(
            representative_wall_s, np.finfo(float).tiny
        ),
    }
    result = candidates.copy()
    result["rom_online_s"] = per_query_times
    result["rom_min_eig"] = minimum_eigenvalues
    for index, name in enumerate(reduced.ENGINEERING_COLUMNS):
        result[f"rom_{name}"] = properties[:, index]
    return result, summary


def tau_sensitivity_study(
    *,
    validation_truth: pd.DataFrame,
    operators: dict[str, np.ndarray],
    thresholds: list[float],
    training_materials: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Change only the Gram rank threshold for one frozen raw snapshot prefix."""
    required = {"raw_Kq", "raw_Bq", "G", "Dq"}
    missing = sorted(required.difference(operators))
    if missing:
        raise KeyError(f"Fixed-prefix sensitivity requires raw Ritz blocks: {missing}")
    summaries: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    raw_rank = int(np.asarray(operators["G"]).shape[0])
    if raw_rank != 6 * int(training_materials):
        raise RuntimeError(
            "The raw Gram dimension does not match the frozen training prefix: "
            f"raw_rank={raw_rank}, expected={6 * int(training_materials)}."
        )
    for threshold in thresholds:
        Kq, Bq, _, gram_meta = reduced._transform_raw_operators_with_rank_policy(
            np.asarray(operators["raw_Kq"], dtype=np.float64),
            np.asarray(operators["raw_Bq"], dtype=np.float64),
            np.asarray(operators["G"], dtype=np.float64),
            allow_rank_reveal=True,
            rank_rtol=float(threshold),
        )
        frame = reduced._evaluate_rom(
            results_df=validation_truth,
            Kq=Kq,
            Bq=Bq,
            Dq=operators["Dq"],
        )
        frame.insert(0, "tau_G", float(threshold))
        frame.insert(1, "training_materials", int(training_materials))
        frame.insert(2, "raw_snapshot_columns", raw_rank)
        frame.insert(3, "pod_rank", int(Kq.shape[1]))
        details.append(frame)
        stats = error_stats(frame, id_column="material_id")
        summaries.append(
            {
                "tau_G": float(threshold),
                "training_materials": int(training_materials),
                "raw_snapshot_columns": raw_rank,
                "pod_rank": int(Kq.shape[1]),
                "discarded_rank": int(gram_meta["discarded_rank"]),
                "gram_lambda_min": float(gram_meta["gram_lambda_min"]),
                "gram_lambda_max": float(gram_meta["gram_lambda_max"]),
                **stats,
            }
        )
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def select_energy_pod_baseline(
    *,
    monitor_truth: pd.DataFrame,
    operators: dict[str, np.ndarray],
    retentions: list[float],
    target_error: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
    """Select conventional POD rank using only the independent monitor set."""
    required = {"raw_Kq", "raw_Bq", "G", "Dq"}
    missing = sorted(required.difference(operators))
    if missing:
        raise KeyError(f"Energy-POD baseline requires raw Ritz blocks: {missing}")
    if monitor_truth.empty:
        raise ValueError("Energy-POD selection requires a non-empty monitor set.")

    rows: list[dict[str, Any]] = []
    candidates: list[tuple[int, float, dict[str, np.ndarray]]] = []
    for retention in sorted({float(value) for value in retentions}):
        Kq, Bq, invR, metadata = (
            reduced._transform_raw_operators_with_energy_retention(
                np.asarray(operators["raw_Kq"], dtype=np.float64),
                np.asarray(operators["raw_Bq"], dtype=np.float64),
                np.asarray(operators["G"], dtype=np.float64),
                retention=retention,
            )
        )
        monitor_rom = reduced._evaluate_rom(
            results_df=monitor_truth,
            Kq=Kq,
            Bq=Bq,
            Dq=operators["Dq"],
        )
        stats = error_stats(monitor_rom, id_column="material_id")
        passes = bool(stats["error_max"] <= float(target_error))
        rows.append(
            {
                "pod_energy_retention": retention,
                "pod_energy_retention_realized": float(
                    metadata["pod_energy_retention_realized"]
                ),
                "pod_rank": int(Kq.shape[1]),
                "monitor_error_mean": float(stats["error_mean"]),
                "monitor_error_max": float(stats["error_max"]),
                "passes_monitor_target": passes,
                "selected_without_heldout": False,
            }
        )
        if passes:
            candidates.append(
                (
                    int(Kq.shape[1]),
                    retention,
                    {
                        "Kq": Kq,
                        "Bq": Bq,
                        "Dq": np.asarray(operators["Dq"], dtype=np.float64),
                        "invR": invR,
                    },
                )
            )

    selected: dict[str, np.ndarray] | None = None
    if candidates:
        selected_rank, selected_retention, selected = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        for row in rows:
            if (
                int(row["pod_rank"]) == selected_rank
                and float(row["pod_energy_retention"]) == selected_retention
            ):
                row["selected_without_heldout"] = True
                break
    return pd.DataFrame(rows), selected


def solve_structural_audit_material(
    *,
    run_dir: Path,
    geometry: common.GeometryData,
    runtime: dict[str, Any],
    material: dict[str, Any],
    seed: int,
    profile: str,
    quiet_solver: bool,
    audit_tag: str | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Solve one high-precision material and retain its six fluctuation fields."""
    nvox = int(geometry.phase.size)
    fields = np.empty((6, 6, nvox), dtype=np.float64)

    def consume(load_id: int, field: np.ndarray) -> None:
        fields[int(load_id)] = np.asarray(field, dtype=np.float64).reshape(6, nvox)

    material_id = int(material["material_id"])
    case_directory = (
        str(audit_tag)
        if audit_tag is not None
        else f"material_{material_id:04d}"
    )
    with quiet_solver_output(bool(quiet_solver)):
        record = common.solve_material(
            material_row=material,
            material_dir=run_dir / "structural_audit_truth" / case_directory,
            geometry=geometry,
            runtime=runtime,
            profile=str(profile),
            seed=int(seed) + material_id,
            save_solution_fields=True,
            persistent_gpu_cache=False,
            solution_field_dtype=np.float64,
            solution_field_consumer=consume,
        )
    return record, fields


def ritz_schur_energy_audit(
    *,
    truth_record: dict[str, Any],
    fom_fields: np.ndarray,
    material: dict[str, Any],
    raw_basis: np.ndarray,
    voxel_order: np.ndarray,
    affine_stress_batch: Any,
    operator_variants: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Compare the Schur difference with the microscopic Ritz error energy."""
    ordered_fom = np.take(fom_fields, voxel_order, axis=2)
    C_fom = reduced._full_ceff_from_row(pd.Series(truth_record))
    C_fom = 0.5 * (C_fom + C_fom.T)
    fom_scale = max(float(np.linalg.norm(C_fom, ord=2)), np.finfo(float).eps)
    coefficients = reduced._material_coefficients(material)
    nvox = int(ordered_fom.shape[-1])
    total_fom = ordered_fom.copy()
    total_fom += np.eye(6, dtype=np.float64)[:, :, None]
    C_fom_stress = np.empty((6, 6), dtype=np.float64)
    for response_load in range(6):
        response = total_fom[response_load : response_load + 1]
        stress = np.zeros_like(response[0])
        for q, coefficient in enumerate(coefficients):
            stress += float(coefficient) * affine_stress_batch(q, response)[0]
        C_fom_stress[:, response_load] = np.mean(stress, axis=1)
    C_fom_stress = 0.5 * (C_fom_stress + C_fom_stress.T)
    energy_stress_mismatch = float(
        np.linalg.norm(C_fom - C_fom_stress, ord="fro")
        / max(np.linalg.norm(C_fom, ord="fro"), np.finfo(float).eps)
    )
    rows: list[dict[str, Any]] = []
    for variant_name, variant in operator_variants.items():
        C_rom, amplitudes, online_s = reduced._rom_ceff(
            coefficients, variant["Kq"], variant["Bq"], variant["Dq"]
        )
        raw_coordinates = np.asarray(variant["invR"], dtype=np.float64) @ amplitudes
        variant_basis = np.asarray(
            variant.get("raw_basis", raw_basis), dtype=np.float64
        )
        errors = np.empty_like(ordered_fom, dtype=np.float64)
        for load_id in range(6):
            rom_field = np.einsum(
                "p,pcv->cv",
                raw_coordinates[:, load_id],
                variant_basis,
                optimize=True,
            )
            errors[load_id] = ordered_fom[load_id] - rom_field
        energy = np.empty((6, 6), dtype=np.float64)
        for response_load in range(6):
            stress = np.zeros_like(errors[response_load])
            response = errors[response_load : response_load + 1]
            for q, coefficient in enumerate(coefficients):
                stress += float(coefficient) * affine_stress_batch(q, response)[0]
            energy[:, response_load] = np.einsum(
                "icv,cv->i", errors, stress, optimize=True
            ) / float(nvox)
        energy = 0.5 * (energy + energy.T)
        difference = 0.5 * ((C_rom - C_fom) + (C_rom - C_fom).T)
        mismatch = float(np.linalg.norm(difference - energy, ord="fro") / fom_scale)
        eig_difference = np.linalg.eigvalsh(difference)
        eig_energy = np.linalg.eigvalsh(energy)
        identity_passes = bool(
            mismatch < 1.0e-8 and eig_energy[0] >= -1.0e-12 * fom_scale
        )
        rows.append(
            {
                "material_id": int(material["material_id"]),
                "material_label": str(material.get("material_label", "")),
                "rom_variant": str(variant_name),
                "pod_rank": int(variant["Kq"].shape[1]),
                "relative_frobenius_error": float(
                    np.linalg.norm(C_rom - C_fom, ord="fro")
                    / max(np.linalg.norm(C_fom, ord="fro"), np.finfo(float).eps)
                ),
                "schur_difference_min_eig": float(eig_difference[0]),
                "schur_eta": float(eig_difference[0] / fom_scale),
                "ritz_error_energy_min_eig": float(eig_energy[0]),
                "energy_identity_relative_mismatch": mismatch,
                "fom_energy_stress_relative_mismatch": energy_stress_mismatch,
                "ritz_schur_preservation_verified": identity_passes,
                "rom_online_s": float(online_s),
            }
        )
        del errors
        gc.collect()
    return pd.DataFrame(rows)


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
        from matplotlib.ticker import FuncFormatter

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        x = curve["training_materials"].to_numpy(dtype=int)
        percent_scale = 100.0
        ax.semilogy(
            x,
            percent_scale * curve["monitor_error_mean"],
            marker="o",
            label="mean",
        )
        ax.semilogy(
            x,
            percent_scale * curve["monitor_error_p95"],
            marker="s",
            label="p95",
        )
        ax.semilogy(
            x,
            percent_scale * curve["monitor_error_max"],
            marker="^",
            label="maximum",
        )
        ax.axhline(
            percent_scale * float(target_error),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"{percent_scale * target_error:g}% target",
        )
        ax.set_xlabel("Sobol training materials used for full-rank POD")
        ax.set_ylabel("Relative Frobenius error on FFT monitor set (%)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
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
    parser.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=bool(pipeline.get("adaptive", False)),
        help="Enable monitor-driven stopping; fixed training is the default.",
    )
    parser.add_argument("--start-materials", type=int, default=int(pipeline.get("start_materials", 2)))
    parser.add_argument(
        "--training-limit",
        type=int,
        default=None,
        help=(
            "Operational cap. By default fixed mode uses training_limit and "
            "adaptive mode uses adaptive_training_limit from the config; zero "
            "lets adaptive mode use the memory-safe candidate limit."
        ),
    )
    parser.add_argument(
        "--monitor-count",
        type=int,
        default=int(pipeline.get("monitor_count", 5)),
        help="Independent preliminary FOM materials used only with --adaptive.",
    )
    parser.add_argument(
        "--final-validation-pool-count",
        type=int,
        default=int(pipeline.get("final_validation_pool_count", 4096)),
    )
    parser.add_argument("--final-validation-count", type=int, default=int(pipeline.get("final_validation_count", 20)))
    parser.add_argument("--candidate-seed", type=int, default=int(pipeline.get("candidate_seed", 20260821)))
    parser.add_argument(
        "--selection-policy",
        choices=("sobol-prefix", "affine-maximin"),
        default="sobol-prefix",
        help="Choose a Sobol prefix or a maximin subset in affine operator space.",
    )
    parser.add_argument("--monitor-seed", type=int, default=int(pipeline.get("monitor_seed", 20260822)))
    parser.add_argument("--final-validation-seed", type=int, default=int(pipeline.get("final_validation_seed", 20260901)))
    parser.add_argument("--rom-timing-count", type=int, default=int(pipeline.get("rom_timing_count", 10000)))
    parser.add_argument("--rom-timing-seed", type=int, default=int(pipeline.get("rom_timing_seed", 20260823)))
    parser.add_argument(
        "--rom-timing-repetitions",
        type=int,
        default=int(pipeline.get("rom_timing_repetitions", 10)),
    )
    parser.add_argument("--rom-chunk-size", type=int, default=int(pipeline.get("rom_chunk_size", 10000)))
    parser.add_argument("--training-profile", choices=tuple(common.SOLVER_PROFILES), default=str(pipeline.get("training_profile", "snapshot")))
    parser.add_argument("--monitor-profile", choices=tuple(common.SOLVER_PROFILES), default=str(pipeline.get("monitor_profile", "snapshot")))
    parser.add_argument("--validation-profile", choices=tuple(common.SOLVER_PROFILES), default=str(pipeline.get("validation_profile", "reference")))
    parser.add_argument("--timing-profile", choices=tuple(common.SOLVER_PROFILES), default=str(pipeline.get("timing_profile", "timing")))
    parser.add_argument("--audit-profile", choices=tuple(common.SOLVER_PROFILES), default=str(pipeline.get("audit_profile", "truth")))
    parser.add_argument(
        "--structural-audit-count",
        type=int,
        default=int(pipeline.get("structural_audit_count", 3)),
    )
    parser.add_argument(
        "--structural-audit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the separate float64 Ritz--Schur audit for this geometry.",
    )
    parser.add_argument(
        "--tau-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompile the frozen snapshot prefix at the requested Gram thresholds.",
    )
    parser.add_argument(
        "--tau-sensitivity-values",
        type=float,
        nargs="+",
        default=[
            float(value)
            for value in pipeline.get(
                "tau_sensitivity_values", [1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7]
            )
        ],
    )
    parser.add_argument(
        "--energy-pod-baseline",
        action=argparse.BooleanOptionalAction,
        default=bool(pipeline.get("energy_pod_baseline", True)),
        help="Select a conventional energy-POD baseline using monitors only.",
    )
    parser.add_argument(
        "--energy-pod-retentions",
        type=float,
        nargs="+",
        default=[
            float(value)
            for value in pipeline.get(
                "energy_pod_retentions", [0.99, 0.999, 0.9999, 0.99999, 0.999999]
            )
        ],
    )
    parser.add_argument("--target-error", type=float, default=float(pipeline.get("target_error", 1.0e-4)))
    parser.add_argument("--basis-tolerance", type=float, default=float(pipeline.get("basis_tolerance", 1.0e-12)))
    parser.add_argument(
        "--basis-dtype",
        choices=("float32", "float64"),
        default=str(pipeline.get("basis_dtype", "float64")),
    )
    parser.add_argument(
        "--full-rank-basis-mode",
        choices=("orthonormal", "raw-ritz"),
        default=str(pipeline.get("full_rank_basis_mode", "raw-ritz")),
        help="Keep the explicit POD basis or whiten raw full-rank snapshots in Ritz.",
    )
    parser.add_argument(
        "--ritz-contraction-dtype",
        choices=("float32", "float64"),
        default=str(pipeline.get("ritz_contraction_dtype", "float32")),
        help="CUDA compute precision for affine Ritz contractions.",
    )
    parser.add_argument(
        "--ritz-gram-compute-dtype",
        choices=("float32", "float64"),
        default=str(pipeline.get("ritz_gram_compute_dtype", "float64")),
        help="Compute precision for the snapshot Gram product.",
    )
    parser.add_argument(
        "--ritz-gram-backend",
        choices=("auto", "cpu", "gpu"),
        default=str(pipeline.get("ritz_gram_backend", "auto")),
        help="Backend used for the snapshot Gram product.",
    )
    parser.add_argument(
        "--ritz-gram-rank-rtol",
        type=float,
        default=float(pipeline.get("ritz_gram_rank_rtol", 1.0e-6)),
        help="Relative machine-zero guard for the full snapshot Gram rank.",
    )
    parser.add_argument(
        "--overlap-cpu-gram-gpu",
        action=argparse.BooleanOptionalAction,
        default=bool(pipeline.get("overlap_cpu_gram_gpu", False)),
        help="Overlap CPU Gram products with GPU affine Ritz contractions.",
    )
    parser.add_argument(
        "--experimental-qr-audit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Recompile the frozen full snapshot span with blocked Householder "
            "TSQR for an experimental POD/raw-Ritz comparison."
        ),
    )
    parser.add_argument(
        "--experimental-qr-block-max-gib",
        type=float,
        default=2.0,
        help="Host temporary-memory cap for each experimental TSQR block.",
    )
    parser.add_argument(
        "--reference-energy-qr",
        action=argparse.BooleanOptionalAction,
        default=bool(pipeline.get("reference_energy_qr", True)),
        help=(
            "Monitor in raw snapshot coordinates and freeze the complete span "
            "once in reference-energy QR coordinates without a snapshot Gram."
        ),
    )
    parser.add_argument(
        "--factorized-ritz",
        action=argparse.BooleanOptionalAction,
        default=bool(pipeline.get("factorized_ritz", True)),
        help=(
            "Use exact constitutive-rank Ritz contractions with GPU-resident "
            "reduced accumulation. Requires --reference-energy-qr."
        ),
    )
    parser.add_argument(
        "--async-ritz",
        action=argparse.BooleanOptionalAction,
        default=bool(pipeline.get("async_ritz", True)),
        help=(
            "Overlap pinned host-to-device snapshot transfers with exact "
            "factorized GPU contractions using two CUDA buffers."
        ),
    )
    parser.add_argument(
        "--experimental-local-frame-ritz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Rotate each new fiber snapshot once into its local Mandel frame "
            "before exact factorized Ritz assembly."
        ),
    )
    parser.add_argument(
        "--experimental-gathered-factor-ritz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Gather orientation-dependent spectral factors per voxel and use "
            "large CUDA contractions without changing snapshot coordinates."
        ),
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
        default=float(pipeline.get("rom_batch_max_gib", 4.0)),
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
    parser.add_argument(
        "--warm-start-route",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Route a fixed Sobol set by affine proximity to improve CG warm starts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.training_limit is None:
        key = "adaptive_training_limit" if bool(args.adaptive) else "training_limit"
        args.training_limit = int(pipeline.get(key, 0) or 0)
    return args


def main() -> int:
    args = parse_args()
    run_pipeline_config = read_config(Path(args.config))["sobol_pod_pipeline"]
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
        if int(args.final_validation_count) < 1:
            raise ValueError("final_validation_count must be positive.")
        if int(args.final_validation_pool_count) < int(args.final_validation_count):
            raise ValueError(
                "final_validation_pool_count must be at least final_validation_count."
            )
        if int(args.rom_timing_repetitions) < 1:
            raise ValueError("rom_timing_repetitions must be positive.")
        if bool(args.structural_audit) and not (
            1 <= int(args.structural_audit_count) <= int(args.final_validation_count)
        ):
            raise ValueError(
                "structural_audit_count must lie within the final validation design."
            )
        if any(
            not np.isfinite(float(value)) or float(value) <= 0.0
            for value in args.tau_sensitivity_values
        ):
            raise ValueError("All tau_G sensitivity thresholds must be positive.")
        if not np.isfinite(float(args.target_error)) or float(args.target_error) <= 0.0:
            raise ValueError("target_error must be finite and positive.")
        if (
            not np.isfinite(float(args.ritz_gram_rank_rtol))
            or float(args.ritz_gram_rank_rtol) <= 0.0
        ):
            raise ValueError("ritz_gram_rank_rtol must be finite and positive.")
        if (
            not np.isfinite(float(args.experimental_qr_block_max_gib))
            or float(args.experimental_qr_block_max_gib) <= 0.0
        ):
            raise ValueError(
                "experimental_qr_block_max_gib must be finite and positive."
            )
        if bool(args.reference_energy_qr):
            if str(args.full_rank_basis_mode) != "raw-ritz":
                raise ValueError(
                    "reference_energy_qr requires full_rank_basis_mode=raw-ritz."
                )
            if bool(args.experimental_qr_audit):
                raise ValueError(
                    "Householder TSQR and reference-energy QR are separate experiments."
                )
            if bool(args.energy_pod_baseline) or bool(args.tau_sensitivity):
                raise ValueError(
                    "Energy-POD and tau sensitivity require the snapshot Gram; "
                    "disable them for reference_energy_qr."
                )
        if bool(args.factorized_ritz) and not bool(
            args.reference_energy_qr
        ):
            raise ValueError(
                "factorized_ritz requires reference_energy_qr."
            )
        if bool(args.async_ritz) and not bool(
            args.factorized_ritz
        ):
            raise ValueError(
                "async_ritz requires factorized_ritz."
            )
        if bool(args.experimental_local_frame_ritz) and not bool(
            args.factorized_ritz
        ):
            raise ValueError(
                "experimental_local_frame_ritz requires factorized_ritz."
            )
        if bool(args.experimental_local_frame_ritz) and bool(args.structural_audit):
            raise ValueError(
                "structural_audit is not implemented for local-frame snapshots."
            )
        if bool(args.experimental_gathered_factor_ritz) and not bool(
            args.factorized_ritz
        ):
            raise ValueError(
                "experimental_gathered_factor_ritz requires factorized_ritz."
            )
        if bool(args.experimental_gathered_factor_ritz) and bool(
            args.experimental_local_frame_ritz
        ):
            raise ValueError(
                "gathered-factor and local-frame Ritz are separate experiments."
            )
        if int(args.candidate_seed) == int(args.final_validation_seed):
            raise ValueError(
                "candidate_seed and final_validation_seed must be distinct."
            )
        if bool(args.adaptive) and len(
            {
                int(args.candidate_seed),
                int(args.monitor_seed),
                int(args.final_validation_seed),
            }
        ) != 3:
            raise ValueError(
                "Adaptive training requires distinct candidate, monitor, and "
                "final-validation seeds."
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
        candidate_parameters = candidates[
            list(reduced.MATERIAL_PARAMETER_COLUMNS)
        ].to_numpy(dtype=np.float64)
        reference_energy_qr_reference = np.mean(
            reduced._material_coefficients_batch(candidate_parameters), axis=0
        )

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
        protocol = resolve_training_protocol(
            bool(args.adaptive), int(args.monitor_count), requested_limit
        )
        fixed_training_protocol = protocol == "fixed"
        if requested_limit and requested_limit > safe_material_limit:
            raise MemoryError(
                f"training_limit={requested_limit} exceeds the raw-snapshot "
                f"memory-safe limit of {safe_material_limit} materials."
            )
        training_limit = requested_limit or safe_material_limit
        minimum_required = (
            int(training_limit)
            if fixed_training_protocol
            else int(args.start_materials)
        )
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
            raise MemoryError("The derived raw-snapshot training limit is not memory-safe.")
        candidates.to_csv(run_dir / "candidate_pool_used.csv", index=False)
        if not pool_hash:
            pool_hash = sha256(run_dir / "candidate_pool_used.csv")

        if str(args.selection_policy) == "affine-maximin":
            sequence = affine_maximin_sequence(candidates, training_limit)
        else:
            sequence = material_sequence(candidates, training_limit)
        fixed_training_set = fixed_training_protocol or (
            int(args.start_materials) == int(training_limit)
        )
        if bool(args.warm_start_route) and fixed_training_set:
            sequence = fixed_warm_start_route(sequence)
        else:
            sequence = sequence.copy()
            sequence.insert(
                0, "sobol_set_position", np.arange(len(sequence), dtype=int)
            )
            sequence.insert(1, "solve_position", np.arange(len(sequence), dtype=int))
        sequence.to_csv(run_dir / "planned_sobol_sequence.csv", index=False)
        if fixed_training_protocol:
            monitor = pd.DataFrame()
        else:
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
            solver_tol=float(common.SOLVER_PROFILES[str(args.training_profile)]["solver_rtol"]),
            fft_backend=str(args.fft_backend),
            load_batch_size=1,
        )
        write_json(
            run_dir / "run_manifest.json",
            {
                "method": (
                    "fixed_sobol_pod_full_rank"
                    if fixed_training_protocol
                    else "adaptive_sobol_pod_full_rank"
                ),
                "adaptive": bool(args.adaptive),
                "run_dir": str(run_dir),
                "geometry_dir": str(geometry_dir),
                **geometry_info,
                "candidate_pool": source,
                "candidate_pool_sha256": pool_hash,
                "selection_policy": str(args.selection_policy),
                "solve_order_policy": (
                    "fixed_set_affine_nearest_neighbor"
                    if bool(args.warm_start_route) and fixed_training_set
                    else "sobol_prefix_order"
                ),
                "training_limit_policy": "memory_safe_candidate_limit"
                if requested_limit == 0
                else "explicit_operational_limit",
                "training_limit": int(training_limit),
                "memory_safe_material_limit": int(safe_material_limit),
                "monitoring_policy": (
                    "none_training_then_final_validation"
                    if fixed_training_protocol
                    else "fixed_independent_fft_pool_after_each_material"
                ),
                "monitor_count": int(len(monitor)),
                "monitor_seed": int(args.monitor_seed),
                "monitor_start_materials": int(args.start_materials),
                "final_validation_policy": "independent_sobol_pool_affine_maximin_after_freeze",
                "final_validation_pool_count": int(args.final_validation_pool_count),
                "final_validation_count": int(args.final_validation_count),
                "final_validation_seed": int(args.final_validation_seed),
                "rom_timing_count": int(args.rom_timing_count),
                "rom_timing_repetitions": int(args.rom_timing_repetitions),
                "training_profile": str(args.training_profile),
                "monitor_profile": str(args.monitor_profile),
                "validation_profile": str(args.validation_profile),
                "timing_profile": str(args.timing_profile),
                "audit_profile": str(args.audit_profile),
                "fft_backend": str(args.fft_backend),
                "target_error": float(args.target_error),
                "basis_tolerance": float(args.basis_tolerance),
                "basis_dtype": str(args.basis_dtype),
                "full_rank_basis_mode": str(args.full_rank_basis_mode),
                "ritz_contraction_dtype": str(args.ritz_contraction_dtype),
                "ritz_gram_compute_dtype": str(args.ritz_gram_compute_dtype),
                "ritz_gram_backend": str(args.ritz_gram_backend),
                "ritz_gram_rank_rtol": float(args.ritz_gram_rank_rtol),
                "overlap_cpu_gram_gpu": bool(args.overlap_cpu_gram_gpu),
                "experimental_qr_audit": bool(args.experimental_qr_audit),
                "experimental_qr_block_max_gib": float(
                    args.experimental_qr_block_max_gib
                ),
                "reference_energy_qr": bool(args.reference_energy_qr),
                "factorized_ritz": bool(
                    args.factorized_ritz
                ),
                "async_ritz": bool(args.async_ritz),
                "experimental_local_frame_ritz": bool(
                    args.experimental_local_frame_ritz
                ),
                "experimental_gathered_factor_ritz": bool(
                    args.experimental_gathered_factor_ritz
                ),
                "reference_energy_qr_reference_policy": "candidate_affine_mean",
                "reference_energy_qr_reference_coefficients": (
                    reference_energy_qr_reference
                    if bool(args.reference_energy_qr)
                    else None
                ),
                "basis_storage": "preallocated_contiguous",
                "basis_projection_passes": (
                    0 if str(args.full_rank_basis_mode) == "raw-ritz" else 1
                ),
                "basis_projection_backend": (
                    "raw_full_rank_no_projection"
                    if str(args.full_rank_basis_mode) == "raw-ritz"
                    else "scipy_blas_gemm_in_place"
                ),
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
                "ritz_contraction_kernel": (
                    "experimental_local_frame_factorized_gpu_async"
                    if bool(args.experimental_local_frame_ritz)
                    else (
                        "experimental_gathered_factorized_gpu_async"
                        if bool(args.experimental_gathered_factor_ritz)
                        else (
                            "exact_factorized_gpu_async"
                            if bool(args.async_ritz)
                            else (
                                "exact_factorized_gpu"
                                if bool(args.factorized_ritz)
                                else "exact_phase_supported_component_batched"
                            )
                        )
                    )
                ),
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
        warm_start_fields: np.ndarray | None = None
        affine_started = time.perf_counter()
        affine = reduced.affine_stress_batch_factory(
            operator_phase,
            operator_ori,
            local_frame_snapshots=bool(args.experimental_local_frame_ritz),
            gathered_factor_ritz=bool(args.experimental_gathered_factor_ritz),
        )
        affine_setup_wall_s = float(time.perf_counter() - affine_started)
        reconstruction_check = reduced.affine_constitutive_reconstruction_error(
            geometry.phase,
            geometry.ori,
            sequence.iloc[0].to_dict(),
        )
        write_json(run_dir / "affine_constitutive_reconstruction.json", reconstruction_check)
        if (
            float(reconstruction_check["relative_frobenius_error"]) >= 1.0e-12
            or float(reconstruction_check["maximum_voxel_group_relative_error"])
            >= 1.0e-12
        ):
            raise RuntimeError(
                "The shared voxelwise affine constitutive reconstruction failed: "
                f"{reconstruction_check}"
            )

        if fixed_training_protocol:
            monitor_truth = pd.DataFrame()
            monitor_fft_stage_wall_s = 0.0
        else:
            monitor_fft_started = time.perf_counter()
            monitor_truth = solve_truth_pool(
                run_dir=run_dir,
                geometry=geometry,
                runtime=runtime,
                materials=monitor,
                seed=int(args.monitor_seed),
                profile=str(args.monitor_profile),
                quiet_solver=bool(args.quiet_solver),
                pool_name="monitor",
            )
            monitor_truth.to_csv(run_dir / "monitor_truth_results.csv", index=False)
            monitor_fft_stage_wall_s = float(
                time.perf_counter() - monitor_fft_started
            )

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
            "local_frame_transform_wall_s": 0.0,
            "operator_assembly_wall_s": 0.0,
            "affine_stress_wall_s": 0.0,
            "ritz_contraction_wall_s": 0.0,
        }

        training_started = time.perf_counter()
        for training_materials, candidate_id in enumerate(candidate_ids, start=1):
            records, operators, warm_start_fields = append_sobol_batch(
                run_dir=run_dir,
                geometry=geometry,
                runtime=runtime,
                candidates=candidates,
                candidate_ids=[candidate_id],
                seed=int(args.candidate_seed),
                profile=str(args.training_profile),
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
                compile_operators=(
                    training_materials == int(training_limit)
                    if fixed_training_protocol
                    else training_materials >= int(args.start_materials)
                ),
                initial_solution_fields=warm_start_fields,
                full_rank_basis_mode=str(args.full_rank_basis_mode),
                ritz_contraction_dtype=str(args.ritz_contraction_dtype),
                ritz_gram_compute_dtype=str(args.ritz_gram_compute_dtype),
                ritz_gram_backend=str(args.ritz_gram_backend),
                ritz_gram_rank_rtol=float(args.ritz_gram_rank_rtol),
                overlap_cpu_gram_gpu=bool(args.overlap_cpu_gram_gpu),
                preserve_raw_coordinates=bool(args.reference_energy_qr),
                factorized_ritz=bool(
                    args.factorized_ritz
                ),
                async_ritz=bool(args.async_ritz),
                experimental_local_frame_ritz=bool(
                    args.experimental_local_frame_ritz
                ),
            )
            snapshot_rows.extend(records)
            for record in records:
                for name in cumulative:
                    cumulative[name] += float(record[name])
            pd.DataFrame(snapshot_rows).to_csv(run_dir / "snapshot_timing.csv", index=False)
            for record in records:
                print(
                    f"[SOBOL-POD] snapshot candidate={int(record['candidate_id'])} | "
                    f"raw_rank={int(record['basis_rank'])} | "
                    f"ritz_rank={int(record['ritz_effective_rank'])} | "
                    f"step={float(record['snapshot_step_wall_s']):.2f}s",
                    flush=True,
                )
            if fixed_training_protocol and training_materials < int(training_limit):
                continue
            if (
                not fixed_training_protocol
                and training_materials < int(args.start_materials)
            ):
                continue
            if operators is None:
                raise RuntimeError("No reduced operators were assembled.")

            if fixed_training_protocol:
                stop_materials = int(training_materials)
                effective_rank = int(operators["Kq"].shape[1])
                curve_rows.append(
                    {
                        "method": "fixed_sobol_pod_full_rank",
                        "training_materials": int(training_materials),
                        "pod_rank": effective_rank,
                        "snapshot_solve_wall_s": cumulative["solve_wall_s"],
                        "snapshot_step_wall_s": cumulative["snapshot_step_wall_s"],
                        "basis_update_wall_s": cumulative["basis_update_wall_s"],
                        "local_frame_transform_wall_s": cumulative[
                            "local_frame_transform_wall_s"
                        ],
                        "operator_assembly_wall_s": cumulative[
                            "operator_assembly_wall_s"
                        ],
                        "affine_stress_wall_s": cumulative["affine_stress_wall_s"],
                        "ritz_contraction_wall_s": cumulative[
                            "ritz_contraction_wall_s"
                        ],
                        "monitor_rom_cumulative_wall_s": 0.0,
                        "monitor_error_mean": np.nan,
                        "monitor_error_median": np.nan,
                        "monitor_error_p95": np.nan,
                        "monitor_error_max": np.nan,
                        "passes_target_max": np.nan,
                        "stop_triggered": True,
                        "rom_online_mean_s": np.nan,
                        "worst_monitor_id": np.nan,
                    }
                )
                print(
                    f"[SOBOL-POD] fixed training complete | "
                    f"materials={training_materials} | rank={effective_rank}",
                    flush=True,
                )
                break

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
            effective_rank = int(operators["Kq"].shape[1])
            frame.insert(2, "pod_rank", effective_rank)
            stats = error_stats(frame, id_column="monitor_id")
            passes = bool(stats["error_max"] <= float(args.target_error))
            last_monitor_stats = stats
            monitor_frames.append(frame)
            curve_rows.append(
                {
                    "method": "adaptive_sobol_pod_full_rank",
                    "training_materials": int(training_materials),
                    "pod_rank": effective_rank,
                    "snapshot_solve_wall_s": cumulative["solve_wall_s"],
                    "snapshot_step_wall_s": cumulative["snapshot_step_wall_s"],
                    "basis_update_wall_s": cumulative["basis_update_wall_s"],
                    "local_frame_transform_wall_s": cumulative[
                        "local_frame_transform_wall_s"
                    ],
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
                f"raw_rank={len(basis)} | ritz_rank={effective_rank} | "
                f"error_max={stats['error_max']:.3e} | "
                f"target={float(args.target_error):.1e}",
                flush=True,
            )
            if passes:
                stop_materials = int(training_materials)
                break
        training_stage_wall_s = float(time.perf_counter() - training_started)
        final_basis_rank = int(
            operators["Kq"].shape[1] if operators is not None else len(basis)
        )

        adaptive_target_reached = (
            None if fixed_training_protocol else stop_materials is not None
        )
        if stop_materials is None:
            stop_materials = int(len(snapshot_rows))
            write_json(
                run_dir / "adaptive_limit_warning.json",
                {
                    "status": "complete_at_limit_without_monitor_pass",
                    "training_materials": int(stop_materials),
                    "training_limit": int(training_limit),
                    "target_error": float(args.target_error),
                    "last_monitor_summary": last_monitor_stats,
                },
            )
            print(
                "[SOBOL-POD] WARNING: adaptive target was not reached by "
                f"material {stop_materials}; freezing the best available ROM "
                "and continuing with independent final validation.",
                flush=True,
            )

        selected = sequence.iloc[:stop_materials].copy()
        selected.to_csv(run_dir / "selected_sobol_sequence.csv", index=False)
        if operators is None:
            raise RuntimeError("Cannot freeze an empty reduced model.")

        reference_energy_qr_metadata: dict[str, Any] = {}
        reference_energy_qr_stage_wall_s = 0.0
        reference_energy_qr_monitor_difference: dict[str, Any] | None = None
        if bool(args.reference_energy_qr):
            required_raw = {"raw_Kq", "raw_Bq"}
            missing_raw = sorted(required_raw.difference(operators))
            if missing_raw:
                raise RuntimeError(
                    "Reference-energy QR requires frozen raw Ritz blocks: "
                    + ", ".join(missing_raw)
                )
            energy_qr_started = time.perf_counter()
            raw_monitor = (
                reduced._evaluate_rom(
                    results_df=monitor_truth,
                    Kq=operators["Kq"],
                    Bq=operators["Bq"],
                    Dq=operators["Dq"],
                )
                if not monitor_truth.empty
                else pd.DataFrame()
            )
            energy_operators, reference_energy_qr_metadata = (
                reduced._reference_energy_qr_recompile(
                    raw_Kq=operators["raw_Kq"],
                    raw_Bq=operators["raw_Bq"],
                    Dq=operators["Dq"],
                    reference_coefficients=reference_energy_qr_reference,
                )
            )
            operators["Kq"] = energy_operators["Kq"]
            operators["Bq"] = energy_operators["Bq"]
            operators["Dq"] = energy_operators["Dq"]
            operators["invR"] = energy_operators["invR"]
            operators["energy_qr_R"] = energy_operators["R"]
            operators["energy_qr_reference_coefficients"] = energy_operators[
                "reference_coefficients"
            ]
            if not monitor_truth.empty:
                energy_monitor = reduced._evaluate_rom(
                    results_df=monitor_truth,
                    Kq=operators["Kq"],
                    Bq=operators["Bq"],
                    Dq=operators["Dq"],
                )
                energy_monitor.insert(
                    0, "monitor_id", monitor["monitor_id"].to_numpy(dtype=int)
                )
                energy_monitor.to_csv(
                    run_dir / "reference_energy_qr_monitor_results.csv",
                    index=False,
                )
                reference_energy_qr_monitor_difference = (
                    rom_tensor_difference_stats(raw_monitor, energy_monitor)
                )
            reference_energy_qr_stage_wall_s = float(
                time.perf_counter() - energy_qr_started
            )
            reference_energy_qr_metadata.update(
                {
                    "principal_route": True,
                    "reference_policy": "candidate_affine_mean",
                    "training_monitor_coordinates": "raw_snapshot",
                    "frozen_coordinates": "reference_energy_qr",
                    "stage_wall_s": reference_energy_qr_stage_wall_s,
                    "monitor_raw_coordinate_difference": (
                        reference_energy_qr_monitor_difference
                    ),
                }
            )
            write_json(
                run_dir / "reference_energy_qr_manifest.json",
                reference_energy_qr_metadata,
            )
            print(
                "[SOBOL-POD] reference-energy QR freeze | "
                f"rank={operators['Kq'].shape[1]} | "
                f"stage={reference_energy_qr_stage_wall_s:.3f}s | "
                "voxel_passes=0",
                flush=True,
            )

        experimental_qr_operators: dict[str, np.ndarray] | None = None
        experimental_qr_metadata: dict[str, Any] = {}
        experimental_qr_monitor = pd.DataFrame()
        experimental_qr_monitor_stats: dict[str, Any] | None = None
        experimental_qr_monitor_difference: dict[str, Any] | None = None
        experimental_qr_model_hash: str | None = None
        experimental_qr_stage_wall_s = 0.0
        if bool(args.experimental_qr_audit):
            required_raw = {"raw_Kq", "raw_Bq", "G"}
            missing_raw = sorted(required_raw.difference(operators))
            if missing_raw:
                raise RuntimeError(
                    "Experimental TSQR requires frozen raw Ritz blocks: "
                    + ", ".join(missing_raw)
                )
            print(
                f"[SOBOL-POD] experimental TSQR audit | rank={len(basis)} | "
                f"block_cap={float(args.experimental_qr_block_max_gib):.2f} GiB",
                flush=True,
            )
            qr_started = time.perf_counter()
            rss_before_kib = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            experimental_qr_operators, experimental_qr_metadata = (
                reduced._experimental_tsqr_recompile(
                    basis=basis.active_fields,
                    raw_Kq=operators["raw_Kq"],
                    raw_Bq=operators["raw_Bq"],
                    Dq=operators["Dq"],
                    G=operators["G"],
                    nvox=int(geometry.phase.size),
                    block_max_gib=float(args.experimental_qr_block_max_gib),
                )
            )
            rss_after_kib = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            experimental_qr_metadata.update(
                {
                    "experimental": True,
                    "official_model_modified": False,
                    "qr_peak_rss_before_kib": rss_before_kib,
                    "qr_peak_rss_after_kib": rss_after_kib,
                    "qr_peak_rss_increment_kib": max(
                        0, rss_after_kib - rss_before_kib
                    ),
                }
            )
            if not monitor_truth.empty:
                nominal_monitor = reduced._evaluate_rom(
                    results_df=monitor_truth,
                    Kq=operators["Kq"],
                    Bq=operators["Bq"],
                    Dq=operators["Dq"],
                )
                experimental_qr_monitor = reduced._evaluate_rom(
                    results_df=monitor_truth,
                    Kq=experimental_qr_operators["Kq"],
                    Bq=experimental_qr_operators["Bq"],
                    Dq=experimental_qr_operators["Dq"],
                )
                experimental_qr_monitor.insert(
                    0, "monitor_id", monitor["monitor_id"].to_numpy(dtype=int)
                )
                experimental_qr_monitor.to_csv(
                    run_dir / "experimental_qr_monitor_results.csv", index=False
                )
                experimental_qr_monitor_stats = error_stats(
                    experimental_qr_monitor,
                    id_column="monitor_id",
                    threshold=float(args.target_error),
                )
                experimental_qr_monitor_difference = rom_tensor_difference_stats(
                    nominal_monitor,
                    experimental_qr_monitor,
                )
            qr_model_path = run_dir / "experimental_qr_operators.npz"
            np.savez_compressed(
                qr_model_path,
                Kq=experimental_qr_operators["Kq"],
                Bq=experimental_qr_operators["Bq"],
                Dq=experimental_qr_operators["Dq"],
                R=experimental_qr_operators["R"],
                coefficient_names=np.asarray(reduced.COEFF_NAMES),
                candidate_ids=selected["candidate_id"].to_numpy(dtype=np.int64),
            )
            experimental_qr_model_hash = sha256(qr_model_path)
            experimental_qr_stage_wall_s = float(time.perf_counter() - qr_started)
            experimental_qr_metadata["qr_audit_stage_wall_s"] = (
                experimental_qr_stage_wall_s
            )
            write_json(
                run_dir / "experimental_qr_manifest.json",
                {
                    "status": "experimental_audit_frozen_before_validation_design",
                    "operators_path": str(qr_model_path),
                    "operators_sha256": experimental_qr_model_hash,
                    "training_materials": int(stop_materials),
                    "rank": int(experimental_qr_operators["Kq"].shape[1]),
                    "metadata": experimental_qr_metadata,
                    "monitor_summary": experimental_qr_monitor_stats,
                    "nominal_monitor_difference": experimental_qr_monitor_difference,
                },
            )
            print(
                "[SOBOL-POD] experimental TSQR complete | "
                f"factor={float(experimental_qr_metadata['qr_factor_wall_s']):.2f}s | "
                f"peak_tmp={int(experimental_qr_metadata['qr_estimated_peak_temporary_bytes']) / 1024**3:.2f} GiB",
                flush=True,
            )

        frozen_payload: dict[str, np.ndarray] = {
            "Kq": operators["Kq"],
            "Bq": operators["Bq"],
            "Dq": operators["Dq"],
            "coefficient_names": np.asarray(reduced.COEFF_NAMES),
            "candidate_ids": selected["candidate_id"].to_numpy(dtype=np.int64),
        }
        for name in (
            "raw_Kq",
            "raw_Bq",
            "G",
            "invR",
            "energy_qr_R",
            "energy_qr_reference_coefficients",
        ):
            if name in operators:
                frozen_payload[name] = operators[name]
        frozen_model_path = run_dir / "reduced_operators.npz"
        np.savez_compressed(frozen_model_path, **frozen_payload)
        frozen_model_hash = sha256(frozen_model_path)
        write_json(
            run_dir / "frozen_model_manifest.json",
            {
                "status": "frozen_before_final_validation_design",
                "reduced_operators_path": str(frozen_model_path),
                "reduced_operators_sha256": frozen_model_hash,
                "selected_sequence_sha256": sha256(
                    run_dir / "selected_sobol_sequence.csv"
                ),
                "training_materials": int(stop_materials),
                "pod_rank": int(final_basis_rank),
                "ritz_contraction_dtype": str(args.ritz_contraction_dtype),
                "ritz_gram_compute_dtype": str(args.ritz_gram_compute_dtype),
                "ritz_gram_backend": str(args.ritz_gram_backend),
                "ritz_gram_rank_rtol": float(args.ritz_gram_rank_rtol),
                "overlap_cpu_gram_gpu": bool(args.overlap_cpu_gram_gpu),
                "reference_energy_qr": bool(args.reference_energy_qr),
                "factorized_ritz": bool(
                    args.factorized_ritz
                ),
                "async_ritz": bool(args.async_ritz),
                "experimental_local_frame_ritz": bool(
                    args.experimental_local_frame_ritz
                ),
                "experimental_gathered_factor_ritz": bool(
                    args.experimental_gathered_factor_ritz
                ),
                "reference_energy_qr_metadata": (
                    reference_energy_qr_metadata
                ),
            },
        )

        energy_pod_summary = pd.DataFrame()
        energy_pod_operators: dict[str, np.ndarray] | None = None
        energy_pod_model_hash: str | None = None
        if bool(args.energy_pod_baseline) and not monitor_truth.empty:
            energy_pod_summary, energy_pod_operators = select_energy_pod_baseline(
                monitor_truth=monitor_truth,
                operators=operators,
                retentions=[float(value) for value in args.energy_pod_retentions],
                target_error=float(args.target_error),
            )
            energy_pod_summary.to_csv(
                run_dir / "energy_pod_monitor_selection.csv", index=False
            )
            if energy_pod_operators is not None:
                energy_pod_model_path = run_dir / "energy_pod_selected_operators.npz"
                np.savez_compressed(energy_pod_model_path, **energy_pod_operators)
                energy_pod_model_hash = sha256(energy_pod_model_path)
                write_json(
                    run_dir / "energy_pod_frozen_manifest.json",
                    {
                        "status": "selected_from_monitor_before_final_validation_design",
                        "operators_sha256": energy_pod_model_hash,
                        "selected_row": energy_pod_summary.loc[
                            energy_pod_summary["selected_without_heldout"]
                        ].iloc[0].to_dict(),
                    },
                )

        validation_candidates, final_validation = maximin_validation_pool(
            pool_count=int(args.final_validation_pool_count),
            count=int(args.final_validation_count),
            seed=int(args.final_validation_seed),
        )
        validation_candidates.to_csv(
            run_dir / "final_validation_candidate_pool.csv", index=False
        )
        final_validation.to_csv(run_dir / "final_validation_pool.csv", index=False)

        curve = pd.DataFrame(curve_rows)
        monitor_rom = (
            pd.concat(monitor_frames, ignore_index=True)
            if monitor_frames
            else pd.DataFrame()
        )
        curve.to_csv(run_dir / "sobol_pod_error_curve.csv", index=False)
        if not monitor_rom.empty:
            monitor_rom.to_csv(run_dir / "monitor_rom_results.csv", index=False)

        fom_timing_results, final_validation_truth = solve_timing_and_reference_pools(
            run_dir=run_dir,
            geometry=geometry,
            runtime=runtime,
            materials=final_validation,
            seed=int(args.final_validation_seed),
            timing_profile=str(args.timing_profile),
            validation_profile=str(args.validation_profile),
            quiet_solver=bool(args.quiet_solver),
        )
        fom_timing_results.to_csv(run_dir / "fom_timing_results.csv", index=False)
        final_validation_truth.to_csv(
            run_dir / "final_validation_truth_results.csv", index=False
        )
        validation_evaluation_started = time.perf_counter()
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
        final_validation_rom["below_1e3"] = (
            final_validation_rom["relative_frobenius_error"] <= 1.0e-3
        )
        final_validation_rom.to_csv(
            run_dir / "final_validation_rom_results.csv", index=False
        )
        final_stats = error_stats(
            final_validation_rom,
            id_column="final_validation_id",
            threshold=float(args.target_error),
        )
        experimental_qr_validation = pd.DataFrame()
        experimental_qr_validation_stats: dict[str, Any] | None = None
        experimental_qr_validation_difference: dict[str, Any] | None = None
        if experimental_qr_operators is not None:
            experimental_qr_validation = reduced._evaluate_rom(
                results_df=final_validation_truth,
                Kq=experimental_qr_operators["Kq"],
                Bq=experimental_qr_operators["Bq"],
                Dq=experimental_qr_operators["Dq"],
            )
            experimental_qr_validation.insert(
                0,
                "final_validation_id",
                final_validation["final_validation_id"].to_numpy(dtype=int),
            )
            experimental_qr_validation.to_csv(
                run_dir / "experimental_qr_final_validation.csv", index=False
            )
            experimental_qr_validation_stats = error_stats(
                experimental_qr_validation,
                id_column="final_validation_id",
                threshold=float(args.target_error),
            )
            experimental_qr_validation_difference = rom_tensor_difference_stats(
                final_validation_rom,
                experimental_qr_validation,
            )
            write_json(
                run_dir / "experimental_qr_validation_summary.json",
                {
                    "experimental_qr_summary": experimental_qr_validation_stats,
                    "nominal_raw_ritz_summary": final_stats,
                    "nominal_prediction_difference": (
                        experimental_qr_validation_difference
                    ),
                },
            )
        energy_pod_validation = pd.DataFrame()
        energy_pod_validation_stats: dict[str, Any] | None = None
        if energy_pod_operators is not None:
            energy_pod_validation = reduced._evaluate_rom(
                results_df=final_validation_truth,
                Kq=energy_pod_operators["Kq"],
                Bq=energy_pod_operators["Bq"],
                Dq=energy_pod_operators["Dq"],
            )
            energy_pod_validation.insert(
                0,
                "final_validation_id",
                final_validation["final_validation_id"].to_numpy(dtype=int),
            )
            energy_pod_validation.to_csv(
                run_dir / "energy_pod_final_validation.csv", index=False
            )
            energy_pod_validation_stats = error_stats(
                energy_pod_validation,
                id_column="final_validation_id",
                threshold=float(args.target_error),
            )
        validation_stage_wall_s = float(
            final_validation_truth["solve_wall_s"].sum()
            + time.perf_counter()
            - validation_evaluation_started
        )
        fom_timing_stage_wall_s = float(fom_timing_results["solve_wall_s"].sum())

        tau_sensitivity_summary = pd.DataFrame()
        tau_sensitivity_details = pd.DataFrame()
        tau_sensitivity_stage_wall_s = 0.0
        if bool(args.tau_sensitivity):
            tau_started = time.perf_counter()
            tau_sensitivity_summary, tau_sensitivity_details = tau_sensitivity_study(
                validation_truth=final_validation_truth,
                operators=operators,
                thresholds=[float(value) for value in args.tau_sensitivity_values],
                training_materials=int(stop_materials),
            )
            tau_sensitivity_summary.to_csv(
                run_dir / "tau_G_sensitivity_summary.csv", index=False
            )
            tau_sensitivity_details.to_csv(
                run_dir / "tau_G_sensitivity_details.csv", index=False
            )
            tau_sensitivity_stage_wall_s = float(time.perf_counter() - tau_started)

        structural_audit_truth = pd.DataFrame()
        structural_audit_results = pd.DataFrame()
        structural_verification_status: dict[str, Any] = {
            "performed": False,
            "nominal_variant": "nominal_float64",
            "identity_relative_mismatch_limit": 1.0e-8,
            "ritz_energy_psd_relative_tolerance": 1.0e-12,
            "preservation_verified": None,
        }
        structural_audit_stage_wall_s = 0.0
        audit_recompile_wall_s = 0.0
        if bool(args.structural_audit):
            audit_started = time.perf_counter()
            diagnostic_seed = int(
                run_pipeline_config.get("diagnostic_validation_seed", 20260824)
            )
            configured_cases = run_pipeline_config.get("diagnostic_audit_cases", {})
            exact_ids = [
                int(value)
                for value in configured_cases.get(str(int(args.geometry_id)), [])
            ]
            audit_basis32 = None
            precision_variants: dict[str, dict[str, Any]] = {}
            precision_snapshot_rows = pd.DataFrame()
            if exact_ids:
                audit_recompile_started = time.perf_counter()
                audit_basis32, precision_variants, precision_snapshot_rows = (
                    build_float32_snapshot_audit_variants(
                    run_dir=run_dir,
                    geometry=geometry,
                    runtime=runtime,
                    candidates=candidates,
                    selected_candidate_ids=selected["candidate_id"].astype(int).tolist(),
                    candidate_seed=int(args.candidate_seed),
                    quiet_solver=bool(args.quiet_solver),
                    voxel_order=voxel_order,
                    operator_phase=operator_phase,
                    operator_ori=operator_ori,
                    affine_stress_batch=affine,
                    affine_stress_max_gib=float(args.affine_stress_max_gib),
                    memory_safety_fraction=float(args.memory_safety_fraction),
                    basis_tolerance=float(args.basis_tolerance),
                    gram_rank_rtol=float(args.ritz_gram_rank_rtol),
                    )
                )
                precision_snapshot_rows.to_csv(
                    run_dir / "structural_audit_float32_snapshot_results.csv",
                    index=False,
                )
                audit_recompile_wall_s = float(
                    time.perf_counter() - audit_recompile_started
                )
            audit_records: list[dict[str, Any]] = []
            if exact_ids:
                diagnostic_pool = independent_pool(
                    max(4096, max(exact_ids) + 1),
                    diagnostic_seed,
                    id_column="diagnostic_pool_id",
                    label_prefix="diagnostic_sobol",
                )
                for material_id in exact_ids:
                    record = diagnostic_pool.loc[
                        diagnostic_pool["material_id"] == material_id
                    ].iloc[0].to_dict()
                    record["audit_source"] = "prespecified_20260824_case"
                    record["audit_seed"] = diagnostic_seed
                    audit_records.append(record)

            eta_threshold = float(
                run_pipeline_config.get("schur_audit_eta_threshold", -1.0e-5)
            )
            eta = final_validation_rom["schur_eta"].to_numpy(dtype=float)
            fresh_indices = set(np.flatnonzero(eta < eta_threshold).tolist())
            finite_eta = np.flatnonzero(np.isfinite(eta))
            if len(finite_eta):
                fresh_indices.add(int(finite_eta[np.argmin(eta[finite_eta])]))
            for index in sorted(fresh_indices):
                record = final_validation.iloc[int(index)].to_dict()
                record["audit_source"] = (
                    "fresh_minimum"
                    if int(index) == int(finite_eta[np.argmin(eta[finite_eta])])
                    else "fresh_eta_below_minus_1e5"
                )
                record["audit_seed"] = int(args.final_validation_seed)
                audit_records.append(record)

            deduplicated: dict[tuple[str, int], dict[str, Any]] = {}
            for record in audit_records:
                key = (str(record["audit_source"]), int(record["material_id"]))
                deduplicated[key] = record
            audit_materials = list(deduplicated.values())
            truth_rows: list[dict[str, Any]] = []
            audit_frames: list[pd.DataFrame] = []
            for audit_index, material in enumerate(
                audit_materials, start=1
            ):
                print(
                    f"[SOBOL-POD] structural audit {audit_index}/"
                    f"{len(audit_materials)} | material={int(material['material_id'])}",
                    flush=True,
                )
                truth_record, truth_fields = solve_structural_audit_material(
                    run_dir=run_dir,
                    geometry=geometry,
                    runtime=runtime,
                    material=material,
                    seed=int(material["audit_seed"]),
                    profile=str(args.audit_profile),
                    quiet_solver=bool(args.quiet_solver),
                    audit_tag=(
                        f"{audit_index - 1:02d}_{material['audit_source']}_"
                        f"material_{int(material['material_id']):04d}"
                    ),
                )
                truth_record["audit_source"] = str(material["audit_source"])
                truth_record["audit_seed"] = int(material["audit_seed"])
                truth_rows.append(truth_record)
                operator_variants = {"nominal_float64": operators}
                if str(material["audit_source"]).startswith("prespecified_"):
                    operator_variants.update(precision_variants)
                audit_frame = ritz_schur_energy_audit(
                    truth_record=truth_record,
                    fom_fields=truth_fields,
                    material=material,
                    raw_basis=basis.active_fields,
                    voxel_order=voxel_order,
                    affine_stress_batch=affine,
                    operator_variants=operator_variants,
                )
                audit_frame.insert(0, "audit_id", audit_index - 1)
                audit_frame.insert(1, "audit_source", str(material["audit_source"]))
                audit_frames.append(audit_frame)
                del truth_fields
                gc.collect()
            del audit_basis32
            (run_dir / "float32_snapshot_audit_basis.dat").unlink(missing_ok=True)
            structural_audit_truth = pd.DataFrame(truth_rows)
            structural_audit_results = pd.concat(audit_frames, ignore_index=True)
            structural_audit_truth.to_csv(
                run_dir / "structural_audit_truth_results.csv", index=False
            )
            structural_audit_results.to_csv(
                run_dir / "structural_audit_ritz_schur_results.csv", index=False
            )
            nominal_audits = structural_audit_results.loc[
                structural_audit_results["rom_variant"] == "nominal_float64"
            ].copy()
            structural_verification_status = {
                "performed": True,
                "nominal_variant": "nominal_float64",
                "identity_relative_mismatch_limit": 1.0e-8,
                "ritz_energy_psd_relative_tolerance": 1.0e-12,
                "audited_material_count": int(len(nominal_audits)),
                "preservation_verified": bool(
                    len(nominal_audits)
                    and nominal_audits[
                        "ritz_schur_preservation_verified"
                    ].astype(bool).all()
                ),
                "maximum_energy_identity_relative_mismatch": float(
                    nominal_audits["energy_identity_relative_mismatch"].max()
                ),
                "minimum_ritz_error_energy_eigenvalue": float(
                    nominal_audits["ritz_error_energy_min_eig"].min()
                ),
                "minimum_schur_eta": float(nominal_audits["schur_eta"].min()),
                "results_csv": str(
                    run_dir / "structural_audit_ritz_schur_results.csv"
                ),
            }
            write_json(
                run_dir / "structural_verification_status.json",
                structural_verification_status,
            )
            structural_audit_stage_wall_s = float(time.perf_counter() - audit_started)

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
            repetitions=int(args.rom_timing_repetitions),
        )
        rom_cloud.to_csv(run_dir / "candidate_rom_cloud.csv", index=False)
        write_json(run_dir / "rom_timing_summary.json", rom_timing)
        rom_timing_stage_wall_s = float(time.perf_counter() - rom_timing_started)

        del basis
        gc.collect()

        plot_path = run_dir / "sobol_pod_error_curve.png"
        if bool(args.write_plot) and not fixed_training_protocol:
            write_plot(curve, plot_path, float(args.target_error))
        timing = curve[
            [
                "training_materials",
                "pod_rank",
                "snapshot_solve_wall_s",
                "snapshot_step_wall_s",
                "basis_update_wall_s",
                "local_frame_transform_wall_s",
                "operator_assembly_wall_s",
                "affine_stress_wall_s",
                "ritz_contraction_wall_s",
                "monitor_rom_cumulative_wall_s",
                "rom_online_mean_s",
            ]
        ].copy()
        timing["monitor_fft_total_wall_s"] = (
            0.0
            if monitor_truth.empty
            else float(monitor_truth["solve_wall_s"].sum())
        )
        timing["final_validation_fft_total_wall_s"] = float(
            final_validation_truth["solve_wall_s"].sum()
        )
        snapshot_timing = pd.DataFrame(snapshot_rows)
        active_q_blocks = snapshot_timing.loc[
            snapshot_timing["affine_q_block_size"] > 0,
            "affine_q_block_size",
        ]
        compilation_wall_s = float(
            geometry_load_wall_s
            + affine_setup_wall_s
            + monitor_fft_stage_wall_s
            + training_stage_wall_s
            + reference_energy_qr_stage_wall_s
        )
        fom_material_median_s = float(fom_timing_results["solve_wall_s"].median())
        rom_material_median_s = float(rom_timing["warm_single_query_median_s"])
        hot_single_speedup = float(
            fom_material_median_s
            / max(rom_material_median_s, np.finfo(float).tiny)
        )
        break_even_queries = (
            int(
                math.ceil(
                    compilation_wall_s
                    / (fom_material_median_s - rom_material_median_s)
                )
            )
            if fom_material_median_s > rom_material_median_s
            else None
        )
        stage_rows = [
            {"stage": "geometry_load", "wall_s": geometry_load_wall_s},
            {"stage": "affine_setup", "wall_s": affine_setup_wall_s},
        ]
        if not fixed_training_protocol:
            stage_rows.append(
                {"stage": "adaptive_monitor_fft_once", "wall_s": monitor_fft_stage_wall_s}
            )
        stage_rows.extend(
            [
                {
                    "stage": (
                        "fixed_sobol_fft_pod_kbd"
                        if fixed_training_protocol
                        else "adaptive_sobol_fft_pod_kbd"
                    ),
                    "wall_s": training_stage_wall_s,
                },
                {
                    "stage": "experimental_full_span_tsqr_audit",
                    "wall_s": experimental_qr_stage_wall_s,
                },
                {
                    "stage": "reference_energy_qr_freeze",
                    "wall_s": reference_energy_qr_stage_wall_s,
                },
                {
                    "stage": "final_independent_fft_validation",
                    "wall_s": validation_stage_wall_s,
                },
                {"stage": "fom_timing_materials", "wall_s": fom_timing_stage_wall_s},
                {"stage": "tau_G_fixed_prefix_sensitivity", "wall_s": tau_sensitivity_stage_wall_s},
                {"stage": "high_precision_structural_audit", "wall_s": structural_audit_stage_wall_s},
                {"stage": "rom_timing_queries", "wall_s": rom_timing_stage_wall_s},
            ]
        )
        stage_timing = pd.DataFrame(stage_rows)
        excel_sheets = {
            "training": curve,
            "timing": timing,
            "stage_timing": stage_timing,
            "sequence": selected,
            "validation_design": final_validation,
            "final_validation_truth": final_validation_truth,
            "final_validation_rom": final_validation_rom,
            "fom_timing": fom_timing_results,
        }
        if not monitor_truth.empty:
            excel_sheets["monitor_truth"] = monitor_truth
            excel_sheets["monitor_rom"] = monitor_rom
        if not tau_sensitivity_summary.empty:
            excel_sheets["tau_G_sensitivity"] = tau_sensitivity_summary
        if not structural_audit_results.empty:
            excel_sheets["structural_audit"] = structural_audit_results
        if not energy_pod_summary.empty:
            excel_sheets["energy_POD_selection"] = energy_pod_summary
        if not energy_pod_validation.empty:
            excel_sheets["energy_POD_validation"] = energy_pod_validation
        if not experimental_qr_monitor.empty:
            excel_sheets["experimental_QR_monitor"] = experimental_qr_monitor
        if not experimental_qr_validation.empty:
            excel_sheets["experimental_QR_validation"] = (
                experimental_qr_validation
            )
        write_excel(run_dir / "sobol_pod_paper_tables.xlsx", excel_sheets)

        summary = {
            "run_dir": str(run_dir),
            "status": "complete",
            "method": (
                "fixed_sobol_pod_full_rank"
                if fixed_training_protocol
                else "adaptive_sobol_pod_full_rank"
            ),
            "adaptive": bool(args.adaptive),
            "adaptive_target_reached": adaptive_target_reached,
            "adaptive_stop_reason": (
                "fixed_training_complete"
                if fixed_training_protocol
                else (
                    "monitor_target_reached"
                    if adaptive_target_reached
                    else "training_limit_reached_without_monitor_pass"
                )
            ),
            **geometry_info,
            "nvox": int(geometry.phase.size),
            "voxel_shape": list(geometry.phase.shape),
            "selection_policy": str(args.selection_policy),
            "training_limit": int(training_limit),
            "memory_safe_material_limit": int(safe_material_limit),
            "final_selected_materials": int(stop_materials),
            "basis_rank": final_basis_rank,
            "basis_dtype": str(args.basis_dtype),
            "full_rank_basis_mode": str(args.full_rank_basis_mode),
            "ritz_contraction_dtype": str(args.ritz_contraction_dtype),
            "ritz_gram_compute_dtype": str(args.ritz_gram_compute_dtype),
            "ritz_gram_backend": str(args.ritz_gram_backend),
            "ritz_gram_rank_rtol": float(args.ritz_gram_rank_rtol),
            "overlap_cpu_gram_gpu": bool(args.overlap_cpu_gram_gpu),
            "reference_energy_qr_enabled": bool(args.reference_energy_qr),
            "factorized_ritz_enabled": bool(
                args.factorized_ritz
            ),
            "async_ritz_enabled": bool(args.async_ritz),
            "experimental_local_frame_ritz_enabled": bool(
                args.experimental_local_frame_ritz
            ),
            "experimental_gathered_factor_ritz_enabled": bool(
                args.experimental_gathered_factor_ritz
            ),
            "reference_energy_qr_reference_policy": "candidate_affine_mean",
            "reference_energy_qr_reference_coefficients": (
                reference_energy_qr_reference
                if bool(args.reference_energy_qr)
                else None
            ),
            "reference_energy_qr_metadata": reference_energy_qr_metadata,
            "reference_energy_qr_monitor_raw_coordinate_difference": (
                reference_energy_qr_monitor_difference
            ),
            "reference_energy_qr_stage_wall_s": (
                reference_energy_qr_stage_wall_s
            ),
            "experimental_qr_audit_enabled": bool(args.experimental_qr_audit),
            "experimental_qr_block_max_gib": float(
                args.experimental_qr_block_max_gib
            ),
            "experimental_qr_model_sha256": experimental_qr_model_hash,
            "experimental_qr_metadata": experimental_qr_metadata,
            "experimental_qr_monitor_summary": experimental_qr_monitor_stats,
            "experimental_qr_monitor_nominal_difference": (
                experimental_qr_monitor_difference
            ),
            "experimental_qr_validation_summary": (
                experimental_qr_validation_stats
            ),
            "experimental_qr_validation_nominal_difference": (
                experimental_qr_validation_difference
            ),
            "experimental_qr_stage_wall_s": experimental_qr_stage_wall_s,
            "frozen_model_sha256": frozen_model_hash,
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
            "final_validation_pool_count": int(len(validation_candidates)),
            "final_validation_seed": int(args.final_validation_seed),
            "final_validation_design": "affine_coefficient_maximin",
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
            "final_validation_below_1e3_count": int(
                final_stats["coverage_1e3_count"]
            ),
            "final_validation_below_1e3_percent": float(
                final_stats["coverage_1e3_percent"]
            ),
            "energy_pod_baseline_enabled": bool(args.energy_pod_baseline),
            "energy_pod_model_sha256": energy_pod_model_hash,
            "energy_pod_monitor_selection": (
                None
                if energy_pod_summary.empty
                else energy_pod_summary.to_dict(orient="records")
            ),
            "energy_pod_final_validation_summary": energy_pod_validation_stats,
            "target_error": float(args.target_error),
            "training_profile": str(args.training_profile),
            "monitor_profile": str(args.monitor_profile),
            "validation_profile": str(args.validation_profile),
            "timing_profile": str(args.timing_profile),
            "audit_profile": str(args.audit_profile),
            "compilation_wall_s": compilation_wall_s,
            "fom_material_median_s": fom_material_median_s,
            "rom_material_median_s": rom_material_median_s,
            "hot_single_speedup": hot_single_speedup,
            "throughput_speedup": hot_single_speedup,
            "break_even_queries": break_even_queries,
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
            "snapshot_gram_product_total_wall_s": float(
                snapshot_timing["gram_product_wall_s"].sum()
            ),
            "snapshot_gram_overlap_wait_total_wall_s": float(
                snapshot_timing["gram_overlap_wait_wall_s"].sum()
            ),
            "snapshot_gram_overlap_hidden_total_wall_s": float(
                snapshot_timing["gram_overlap_hidden_wall_s"].sum()
            ),
            "geometry_load_wall_s": geometry_load_wall_s,
            "affine_setup_wall_s": affine_setup_wall_s,
            "monitor_fft_total_wall_s": (
                0.0
                if monitor_truth.empty
                else float(monitor_truth["solve_wall_s"].sum())
            ),
            "monitor_fft_stage_wall_s": monitor_fft_stage_wall_s,
            "monitor_rom_total_wall_s": monitor_rom_total_wall_s,
            "training_stage_wall_s": training_stage_wall_s,
            "final_validation_fft_total_wall_s": float(
                final_validation_truth["solve_wall_s"].sum()
            ),
            "final_validation_stage_wall_s": validation_stage_wall_s,
            "fom_timing_stage_wall_s": fom_timing_stage_wall_s,
            "tau_sensitivity_enabled": bool(args.tau_sensitivity),
            "tau_sensitivity_stage_wall_s": tau_sensitivity_stage_wall_s,
            "structural_audit_enabled": bool(args.structural_audit),
            "structural_audit_count": int(len(structural_audit_truth)),
            "structural_audit_recompile_wall_s": audit_recompile_wall_s,
            "structural_audit_stage_wall_s": structural_audit_stage_wall_s,
            "structural_verification": structural_verification_status,
            "rom_timing_stage_wall_s": rom_timing_stage_wall_s,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "pipeline_wall_s_before_final_write": float(time.perf_counter() - pipeline_started),
            "final_validation_summary": final_stats,
            "fft_backend": str(args.fft_backend),
            "rom_timing": rom_timing,
            "affine_constitutive_reconstruction": reconstruction_check,
            "curve_csv": str(run_dir / "sobol_pod_error_curve.csv"),
            "monitor_truth_csv": (
                None
                if monitor_truth.empty
                else str(run_dir / "monitor_truth_results.csv")
            ),
            "monitor_rom_csv": (
                None
                if monitor_rom.empty
                else str(run_dir / "monitor_rom_results.csv")
            ),
            "final_validation_truth_csv": str(
                run_dir / "final_validation_truth_results.csv"
            ),
            "final_validation_rom_csv": str(
                run_dir / "final_validation_rom_results.csv"
            ),
            "energy_pod_monitor_selection_csv": (
                str(run_dir / "energy_pod_monitor_selection.csv")
                if not energy_pod_summary.empty
                else None
            ),
            "energy_pod_final_validation_csv": (
                str(run_dir / "energy_pod_final_validation.csv")
                if not energy_pod_validation.empty
                else None
            ),
            "experimental_qr_monitor_csv": (
                str(run_dir / "experimental_qr_monitor_results.csv")
                if not experimental_qr_monitor.empty
                else None
            ),
            "experimental_qr_final_validation_csv": (
                str(run_dir / "experimental_qr_final_validation.csv")
                if not experimental_qr_validation.empty
                else None
            ),
            "reference_energy_qr_monitor_csv": (
                str(run_dir / "reference_energy_qr_monitor_results.csv")
                if bool(args.reference_energy_qr) and not monitor_truth.empty
                else None
            ),
            "fom_timing_csv": str(run_dir / "fom_timing_results.csv"),
            "tau_sensitivity_summary_csv": (
                str(run_dir / "tau_G_sensitivity_summary.csv")
                if not tau_sensitivity_summary.empty
                else None
            ),
            "structural_audit_csv": (
                str(run_dir / "structural_audit_ritz_schur_results.csv")
                if not structural_audit_results.empty
                else None
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
