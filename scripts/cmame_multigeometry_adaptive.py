#!/usr/bin/env python3
"""Fast causal multi-geometry ROM validation for the CMAME study.

The protocol is intentionally economical:

1. build a fixed Sobol material pool for cheap ROM screening;
2. for each fixed geometry, collect an initial Sobol block of full-order
   snapshots;
3. enrich in small Schur-greedy batches without candidate FOM errors;
4. validate the frozen operator on independent FFT truth materials; and
5. show the validation points inside a large ROM-query cloud.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import cmame_campaign_common as common
import schur_estimator_compiler as compile_estimator
import schur_energy_indicators as qoi
import rom_reduced_operator as reduced
import rom_validation_utils as validate
from constitutive_transfer.schur_estimator import IncrementalTwoKernelCompiler


GEOMETRY_DIR_DEFAULT = ROOT / "results" / "cmame_method" / "geometries_binary_res6"
OUT_DEFAULT = ROOT / "results" / "cmame_method" / "multigeometry_validation_binary_res6_optimized"
PHYSICAL_NAMES = tuple(common.sweep.MATERIAL_BOUNDS)
TARGET_ERROR = 1.0e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _material_key(frame: pd.DataFrame) -> set[tuple[float, ...]]:
    return {
        tuple(float(row[name]) for name in PHYSICAL_NAMES)
        for row in frame.to_dict(orient="records")
    }


def _candidate_pool(count: int, seed: int) -> pd.DataFrame:
    frame = validate._build_independent_materials(int(count), int(seed)).copy()
    frame.insert(0, "candidate_id", np.arange(len(frame), dtype=int))
    frame["material_id"] = frame["candidate_id"].astype(int)
    frame["material_label"] = [
        f"candidate_sobol_{idx:05d}" for idx in frame["candidate_id"].astype(int)
    ]
    return frame


def _validation_pool(
    count: int,
    seed: int,
    *,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    frame = validate._build_independent_materials(int(count), int(seed)).copy()
    frame.insert(0, "validation_id", np.arange(len(frame), dtype=int))
    frame["material_id"] = frame["validation_id"].astype(int)
    frame["material_label"] = [
        f"validation_sobol_{idx:04d}" for idx in frame["validation_id"].astype(int)
    ]
    overlap = _material_key(frame).intersection(_material_key(candidates))
    if overlap:
        raise RuntimeError("Candidate and held-out validation material pools overlap.")
    return frame


def _ensure_material_design(
    *,
    out_dir: Path,
    candidate_count: int,
    validation_count: int,
    candidate_seed: int,
    validation_seed: int,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_path = out_dir / "candidate_pool.csv"
    validation_path = out_dir / "held_out_validation_pool.csv"
    manifest_path = out_dir / "material_design_manifest.json"
    if (
        candidate_path.is_file()
        and validation_path.is_file()
        and manifest_path.is_file()
        and not overwrite
    ):
        candidates = pd.read_csv(candidate_path)
        validation = pd.read_csv(validation_path)
        return candidates, validation

    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_pool(candidate_count, candidate_seed)
    validation = _validation_pool(
        validation_count,
        validation_seed,
        candidates=candidates,
    )
    candidates.to_csv(candidate_path, index=False)
    validation.to_csv(validation_path, index=False)
    common.write_json(
        manifest_path,
        {
            "status": "frozen_material_design_no_fom_errors",
            "candidate_count": int(len(candidates)),
            "held_out_validation_count": int(len(validation)),
            "candidate_seed": int(candidate_seed),
            "validation_seed": int(validation_seed),
            "candidate_pool_sha256": _sha256(candidate_path),
            "held_out_validation_pool_sha256": _sha256(validation_path),
            "candidate_validation_overlap_count": 0,
            "selection_may_use_candidate_fom_errors": False,
        },
    )
    return candidates, validation


def _axis_name(index: int) -> str:
    return ("x", "y", "z")[int(index)]


def _morphology_label(row: pd.Series | dict[str, Any]) -> str:
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    vf = float(data.get("Vf_realized", data.get("Vf_target", np.nan)))
    ar = float(data.get("aspect_ratio", data.get("AR", np.nan)))
    cluster = float(data.get("cluster_fraction_target", data.get("cluster_fraction", 0.0)))
    a2 = np.array(
        [
            float(data.get("A2_11_realized", data.get("A2_11", 1.0 / 3.0))),
            float(data.get("A2_22_realized", data.get("A2_22", 1.0 / 3.0))),
            float(data.get("A2_33_realized", data.get("A2_33", 1.0 / 3.0))),
        ],
        dtype=float,
    )
    density = "dilute" if vf <= 0.13 else "dense" if vf >= 0.25 else "mid-vf"
    length = "short" if ar <= 10.0 else "long" if ar >= 20.0 else "medium-ar"
    spatial = "clustered" if cluster >= 0.50 else "uniform" if cluster <= 0.10 else "mild-cluster"
    if float(a2.max()) >= 0.65:
        orient = f"aligned-{_axis_name(int(a2.argmax()))}"
    elif float(a2.min()) <= 0.10:
        orient = f"planar-normal-{_axis_name(int(a2.argmin()))}"
    else:
        orient = "triaxial"
    return f"{density}_{length}_{spatial}_{orient}"


def _geometry_design_row(
    *,
    geometry_id: int,
    geometry_dir: Path,
    design: pd.DataFrame,
    descriptors: pd.DataFrame,
) -> dict[str, Any]:
    design_row = design.loc[design["geometry_id"].astype(int) == int(geometry_id)]
    descriptor_row = descriptors.loc[
        descriptors["geometry_id"].astype(int) == int(geometry_id)
    ]
    if len(design_row) != 1 or len(descriptor_row) != 1:
        raise KeyError(f"Missing design/descriptors for geometry_{geometry_id:02d}.")
    row = design_row.iloc[0].to_dict()
    desc = descriptor_row.iloc[0].to_dict()
    generation_path = geometry_dir / f"geometry_{geometry_id:02d}" / "generation_result.json"
    generation = _read_json(generation_path) if generation_path.is_file() else {}
    box_um = float(row.get("box_um", row.get("caja_um", 18.059971)))
    grid_size = int(round(float(row.get("grid_size", row.get("nvox", 91)))))
    fiber_diameter = float(row.get("fiber_diameter_um", row.get("d_um", 1.0)))
    fiber_length = float(row.get("fiber_length_um", row.get("L_um", row["aspect_ratio"])))
    ar = float(row.get("aspect_ratio", fiber_length / max(fiber_diameter, 1.0e-12)))
    return {
        "config_id": "cmame_multigeometry_validation",
        "label": str(row.get("geometry_label", f"geometry_{geometry_id:02d}")),
        "design_id": int(geometry_id),
        "sobol_index": int(geometry_id),
        "is_operable": True,
        "reject_reason": "",
        "BOX_FACTOR": float(box_um / max(fiber_length, 1.0e-12)),
        "DF_VOXEL_TARGET": float(grid_size / box_um * fiber_diameter),
        "NVOX_MULTIPLE": 1,
        "caja_um": box_um,
        "nvox": grid_size,
        "nvox_ref": grid_size,
        "res": float(grid_size / box_um),
        "res_ref": float(grid_size / box_um),
        "voxel_um": float(box_um / grid_size),
        "df_voxel": float(fiber_diameter * grid_size / box_um),
        "d_um": fiber_diameter,
        "L_um": fiber_length,
        "fiber_length_lf": fiber_length,
        "AR": ar,
        "Lf_Ldom": float(fiber_length / box_um),
        "Vf_target": float(row["Vf_target"]),
        "a11": float(row["A2_11"]),
        "a22": float(row["A2_22"]),
        "a33": float(row["A2_33"]),
        "target_fibers_nominal": int(generation.get("n_fibers", desc.get("n_fibers", -1))),
    }


def _geometry_cases(geometry_dir: Path, geometry_ids: list[int] | None) -> list[dict[str, Any]]:
    design = pd.read_csv(geometry_dir / "geometry_design.csv")
    descriptors = pd.read_csv(geometry_dir / "geometry_realized_descriptors.csv")
    selected = (
        {int(value) for value in geometry_ids}
        if geometry_ids is not None
        else {int(value) for value in descriptors["geometry_id"]}
    )
    cases: list[dict[str, Any]] = []
    for _, desc_row in descriptors.sort_values("geometry_id").iterrows():
        geometry_id = int(desc_row["geometry_id"])
        if geometry_id not in selected:
            continue
        case_dir = geometry_dir / f"geometry_{geometry_id:02d}"
        phase_path = case_dir / "phase.npy"
        ori_path = case_dir / "ori.npy"
        if not phase_path.is_file() or not ori_path.is_file():
            raise FileNotFoundError(f"Missing phase/ori for {case_dir}.")
        cases.append(
            {
                "geometry_id": geometry_id,
                "case_id": f"geometry_{geometry_id:02d}",
                "case_dir": case_dir,
                "morphology_label": _morphology_label(desc_row),
                "design_row": _geometry_design_row(
                    geometry_id=geometry_id,
                    geometry_dir=geometry_dir,
                    design=design,
                    descriptors=descriptors,
                ),
                **desc_row.to_dict(),
            }
        )
    if not cases:
        raise RuntimeError("No geometry cases selected.")
    return cases


def _load_geometry(case: dict[str, Any]) -> common.GeometryData:
    case_dir = Path(case["case_dir"]).resolve()
    phase_path = case_dir / "phase.npy"
    ori_path = case_dir / "ori.npy"
    phase = np.load(phase_path).astype(np.uint8)
    ori = np.load(ori_path).astype(np.float32)
    manifest = {
        "phase_sha256": _sha256(phase_path),
        "ori_sha256": _sha256(ori_path),
        "grid_shape": list(phase.shape),
        "case_id": str(case["case_id"]),
        "morphology_label": str(case["morphology_label"]),
    }
    return common.GeometryData(
        source_run_dir=case_dir.parent,
        geometry_dir=case_dir,
        design_row=dict(case["design_row"]),
        manifest=manifest,
        phase=phase,
        ori=ori,
    )


def _load_basis(method_dir: Path) -> list[np.ndarray]:
    return [
        np.load(path).astype(np.float64)
        for path in sorted((method_dir / "basis_fields").glob("basis_*.npy"))
    ]


def _truncate_operators(operators: dict[str, np.ndarray], rank: int) -> dict[str, np.ndarray]:
    rank = int(rank)
    return {
        "Kq": operators["Kq"][:, :rank, :rank],
        "Bq": operators["Bq"][:, :rank, :],
        "Dq": operators["Dq"],
    }


def _truncate_estimator(estimator: Any, rank: int) -> Any:
    rank = int(rank)
    if rank >= int(estimator.rank):
        return estimator
    return estimator.__class__(
        estimator.BB,
        estimator.BK[..., :rank],
        estimator.KK[..., :rank, :rank],
        estimator.coefficient_names,
    )


def _score_candidate_frame(
    *,
    frame: pd.DataFrame,
    operators: dict[str, np.ndarray],
    estimator: Any,
    rank: int,
    progress_label: str,
) -> tuple[pd.DataFrame, float]:
    started = time.perf_counter()
    if frame.empty:
        return pd.DataFrame(), 0.0

    # All candidates use the same compiled operators and reference family.
    # Keeping this as one batched dense operation removes the Python loop and
    # evaluates the 129 reference spectra in a single array expression.
    rows = frame.to_dict(orient="records")
    coeffs = np.stack(
        [reduced._material_coefficients(row) for row in rows],
        axis=0,
    )
    C_rom, amplitudes, rom_wall = reduced._rom_ceff_batch(
        coeffs,
        operators["Kq"],
        operators["Bq"],
        operators["Dq"],
    )
    kernels = estimator.kernel_energy_matrices_batch(coeffs, amplitudes)
    matrix_stiffness, fiber_stiffness = qoi._local_phase_stiffnesses_batch(coeffs)
    selection = qoi.optimize_isotropic_reference_batch(
        kernels,
        (matrix_stiffness, fiber_stiffness),
        beta_margin=1.0e-10,
    )
    energy_matrix = selection["energy_matrices"]
    upper_bound_matrix = selection["upper_bound_matrices"]
    rom_norm = np.maximum(np.linalg.norm(C_rom, axis=(-2, -1)), np.finfo(float).eps)
    energy_abs = np.linalg.norm(energy_matrix, axis=(-2, -1))
    bound_abs = np.linalg.norm(upper_bound_matrix, axis=(-2, -1))
    rom_eigenvalues = np.linalg.eigvalsh(C_rom)
    candidate_count = len(frame)
    elapsed = float(time.perf_counter() - started)
    per_candidate_rom = float(rom_wall) / float(candidate_count)
    records: dict[str, Any] = {
        "candidate_id": frame["candidate_id"].astype(int).to_numpy(copy=True),
        "material_id": frame["material_id"].astype(int).to_numpy(copy=True),
        "material_label": frame["material_label"].astype(str).to_numpy(copy=True),
        "energy_indicator_abs": energy_abs,
        "energy_indicator_rel": energy_abs / rom_norm,
        "schur_bound_indicator_abs": bound_abs,
        "schur_bound_indicator_rel": bound_abs / rom_norm,
        "field_residual_norm": np.full(candidate_count, np.nan),
        "field_residual_rel": np.full(candidate_count, np.nan),
        "true_relative_error": np.full(candidate_count, np.nan),
        "rom_online_s": np.full(candidate_count, per_candidate_rom),
        "indicator_wall_s": np.full(candidate_count, elapsed / float(candidate_count)),
        "batch_rom_wall_s": np.full(candidate_count, float(rom_wall)),
        "batch_indicator_wall_s": np.full(candidate_count, elapsed),
        "indicator_mode": np.full(candidate_count, "dense", dtype=object),
        "load_batch_size": np.full(candidate_count, 6, dtype=int),
        "reference_lambda0": selection["lambda0"],
        "reference_mu0": selection["mu0"],
        "reference_poisson0": selection["poisson0"],
        "reference_index": selection["index"],
        "reference_strategy": np.full(
            candidate_count, "optimized_129_isotropic_shapes", dtype=object
        ),
        "beta": selection["beta"],
        "beta_safe": selection["beta_safe"],
        "rom_min_eig": rom_eigenvalues[:, 0],
        "rom_max_eig": rom_eigenvalues[:, -1],
        "candidate_score": bound_abs / rom_norm,
        "score_mode": np.full(candidate_count, "bound", dtype=object),
        "mixed_indicator_rel": bound_abs / rom_norm,
    }
    for column in frame.columns:
        if column not in records:
            records[column] = frame[column].to_numpy(copy=True)
    ranking = pd.DataFrame(records).sort_values(
        ["candidate_score", "candidate_id"], ascending=[False, True]
    )
    ranking.insert(0, "score_rank", np.arange(1, len(ranking) + 1, dtype=int))
    return ranking, float(time.perf_counter() - started)


def _write_ranking(path: Path, ranking: pd.DataFrame) -> None:
    keep = [
        "score_rank",
        "candidate_id",
        "material_id",
        "candidate_score",
        "schur_bound_indicator_rel",
        "energy_indicator_rel",
        "reference_index",
        "reference_poisson0",
        "beta_safe",
        "rom_online_s",
        "indicator_wall_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    ranking[[column for column in keep if column in ranking.columns]].to_csv(
        path,
        index=False,
    )


def _score_candidates_batch(
    *,
    candidates: pd.DataFrame,
    selected_ids: set[int],
    operators: dict[str, np.ndarray],
    estimator: Any,
    rank: int,
    output_path: Path,
    batch_size: int,
    prefilter_rank: int,
    exact_score_limit: int,
) -> tuple[list[int], pd.DataFrame, dict[str, Any]]:
    remaining = candidates.loc[~candidates["candidate_id"].isin(selected_ids)].copy()
    if remaining.empty:
        return [], pd.DataFrame(), {"candidate_scan_wall_s": 0.0, "max_candidate_score": np.nan}
    started = time.perf_counter()
    exact_frame = remaining
    prefilter_wall = 0.0
    prefilter_count = 0
    exact_limit = int(exact_score_limit)
    pre_rank = max(1, min(int(prefilter_rank), int(rank)))
    use_prefilter = (
        exact_limit > 0
        and len(remaining) > exact_limit
        and int(rank) > pre_rank
    )
    if use_prefilter:
        prefilter_path = output_path.with_name(
            f"{output_path.stem}_prefilter_r{pre_rank}{output_path.suffix}"
        )
        prefilter_ranking, prefilter_wall = _score_candidate_frame(
            frame=remaining,
            operators=_truncate_operators(operators, pre_rank),
            estimator=_truncate_estimator(estimator, pre_rank),
            rank=pre_rank,
            progress_label=f"prefilter-r{pre_rank}",
        )
        _write_ranking(prefilter_path, prefilter_ranking)
        prefilter_count = int(len(prefilter_ranking))
        exact_ids = set(
            int(value)
            for value in prefilter_ranking["candidate_id"].iloc[:exact_limit].to_list()
        )
        exact_frame = remaining.loc[remaining["candidate_id"].isin(exact_ids)].copy()

    exact_ranking, exact_wall = _score_candidate_frame(
        frame=exact_frame,
        operators=operators,
        estimator=estimator,
        rank=int(rank),
        progress_label=f"exact-r{int(rank)}",
    )
    _write_ranking(output_path, exact_ranking)
    scan_wall = float(time.perf_counter() - started)
    selected = [
        int(value)
        for value in exact_ranking["candidate_id"].iloc[: max(1, int(batch_size))].to_list()
    ]
    meta = {
        "candidate_scan_wall_s": scan_wall,
        "candidate_count_scored": int(len(remaining)),
        "candidate_prefilter_count": int(prefilter_count),
        "candidate_exact_count": int(len(exact_ranking)),
        "candidate_prefilter_rank": int(pre_rank if use_prefilter else int(rank)),
        "candidate_prefilter_wall_s": float(prefilter_wall),
        "candidate_exact_wall_s": float(exact_wall),
        "candidate_two_stage": bool(use_prefilter),
        "max_candidate_score": float(exact_ranking["candidate_score"].iloc[0]),
        "median_candidate_score": float(exact_ranking["candidate_score"].median()),
        "p95_candidate_score": float(exact_ranking["candidate_score"].quantile(0.95)),
    }
    return selected, exact_ranking, meta


def _append_snapshot_to_basis(
    *,
    basis: list[np.ndarray],
    candidate_id: int,
    out_dir: Path,
    method_dir: Path,
    tolerance: float,
) -> list[np.ndarray]:
    before = len(basis)
    new_fields = common._append_orthonormal(
        basis,
        common.load_snapshot_fields(common._snapshot_dir(out_dir, candidate_id)),
        tolerance=float(tolerance),
    )
    if not new_fields:
        raise RuntimeError(f"Candidate {candidate_id} did not add basis directions.")
    common._save_basis_block(method_dir, before, new_fields)
    return new_fields


def _compiler_for(
    *,
    method_dir: Path,
    geometry: common.GeometryData,
    basis: list[np.ndarray],
    max_rank: int,
    atom_batch_size: int,
    feature_block: int,
    fft_workers: int,
) -> IncrementalTwoKernelCompiler:
    compiler_work = method_dir / "incremental_estimator_work"
    matrix_idx, fiber_groups = qoi._geometry_groups(geometry.phase, geometry.ori)
    resume = compiler_work.is_dir()
    state_path = compiler_work / "state.json"
    if resume:
        try:
            state = _read_json(state_path)
            expected = {
                "shape": [int(value) for value in geometry.phase.shape],
                "coefficient_names": list(reduced.COEFF_NAMES),
                "max_rank": int(max_rank),
            }
            mismatch = [
                key for key, value in expected.items()
                if state.get(key) != value
            ]
        except Exception:
            mismatch = ["state"]
        if mismatch:
            print(
                "[MULTIGEO] rebuilding incremental estimator work "
                f"for {method_dir.name}; mismatch={','.join(mismatch)}",
                flush=True,
            )
            shutil.rmtree(compiler_work)
            resume = False
    compiler = IncrementalTwoKernelCompiler(
        work_dir=compiler_work,
        shape=tuple(int(value) for value in geometry.phase.shape),
        coefficient_names=reduced.COEFF_NAMES,
        affine_stress_batch=compile_estimator._affine_stress_factory(matrix_idx, fiber_groups),
        max_rank=int(max_rank),
        atom_batch_size=int(atom_batch_size),
        feature_block=int(feature_block),
        fft_workers=int(fft_workers),
        resume=resume,
    )
    if compiler.rank == len(basis):
        return compiler
    if compiler.rank != 0:
        print(
            "[MULTIGEO] rebuilding incremental estimator work "
            f"for {method_dir.name}; compiler_rank={compiler.rank} basis_rank={len(basis)}",
            flush=True,
        )
        compiler.close()
        shutil.rmtree(compiler_work)
        compiler = IncrementalTwoKernelCompiler(
            work_dir=compiler_work,
            shape=tuple(int(value) for value in geometry.phase.shape),
            coefficient_names=reduced.COEFF_NAMES,
            affine_stress_batch=compile_estimator._affine_stress_factory(matrix_idx, fiber_groups),
            max_rank=int(max_rank),
            atom_batch_size=int(atom_batch_size),
            feature_block=int(feature_block),
            fft_workers=int(fft_workers),
            resume=False,
        )
    if compiler.rank == 0 and basis:
        compiler.append(np.stack(basis).reshape(len(basis), 6, -1))
        return compiler
    raise RuntimeError(
        f"Compiler/basis mismatch in {method_dir}: compiler={compiler.rank}, basis={len(basis)}."
    )


def _solve_selection(
    *,
    case: dict[str, Any],
    out_dir: Path,
    candidates: pd.DataFrame,
    runtime: dict[str, Any],
    initial_materials: int,
    max_materials: int,
    adaptive_batch_size: int,
    max_adaptive_rounds: int,
    prefilter_rank: int,
    exact_score_limit: int,
    target_error: float,
    stop_score_factor: float,
    orthonormal_tolerance: float,
    seed: int,
    atom_batch_size: int,
    feature_block: int,
    keep_compiler_work: bool,
    basis_profile: str,
    fft_workers: int,
) -> dict[str, Any]:
    case_out = out_dir / str(case["case_id"])
    method_dir = case_out / "qoi_schur_batched"
    method_dir.mkdir(parents=True, exist_ok=True)
    selection_path = method_dir / "selection.csv"
    summary_path = case_out / "selection_summary.json"
    final_operator = case_out / "reduced_operators.npz"
    final_estimator = case_out / "online_estimator.npz"
    if summary_path.is_file() and final_operator.is_file() and final_estimator.is_file():
        cached = _read_json(summary_path)
        if (
            cached.get("status") == "complete"
            and int(cached.get("candidate_count", -1)) == len(candidates)
            and int(cached.get("initial_materials", -1)) == int(initial_materials)
            and int(cached.get("max_materials", -1)) == int(max_materials)
            and int(cached.get("max_adaptive_rounds", -1)) == int(max_adaptive_rounds)
            and int(cached.get("exact_score_limit", -1)) == int(exact_score_limit)
            and int(cached.get("prefilter_rank", -1)) == int(prefilter_rank)
            and cached.get("basis_solver_profile") == str(basis_profile)
        ):
            print(f"[MULTIGEO] reuse selection {case['case_id']}", flush=True)
            return cached

    geometry = _load_geometry(case)
    rows = pd.read_csv(selection_path).to_dict(orient="records") if selection_path.is_file() else []
    selected_ids = {int(row["candidate_id"]) for row in rows}
    basis = _load_basis(method_dir)
    compiler = _compiler_for(
        method_dir=method_dir,
        geometry=geometry,
        basis=basis,
        max_rank=6 * int(max_materials),
        atom_batch_size=atom_batch_size,
        feature_block=feature_block,
        fft_workers=fft_workers,
    )
    operator_path = method_dir / "current_reduced_operators.npz"
    operators = common._load_operators(operator_path) if operator_path.is_file() else None
    stop_reason = "budget"
    last_scan: dict[str, Any] = {
        "candidate_scan_wall_s": 0.0,
        "max_candidate_score": np.nan,
    }
    batch_index = int(max([row.get("batch_index", 0) for row in rows], default=0))
    adaptive_rounds_done = int(
        sum(
            1
            for row in rows
            if str(row.get("selection_phase", "")) == "batched_qoi_schur"
            and float(row.get("candidate_scan_wall_s", 0.0) or 0.0) > 0.0
        )
    )

    while len(rows) < int(max_materials):
        batch_index += 1
        scan_meta = {"candidate_scan_wall_s": 0.0, "max_candidate_score": np.nan}
        ranking = pd.DataFrame()
        if len(rows) < int(initial_materials):
            remaining_initial = int(initial_materials) - len(rows)
            pool = candidates.loc[~candidates["candidate_id"].isin(selected_ids)]
            batch_ids = [
                int(value)
                for value in pool["candidate_id"].iloc[:remaining_initial].to_list()
            ]
            phase = "initial_sobol_block"
        else:
            if adaptive_rounds_done >= int(max_adaptive_rounds):
                stop_reason = "adaptive_round_budget"
                break
            if operators is None or compiler.rank != len(basis):
                raise RuntimeError("Adaptive scan requested before operators/estimator are ready.")
            ranking_path = method_dir / f"candidate_ranking_after_budget_{len(rows):02d}.csv"
            batch_ids, ranking, scan_meta = _score_candidates_batch(
                candidates=candidates,
                selected_ids=selected_ids,
                operators=operators,
                estimator=compiler.estimator(),
                rank=len(basis),
                output_path=ranking_path,
                batch_size=min(int(adaptive_batch_size), int(max_materials) - len(rows)),
                prefilter_rank=int(prefilter_rank),
                exact_score_limit=int(exact_score_limit),
            )
            last_scan = dict(scan_meta)
            score_gate = float(target_error) * float(stop_score_factor)
            if float(scan_meta["max_candidate_score"]) <= score_gate:
                stop_reason = "candidate_schur_score_gate"
                break
            adaptive_rounds_done += 1
            phase = "batched_qoi_schur"

        if not batch_ids:
            stop_reason = "candidate_pool_exhausted"
            break

        batch_new_fields: list[np.ndarray] = []
        batch_started = time.perf_counter()
        for local_index, candidate_id in enumerate(batch_ids, start=1):
            if len(rows) >= int(max_materials):
                break
            record = common._ensure_snapshot(
                candidate_id=candidate_id,
                candidates=candidates,
                out_dir=case_out,
                geometry=geometry,
                runtime=runtime,
                seed=int(seed),
                profile=str(basis_profile),
            )
            field_started = time.perf_counter()
            new_fields = _append_snapshot_to_basis(
                basis=basis,
                candidate_id=candidate_id,
                out_dir=case_out,
                method_dir=method_dir,
                tolerance=orthonormal_tolerance,
            )
            basis_update_s = float(time.perf_counter() - field_started)
            selected_ids.add(candidate_id)
            score_row = {}
            if not ranking.empty:
                match = ranking.loc[ranking["candidate_id"] == candidate_id]
                if len(match) == 1:
                    score_row = match.iloc[0].to_dict()
            rows.append(
                {
                    "method": "qoi_schur_batched",
                    "offline_index": int(len(rows) + 1),
                    "batch_index": int(batch_index),
                    "batch_local_index": int(local_index),
                    "selection_phase": phase,
                    "candidate_id": int(candidate_id),
                    "selection_uses_fom_error": False,
                    "candidate_score": float(score_row.get("candidate_score", np.nan)),
                    "schur_bound_indicator_rel": float(
                        score_row.get("schur_bound_indicator_rel", np.nan)
                    ),
                    "max_candidate_score_at_scan": float(
                        scan_meta.get("max_candidate_score", np.nan)
                    ),
                    "candidate_scan_wall_s": float(
                        scan_meta.get("candidate_scan_wall_s", 0.0)
                        if local_index == 1
                        else 0.0
                    ),
                    "candidate_prefilter_wall_s": float(
                        scan_meta.get("candidate_prefilter_wall_s", 0.0)
                        if local_index == 1
                        else 0.0
                    ),
                    "candidate_exact_wall_s": float(
                        scan_meta.get("candidate_exact_wall_s", 0.0)
                        if local_index == 1
                        else 0.0
                    ),
                    "candidate_exact_count": int(scan_meta.get("candidate_exact_count", 0)),
                    "candidate_prefilter_count": int(scan_meta.get("candidate_prefilter_count", 0)),
                    "candidate_two_stage": bool(scan_meta.get("candidate_two_stage", False)),
                    "basis_solver_profile": str(record["solver_profile"]),
                    "basis_solver_rtol": float(record["solver_rtol"]),
                    "basis_solver_all_converged": bool(record["solver_all_converged"]),
                    "basis_solver_max_relative_residual": float(
                        record["solver_max_relative_residual"]
                    ),
                    "snapshot_solve_wall_s": float(record["solve_wall_s"]),
                    "basis_update_wall_s": basis_update_s,
                    "new_directions": int(len(new_fields)),
                    "basis_rank": int(len(basis)),
                    "estimator_append_wall_s": 0.0,
                    "reduced_assembly_wall_s": 0.0,
                }
            )
            batch_new_fields.extend(new_fields)
            pd.DataFrame(rows).to_csv(selection_path, index=False)
            print(
                f"[MULTIGEO] {case['case_id']} selected={candidate_id} "
                f"budget={len(rows)}/{max_materials} r={len(basis)}",
                flush=True,
            )

        if not batch_new_fields:
            break
        append_meta = compiler.append(
            np.stack(batch_new_fields).reshape(len(batch_new_fields), 6, -1)
        )
        assembly_started = time.perf_counter()
        operators, assembly_meta = common._save_current_operators(
            operator_path,
            phase=geometry.phase,
            ori=geometry.ori.astype(np.float64),
            basis=basis,
            existing=operators,
            new_fields=batch_new_fields,
            affine_stress_batch=compiler.affine_stress_batch,
        )
        assembly_wall = float(time.perf_counter() - assembly_started)
        if rows:
            rows[-1]["estimator_append_wall_s"] = float(append_meta["append_wall_s"])
            rows[-1]["reduced_assembly_wall_s"] = assembly_wall
            rows[-1]["operator_assembly_reported_s"] = float(assembly_meta["assembly_wall_s"])
            rows[-1]["batch_wall_s"] = float(time.perf_counter() - batch_started)
            pd.DataFrame(rows).to_csv(selection_path, index=False)

    compiler.save(final_estimator)
    compiler.close()
    if operator_path.is_file():
        shutil.copy2(operator_path, final_operator)
    if not keep_compiler_work:
        compiler.cleanup()

    selection = pd.DataFrame(rows)
    if selection.empty:
        raise RuntimeError(f"No selection rows were created for {case['case_id']}.")
    summary = {
        "status": "complete",
        "case_id": str(case["case_id"]),
        "geometry_id": int(case["geometry_id"]),
        "morphology_label": str(case["morphology_label"]),
        "candidate_count": int(len(candidates)),
        "initial_materials": int(initial_materials),
        "max_materials": int(max_materials),
        "adaptive_batch_size": int(adaptive_batch_size),
        "max_adaptive_rounds": int(max_adaptive_rounds),
        "adaptive_rounds_done": int(adaptive_rounds_done),
        "prefilter_rank": int(prefilter_rank),
        "exact_score_limit": int(exact_score_limit),
        "selected_materials": int(len(selection)),
        "selected_candidate_ids": [int(value) for value in selection["candidate_id"]],
        "basis_rank": int(selection["basis_rank"].iloc[-1]),
        "target_error": float(target_error),
        "stop_score_factor": float(stop_score_factor),
        "stop_score_gate": float(target_error) * float(stop_score_factor),
        "stop_reason": stop_reason,
        "last_max_candidate_score": float(last_scan.get("max_candidate_score", np.nan)),
        "selection_uses_fom_error": False,
        "basis_solver_profile": str(basis_profile),
        "basis_solver_rtol": float(
            common.SOLVER_PROFILES[str(basis_profile)]["solver_rtol"]
        ),
        "basis_all_converged": bool(selection["basis_solver_all_converged"].all()),
        "basis_max_relative_residual": float(
            selection["basis_solver_max_relative_residual"].max()
        ),
        "snapshot_all_converged": True,
        "snapshot_max_relative_residual": float(
            selection.get("solver_max_relative_residual", pd.Series([np.nan])).max()
        ),
        "offline_snapshot_solve_wall_s": float(selection["snapshot_solve_wall_s"].sum()),
        "offline_candidate_scan_wall_s": float(selection["candidate_scan_wall_s"].sum()),
        "offline_candidate_prefilter_wall_s": float(selection.get("candidate_prefilter_wall_s", pd.Series([0.0])).sum()),
        "offline_candidate_exact_wall_s": float(selection.get("candidate_exact_wall_s", pd.Series([0.0])).sum()),
        "offline_basis_update_wall_s": float(selection["basis_update_wall_s"].sum()),
        "offline_estimator_append_wall_s": float(selection["estimator_append_wall_s"].sum()),
        "offline_reduced_assembly_wall_s": float(selection["reduced_assembly_wall_s"].sum()),
        "phase_sha256": geometry.manifest["phase_sha256"],
        "ori_sha256": geometry.manifest["ori_sha256"],
        "grid_shape": list(geometry.phase.shape),
    }
    common.write_json(summary_path, summary)
    print(
        f"[MULTIGEO] selection complete {case['case_id']} | "
        f"materials={summary['selected_materials']} r={summary['basis_rank']} "
        f"stop={stop_reason}",
        flush=True,
    )
    return summary


def _solve_truth(
    *,
    case: dict[str, Any],
    out_dir: Path,
    validation: pd.DataFrame,
    runtime: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    case_out = out_dir / str(case["case_id"])
    truth_path = case_out / "validation_truth_results.csv"
    if truth_path.is_file():
        return pd.read_csv(truth_path)
    geometry = _load_geometry(case)
    records: list[dict[str, Any]] = []
    for _, row in validation.iterrows():
        validation_id = int(row["validation_id"])
        material = row.to_dict()
        material["validation_id"] = validation_id
        record = common.solve_material(
            material_row=material,
            material_dir=case_out / "held_out_truth" / f"material_{validation_id:04d}",
            geometry=geometry,
            runtime=runtime,
            profile="truth",
            seed=int(seed) + validation_id,
            save_solution_fields=False,
        )
        records.append(record)
        pd.DataFrame(records).to_csv(truth_path, index=False)
        print(
            f"[MULTIGEO-TRUTH] {case['case_id']} {len(records)}/{len(validation)}",
            flush=True,
        )
    return pd.DataFrame(records)


def _candidate_rom_cloud(
    *,
    candidates: pd.DataFrame,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        coeffs = reduced._material_coefficients(row.to_dict())
        C_rom, _, online_s = reduced._rom_ceff(coeffs, Kq, Bq, Dq)
        props = reduced.engineering_constants_from_Cmandel(C_rom)
        out = {
            "candidate_id": int(row["candidate_id"]),
            "material_id": int(row["material_id"]),
            "rom_online_s": float(online_s),
            "rom_min_eig": float(np.linalg.eigvalsh(C_rom).min()),
        }
        for key in reduced.ENGINEERING_COLUMNS:
            out[f"rom_{key}"] = float(props.get(key, np.nan))
        for ii in range(6):
            for jj in range(6):
                out[f"Crom_{ii + 1}{jj + 1}"] = float(C_rom[ii, jj])
        rows.append(out)
    return pd.DataFrame(rows)


def _plot_overlay(
    *,
    case_out: Path,
    case_id: str,
    cloud: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"[MULTIGEO][WARN] matplotlib unavailable: {exc}", flush=True)
        return
    if cloud.empty or validation.empty:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    ax.scatter(
        cloud["rom_E1"],
        cloud["rom_E2"],
        s=4,
        color="0.72",
        alpha=0.35,
        linewidths=0,
        label=f"{len(cloud)} ROM queries",
    )
    scatter = ax.scatter(
        validation["rom_E1"],
        validation["rom_E2"],
        c=validation["relative_frobenius_error"],
        cmap="viridis",
        s=42,
        edgecolor="black",
        linewidth=0.35,
        label="30 independent FFT truths",
    )
    ax.set_xlabel(r"$E_1^{\mathrm{ROM}}$")
    ax.set_ylabel(r"$E_2^{\mathrm{ROM}}$")
    ax.set_title(case_id)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("relative tensor error")
    ax.legend(loc="best", frameon=False)
    fig.savefig(case_out / "candidate_cloud_validation_overlay.png", dpi=240)
    fig.savefig(case_out / "candidate_cloud_validation_overlay.pdf")
    plt.close(fig)


def _evaluate_case(
    *,
    case: dict[str, Any],
    out_dir: Path,
    candidates: pd.DataFrame,
    make_plots: bool,
    write_candidate_cloud: bool,
) -> dict[str, Any]:
    case_out = out_dir / str(case["case_id"])
    operator_path = case_out / "reduced_operators.npz"
    truth_path = case_out / "validation_truth_results.csv"
    if not operator_path.is_file():
        raise FileNotFoundError(f"Missing final operators: {operator_path}")
    if not truth_path.is_file():
        raise FileNotFoundError(f"Missing validation truth: {truth_path}")
    operators = common._load_operators(operator_path)
    truth = pd.read_csv(truth_path)
    validation = reduced._evaluate_rom(
        results_df=truth,
        Kq=operators["Kq"],
        Bq=operators["Bq"],
        Dq=operators["Dq"],
    )
    if "validation_id" in truth.columns:
        validation.insert(0, "validation_id", truth["validation_id"].astype(int).to_numpy())
    validation.to_csv(case_out / "validation_rom_results.csv", index=False)

    cloud_path = case_out / "candidate_rom_cloud.csv"
    if cloud_path.is_file() and not write_candidate_cloud:
        cloud = pd.read_csv(cloud_path)
    else:
        cloud = _candidate_rom_cloud(
            candidates=candidates,
            Kq=operators["Kq"],
            Bq=operators["Bq"],
            Dq=operators["Dq"],
        )
        cloud.to_csv(cloud_path, index=False)

    if make_plots:
        _plot_overlay(
            case_out=case_out,
            case_id=str(case["case_id"]),
            cloud=cloud,
            validation=validation,
        )

    errors = validation["relative_frobenius_error"].to_numpy(dtype=float)
    selection_summary = _read_json(case_out / "selection_summary.json")
    summary = {
        **selection_summary,
        "validation_materials": int(len(validation)),
        "validation_mean_error": float(np.mean(errors)),
        "validation_median_error": float(np.median(errors)),
        "validation_p95_error": float(np.quantile(errors, 0.95)),
        "validation_max_error": float(np.max(errors)),
        "validation_pass_1e4": bool(np.max(errors) <= TARGET_ERROR),
        "validation_worst_id": int(validation.iloc[int(np.argmax(errors))].get("validation_id", -1)),
        "validation_rom_online_median_s": float(validation["rom_online_s"].median()),
        "candidate_rom_count": int(len(cloud)),
        "candidate_rom_online_median_s": float(cloud["rom_online_s"].median()),
        "candidate_rom_min_eig": float(cloud["rom_min_eig"].min()),
    }
    common.write_json(case_out / "case_summary.json", summary)
    return summary


def _write_global_summary(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    frame = pd.DataFrame(summaries).sort_values("case_id")
    frame.to_csv(out_dir / "multigeometry_summary.csv", index=False)
    common.write_json(
        out_dir / "campaign_manifest.json",
        {
            "status": "complete",
            "geometry_count": int(len(frame)),
            "candidate_count": int(frame["candidate_count"].iloc[0]),
            "validation_materials_per_geometry": int(frame["validation_materials"].iloc[0])
            if "validation_materials" in frame
            else None,
            "target_error": TARGET_ERROR,
            "selection_uses_fom_error": False,
            "geometries_passing_1e4_validation": int(
                frame.get("validation_pass_1e4", pd.Series(dtype=bool)).sum()
            )
            if "validation_pass_1e4" in frame
            else None,
            "summary_csv_sha256": _sha256(out_dir / "multigeometry_summary.csv"),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", type=Path, default=GEOMETRY_DIR_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--stage", choices=("design", "select", "truth", "evaluate", "all"), default="all")
    parser.add_argument("--geometry-ids", type=int, nargs="*", default=None)
    parser.add_argument("--candidate-count", type=int, default=1024)
    parser.add_argument("--validation-count", type=int, default=30)
    parser.add_argument("--candidate-seed", type=int, default=20260821)
    parser.add_argument("--validation-seed", type=int, default=20260822)
    parser.add_argument("--initial-materials", type=int, default=10)
    parser.add_argument("--max-materials", type=int, default=20)
    parser.add_argument("--adaptive-batch-size", type=int, default=10)
    parser.add_argument(
        "--max-adaptive-rounds",
        type=int,
        default=1,
        help=(
            "Number of full candidate-pool Schur scans after the initial block. "
            "Use 2 with --max-materials 30 only when validation shows that one "
            "adaptive batch is insufficient."
        ),
    )
    parser.add_argument(
        "--prefilter-rank",
        type=int,
        default=60,
        help=(
            "Rank used for the cheap Schur prefilter when the current rank is larger. "
            "The candidate pool is scored at this rank before full-rank scoring."
        ),
    )
    parser.add_argument(
        "--exact-score-limit",
        type=int,
        default=1024,
        help=(
            "Maximum candidates receiving full-rank Schur scores after prefiltering. "
            "Set 0 to disable two-stage scoring and score every remaining candidate exactly."
        ),
    )
    parser.add_argument("--target-error", type=float, default=TARGET_ERROR)
    parser.add_argument(
        "--stop-score-factor",
        type=float,
        default=2.0,
        help=(
            "Stop after a scan when max Schur score <= factor*target. "
            "This is a predeclared heuristic gate; held-out FFT errors remain untouched."
        ),
    )
    parser.add_argument("--orthonormal-tol", type=float, default=1.0e-10)
    parser.add_argument(
        "--basis-profile",
        choices=tuple(common.SOLVER_PROFILES),
        default="timing",
        help=(
            "FFTHomPy profile used to generate the offline basis. The default "
            "timing profile is float32/rtol=1e-5 and is below the 1e-4 ROM floor; "
            "use rom_floor for a fast 1e-4 feasibility pilot or snapshot for "
            "float64/rtol=1e-8 confirmation runs."
        ),
    )
    parser.add_argument("--atom-batch-size", type=int, default=6)
    parser.add_argument("--feature-block", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--geometry-backend", choices=("numba", "cupy", "auto"), default="numba")
    parser.add_argument("--generator-cores", type=int, default=2)
    parser.add_argument("--blas-threads", type=int, default=8)
    parser.add_argument("--fft-workers", type=int, default=16)
    parser.add_argument("--venv-path", type=Path, default=common.DEFAULT_VENV_PATH)
    parser.add_argument("--overwrite-design", action="store_true")
    parser.add_argument("--keep-compiler-work", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--write-candidate-cloud", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blas_controller, blas_info = common.configure_blas_threads(args.blas_threads)
    out_dir = args.out_dir.resolve()
    if args.smoke:
        args.candidate_count = min(int(args.candidate_count), 64)
        args.validation_count = min(int(args.validation_count), 4)
        args.initial_materials = min(int(args.initial_materials), 2)
        args.max_materials = min(int(args.max_materials), 4)
        args.adaptive_batch_size = min(int(args.adaptive_batch_size), 1)
        args.max_adaptive_rounds = min(int(args.max_adaptive_rounds), 1)
        args.exact_score_limit = min(int(args.exact_score_limit), 32)
        if args.geometry_ids is None:
            args.geometry_ids = [0]

    if int(args.initial_materials) < 1:
        raise ValueError("--initial-materials must be positive.")
    if int(args.max_materials) < int(args.initial_materials):
        raise ValueError("--max-materials must be >= --initial-materials.")
    if int(args.adaptive_batch_size) < 1:
        raise ValueError("--adaptive-batch-size must be positive.")
    if int(args.max_adaptive_rounds) < 0:
        raise ValueError("--max-adaptive-rounds must be non-negative.")
    if int(args.prefilter_rank) < 1:
        raise ValueError("--prefilter-rank must be positive.")
    if int(args.exact_score_limit) < 0:
        raise ValueError("--exact-score-limit must be non-negative.")
    if float(args.target_error) <= 0.0 or float(args.stop_score_factor) <= 0.0:
        raise ValueError("--target-error and --stop-score-factor must be positive.")
    if str(args.basis_profile) == "truth":
        raise ValueError("--basis-profile truth is reserved for held-out validation.")

    candidates, validation = _ensure_material_design(
        out_dir=out_dir,
        candidate_count=int(args.candidate_count),
        validation_count=int(args.validation_count),
        candidate_seed=int(args.candidate_seed),
        validation_seed=int(args.validation_seed),
        overwrite=bool(args.overwrite_design),
    )
    cases = _geometry_cases(args.geometry_dir.resolve(), args.geometry_ids)
    geometry_table = pd.DataFrame(
        [
            {
                "case_id": case["case_id"],
                "geometry_id": case["geometry_id"],
                "morphology_label": case["morphology_label"],
                "Vf_realized": case.get("Vf_realized", np.nan),
                "aspect_ratio": case.get("aspect_ratio", np.nan),
                "cluster_fraction_target": case.get("cluster_fraction_target", np.nan),
                "interface_density": case.get("interface_density", np.nan),
                "n_fibers": case.get("n_fibers", np.nan),
            }
            for case in cases
        ]
    )
    geometry_table.to_csv(out_dir / "geometry_protocol_table.csv", index=False)

    if args.stage == "design":
        common.write_json(
            out_dir / "campaign_manifest.json",
            {
                "status": "design_only",
                "geometry_count": int(len(cases)),
                "candidate_count": int(len(candidates)),
                "validation_count": int(len(validation)),
                "initial_materials": int(args.initial_materials),
                "max_materials": int(args.max_materials),
                "adaptive_batch_size": int(args.adaptive_batch_size),
                "max_adaptive_rounds": int(args.max_adaptive_rounds),
                "prefilter_rank": int(args.prefilter_rank),
                "exact_score_limit": int(args.exact_score_limit),
                "target_error": float(args.target_error),
                "stop_score_factor": float(args.stop_score_factor),
                "basis_solver_profile": str(args.basis_profile),
                "blas_threads": int(args.blas_threads),
                "fft_workers": int(args.fft_workers),
                "blas_runtimes": blas_info,
                "basis_solver_rtol": float(
                    common.SOLVER_PROFILES[str(args.basis_profile)]["solver_rtol"]
                ),
                "selection_uses_fom_error": False,
            },
        )
        print(f"[MULTIGEO] design ready | out={out_dir}", flush=True)
        del blas_controller
        return 0

    needs_gpu = args.stage in {"select", "truth", "all"}
    runtime: dict[str, Any] | None = None
    if needs_gpu:
        common.prepare_runtime(
            args.venv_path,
            Path(__file__),
            marker_name="CMAME_MULTIGEO_CUDA_READY",
        )
        runtime = common.configure_runtime(
            geometry_backend=args.geometry_backend,
            generator_cores=args.generator_cores,
        )

    summaries: list[dict[str, Any]] = []
    if args.stage in {"select", "all"}:
        assert runtime is not None
        for case_index, case in enumerate(cases):
            summaries.append(
                _solve_selection(
                    case=case,
                    out_dir=out_dir,
                    candidates=candidates,
                    runtime=runtime,
                    initial_materials=int(args.initial_materials),
                    max_materials=int(args.max_materials),
                    adaptive_batch_size=int(args.adaptive_batch_size),
                    max_adaptive_rounds=int(args.max_adaptive_rounds),
                    prefilter_rank=int(args.prefilter_rank),
                    exact_score_limit=int(args.exact_score_limit),
                    target_error=float(args.target_error),
                    stop_score_factor=float(args.stop_score_factor),
                    orthonormal_tolerance=float(args.orthonormal_tol),
                    seed=int(args.seed) + 100_000 * case_index,
                    atom_batch_size=int(args.atom_batch_size),
                    feature_block=int(args.feature_block),
                    keep_compiler_work=bool(args.keep_compiler_work),
                    basis_profile=str(args.basis_profile),
                    fft_workers=int(args.fft_workers),
                )
            )

    if args.stage in {"truth", "all"}:
        assert runtime is not None
        for case_index, case in enumerate(cases):
            _solve_truth(
                case=case,
                out_dir=out_dir,
                validation=validation,
                runtime=runtime,
                seed=int(args.seed) + 1_000_000 + 100_000 * case_index,
            )

    if args.stage in {"evaluate", "all"}:
        summaries = [
            _evaluate_case(
                case=case,
                out_dir=out_dir,
                candidates=candidates,
                make_plots=not bool(args.no_plots),
                write_candidate_cloud=bool(args.write_candidate_cloud),
            )
            for case in cases
        ]

    _write_global_summary(out_dir, summaries)
    print(f"[MULTIGEO] stage={args.stage} done | out={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
