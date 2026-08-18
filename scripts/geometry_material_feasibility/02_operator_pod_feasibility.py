#!/usr/bin/env python3
"""Test operator-POD transfer to one unseen volume-fraction geometry."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_DEFAULT = HERE / "config.json"
INTERPRETABLE = ROOT / "scripts" / "cmame_interpretable_pipeline"
if str(INTERPRETABLE) not in sys.path:
    sys.path.insert(0, str(INTERPRETABLE))

from env_bootstrap import ensure_configured_venv

ensure_configured_venv(CONFIG_DEFAULT)

sys.path = [value for value in sys.path if value != str(INTERPRETABLE)]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_triangular

import cmame_campaign_common as common
import fft_homogenization_solver as sweep
import rom_reduced_operator as reduced
import rom_validation_utils as validate


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    common.write_json(path, payload)


@contextmanager
def quiet_output(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    with Path(os.devnull).open("w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def material_payload(row: pd.Series) -> dict[str, Any]:
    return {
        **{name: float(row[name]) for name in sweep.MATERIAL_BOUNDS},
        "material_id": int(row["material_id"]),
        "material_label": str(row["material_label"]),
    }


def parameter_matrix(materials: pd.DataFrame) -> np.ndarray:
    return materials.loc[:, list(reduced.MATERIAL_PARAMETER_COLUMNS)].to_numpy(dtype=float)


def relative_frobenius(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    difference = np.asarray(predicted) - np.asarray(reference)
    numerator = np.linalg.norm(difference.reshape(len(difference), -1), axis=1)
    denominator = np.linalg.norm(np.asarray(reference).reshape(len(reference), -1), axis=1)
    return numerator / np.maximum(denominator, np.finfo(float).eps)


def summarize_errors(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    errors = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(errors)),
        f"{prefix}_median": float(np.median(errors)),
        f"{prefix}_p95": float(np.quantile(errors, 0.95)),
        f"{prefix}_max": float(np.max(errors)),
        f"{prefix}_pass_1e3_count": int(np.count_nonzero(errors <= 1.0e-3)),
        f"{prefix}_pass_1e3_percent": float(100.0 * np.mean(errors <= 1.0e-3)),
        f"{prefix}_pass_1e4_count": int(np.count_nonzero(errors <= 1.0e-4)),
        f"{prefix}_pass_1e4_percent": float(100.0 * np.mean(errors <= 1.0e-4)),
    }


def validate_native_meshes(
    geometry_table: pd.DataFrame,
    geometries: list[common.GeometryData],
) -> dict[str, Any]:
    shapes = {tuple(geometry.phase.shape) for geometry in geometries}
    boxes = geometry_table["box_um"].to_numpy(dtype=float)
    resolutions = geometry_table["resolution_vox_per_um"].to_numpy(dtype=float)
    compatible = (
        len(shapes) == 1
        and np.allclose(boxes, boxes[0], rtol=0.0, atol=1.0e-12)
        and np.allclose(resolutions, resolutions[0], rtol=0.0, atol=1.0e-12)
    )
    audit = {
        "mesh_assignment": "independent_per_geometry",
        "native_shapes": [list(shape) for shape in sorted(shapes)],
        "box_um_values": sorted(set(float(value) for value in boxes)),
        "resolution_vox_per_um_values": sorted(
            set(float(value) for value in resolutions)
        ),
        "direct_common_physical_space_compatible": bool(compatible),
        "operator_local_ordering": "independent_per_geometry",
    }
    if not compatible:
        raise RuntimeError(
            "The independently selected native meshes do not share the same physical "
            "grid. Direct common-POD transfer is disabled; an explicit conservative "
            "cross-mesh transfer operator must be validated first."
        )
    return audit


def build_common_basis(
    snapshots: np.ndarray,
    *,
    rank_tolerance: float,
    energy_fraction: float,
    rank_cap: int,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    started = time.perf_counter()
    values = np.asarray(snapshots, dtype=np.float32)
    dimension = int(values.shape[1])
    gram_started = time.perf_counter()
    gram = np.asarray(values @ values.T, dtype=np.float64) / float(dimension)
    gram = 0.5 * (gram + gram.T)
    gram_wall_s = float(time.perf_counter() - gram_started)

    eig_started = time.perf_counter()
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    singular_values = np.sqrt(eigenvalues)
    eig_wall_s = float(time.perf_counter() - eig_started)
    if not len(singular_values) or singular_values[0] <= 0.0:
        raise RuntimeError("The common snapshot matrix has zero numerical rank.")

    numerical_rank = int(
        np.count_nonzero(singular_values / singular_values[0] > float(rank_tolerance))
    )
    total_energy = float(np.sum(eigenvalues))
    cumulative = np.cumsum(eigenvalues) / max(total_energy, np.finfo(float).eps)
    energy_rank = int(np.searchsorted(cumulative, float(energy_fraction)) + 1)
    unconstrained_rank = min(numerical_rank, energy_rank)
    configured_rank_cap = int(rank_cap)
    rank = (
        unconstrained_rank
        if configured_rank_cap <= 0
        else min(unconstrained_rank, configured_rank_cap)
    )
    if rank < 1:
        raise RuntimeError("The selected common basis rank is zero.")

    modes_started = time.perf_counter()
    left = np.ascontiguousarray(eigenvectors[:, :rank].T, dtype=np.float32)
    basis = left @ values
    basis /= singular_values[:rank, None].astype(np.float32)
    basis_gram = np.asarray(basis @ basis.T, dtype=np.float64) / float(dimension)
    basis_gram = 0.5 * (basis_gram + basis_gram.T)
    correction = np.linalg.cholesky(basis_gram)
    basis = solve_triangular(
        correction.astype(np.float32),
        basis,
        lower=True,
        overwrite_b=True,
        check_finite=False,
    )
    modes_wall_s = float(time.perf_counter() - modes_started)
    orthogonality = np.asarray(basis @ basis.T, dtype=np.float64) / float(dimension)
    orthogonality_error = float(
        np.linalg.norm(orthogonality - np.eye(rank), ord="fro")
        / math.sqrt(float(rank))
    )
    spectrum = pd.DataFrame(
        {
            "mode": np.arange(1, len(singular_values) + 1, dtype=int),
            "singular_value": singular_values,
            "relative_singular_value": singular_values / singular_values[0],
            "cumulative_energy": cumulative,
            "retained": np.arange(len(singular_values)) < rank,
        }
    )
    metadata = {
        "snapshot_count": int(values.shape[0]),
        "snapshot_dimension": dimension,
        "snapshot_dtype": str(values.dtype),
        "numerical_rank": numerical_rank,
        "energy_rank": energy_rank,
        "unconstrained_rank": unconstrained_rank,
        "selected_rank": rank,
        "rank_cap": configured_rank_cap,
        "rank_cap_policy": "unlimited" if configured_rank_cap <= 0 else "explicit",
        "rank_cap_active": bool(
            configured_rank_cap > 0 and rank < unconstrained_rank
        ),
        "rank_tolerance": float(rank_tolerance),
        "energy_fraction_target": float(energy_fraction),
        "retained_energy": float(cumulative[rank - 1]),
        "orthogonality_error_rel": orthogonality_error,
        "gram_wall_s": gram_wall_s,
        "eigendecomposition_wall_s": eig_wall_s,
        "mode_reconstruction_wall_s": modes_wall_s,
        "basis_total_wall_s": float(time.perf_counter() - started),
    }
    return np.ascontiguousarray(basis, dtype=np.float32), metadata, spectrum


def symmetric_vectorize(values: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    matrices = np.asarray(values, dtype=np.float64)
    size = int(matrices.shape[-1])
    indices = np.triu_indices(size)
    vectors = matrices[..., indices[0], indices[1]].copy()
    off_diagonal = indices[0] != indices[1]
    vectors[..., off_diagonal] *= math.sqrt(2.0)
    return vectors, indices


def symmetric_restore(
    vectors: np.ndarray,
    indices: tuple[np.ndarray, np.ndarray],
    size: int,
) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64).copy()
    off_diagonal = indices[0] != indices[1]
    values[..., off_diagonal] /= math.sqrt(2.0)
    matrices = np.zeros(values.shape[:-1] + (size, size), dtype=np.float64)
    matrices[..., indices[0], indices[1]] = values
    matrices[..., indices[1], indices[0]] = values
    return matrices


def flatten_operator(values: np.ndarray, symmetric: bool) -> tuple[np.ndarray, Any]:
    arrays = np.asarray(values, dtype=np.float64)
    if symmetric:
        vectors, indices = symmetric_vectorize(arrays)
        return vectors, {"indices": indices, "size": int(arrays.shape[-1])}
    return arrays.reshape(len(arrays), -1), {"shape": tuple(arrays.shape[1:])}


def restore_operator(vector: np.ndarray, metadata: dict[str, Any], symmetric: bool) -> np.ndarray:
    if symmetric:
        return symmetric_restore(
            np.asarray(vector)[None, :],
            metadata["indices"],
            int(metadata["size"]),
        )[0]
    return np.asarray(vector, dtype=np.float64).reshape(metadata["shape"])


def interpolate_vector_family(
    x_train: np.ndarray,
    vectors: np.ndarray,
    x_target: float,
    modes: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_values = np.asarray(x_train, dtype=float)
    data = np.asarray(vectors, dtype=np.float64)
    mean = np.mean(data, axis=0)
    centered = data - mean
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    numerical_rank = int(
        np.count_nonzero(
            singular_values
            > max(singular_values[0] * 1.0e-12, np.finfo(float).eps)
        )
    ) if len(singular_values) and singular_values[0] > 0.0 else 0
    retained = min(int(modes), numerical_rank)
    if retained:
        operator_modes = right[:retained]
        coefficients = centered @ operator_modes.T
        predicted_coefficients = np.asarray(
            PchipInterpolator(x_values, coefficients, axis=0, extrapolate=True)(
                float(x_target)
            )
        )
        prediction = mean + predicted_coefficients @ operator_modes
    else:
        operator_modes = np.empty((0, data.shape[1]), dtype=np.float64)
        coefficients = np.empty((len(data), 0), dtype=np.float64)
        prediction = mean.copy()
    energy = singular_values**2
    cumulative = (
        np.cumsum(energy) / np.sum(energy)
        if np.sum(energy) > 0.0
        else np.zeros_like(energy)
    )
    return prediction, {
        "mean": mean,
        "modes": operator_modes,
        "coefficients": coefficients,
        "singular_values": singular_values,
        "cumulative_energy": cumulative,
        "numerical_rank": numerical_rank,
        "retained_modes": retained,
    }


def select_operator_modes(
    x_train: np.ndarray,
    vectors: np.ndarray,
    max_modes: int,
) -> tuple[int, pd.DataFrame]:
    maximum = min(int(max_modes), max(0, len(x_train) - 2))
    rows: list[dict[str, Any]] = []
    for modes in range(maximum + 1):
        errors: list[float] = []
        for omitted in range(len(x_train)):
            mask = np.arange(len(x_train)) != omitted
            prediction, _ = interpolate_vector_family(
                x_train[mask], vectors[mask], float(x_train[omitted]), modes
            )
            reference = vectors[omitted]
            error = float(
                np.linalg.norm(prediction - reference)
                / max(np.linalg.norm(reference), np.finfo(float).eps)
            )
            errors.append(error)
        rows.append(
            {
                "modes": int(modes),
                "cv_mean_rel_error": float(np.mean(errors)),
                "cv_p95_rel_error": float(np.quantile(errors, 0.95)),
                "cv_max_rel_error": float(np.max(errors)),
            }
        )
    frame = pd.DataFrame(rows)
    best = frame.sort_values(
        ["cv_p95_rel_error", "cv_mean_rel_error", "modes"], kind="stable"
    ).iloc[0]
    return int(best["modes"]), frame


def operator_energy_ranks(
    singular_values: np.ndarray,
    targets: list[float],
) -> dict[str, int]:
    energy = np.asarray(singular_values, dtype=float) ** 2
    if not len(energy) or np.sum(energy) <= 0.0:
        return {f"rank_energy_{target:g}": 0 for target in targets}
    cumulative = np.cumsum(energy) / np.sum(energy)
    return {
        f"rank_energy_{target:g}": int(np.searchsorted(cumulative, target) + 1)
        for target in targets
    }


def fit_all_operator_families(
    *,
    volume_fractions: np.ndarray,
    held_out_index: int,
    K_all: np.ndarray,
    B_all: np.ndarray,
    D_all: np.ndarray,
    max_modes: int,
    energy_targets: list[float],
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask = np.arange(len(volume_fractions)) != int(held_out_index)
    x_train = volume_fractions[train_mask]
    x_target = float(volume_fractions[held_out_index])
    reconstructed: dict[str, list[np.ndarray]] = {"K": [], "B": [], "D": []}
    error_rows: list[dict[str, Any]] = []
    cv_frames: list[pd.DataFrame] = []
    spectrum_rows: list[dict[str, Any]] = []
    families = (("K", K_all, True), ("B", B_all, False), ("D", D_all, True))
    model_dir.mkdir(parents=True, exist_ok=True)

    for block_name, all_values, symmetric in families:
        for coefficient in range(all_values.shape[1]):
            vectors, vector_metadata = flatten_operator(
                all_values[:, coefficient], symmetric
            )
            train_vectors = vectors[train_mask]
            selected_modes, cv = select_operator_modes(
                x_train, train_vectors, int(max_modes)
            )
            cv.insert(0, "affine_coefficient", int(coefficient))
            cv.insert(0, "block", block_name)
            cv_frames.append(cv)
            prediction, model = interpolate_vector_family(
                x_train, train_vectors, x_target, selected_modes
            )
            restored = restore_operator(prediction, vector_metadata, symmetric)
            reconstructed[block_name].append(restored)
            reference = np.asarray(all_values[held_out_index, coefficient])
            held_out_error = float(
                np.linalg.norm(restored - reference)
                / max(np.linalg.norm(reference), np.finfo(float).eps)
            )
            ranks = operator_energy_ranks(
                np.asarray(model["singular_values"]), energy_targets
            )
            error_rows.append(
                {
                    "block": block_name,
                    "affine_coefficient": int(coefficient),
                    "coefficient_name": reduced.COEFF_NAMES[coefficient],
                    "selected_modes": int(selected_modes),
                    "training_numerical_rank": int(model["numerical_rank"]),
                    "held_out_relative_operator_error": held_out_error,
                    **ranks,
                }
            )
            for mode, (singular, cumulative) in enumerate(
                zip(model["singular_values"], model["cumulative_energy"], strict=True),
                start=1,
            ):
                spectrum_rows.append(
                    {
                        "block": block_name,
                        "affine_coefficient": int(coefficient),
                        "coefficient_name": reduced.COEFF_NAMES[coefficient],
                        "mode": int(mode),
                        "singular_value": float(singular),
                        "cumulative_energy": float(cumulative),
                    }
                )
            np.savez_compressed(
                model_dir / f"operator_pod_{block_name}_q{coefficient}.npz",
                x_train=x_train,
                mean=model["mean"],
                modes=model["modes"],
                coefficients=model["coefficients"],
                singular_values=model["singular_values"],
                selected_modes=np.asarray(selected_modes),
                x_target=np.asarray(x_target),
                predicted_vector=prediction,
            )

    return (
        np.stack(reconstructed["K"]),
        np.stack(reconstructed["B"]),
        np.stack(reconstructed["D"]),
        pd.DataFrame(error_rows),
        pd.concat(cv_frames, ignore_index=True),
        pd.DataFrame(spectrum_rows),
    )


def plot_results(
    out_dir: Path,
    basis_spectrum: pd.DataFrame,
    operator_spectrum: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.semilogy(
        basis_spectrum["mode"],
        basis_spectrum["relative_singular_value"],
        color="#176B87",
        linewidth=1.6,
    )
    retained = basis_spectrum.loc[basis_spectrum["retained"], "mode"]
    if len(retained):
        axis.axvline(int(retained.max()), color="#C84B31", linestyle="--", linewidth=1.2)
    axis.set_xlabel("Common POD mode")
    axis.set_ylabel("Relative singular value")
    axis.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "common_basis_spectrum.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True)
    for axis, block in zip(axes, ("K", "B", "D"), strict=True):
        selected = operator_spectrum[operator_spectrum["block"] == block]
        for _, group in selected.groupby("affine_coefficient"):
            first = float(group["singular_value"].iloc[0])
            if first > 0.0:
                axis.semilogy(group["mode"], group["singular_value"] / first, alpha=0.75)
        axis.set_title(block)
        axis.set_xlabel("Operator mode")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Relative singular value")
    fig.tight_layout()
    fig.savefig(figures / "operator_pod_spectra.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for column, label, color in (
        ("error_exact_geometry_rom_vs_fft", "ROM with exact geometry", "#176B87"),
        (
            "error_linear_neighbor_geometry_rom_vs_fft",
            "Linear neighboring operators",
            "#4C956C",
        ),
        ("error_interpolated_geometry_vs_exact_rom", "Geometry interpolation", "#D18B00"),
        ("error_interpolated_geometry_rom_vs_fft", "Operator-POD total", "#B23A48"),
    ):
        values = np.sort(validation[column].to_numpy(dtype=float))
        probability = np.arange(1, len(values) + 1) / float(len(values))
        axis.semilogx(values, probability, label=label, color=color, linewidth=1.7)
    axis.axvline(1.0e-4, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Relative Frobenius error")
    axis.set_ylabel("Empirical CDF")
    axis.set_ylim(0.0, 1.01)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "unseen_geometry_validation_errors.png", dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config.resolve())
    common.prepare_runtime(
        project_path(config["paths"]["venv_path"]),
        Path(__file__),
        marker_name="GEOMETRY_MATERIAL_FEASIBILITY_CUDA_READY",
    )
    experiment = dict(config["experiment"])
    out_root = project_path(config["paths"]["out_root"]).resolve()
    geometry_root = out_root / "geometries"
    geometry_table = pd.read_csv(geometry_root / "geometry_family.csv")
    family_manifest = json.loads(
        (geometry_root / "family_manifest.json").read_text(encoding="utf-8")
    )
    if not bool(family_manifest.get("all_accepted", False)):
        raise RuntimeError("The geometry family has not passed its local tolerances.")

    run_name = args.run_name or datetime.now().strftime("run_operator_pod_%Y%m%d_%H%M%S")
    run_dir = out_root / "runs" / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=bool(args.overwrite))
    started = time.perf_counter()
    timings: list[dict[str, Any]] = []

    geometries = [
        common.load_fixed_geometry(geometry_root / f"geometry_{int(row.geometry_id):02d}")
        for row in geometry_table.itertuples(index=False)
    ]
    mesh_audit = validate_native_meshes(geometry_table, geometries)
    held_out_indices = np.flatnonzero(geometry_table["is_held_out"].to_numpy(dtype=bool))
    if len(held_out_indices) != 1:
        raise ValueError("Exactly one geometry must be held out.")
    held_out_index = int(held_out_indices[0])
    training_indices = [
        index for index in range(len(geometries)) if index != held_out_index
    ]
    nvox = int(geometries[0].phase.size)

    snapshot_count = int(args.snapshot_materials or experiment["snapshot_material_count"])
    validation_count = int(args.validation_materials or experiment["validation_material_count"])
    rank_cap = int(
        experiment["basis_rank_cap"] if args.rank_cap is None else args.rank_cap
    )
    snapshot_materials = validate._build_independent_materials(
        snapshot_count, int(experiment["snapshot_material_seed"])
    )
    validation_materials = validate._build_independent_materials(
        validation_count, int(experiment["validation_material_seed"])
    )
    snapshot_materials.to_csv(run_dir / "snapshot_materials.csv", index=False)
    validation_materials.to_csv(run_dir / "validation_materials.csv", index=False)

    cpu_info = common.cpu_resource_info()
    blas_threads = common.resolve_cpu_workers(
        experiment.get("blas_threads", "auto"), resource_info=cpu_info
    )
    _blas_controller, blas_info = common.configure_blas_threads(blas_threads)
    runtime = common.configure_runtime(
        solver_tol=float(common.SOLVER_PROFILES[str(experiment["solver_profile"])]["solver_rtol"]),
        fft_backend=str(experiment["fft_backend"]),
    )

    write_json(
        run_dir / "run_manifest.json",
        {
            "campaign_id": config["campaign_id"],
            "method": "common_snapshot_POD_plus_operator_POD",
            "status": "running",
            "run_dir": str(run_dir),
            "geometry_root": str(geometry_root),
            "held_out_geometry_id": int(geometry_table.iloc[held_out_index]["geometry_id"]),
            "held_out_volume_fraction": float(
                geometry_table.iloc[held_out_index]["Vf_target"]
            ),
            "training_geometry_ids": [
                int(geometry_table.iloc[index]["geometry_id"])
                for index in training_indices
            ],
            "snapshot_material_count": snapshot_count,
            "snapshot_selection_policy": "fixed_sobol_prefix_no_adaptivity",
            "training_fft_monitor_count": 0,
            "training_fft_validation_count": 0,
            "validation_policy": "held_out_geometry_only_after_model_freeze",
            "validation_material_count": validation_count,
            "solver_profile": str(experiment["solver_profile"]),
            "solver_dtype": common.SOLVER_PROFILES[str(experiment["solver_profile"])]["solver_real_dtype"],
            "solver_rtol": common.SOLVER_PROFILES[str(experiment["solver_profile"])]["solver_rtol"],
            "basis_dtype": str(experiment["basis_dtype"]),
            "basis_rank_cap": rank_cap,
            "mesh_audit": mesh_audit,
            "blas_threads": blas_threads,
            "blas_info": blas_info,
            "cpu_resources": cpu_info,
        },
    )

    total_snapshots = len(training_indices) * snapshot_count * 6
    snapshots = np.empty((total_snapshots, 6 * nvox), dtype=np.float32)
    snapshot_rows: list[dict[str, Any]] = []
    snapshot_position = 0
    training_fft_started = time.perf_counter()
    for geometry_index in training_indices:
        geometry_row = geometry_table.iloc[geometry_index]
        geometry = geometries[geometry_index]
        for material_index, material_row in snapshot_materials.iterrows():
            start_position = snapshot_position

            def consume_field(load_id: int, field: np.ndarray) -> None:
                snapshots[start_position + int(load_id)] = np.asarray(
                    field, dtype=np.float32
                ).reshape(-1)

            material_dir = (
                run_dir
                / "training_fft"
                / f"geometry_{int(geometry_row['geometry_id']):02d}"
                / f"material_{int(material_index):04d}"
            )
            with quiet_output(bool(args.quiet_solver)):
                record = common.solve_material(
                    material_row=material_payload(material_row),
                    material_dir=material_dir,
                    geometry=geometry,
                    runtime=runtime,
                    profile=str(experiment["solver_profile"]),
                    seed=int(experiment["snapshot_material_seed"]) + int(material_index),
                    save_solution_fields=True,
                    persistent_gpu_cache=False,
                    solution_field_dtype=np.float32,
                    solution_field_consumer=consume_field,
                )
            snapshot_position += 6
            snapshot_rows.append(
                {
                    "geometry_id": int(geometry_row["geometry_id"]),
                    "Vf_target": float(geometry_row["Vf_target"]),
                    "grid_size": int(geometry_row["grid_size"]),
                    "material_id": int(material_index),
                    "snapshot_start": int(start_position),
                    "snapshot_end": int(snapshot_position - 1),
                    "solve_wall_s": float(record["solve_wall_s"]),
                    "solver_max_iterations": int(record["solver_max_iterations"]),
                    "solver_max_relative_residual": float(
                        record["solver_max_relative_residual"]
                    ),
                }
            )
            print(
                f"[TRAIN FFT] geometry={int(geometry_row['geometry_id']):02d} "
                f"Vf={100.0 * float(geometry_row['Vf_target']):.0f}% "
                f"material={material_index + 1}/{snapshot_count} "
                f"time={float(record['solve_wall_s']):.2f}s",
                flush=True,
            )
    training_fft_wall_s = float(time.perf_counter() - training_fft_started)
    timings.append({"stage": "training_fft", "wall_s": training_fft_wall_s})
    snapshot_frame = pd.DataFrame(snapshot_rows)
    snapshot_frame.to_csv(run_dir / "training_fft_timing.csv", index=False)

    basis, basis_metadata, basis_spectrum = build_common_basis(
        snapshots,
        rank_tolerance=float(experiment["basis_rank_tolerance"]),
        energy_fraction=float(experiment["basis_energy_fraction"]),
        rank_cap=rank_cap,
    )
    del snapshots
    gc.collect()
    write_json(run_dir / "common_basis_metadata.json", basis_metadata)
    basis_spectrum.to_csv(run_dir / "common_basis_spectrum.csv", index=False)
    timings.append({"stage": "common_basis", "wall_s": basis_metadata["basis_total_wall_s"]})
    print(
        f"[COMMON POD] rank={basis_metadata['selected_rank']} "
        f"energy={basis_metadata['retained_energy']:.8f} "
        f"time={basis_metadata['basis_total_wall_s']:.2f}s",
        flush=True,
    )

    rank = int(basis.shape[0])
    basis_fields = basis.reshape(rank, 6, nvox)
    K_all = np.empty((len(geometries), 7, rank, rank), dtype=np.float64)
    B_all = np.empty((len(geometries), 7, rank, 6), dtype=np.float64)
    D_all = np.empty((len(geometries), 7, 6, 6), dtype=np.float64)
    assembly_rows: list[dict[str, Any]] = []
    assembly_started = time.perf_counter()
    operator_dir = run_dir / "native_reduced_operators"
    operator_dir.mkdir(parents=True, exist_ok=True)
    for geometry_index, geometry in enumerate(geometries):
        row = geometry_table.iloc[geometry_index]
        local_order = reduced.phase_orientation_voxel_order(geometry.phase, geometry.ori)
        local_basis = np.ascontiguousarray(basis_fields[:, :, local_order])
        local_phase = geometry.phase.reshape(-1)[local_order]
        local_ori = geometry.ori.reshape(-1, 3)[local_order]
        affine = reduced.affine_stress_batch_factory(local_phase, local_ori)
        Kq, Bq, Dq, metadata = reduced._assemble_reduced_operators(
            phase=local_phase,
            ori=local_ori,
            basis=local_basis,
            affine_stress_batch=affine,
            affine_q_block_size=int(experiment["affine_q_block_size"]),
        )
        K_all[geometry_index] = Kq
        B_all[geometry_index] = Bq
        D_all[geometry_index] = Dq
        np.savez_compressed(
            operator_dir / f"geometry_{int(row['geometry_id']):02d}.npz",
            Kq=Kq,
            Bq=Bq,
            Dq=Dq,
        )
        assembly_rows.append(
            {
                "geometry_id": int(row["geometry_id"]),
                "Vf_target": float(row["Vf_target"]),
                "grid_size": int(row["grid_size"]),
                "local_voxel_order": "phase_then_orientation",
                **metadata,
            }
        )
        print(
            f"[KBD] geometry={int(row['geometry_id']):02d} "
            f"Vf={100.0 * float(row['Vf_target']):.0f}% "
            f"time={float(metadata['assembly_wall_s']):.2f}s",
            flush=True,
        )
        del local_basis, local_phase, local_ori, affine
        gc.collect()
    assembly_wall_s = float(time.perf_counter() - assembly_started)
    timings.append({"stage": "native_operator_assembly", "wall_s": assembly_wall_s})
    assembly_frame = pd.DataFrame(assembly_rows)
    assembly_frame.to_csv(run_dir / "native_operator_assembly_timing.csv", index=False)

    operator_pod_started = time.perf_counter()
    (
        K_interpolated,
        B_interpolated,
        D_interpolated,
        operator_errors,
        operator_cv,
        operator_spectrum,
    ) = fit_all_operator_families(
        volume_fractions=geometry_table["Vf_target"].to_numpy(dtype=float),
        held_out_index=held_out_index,
        K_all=K_all,
        B_all=B_all,
        D_all=D_all,
        max_modes=int(experiment["operator_max_modes"]),
        energy_targets=[float(value) for value in experiment["operator_energy_targets"]],
        model_dir=run_dir / "operator_pod_models",
    )
    operator_pod_wall_s = float(time.perf_counter() - operator_pod_started)
    timings.append({"stage": "operator_pod_and_interpolation", "wall_s": operator_pod_wall_s})
    operator_errors.to_csv(run_dir / "operator_held_out_errors.csv", index=False)
    operator_cv.to_csv(run_dir / "operator_mode_cross_validation.csv", index=False)
    operator_spectrum.to_csv(run_dir / "operator_pod_spectra.csv", index=False)
    np.savez_compressed(
        run_dir / "held_out_interpolated_operators.npz",
        Kq=K_interpolated,
        Bq=B_interpolated,
        Dq=D_interpolated,
    )

    volume_fractions = geometry_table["Vf_target"].to_numpy(dtype=float)
    target_vf = float(volume_fractions[held_out_index])
    lower_candidates = [
        index for index in training_indices if volume_fractions[index] < target_vf
    ]
    upper_candidates = [
        index for index in training_indices if volume_fractions[index] > target_vf
    ]
    if not lower_candidates or not upper_candidates:
        raise RuntimeError("The held-out geometry must be bracketed by training geometries.")
    lower_index = max(lower_candidates, key=lambda index: volume_fractions[index])
    upper_index = min(upper_candidates, key=lambda index: volume_fractions[index])
    interpolation_weight = (
        (target_vf - volume_fractions[lower_index])
        / (volume_fractions[upper_index] - volume_fractions[lower_index])
    )
    K_linear = (
        (1.0 - interpolation_weight) * K_all[lower_index]
        + interpolation_weight * K_all[upper_index]
    )
    B_linear = (
        (1.0 - interpolation_weight) * B_all[lower_index]
        + interpolation_weight * B_all[upper_index]
    )
    D_linear = (
        (1.0 - interpolation_weight) * D_all[lower_index]
        + interpolation_weight * D_all[upper_index]
    )
    linear_operator_rows: list[dict[str, Any]] = []
    for block_name, predicted, reference in (
        ("K", K_linear, K_all[held_out_index]),
        ("B", B_linear, B_all[held_out_index]),
        ("D", D_linear, D_all[held_out_index]),
    ):
        for coefficient in range(predicted.shape[0]):
            error = float(
                np.linalg.norm(predicted[coefficient] - reference[coefficient])
                / max(
                    np.linalg.norm(reference[coefficient]), np.finfo(float).eps
                )
            )
            linear_operator_rows.append(
                {
                    "block": block_name,
                    "affine_coefficient": int(coefficient),
                    "coefficient_name": reduced.COEFF_NAMES[coefficient],
                    "held_out_relative_operator_error": error,
                    "lower_geometry_id": int(
                        geometry_table.iloc[lower_index]["geometry_id"]
                    ),
                    "upper_geometry_id": int(
                        geometry_table.iloc[upper_index]["geometry_id"]
                    ),
                    "upper_weight": float(interpolation_weight),
                }
            )
    linear_operator_errors = pd.DataFrame(linear_operator_rows)
    linear_operator_errors.to_csv(
        run_dir / "linear_neighbor_operator_errors.csv", index=False
    )
    np.savez_compressed(
        run_dir / "held_out_linear_neighbor_operators.npz",
        Kq=K_linear,
        Bq=B_linear,
        Dq=D_linear,
        lower_volume_fraction=np.asarray(volume_fractions[lower_index]),
        upper_volume_fraction=np.asarray(volume_fractions[upper_index]),
        upper_weight=np.asarray(interpolation_weight),
    )
    del basis, basis_fields
    gc.collect()

    held_geometry = geometries[held_out_index]
    validation_rows: list[dict[str, Any]] = []
    truth_tensors: list[np.ndarray] = []
    validation_fft_started = time.perf_counter()
    for material_index, material_row in validation_materials.iterrows():
        material_dir = run_dir / "validation_fft" / f"material_{int(material_index):04d}"
        with quiet_output(bool(args.quiet_solver)):
            record = common.solve_material(
                material_row=material_payload(material_row),
                material_dir=material_dir,
                geometry=held_geometry,
                runtime=runtime,
                profile=str(experiment["solver_profile"]),
                seed=int(experiment["validation_material_seed"]) + int(material_index),
                save_solution_fields=False,
                persistent_gpu_cache=False,
            )
        truth_tensors.append(np.load(record["Ceff_path"]))
        validation_rows.append(
            {
                "material_id": int(material_index),
                "material_label": str(material_row["material_label"]),
                "fft_solve_wall_s": float(record["solve_wall_s"]),
                "fft_max_iterations": int(record["solver_max_iterations"]),
                "fft_max_relative_residual": float(
                    record["solver_max_relative_residual"]
                ),
            }
        )
        print(
            f"[VALIDATION FFT] material={material_index + 1}/{validation_count} "
            f"time={float(record['solve_wall_s']):.2f}s",
            flush=True,
        )
    validation_fft_wall_s = float(time.perf_counter() - validation_fft_started)
    timings.append({"stage": "held_out_validation_fft", "wall_s": validation_fft_wall_s})

    coefficients = reduced._material_coefficients_batch(
        parameter_matrix(validation_materials)
    )
    exact_rom, _, exact_online_wall_s = reduced._rom_ceff_batch(
        coefficients,
        K_all[held_out_index],
        B_all[held_out_index],
        D_all[held_out_index],
    )
    interpolated_rom, _, interpolated_online_wall_s = reduced._rom_ceff_batch(
        coefficients,
        K_interpolated,
        B_interpolated,
        D_interpolated,
    )
    linear_rom, _, linear_online_wall_s = reduced._rom_ceff_batch(
        coefficients,
        K_linear,
        B_linear,
        D_linear,
    )
    timings.extend(
        [
            {"stage": "exact_geometry_rom_batch", "wall_s": exact_online_wall_s},
            {
                "stage": "interpolated_geometry_rom_batch",
                "wall_s": interpolated_online_wall_s,
            },
            {
                "stage": "linear_neighbor_geometry_rom_batch",
                "wall_s": linear_online_wall_s,
            },
        ]
    )
    truth = np.asarray(truth_tensors, dtype=np.float64)
    error_basis = relative_frobenius(exact_rom, truth)
    error_geometry = relative_frobenius(interpolated_rom, exact_rom)
    error_total = relative_frobenius(interpolated_rom, truth)
    error_linear_geometry = relative_frobenius(linear_rom, exact_rom)
    error_linear_total = relative_frobenius(linear_rom, truth)
    validation_frame = pd.DataFrame(validation_rows)
    validation_frame["error_exact_geometry_rom_vs_fft"] = error_basis
    validation_frame["error_interpolated_geometry_vs_exact_rom"] = error_geometry
    validation_frame["error_interpolated_geometry_rom_vs_fft"] = error_total
    validation_frame[
        "error_linear_neighbor_geometry_vs_exact_rom"
    ] = error_linear_geometry
    validation_frame["error_linear_neighbor_geometry_rom_vs_fft"] = error_linear_total
    validation_frame["exact_rom_min_eig"] = [
        float(np.linalg.eigvalsh(matrix).min()) for matrix in exact_rom
    ]
    validation_frame["interpolated_rom_min_eig"] = [
        float(np.linalg.eigvalsh(matrix).min()) for matrix in interpolated_rom
    ]
    validation_frame["linear_neighbor_rom_min_eig"] = [
        float(np.linalg.eigvalsh(matrix).min()) for matrix in linear_rom
    ]
    tensor_columns: dict[str, np.ndarray] = {}
    for label, matrices in (
        ("fft", truth),
        ("exact_rom", exact_rom),
        ("interpolated_rom", interpolated_rom),
        ("linear_neighbor_rom", linear_rom),
    ):
        for ii in range(6):
            for jj in range(6):
                tensor_columns[f"{label}_C{ii + 1}{jj + 1}"] = matrices[:, ii, jj]
    validation_frame = pd.concat(
        [validation_frame, pd.DataFrame(tensor_columns)], axis=1
    )
    validation_frame.to_csv(run_dir / "held_out_validation_results.csv", index=False)

    summary = {
        "held_out_geometry_id": int(geometry_table.iloc[held_out_index]["geometry_id"]),
        "held_out_Vf_target": float(geometry_table.iloc[held_out_index]["Vf_target"]),
        "held_out_grid_size": int(geometry_table.iloc[held_out_index]["grid_size"]),
        "training_geometry_count": len(training_indices),
        "snapshot_material_count_per_geometry": snapshot_count,
        "validation_material_count": validation_count,
        "common_basis_rank": rank,
        "common_basis_retained_energy": float(basis_metadata["retained_energy"]),
        "operator_error_max": float(
            operator_errors["held_out_relative_operator_error"].max()
        ),
        "operator_error_p95": float(
            operator_errors["held_out_relative_operator_error"].quantile(0.95)
        ),
        **summarize_errors(error_basis, "basis_error"),
        **summarize_errors(error_geometry, "geometry_interpolation_error"),
        **summarize_errors(error_total, "total_error"),
        **summarize_errors(error_linear_geometry, "linear_geometry_error"),
        **summarize_errors(error_linear_total, "linear_total_error"),
        "all_exact_rom_spd": bool(np.all(validation_frame["exact_rom_min_eig"] > 0.0)),
        "all_interpolated_rom_spd": bool(
            np.all(validation_frame["interpolated_rom_min_eig"] > 0.0)
        ),
        "all_linear_neighbor_rom_spd": bool(
            np.all(validation_frame["linear_neighbor_rom_min_eig"] > 0.0)
        ),
        "linear_neighbor_lower_geometry_id": int(
            geometry_table.iloc[lower_index]["geometry_id"]
        ),
        "linear_neighbor_upper_geometry_id": int(
            geometry_table.iloc[upper_index]["geometry_id"]
        ),
        "linear_neighbor_upper_weight": float(interpolation_weight),
        "linear_operator_error_max": float(
            linear_operator_errors["held_out_relative_operator_error"].max()
        ),
        "linear_operator_error_p95": float(
            linear_operator_errors["held_out_relative_operator_error"].quantile(0.95)
        ),
        "training_fft_wall_s": training_fft_wall_s,
        "common_basis_wall_s": float(basis_metadata["basis_total_wall_s"]),
        "native_operator_assembly_wall_s": assembly_wall_s,
        "operator_pod_wall_s": operator_pod_wall_s,
        "validation_fft_wall_s": validation_fft_wall_s,
        "exact_rom_batch_wall_s": exact_online_wall_s,
        "interpolated_rom_batch_wall_s": interpolated_online_wall_s,
        "linear_neighbor_rom_batch_wall_s": linear_online_wall_s,
        "total_wall_s": float(time.perf_counter() - started),
    }
    summary_frame = pd.DataFrame([summary])
    summary_frame.to_csv(run_dir / "feasibility_summary.csv", index=False)
    timing_frame = pd.DataFrame(timings)
    timing_frame.to_csv(run_dir / "stage_timings.csv", index=False)

    plot_results(run_dir, basis_spectrum, operator_spectrum, validation_frame)
    with pd.ExcelWriter(run_dir / "feasibility_report.xlsx") as writer:
        summary_frame.to_excel(writer, sheet_name="summary", index=False)
        geometry_table.to_excel(writer, sheet_name="geometries", index=False)
        snapshot_materials.to_excel(writer, sheet_name="snapshot_materials", index=False)
        snapshot_frame.to_excel(writer, sheet_name="training_fft", index=False)
        basis_spectrum.to_excel(writer, sheet_name="basis_spectrum", index=False)
        assembly_frame.to_excel(writer, sheet_name="operator_assembly", index=False)
        operator_errors.to_excel(writer, sheet_name="operator_errors", index=False)
        linear_operator_errors.to_excel(
            writer, sheet_name="linear_operator_errors", index=False
        )
        operator_cv.to_excel(writer, sheet_name="operator_cv", index=False)
        validation_frame.to_excel(writer, sheet_name="validation", index=False)
        timing_frame.to_excel(writer, sheet_name="timings", index=False)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete",
            "common_basis": basis_metadata,
            "summary": summary,
            "heavy_snapshot_fields_retained": False,
            "common_basis_retained": False,
        }
    )
    write_json(manifest_path, manifest)
    (out_root / "latest_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    print(
        f"[DONE] total p95={summary['total_error_p95']:.3e} | "
        f"max={summary['total_error_max']:.3e} | run={run_dir}",
        flush=True,
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--snapshot-materials", type=int, default=None)
    parser.add_argument("--validation-materials", type=int, default=None)
    parser.add_argument("--rank-cap", type=int, default=None)
    parser.add_argument("--quiet-solver", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
