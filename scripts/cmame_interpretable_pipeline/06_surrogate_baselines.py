#!/usr/bin/env python3
"""Compare the frozen Sobol--Ritz ROM with data-only surrogate baselines."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import pdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rom_reduced_operator as reduced


TRIANGLE = np.triu_indices(6)
METHOD_LABELS = {
    "sobol_pod": "Sobol--Ritz",
    "rbf_cv": "RBF (LOOCV)",
    "kriging_selected": "Kriging (LML)",
}

RBF_KERNELS_WITHOUT_SHAPE = ("linear", "thin_plate_spline", "cubic", "quintic")
RBF_KERNELS_WITH_SHAPE = (
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian",
)
RBF_EPSILON_MULTIPLIERS = (
    0.000244140625,
    0.00048828125,
    0.0009765625,
    0.001953125,
    0.00390625,
    0.0078125,
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
)
RBF_SMOOTHING_VALUES = (0.0, 1.0e-8, 1.0e-5, 1.0e-3, 1.0e-1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--base-run-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-figure-dir", type=Path, default=None)
    parser.add_argument("--geometry-ids", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--minimum-prefix", type=int, default=2)
    parser.add_argument("--tau-G", type=float, default=1.0e-6)
    parser.add_argument("--kriging-selection-restarts", type=int, default=2)
    parser.add_argument("--kriging-restarts", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260830)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Independent geometry workers; keep BLAS threads at one when jobs > 1.",
    )
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_to_tensors(frame: pd.DataFrame, prefix: str = "Ceff") -> np.ndarray:
    tensors = np.empty((len(frame), 6, 6), dtype=np.float64)
    for ii in range(6):
        for jj in range(6):
            tensors[:, ii, jj] = frame[f"{prefix}_{ii + 1}{jj + 1}"].to_numpy(
                dtype=np.float64
            )
    return 0.5 * (tensors + np.swapaxes(tensors, -1, -2))


def tensors_to_independent(tensors: np.ndarray) -> np.ndarray:
    values = np.asarray(tensors, dtype=np.float64)
    return values[:, TRIANGLE[0], TRIANGLE[1]]


def independent_to_tensors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    tensors = np.zeros((len(values), 6, 6), dtype=np.float64)
    tensors[:, TRIANGLE[0], TRIANGLE[1]] = values
    tensors[:, TRIANGLE[1], TRIANGLE[0]] = values
    return tensors


def affine_coefficients(frame: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            reduced._material_coefficients(row)
            for row in frame.to_dict(orient="records")
        ]
    ).astype(np.float64, copy=False)


def normalize_affine_inputs(
    frame: pd.DataFrame, candidate_pool: pd.DataFrame
) -> np.ndarray:
    """Normalize inputs with the declared candidate domain, without FOM labels."""
    pool = affine_coefficients(candidate_pool)
    lower = np.min(pool, axis=0)
    scale = np.maximum(np.max(pool, axis=0) - lower, np.finfo(float).eps)
    return (affine_coefficients(frame) - lower) / scale


def standardize_outputs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    floor = 1.0e-12 * np.maximum(np.abs(mean), 1.0)
    scale = np.maximum(scale, floor)
    return (values - mean) / scale, mean, scale


def _rbf_epsilon_reference(train_x: np.ndarray) -> float:
    distances = pdist(np.asarray(train_x, dtype=np.float64))
    positive = distances[distances > np.finfo(float).eps]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def rbf_configurations(train_x: np.ndarray) -> list[dict[str, float | int | str]]:
    """Return the declared RBF hyperparameter search space for one prefix."""
    train_x = np.asarray(train_x, dtype=np.float64)
    epsilon_reference = _rbf_epsilon_reference(train_x)
    configurations: list[dict[str, float | int | str]] = []

    for kernel in RBF_KERNELS_WITHOUT_SHAPE:
        degrees = [-1]
        if kernel == "linear":
            degrees.append(0)
        if len(train_x) - 1 >= train_x.shape[1] + 1 and kernel in {
            "linear",
            "thin_plate_spline",
            "cubic",
        }:
            degrees.append(1)
        for degree in degrees:
            for smoothing in RBF_SMOOTHING_VALUES:
                configurations.append(
                    {
                        "kernel": kernel,
                        "epsilon": 1.0,
                        "epsilon_multiplier": 0.0,
                        "smoothing": float(smoothing),
                        "degree": int(degree),
                    }
                )

    shape_degrees = [-1, 0]
    if len(train_x) - 1 >= train_x.shape[1] + 1:
        shape_degrees.append(1)
    for kernel in RBF_KERNELS_WITH_SHAPE:
        for degree in shape_degrees:
            for multiplier in RBF_EPSILON_MULTIPLIERS:
                epsilon = float(multiplier) / epsilon_reference
                for smoothing in RBF_SMOOTHING_VALUES:
                    configurations.append(
                        {
                            "kernel": kernel,
                            "epsilon": epsilon,
                            "epsilon_multiplier": float(multiplier),
                            "smoothing": float(smoothing),
                            "degree": int(degree),
                        }
                    )
    return configurations


def _fit_rbf(
    train_x: np.ndarray,
    train_y: np.ndarray,
    configuration: dict[str, float | int | str],
) -> RBFInterpolator:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return RBFInterpolator(
            np.asarray(train_x, dtype=np.float64),
            np.asarray(train_y, dtype=np.float64),
            kernel=str(configuration["kernel"]),
            epsilon=float(configuration["epsilon"]),
            smoothing=float(configuration["smoothing"]),
            degree=int(configuration["degree"]),
        )


def _rbf_loocv_errors(
    train_x: np.ndarray,
    train_tensors: np.ndarray,
    configuration: dict[str, float | int | str],
) -> np.ndarray:
    predictions = np.empty_like(train_tensors, dtype=np.float64)
    for heldout_index in range(len(train_x)):
        keep = np.arange(len(train_x)) != heldout_index
        fold_values, fold_mean, fold_scale = standardize_outputs(
            tensors_to_independent(train_tensors[keep])
        )
        model = _fit_rbf(train_x[keep], fold_values, configuration)
        standardized = np.asarray(
            model(np.asarray(train_x[heldout_index : heldout_index + 1])),
            dtype=np.float64,
        )
        predictions[heldout_index] = independent_to_tensors(
            standardized * fold_scale + fold_mean
        )[0]
    return relative_frobenius_errors(predictions, train_tensors)


def select_rbf_configuration(
    train_x: np.ndarray, train_tensors: np.ndarray
) -> tuple[dict[str, float | int | str], float, float]:
    """Select RBF hyperparameters by deterministic training-only LOOCV."""
    ranked: list[
        tuple[float, float, str, dict[str, float | int | str]]
    ] = []
    for configuration in rbf_configurations(train_x):
        try:
            errors = _rbf_loocv_errors(train_x, train_tensors, configuration)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(errors)):
            continue
        serialized = json.dumps(configuration, sort_keys=True)
        ranked.append(
            (float(np.mean(errors)), float(np.max(errors)), serialized, configuration)
        )
    if not ranked:
        raise RuntimeError("No RBF hyperparameter configuration completed LOOCV.")
    mean_error, maximum_error, _, selected = min(ranked)
    return dict(selected), mean_error, maximum_error


def rbf_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    configuration: dict[str, float | int | str] | None = None,
) -> np.ndarray:
    """Evaluate one declared RBF configuration."""
    if configuration is None:
        configuration = {
            "kernel": "linear",
            "epsilon": 1.0,
            "smoothing": 0.0,
            "degree": 0,
        }
    model = _fit_rbf(train_x, train_y, configuration)
    return np.asarray(model(query_x), dtype=np.float64)


def kriging_kernel_candidates(
    n_features: int,
) -> list[tuple[str, object]]:
    """Covariance families considered by training marginal likelihood."""
    isotropic = 1.0
    ard = np.ones(int(n_features), dtype=np.float64)
    length_bounds = (1.0e-2, 1.0e2)
    candidates: list[tuple[str, object]] = [
        ("Matern-1/2 isotropic", Matern(isotropic, length_bounds, nu=0.5)),
        ("Matern-3/2 isotropic", Matern(isotropic, length_bounds, nu=1.5)),
        ("Matern-5/2 isotropic", Matern(isotropic, length_bounds, nu=2.5)),
        ("squared-exponential isotropic", RBF(isotropic, length_bounds)),
        (
            "rational-quadratic isotropic",
            RationalQuadratic(
                length_scale=1.0,
                alpha=1.0,
                length_scale_bounds=length_bounds,
                alpha_bounds=(1.0e-2, 1.0e2),
            ),
        ),
        ("Matern-1/2 ARD", Matern(ard, length_bounds, nu=0.5)),
        ("Matern-3/2 ARD", Matern(ard, length_bounds, nu=1.5)),
        ("Matern-5/2 ARD", Matern(ard, length_bounds, nu=2.5)),
        ("squared-exponential ARD", RBF(ard, length_bounds)),
    ]
    kernels: list[tuple[str, object]] = []
    for name, covariance in candidates:
        kernels.append(
            (
                name,
                ConstantKernel(1.0, (1.0e-3, 1.0e3)) * covariance
                + WhiteKernel(1.0e-6, (1.0e-10, 1.0e-1)),
            )
        )
    kernels.append(
        (
            "Matern-5/2 plus linear trend",
            ConstantKernel(1.0, (1.0e-3, 1.0e3))
            * Matern(1.0, length_bounds, nu=2.5)
            + ConstantKernel(1.0, (1.0e-3, 1.0e3))
            * DotProduct(sigma_0=1.0, sigma_0_bounds=(1.0e-3, 1.0e3))
            + WhiteKernel(1.0e-6, (1.0e-10, 1.0e-1)),
        )
    )
    return kernels


def _fit_kriging(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    kernel: object,
    restarts: int,
    random_seed: int,
) -> GaussianProcessRegressor:
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1.0e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=int(restarts),
        normalize_y=False,
        random_state=int(random_seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(train_x, train_y)
    return model


def select_kriging_and_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    selection_restarts: int,
    final_restarts: int,
    random_seed: int,
) -> tuple[np.ndarray, str, str, float, list[dict[str, float | str | None]]]:
    """Select a covariance family and optimize its continuous hyperparameters."""
    ranked: list[tuple[float, str, object]] = []
    candidate_scores: list[dict[str, float | str | None]] = []
    for index, (name, kernel) in enumerate(
        kriging_kernel_candidates(np.asarray(train_x).shape[1])
    ):
        try:
            model = _fit_kriging(
                train_x,
                train_y,
                kernel=kernel,
                restarts=selection_restarts,
                random_seed=int(random_seed) + index,
            )
        except (ValueError, np.linalg.LinAlgError):
            candidate_scores.append(
                {"family": name, "log_marginal_likelihood": None}
            )
            continue
        score = float(model.log_marginal_likelihood_value_)
        candidate_scores.append({"family": name, "log_marginal_likelihood": score})
        if np.isfinite(score):
            ranked.append((-score, name, kernel))
    if not ranked:
        raise RuntimeError("No Kriging covariance family completed optimization.")
    negative_score, selected_name, selected_kernel = min(ranked, key=lambda row: (row[0], row[1]))
    final_model = _fit_kriging(
        train_x,
        train_y,
        kernel=selected_kernel,
        restarts=final_restarts,
        random_seed=int(random_seed) + 1000,
    )
    prediction = np.asarray(final_model.predict(query_x), dtype=np.float64)
    return (
        prediction,
        str(final_model.kernel_),
        selected_name,
        float(-negative_score),
        candidate_scores,
    )


def relative_frobenius_errors(
    prediction: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    numerator = np.linalg.norm(
        (prediction - reference).reshape(len(reference), -1), axis=1
    )
    denominator = np.maximum(
        np.linalg.norm(reference.reshape(len(reference), -1), axis=1),
        np.finfo(float).eps,
    )
    return numerator / denominator


def summarize_details(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    keys = ["geometry_id", "training_materials", "method"]
    for key, frame in details.groupby(keys, sort=True):
        errors = frame["relative_frobenius_error"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "geometry_id": int(key[0]),
                "training_materials": int(key[1]),
                "method": str(key[2]),
                "method_label": METHOD_LABELS[str(key[2])],
                "heldout_count": int(len(frame)),
                "error_mean": float(np.mean(errors)),
                "error_median": float(np.median(errors)),
                "error_max": float(np.max(errors)),
                "below_1e-4_count": int(np.count_nonzero(errors <= 1.0e-4)),
                "spd_count": int(np.count_nonzero(frame["prediction_min_eig"] > 0.0)),
                "prediction_min_eig": float(frame["prediction_min_eig"].min()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_prediction(
    *,
    geometry_id: int,
    training_materials: int,
    method: str,
    validation: pd.DataFrame,
    prediction: np.ndarray,
    reference: np.ndarray,
    fit_wall_s: float,
    predict_wall_s: float,
    fitted_kernel: str = "",
    selected_configuration: str = "",
    selection_score: float = np.nan,
) -> pd.DataFrame:
    errors = relative_frobenius_errors(prediction, reference)
    minimum_eigenvalues = np.linalg.eigvalsh(prediction)[:, 0]
    return pd.DataFrame(
        {
            "geometry_id": int(geometry_id),
            "training_materials": int(training_materials),
            "method": str(method),
            "final_validation_id": validation["final_validation_id"].to_numpy(
                dtype=int
            ),
            "validation_pool_id": validation["validation_pool_id"].to_numpy(
                dtype=int
            ),
            "relative_frobenius_error": errors,
            "prediction_min_eig": minimum_eigenvalues,
            "prediction_spd": minimum_eigenvalues > 0.0,
            "below_1e-4": errors <= 1.0e-4,
            "fit_wall_s": float(fit_wall_s),
            "predict_wall_s": float(predict_wall_s),
            "fitted_kernel": str(fitted_kernel),
            "selected_configuration": str(selected_configuration),
            "selection_score": float(selection_score),
        }
    )


def prefix_rom_prediction(
    *,
    operators: dict[str, np.ndarray],
    validation: pd.DataFrame,
    training_materials: int,
    tau_g: float,
) -> tuple[np.ndarray, int, float]:
    raw_columns = 6 * int(training_materials)
    started = time.perf_counter()
    raw_Kq = np.asarray(
        operators["raw_Kq"][:, :raw_columns, :raw_columns]
    )
    raw_Bq = np.asarray(operators["raw_Bq"][:, :raw_columns])
    if "energy_qr_reference_coefficients" in operators:
        energy_operators, _ = reduced._reference_energy_qr_recompile(
            raw_Kq=raw_Kq,
            raw_Bq=raw_Bq,
            Dq=np.asarray(operators["Dq"]),
            reference_coefficients=np.asarray(
                operators["energy_qr_reference_coefficients"]
            ),
        )
        Kq = energy_operators["Kq"]
        Bq = energy_operators["Bq"]
    else:
        Kq, Bq, _, _ = reduced._transform_raw_operators_with_rank_policy(
            raw_Kq,
            raw_Bq,
            np.asarray(operators["G"][:raw_columns, :raw_columns]),
            allow_rank_reveal=True,
            rank_rtol=float(tau_g),
        )
    frame = reduced._evaluate_rom(
        results_df=validation,
        Kq=Kq,
        Bq=Bq,
        Dq=operators["Dq"],
    )
    prediction = frame_to_tensors(frame, prefix="Crom")
    return prediction, int(Kq.shape[1]), float(time.perf_counter() - started)


def benchmark_geometry(
    *,
    run_dir: Path,
    geometry_id: int,
    minimum_prefix: int,
    tau_g: float,
    kriging_selection_restarts: int,
    kriging_restarts: int,
    random_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    training = pd.read_csv(run_dir / "snapshot_timing.csv")
    validation = pd.read_csv(run_dir / "final_validation_truth_results.csv")
    candidate_pool = pd.read_csv(run_dir / "candidate_pool_used.csv")
    reference = frame_to_tensors(validation)
    validation_x = normalize_affine_inputs(validation, candidate_pool)
    manifests: list[dict[str, object]] = []
    details: list[pd.DataFrame] = []
    with np.load(run_dir / "reduced_operators.npz") as archive:
        operators = {name: archive[name] for name in archive.files}

    for count in range(int(minimum_prefix), len(training) + 1):
        prefix = training.iloc[:count].copy()
        train_x = normalize_affine_inputs(prefix, candidate_pool)
        train_tensor = frame_to_tensors(prefix)
        train_y, output_mean, output_scale = standardize_outputs(
            tensors_to_independent(train_tensor)
        )

        started = time.perf_counter()
        rom_prediction, rank, rom_wall_s = prefix_rom_prediction(
            operators=operators,
            validation=validation,
            training_materials=count,
            tau_g=tau_g,
        )
        details.append(
            evaluate_prediction(
                geometry_id=geometry_id,
                training_materials=count,
                method="sobol_pod",
                validation=validation,
                prediction=rom_prediction,
                reference=reference,
                fit_wall_s=float(time.perf_counter() - started),
                predict_wall_s=rom_wall_s,
            )
        )

        started = time.perf_counter()
        rbf_configuration, rbf_cv_mean, rbf_cv_max = select_rbf_configuration(
            train_x, train_tensor
        )
        rbf_model = _fit_rbf(train_x, train_y, rbf_configuration)
        rbf_fit_s = float(time.perf_counter() - started)
        started = time.perf_counter()
        rbf_values = np.asarray(rbf_model(validation_x), dtype=np.float64)
        rbf_predict_s = float(time.perf_counter() - started)
        rbf_prediction = independent_to_tensors(
            rbf_values * output_scale + output_mean
        )
        details.append(
            evaluate_prediction(
                geometry_id=geometry_id,
                training_materials=count,
                method="rbf_cv",
                validation=validation,
                prediction=rbf_prediction,
                reference=reference,
                fit_wall_s=rbf_fit_s,
                predict_wall_s=rbf_predict_s,
                selected_configuration=json.dumps(
                    rbf_configuration, sort_keys=True
                ),
                selection_score=rbf_cv_mean,
            )
        )

        started = time.perf_counter()
        (
            kriging_values,
            fitted_kernel,
            kriging_family,
            kriging_lml,
            kriging_candidate_scores,
        ) = select_kriging_and_predict(
            train_x,
            train_y,
            validation_x,
            selection_restarts=kriging_selection_restarts,
            final_restarts=kriging_restarts,
            random_seed=int(random_seed) + 100 * int(geometry_id) + count,
        )
        kriging_total_s = float(time.perf_counter() - started)
        kriging_prediction = independent_to_tensors(
            kriging_values * output_scale + output_mean
        )
        details.append(
            evaluate_prediction(
                geometry_id=geometry_id,
                training_materials=count,
                method="kriging_selected",
                validation=validation,
                prediction=kriging_prediction,
                reference=reference,
                fit_wall_s=kriging_total_s,
                predict_wall_s=0.0,
                fitted_kernel=fitted_kernel,
                selected_configuration=kriging_family,
                selection_score=kriging_lml,
            )
        )
        manifests.append(
            {
                "geometry_id": int(geometry_id),
                "training_materials": int(count),
                "pod_rank": int(rank),
                "rbf_configuration": rbf_configuration,
                "rbf_loocv_mean_relative_error": float(rbf_cv_mean),
                "rbf_loocv_max_relative_error": float(rbf_cv_max),
                "kriging_selected_family": kriging_family,
                "kriging_selected_log_marginal_likelihood": float(kriging_lml),
                "kriging_kernel": fitted_kernel,
                "kriging_candidate_scores": kriging_candidate_scores,
            }
        )
        print(
            f"[BASELINE] G{geometry_id:02d} prefix={count}/{len(training)} "
            f"rank={rank} RBF={rbf_configuration['kernel']} "
            f"Kriging={kriging_family}",
            flush=True,
        )
    return pd.concat(details, ignore_index=True), manifests


def final_prefix_rows(details: pd.DataFrame) -> pd.DataFrame:
    final_counts = details.groupby("geometry_id")["training_materials"].max()
    return pd.concat(
        [
            details[
                (details["geometry_id"] == geometry_id)
                & (details["training_materials"] == count)
            ]
            for geometry_id, count in final_counts.items()
        ],
        ignore_index=True,
    )


def write_learning_figure(summary: pd.DataFrame, output: Path) -> None:
    colors = {
        "sobol_pod": "#0072B2",
        "rbf_cv": "#D55E00",
        "kriging_selected": "#009E73",
    }
    markers = {"sobol_pod": "o", "rbf_cv": "s", "kriging_selected": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.0), constrained_layout=True)
    for axis, geometry_id in zip(axes, (3, 9), strict=True):
        selected = summary[summary["geometry_id"] == geometry_id]
        for method in METHOD_LABELS:
            data = selected[selected["method"] == method].sort_values(
                "training_materials"
            )
            axis.plot(
                data["training_materials"],
                data["error_mean"],
                color=colors[method],
                marker=markers[method],
                markersize=3.5,
                linewidth=1.2,
                label=METHOD_LABELS[method],
            )
        axis.axhline(1.0e-4, color="black", linestyle="--", linewidth=0.9)
        axis.set_yscale("log")
        axis.set_xlabel("Training materials")
        axis.set_title(f"G{geometry_id:02d}")
        axis.grid(True, which="both", linewidth=0.35, alpha=0.45)
    axes[0].set_ylabel("Mean held-out relative tensor error")
    axes[0].legend(frameon=False, fontsize=7)
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def main() -> int:
    args = _parser().parse_args()
    runs_root = args.runs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_details: list[pd.DataFrame] = []
    prefix_manifests: list[dict[str, object]] = []
    source_hashes: list[dict[str, object]] = []

    geometry_ids = [int(value) for value in args.geometry_ids]
    run_directories = {
        geometry_id: runs_root / f"{args.base_run_name}_geometry_{geometry_id:02d}"
        for geometry_id in geometry_ids
    }
    for geometry_id, run_dir in run_directories.items():
        source_hashes.append(
            {
                "geometry_id": int(geometry_id),
                "snapshot_timing_sha256": sha256(run_dir / "snapshot_timing.csv"),
                "validation_truth_sha256": sha256(
                    run_dir / "final_validation_truth_results.csv"
                ),
                "frozen_operator_sha256": sha256(run_dir / "reduced_operators.npz"),
            }
        )

    def collect_result(result: tuple[pd.DataFrame, list[dict[str, object]]]) -> None:
        details, manifests = result
        all_details.append(details)
        prefix_manifests.extend(manifests)

    jobs = max(1, min(int(args.jobs), len(geometry_ids)))
    if jobs == 1:
        for geometry_id in geometry_ids:
            collect_result(
                benchmark_geometry(
                    run_dir=run_directories[geometry_id],
                    geometry_id=geometry_id,
                    minimum_prefix=int(args.minimum_prefix),
                    tau_g=float(args.tau_G),
                    kriging_selection_restarts=int(args.kriging_selection_restarts),
                    kriging_restarts=int(args.kriging_restarts),
                    random_seed=int(args.random_seed),
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    benchmark_geometry,
                    run_dir=run_directories[geometry_id],
                    geometry_id=geometry_id,
                    minimum_prefix=int(args.minimum_prefix),
                    tau_g=float(args.tau_G),
                    kriging_selection_restarts=int(args.kriging_selection_restarts),
                    kriging_restarts=int(args.kriging_restarts),
                    random_seed=int(args.random_seed),
                ): geometry_id
                for geometry_id in geometry_ids
            }
            for future in as_completed(futures):
                collect_result(future.result())

    prefix_manifests.sort(
        key=lambda row: (int(row["geometry_id"]), int(row["training_materials"]))
    )

    details = pd.concat(all_details, ignore_index=True).sort_values(
        ["geometry_id", "training_materials", "method", "final_validation_id"]
    ).reset_index(drop=True)
    summary = summarize_details(details)
    final_details = final_prefix_rows(details)
    final_summary = summarize_details(final_details)
    global_rows: list[dict[str, object]] = []
    for method, frame in final_details.groupby("method", sort=True):
        errors = frame["relative_frobenius_error"].to_numpy(dtype=np.float64)
        global_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "comparison_count": int(len(frame)),
                "error_mean": float(np.mean(errors)),
                "error_median": float(np.median(errors)),
                "error_p95": float(np.quantile(errors, 0.95)),
                "error_max": float(np.max(errors)),
                "below_1e-4_count": int(np.count_nonzero(errors <= 1.0e-4)),
                "spd_count": int(np.count_nonzero(frame["prediction_spd"])),
                "prediction_min_eig": float(frame["prediction_min_eig"].min()),
            }
        )
    global_summary = pd.DataFrame(global_rows)

    details.to_csv(output_dir / "surrogate_baseline_details.csv", index=False)
    summary.to_csv(output_dir / "surrogate_baseline_learning_curves.csv", index=False)
    final_summary.to_csv(
        output_dir / "surrogate_baseline_final_by_geometry.csv", index=False
    )
    global_summary.to_csv(
        output_dir / "surrogate_baseline_global_summary.csv", index=False
    )
    hyperparameter_rows: list[dict[str, object]] = []
    for entry in prefix_manifests:
        rbf_configuration = dict(entry["rbf_configuration"])
        hyperparameter_rows.append(
            {
                "geometry_id": int(entry["geometry_id"]),
                "training_materials": int(entry["training_materials"]),
                "pod_rank": int(entry["pod_rank"]),
                "rbf_kernel": str(rbf_configuration["kernel"]),
                "rbf_degree": int(rbf_configuration["degree"]),
                "rbf_epsilon": float(rbf_configuration["epsilon"]),
                "rbf_epsilon_multiplier": float(
                    rbf_configuration["epsilon_multiplier"]
                ),
                "rbf_smoothing": float(rbf_configuration["smoothing"]),
                "rbf_loocv_mean_relative_error": float(
                    entry["rbf_loocv_mean_relative_error"]
                ),
                "rbf_loocv_max_relative_error": float(
                    entry["rbf_loocv_max_relative_error"]
                ),
                "kriging_selected_family": str(entry["kriging_selected_family"]),
                "kriging_log_marginal_likelihood": float(
                    entry["kriging_selected_log_marginal_likelihood"]
                ),
                "kriging_fitted_kernel": str(entry["kriging_kernel"]),
            }
        )
    hyperparameters = pd.DataFrame(hyperparameter_rows).sort_values(
        ["geometry_id", "training_materials"]
    )
    hyperparameters.to_csv(
        output_dir / "surrogate_baseline_prefix_hyperparameters.csv", index=False
    )
    hyperparameters.groupby("geometry_id", sort=True).tail(1).to_csv(
        output_dir / "surrogate_baseline_final_hyperparameters.csv", index=False
    )
    protocol = {
        "status": "post_freeze_data_only_baseline",
        "base_run_name": str(args.base_run_name),
        "geometry_ids": [int(value) for value in args.geometry_ids],
        "geometry_parallel_jobs": int(jobs),
        "minimum_prefix": int(args.minimum_prefix),
        "legacy_tau_G": float(args.tau_G),
        "rom_prefix_coordinates": (
            "reference_energy_cholesky_qr when present; legacy Euclidean "
            "Gram rank policy otherwise"
        ),
        "input_coordinates": "seven affine coefficients normalized by candidate-pool coordinate ranges",
        "outputs": "21 independent upper-triangular Mandel stiffness entries standardized on each training prefix",
        "rbf": {
            "selection": "minimum mean leave-one-out relative Frobenius tensor error on the current training prefix",
            "kernels_without_shape_parameter": list(RBF_KERNELS_WITHOUT_SHAPE),
            "kernels_with_shape_parameter": list(RBF_KERNELS_WITH_SHAPE),
            "epsilon_multipliers_over_inverse_median_pair_distance": list(
                RBF_EPSILON_MULTIPLIERS
            ),
            "smoothing_values": list(RBF_SMOOTHING_VALUES),
            "polynomial_degrees": "-1 and admissible 0/1 values for the current leave-one-out fold size",
        },
        "kriging": {
            "candidate_families": [
                name for name, _ in kriging_kernel_candidates(7)
            ],
            "family_selection": "maximum optimized training log marginal likelihood on the current prefix",
            "continuous_hyperparameters": "covariance amplitude, isotropic or ARD length scales, rational-quadratic alpha when applicable, and nugget",
            "selection_optimizer_restarts": int(args.kriging_selection_restarts),
            "final_optimizer_restarts": int(args.kriging_restarts),
            "random_seed": int(args.random_seed),
        },
        "heldout_used_for_model_selection": False,
        "new_fom_solves": 0,
        "source_hashes": source_hashes,
        "prefix_models": prefix_manifests,
    }
    (output_dir / "surrogate_baseline_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )

    figure_path = output_dir / "surrogate_sample_efficiency"
    write_learning_figure(summary, figure_path)
    if args.paper_figure_dir is not None:
        paper_dir = args.paper_figure_dir.resolve()
        paper_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "png"):
            target = paper_dir / f"numerical_surrogate_sample_efficiency.{suffix}"
            target.write_bytes(figure_path.with_suffix(f".{suffix}").read_bytes())

    print(global_summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
