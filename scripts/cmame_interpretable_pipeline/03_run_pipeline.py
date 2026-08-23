#!/usr/bin/env python3
"""Step 3: run the fixed Sobol + full-rank POD campaign."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from env_bootstrap import ensure_configured_venv


HERE = Path(__file__).resolve().parent
CONFIG_DEFAULT = HERE / "campaign_config.json"
ensure_configured_venv(CONFIG_DEFAULT)

import pandas as pd

from validation_reporting import empirical_coverage


os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmame-matplotlib")


ROOT = Path(__file__).resolve().parents[2]
SOBOL_POD_SCRIPT = HERE / "04_sobol_pod_pipeline.py"


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def out_root(config: dict[str, Any], override: Path | None) -> Path:
    return (override or project_path(config["paths"]["out_root"])).resolve()


def pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    if "sobol_pod_pipeline" not in config:
        raise KeyError("campaign_config.json must define sobol_pod_pipeline.")
    return config["sobol_pod_pipeline"]


def run_name_now() -> str:
    return "run_sobol_pod_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def geometry_design(destination: Path) -> pd.DataFrame:
    path = destination / "geometries" / "geometry_design.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing design table: {path}. Run step 01 first.")
    return pd.read_csv(path)


def selected_geometry_ids(destination: Path, values: list[int] | None) -> list[int]:
    design = geometry_design(destination)
    if values is None:
        return [int(value) for value in design["geometry_id"]]
    return sorted(set(int(value) for value in values))


def validate_geometry_set(destination: Path, geometry_ids: list[int]) -> None:
    geometry_dir = destination / "geometries"
    descriptors_path = geometry_dir / "geometry_realized_descriptors.csv"
    if not descriptors_path.is_file():
        raise FileNotFoundError(
            f"Missing realized descriptors: {descriptors_path}. Run step 02 first."
        )
    descriptors = pd.read_csv(descriptors_path)
    realized = set(int(value) for value in descriptors["geometry_id"])
    missing = sorted(set(geometry_ids) - realized)
    if missing:
        raise RuntimeError(
            "Realized descriptor table does not include requested geometries: "
            f"{missing}."
        )
    if "accepted_local" in descriptors:
        accepted = (
            descriptors["accepted_local"].astype(str).str.lower().eq("true")
        )
        rejected = sorted(
            int(value)
            for value in descriptors.loc[
                descriptors["geometry_id"].isin(geometry_ids) & ~accepted,
                "geometry_id",
            ]
        )
        if rejected:
            raise RuntimeError(
                "Requested geometries failed their generation tolerances: "
                f"{rejected}. Regenerate them before the paper campaign."
            )
    missing_arrays = []
    for geometry_id in geometry_ids:
        case_dir = geometry_dir / f"geometry_{geometry_id:02d}"
        if not (case_dir / "phase.npy").is_file() or not (case_dir / "ori.npy").is_file():
            missing_arrays.append(geometry_id)
    if missing_arrays:
        raise RuntimeError(
            "Some requested geometries are missing phase.npy/ori.npy: "
            f"{missing_arrays}."
        )


def append_bool(command: list[str], enabled: bool, flag: str, no_flag: str) -> None:
    command.append(flag if enabled else no_flag)


def command_for_geometry(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    destination: Path,
    run_name: str,
    geometry_id: int,
    single_geometry: bool,
) -> list[str]:
    pipe = pipeline_config(config).copy()
    adaptive = bool(pipe.get("adaptive", False))
    if args.smoke:
        limit_key = "adaptive_training_limit" if adaptive else "training_limit"
        configured_limit = int(pipe.get(limit_key, 0) or 0)
        pipe[limit_key] = min(configured_limit, 4) if configured_limit else 4
        if adaptive:
            pipe["monitor_count"] = min(int(pipe.get("monitor_count", 5)), 2)
        pipe["final_validation_count"] = min(
            int(pipe.get("final_validation_count", 16)), 2
        )
        pipe["rom_timing_count"] = min(int(pipe.get("rom_timing_count", 10000)), 100)
        pipe["rom_timing_repetitions"] = 1
        pipe["target_error"] = 1.0
        pipe["training_profile"] = "rom_floor"
        pipe["monitor_profile"] = "rom_floor"
        pipe["validation_profile"] = "rom_floor"
        pipe["timing_profile"] = "rom_floor"
        pipe["audit_profile"] = "rom_floor"
        pipe["structural_audit_geometry_ids"] = []
        pipe["tau_sensitivity_geometry_ids"] = []

    if not adaptive:
        fixed_limit = int(pipe.get("training_limit", 0) or 0)
        if fixed_limit < 1:
            raise ValueError("Fixed training requires a positive training_limit.")
        pipe["start_materials"] = fixed_limit

    actual_run_name = run_name if single_geometry else f"{run_name}_geometry_{geometry_id:02d}"
    command = [
        sys.executable,
        str(SOBOL_POD_SCRIPT),
        "--config",
        str(args.config),
        "--out-root",
        str(destination),
        "--geometry-id",
        str(int(geometry_id)),
        "--geometry-dir",
        str(destination / "geometries" / f"geometry_{geometry_id:02d}"),
        "--run-name",
        actual_run_name,
        "--candidate-count",
        str(int(pipe.get("candidate_count", 1024))),
        "--candidate-seed",
        str(int(pipe.get("candidate_seed", 20260821))),
        "--monitor-count",
        str(int(pipe.get("monitor_count", 5))),
        "--monitor-seed",
        str(int(pipe.get("monitor_seed", 20260822))),
        "--final-validation-count",
        str(int(pipe.get("final_validation_count", 20))),
        "--final-validation-pool-count",
        str(int(pipe.get("final_validation_pool_count", 4096))),
        "--final-validation-seed",
        str(int(pipe.get("final_validation_seed", 20260901))),
        "--training-profile",
        str(pipe.get("training_profile", "snapshot")),
        "--monitor-profile",
        str(pipe.get("monitor_profile", "snapshot")),
        "--validation-profile",
        str(pipe.get("validation_profile", "reference")),
        "--timing-profile",
        str(pipe.get("timing_profile", "timing")),
        "--audit-profile",
        str(pipe.get("audit_profile", "truth")),
        "--structural-audit-count",
        str(int(pipe.get("structural_audit_count", 3))),
        "--target-error",
        str(float(pipe.get("target_error", 1.0e-4))),
        "--basis-tolerance",
        str(float(pipe.get("basis_tolerance", 1.0e-12))),
        "--basis-dtype",
        str(pipe.get("basis_dtype", "float64")),
        "--full-rank-basis-mode",
        str(pipe.get("full_rank_basis_mode", "raw-ritz")),
        "--ritz-contraction-dtype",
        str(pipe.get("ritz_contraction_dtype", "float32")),
        "--ritz-gram-compute-dtype",
        str(pipe.get("ritz_gram_compute_dtype", "float64")),
        "--ritz-gram-backend",
        str(pipe.get("ritz_gram_backend", "auto")),
        "--ritz-gram-rank-rtol",
        str(float(pipe.get("ritz_gram_rank_rtol", 1.0e-6))),
        "--pod-batch-max-gib",
        str(float(pipe.get("pod_batch_max_gib", 8.0))),
        "--affine-stress-max-gib",
        str(float(pipe.get("affine_stress_max_gib", 8.0))),
        "--rom-batch-max-gib",
        str(float(pipe.get("rom_batch_max_gib", 4.0))),
        "--rom-backend",
        str(pipe.get("rom_backend", "gpu")),
        "--fft-backend",
        str(pipe.get("fft_backend", "gpu")),
        "--memory-safety-fraction",
        str(float(pipe.get("memory_safety_fraction", 0.8))),
        "--blas-threads",
        str(pipe.get("blas_threads", "auto")),
        "--start-materials",
        str(int(pipe.get("start_materials", 2))),
        "--training-limit",
        str(
            int(
                pipe.get(
                    "adaptive_training_limit" if adaptive else "training_limit",
                    0,
                )
                or 0
            )
        ),
        "--rom-timing-count",
        str(int(pipe.get("rom_timing_count", 10000))),
        "--rom-timing-seed",
        str(int(pipe.get("rom_timing_seed", 20260823))),
        "--rom-timing-repetitions",
        str(int(pipe.get("rom_timing_repetitions", 10))),
        "--rom-chunk-size",
        str(int(pipe.get("rom_chunk_size", 10000))),
    ]
    append_bool(
        command,
        adaptive,
        "--adaptive",
        "--no-adaptive",
    )
    append_bool(
        command,
        int(geometry_id)
        in {int(value) for value in pipe.get("structural_audit_geometry_ids", [])},
        "--structural-audit",
        "--no-structural-audit",
    )
    append_bool(
        command,
        int(geometry_id)
        in {int(value) for value in pipe.get("tau_sensitivity_geometry_ids", [])},
        "--tau-sensitivity",
        "--no-tau-sensitivity",
    )
    append_bool(
        command,
        bool(pipe.get("energy_pod_baseline", True)),
        "--energy-pod-baseline",
        "--no-energy-pod-baseline",
    )
    command.extend(
        [
            "--tau-sensitivity-values",
            *[
                str(float(value))
                for value in pipe.get(
                    "tau_sensitivity_values", [1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7]
                )
            ],
        ]
    )
    command.extend(
        [
            "--energy-pod-retentions",
            *[
                str(float(value))
                for value in pipe.get(
                    "energy_pod_retentions",
                    [0.99, 0.999, 0.9999, 0.99999, 0.999999],
                )
            ],
        ]
    )
    append_bool(command, bool(pipe.get("save_operators", True)), "--save-operators", "--no-save-operators")
    append_bool(
        command,
        bool(pipe.get("write_per_geometry_plots", False)),
        "--write-plot",
        "--no-write-plot",
    )
    append_bool(
        command,
        bool(pipe.get("cleanup_snapshot_fields", True)),
        "--cleanup-snapshot-fields",
        "--no-cleanup-snapshot-fields",
    )
    append_bool(command, bool(pipe.get("quiet_solver", True)), "--quiet-solver", "--no-quiet-solver")
    if args.overwrite:
        command.append("--overwrite")
    return command


def run(command: list[str]) -> float:
    print("[STEP3] " + " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=str(ROOT), check=True)
    return float(time.perf_counter() - started)


def collect_run_summary(
    *,
    destination: Path,
    run_name: str,
    geometry_id: int,
    wall_s: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = destination / "runs" / run_name
    curve = read_csv_if_exists(run_dir / "sobol_pod_error_curve.csv")
    if not curve.empty:
        curve.insert(0, "geometry_id", int(geometry_id))
        curve.insert(1, "run_name", run_name)
        curve.insert(2, "run_dir", str(run_dir))
    summary_path = run_dir / "sobol_pod_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    summary.pop("automatic_precision_rebuild", None)
    summary.update(
        {
            "geometry_id": int(geometry_id),
            "run_name": run_name,
            "run_dir": str(run_dir),
            "wrapper_wall_s": float(wall_s),
        }
    )
    return curve, summary


def first_threshold_crossing_curve(
    curve: pd.DataFrame, target_error: float
) -> pd.DataFrame:
    """Keep each geometry history through its first accepted monitor pass."""
    if curve.empty:
        result = curve.copy()
        result["one_pass_stop"] = pd.Series(dtype=bool)
        return result

    required = {"geometry_id", "training_materials", "monitor_error_max"}
    missing = sorted(required.difference(curve.columns))
    if missing:
        raise ValueError(f"Error curve is missing required columns: {missing}")

    groups: list[pd.DataFrame] = []
    for _, group in curve.groupby("geometry_id", sort=True):
        group = group.sort_values("training_materials").copy()
        if not group["monitor_error_max"].notna().any():
            group = group.tail(1).copy()
            group["one_pass_stop"] = True
            groups.append(group)
            continue
        accepted = (
            group["monitor_error_max"].to_numpy(dtype=float) <= float(target_error)
        )
        crossing_positions = accepted.nonzero()[0]
        if crossing_positions.size:
            group = group.iloc[: int(crossing_positions[0]) + 1].copy()
        group["one_pass_stop"] = False
        if crossing_positions.size:
            group.iloc[-1, group.columns.get_loc("one_pass_stop")] = True
        groups.append(group)
    return pd.concat(groups, ignore_index=True)


def one_pass_timing_table(
    summary: pd.DataFrame,
    full_curve: pd.DataFrame,
    first_crossing_curve: pd.DataFrame,
    rom_query_count: int,
) -> pd.DataFrame:
    """Build per-geometry wall-time rows for first-crossing stopping."""
    columns = [
        "geometry_id",
        "geometry_label",
        "stop_materials",
        "pod_rank",
        "monitor_error_max",
        "geometry_affine_setup_s",
        "monitor_count",
        "monitor_fft_once_s",
        "adaptive_training_s",
        "training_snapshot_fft_s",
        "pod_basis_update_s",
        "reduced_operator_assembly_s",
        "affine_stress_action_s",
        "ritz_contraction_s",
        "monitor_rom_updates_s",
        "training_other_s",
        "final_validation_count",
        "final_validation_s",
        "rom_query_count",
        "rom_timing_s",
        "other_pipeline_io_s",
        "one_pass_pipeline_total_s",
        "one_pass_pipeline_total_min",
        "recorded_pipeline_total_s",
        "removed_confirmation_s",
        "timing_basis",
    ]
    if summary.empty or first_crossing_curve.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for record in summary.to_dict(orient="records"):
        geometry_id = int(record["geometry_id"])
        full_group = full_curve[full_curve["geometry_id"] == geometry_id].sort_values(
            "training_materials"
        )
        first_group = first_crossing_curve[
            first_crossing_curve["geometry_id"] == geometry_id
        ].sort_values("training_materials")
        if full_group.empty or first_group.empty:
            continue

        full_stop = full_group.iloc[-1]
        first_stop = first_group.iloc[-1]
        recorded_training_s = float(record["training_stage_wall_s"])
        training_overhead_s = max(
            recorded_training_s - float(full_stop["snapshot_step_wall_s"]), 0.0
        )
        adaptive_training_s = (
            float(first_stop["snapshot_step_wall_s"]) + training_overhead_s
        )
        snapshot_fft_s = float(first_stop["snapshot_solve_wall_s"])
        basis_update_s = float(first_stop["basis_update_wall_s"])
        operator_assembly_s = float(first_stop["operator_assembly_wall_s"])
        training_other_s = adaptive_training_s - (
            snapshot_fft_s + basis_update_s + operator_assembly_s
        )
        removed_confirmation_s = max(
            float(full_stop["snapshot_step_wall_s"])
            - float(first_stop["snapshot_step_wall_s"]),
            0.0,
        )
        recorded_total_s = float(record["wrapper_wall_s"])
        historical_stop_materials = int(record["stop_materials"])
        stop_materials = int(first_stop["training_materials"])
        was_one_pass = historical_stop_materials == stop_materials
        one_pass_total_s = recorded_total_s - removed_confirmation_s
        timing_basis = "measured" if was_one_pass else "retrospective_cumulative_estimate"

        geometry_setup_s = float(record["geometry_load_wall_s"]) + float(
            record["affine_setup_wall_s"]
        )
        monitor_s = float(record["monitor_fft_stage_wall_s"])
        validation_s = float(record["final_validation_stage_wall_s"])
        rom_s = float(record["rom_timing_stage_wall_s"])
        other_s = one_pass_total_s - (
            geometry_setup_s
            + monitor_s
            + adaptive_training_s
            + validation_s
            + rom_s
        )
        rows.append(
            {
                "geometry_id": geometry_id,
                "geometry_label": str(record["geometry_label"]),
                "stop_materials": stop_materials,
                "pod_rank": int(first_stop["pod_rank"]),
                "monitor_error_max": float(first_stop["monitor_error_max"]),
                "geometry_affine_setup_s": geometry_setup_s,
                "monitor_count": int(record["monitor_count"]),
                "monitor_fft_once_s": monitor_s,
                "adaptive_training_s": adaptive_training_s,
                "training_snapshot_fft_s": snapshot_fft_s,
                "pod_basis_update_s": basis_update_s,
                "reduced_operator_assembly_s": operator_assembly_s,
                "affine_stress_action_s": float(first_stop["affine_stress_wall_s"]),
                "ritz_contraction_s": float(first_stop["ritz_contraction_wall_s"]),
                "monitor_rom_updates_s": float(
                    first_stop["monitor_rom_cumulative_wall_s"]
                ),
                "training_other_s": training_other_s,
                "final_validation_count": int(record["final_validation_count"]),
                "final_validation_s": validation_s,
                "rom_query_count": int(rom_query_count),
                "rom_timing_s": rom_s,
                "other_pipeline_io_s": other_s,
                "one_pass_pipeline_total_s": one_pass_total_s,
                "one_pass_pipeline_total_min": one_pass_total_s / 60.0,
                "recorded_pipeline_total_s": recorded_total_s,
                "removed_confirmation_s": removed_confirmation_s,
                "timing_basis": timing_basis,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def write_campaign_plot(curve: pd.DataFrame, path: Path, target_error: float) -> None:
    if curve.empty or not curve["monitor_error_max"].notna().any():
        return
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        percent_scale = 100.0
        stop_markers: list[tuple[pd.DataFrame, str]] = []
        for geometry_id, group in curve.groupby("geometry_id", sort=True):
            group = group.sort_values("training_materials")
            (line,) = ax.semilogy(
                group["training_materials"].to_numpy(dtype=int),
                percent_scale * group["monitor_error_max"].to_numpy(dtype=float),
                marker="o",
                linewidth=1.5,
                markersize=3.8,
                label=f"g{int(geometry_id):02d}",
            )
            if "one_pass_stop" in group:
                stop = group[group["one_pass_stop"].astype(bool)]
                if not stop.empty:
                    stop_markers.append((stop, line.get_color()))
        ax.axhline(
            percent_scale * float(target_error),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"{percent_scale * target_error:g}% stop target",
        )
        for marker_index, (stop, color) in enumerate(stop_markers):
            ax.scatter(
                stop["training_materials"].to_numpy(dtype=int),
                percent_scale * stop["monitor_error_max"].to_numpy(dtype=float),
                marker="X",
                s=48,
                color=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
                label="first accepted pass" if marker_index == 0 else None,
            )
        ax.set_xlabel("Sobol training materials used for POD")
        ax.set_ylabel("Maximum relative Frobenius error on FFT monitor set (%)")
        ax.set_title("One-pass stopping at the first tolerance crossing")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - paper artifact only
        print(f"[STEP3] summary plot skipped: {exc}", flush=True)


def validation_coverage_tables(
    summaries: list[dict[str, Any]], target_error: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for summary in summaries:
        path = Path(str(summary.get("final_validation_rom_csv", "")))
        frame = read_csv_if_exists(path)
        if frame.empty or "relative_frobenius_error" not in frame:
            continue
        geometry_id = int(summary["geometry_id"])
        frame.insert(0, "geometry_id", geometry_id)
        frame["relative_frobenius_error_percent"] = (
            100.0 * frame["relative_frobenius_error"]
        )
        frame["below_target"] = (
            frame["relative_frobenius_error"] <= float(target_error)
        )
        frames.append(frame)
        for threshold in sorted({float(target_error), 1.0e-3}):
            coverage_rows.append(
                {
                    "scope": f"geometry_{geometry_id:02d}",
                    "criterion": f"error_le_{threshold:.0e}",
                    **empirical_coverage(
                        frame["relative_frobenius_error"], threshold
                    ),
                }
            )

    details = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not details.empty:
        for threshold in sorted({float(target_error), 1.0e-3}):
            coverage_rows.append(
                {
                    "scope": "all_geometry_material_pairs",
                    "criterion": f"error_le_{threshold:.0e}",
                    **empirical_coverage(
                        details["relative_frobenius_error"], threshold
                    ),
                }
            )
    return details, pd.DataFrame(coverage_rows)


def performance_table(summary: pd.DataFrame) -> pd.DataFrame:
    if not summary.empty:
        summary = summary.copy()
        summary["cuda_cold_start_median_s"] = summary["rom_timing"].map(
            lambda value: value.get("cold_start_median_s")
            if isinstance(value, dict)
            else None
        )
        summary["rom_hot_single_median_s"] = summary["rom_timing"].map(
            lambda value: value.get("warm_single_query_median_s")
            if isinstance(value, dict)
            else None
        )
        summary["rom_hot_single_p95_s"] = summary["rom_timing"].map(
            lambda value: value.get("warm_single_query_p95_s")
            if isinstance(value, dict)
            else None
        )
        summary["rom_batch_10000_median_s"] = summary["rom_timing"].map(
            lambda value: value.get("rom_online_total_s")
            if isinstance(value, dict)
            else None
        )
    columns = [
        "geometry_id",
        "geometry_label",
        "nvox",
        "stop_materials",
        "basis_rank",
        "fom_material_median_s",
        "compilation_wall_s",
        "cuda_cold_start_median_s",
        "rom_hot_single_median_s",
        "rom_hot_single_p95_s",
        "rom_batch_10000_median_s",
        "hot_single_speedup",
        "break_even_queries",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    missing = sorted(set(columns).difference(summary.columns))
    if missing:
        raise KeyError(f"Campaign summaries are missing performance fields: {missing}")
    return summary[columns].sort_values("geometry_id").reset_index(drop=True)


def collect_optional_diagnostics(
    summaries: list[dict[str, Any]], key: str
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for summary in summaries:
        path_value = summary.get(key)
        if not path_value:
            continue
        frame = read_csv_if_exists(Path(str(path_value)))
        if frame.empty:
            continue
        frame.insert(0, "geometry_id", int(summary["geometry_id"]))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_performance_plot(performance: pd.DataFrame, path: Path) -> None:
    if performance.empty:
        return
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
        left, right = axes
        left.scatter(
            performance["nvox"], performance["fom_material_median_s"], s=42
        )
        right.scatter(
            performance["basis_rank"],
            1.0e6 * performance["rom_hot_single_median_s"],
            s=42,
        )
        for row in performance.itertuples(index=False):
            label = f"G{int(row.geometry_id):02d}"
            left.annotate(label, (row.nvox, row.fom_material_median_s), xytext=(4, 3), textcoords="offset points", fontsize=7)
            right.annotate(label, (row.basis_rank, 1.0e6 * row.rom_hot_single_median_s), xytext=(4, 3), textcoords="offset points", fontsize=7)
        left.set_xlabel(r"FFT voxel count $N_{\mathrm{vox}}$")
        left.set_ylabel(r"Median FOM time $t_{\mathrm{FOM}}$ [s]")
        right.set_xlabel(r"Reduced rank $r$")
        right.set_ylabel(r"Median ROM time $t_{\mathrm{ROM}}$ [$\mu$s]")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=240)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - paper artifact only
        print(f"[STEP3] performance plot skipped: {exc}", flush=True)


def write_campaign_summary(
    *,
    destination: Path,
    base_run_name: str,
    config: dict[str, Any],
    command_rows: list[dict[str, Any]],
    curves: list[pd.DataFrame],
    summaries: list[dict[str, Any]],
) -> Path:
    summary_dir = destination / "runs" / f"{base_run_name}_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    curve = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    commands = pd.DataFrame(command_rows)
    target_error = float(
        config.get("sobol_pod_pipeline", {}).get("target_error", 1.0e-4)
    )
    validation, validation_coverage = validation_coverage_tables(
        summaries, target_error
    )
    performance = performance_table(summary)
    tau_sensitivity = collect_optional_diagnostics(
        summaries, "tau_sensitivity_summary_csv"
    )
    structural_audit = collect_optional_diagnostics(
        summaries, "structural_audit_csv"
    )
    energy_pod_selection = collect_optional_diagnostics(
        summaries, "energy_pod_monitor_selection_csv"
    )
    energy_pod_validation = collect_optional_diagnostics(
        summaries, "energy_pod_final_validation_csv"
    )
    first_crossing_curve = first_threshold_crossing_curve(curve, target_error)
    timing = one_pass_timing_table(
        summary,
        curve,
        first_crossing_curve,
        int(config.get("sobol_pod_pipeline", {}).get("rom_timing_count", 10000)),
    )
    curve.to_csv(summary_dir / "sobol_pod_multigeometry_curve.csv", index=False)
    first_crossing_curve.to_csv(
        summary_dir / "sobol_pod_multigeometry_curve_first_crossing.csv", index=False
    )
    summary.to_csv(summary_dir / "sobol_pod_multigeometry_summary.csv", index=False)
    timing.to_csv(
        summary_dir / "sobol_pod_multigeometry_timing_one_pass.csv", index=False
    )
    commands.to_csv(summary_dir / "execution_commands.csv", index=False)
    validation.to_csv(
        summary_dir / "sobol_pod_multigeometry_validation.csv", index=False
    )
    validation_coverage.to_csv(
        summary_dir / "sobol_pod_validation_coverage.csv", index=False
    )
    performance.to_csv(
        summary_dir / "sobol_pod_multigeometry_performance.csv", index=False
    )
    tau_sensitivity.to_csv(
        summary_dir / "sobol_pod_tau_G_sensitivity.csv", index=False
    )
    structural_audit.to_csv(
        summary_dir / "sobol_pod_structural_audit.csv", index=False
    )
    energy_pod_selection.to_csv(
        summary_dir / "energy_pod_monitor_selection.csv", index=False
    )
    energy_pod_validation.to_csv(
        summary_dir / "energy_pod_validation.csv", index=False
    )
    if bool(config.get("sobol_pod_pipeline", {}).get("write_summary_plot", True)):
        write_campaign_plot(
            first_crossing_curve,
            summary_dir / "sobol_pod_multigeometry_error_curves.png",
            target_error,
        )
        write_performance_plot(
            performance, summary_dir / "sobol_pod_performance_scaling.png"
        )
    with pd.ExcelWriter(summary_dir / "sobol_pod_multigeometry_tables.xlsx") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        curve.to_excel(writer, sheet_name="error_curve", index=False)
        first_crossing_curve.to_excel(
            writer, sheet_name="first_crossing_curve", index=False
        )
        timing.to_excel(writer, sheet_name="one_pass_timing", index=False)
        validation.to_excel(writer, sheet_name="final_validation", index=False)
        validation_coverage.to_excel(writer, sheet_name="validation_coverage", index=False)
        performance.to_excel(writer, sheet_name="performance", index=False)
        tau_sensitivity.to_excel(writer, sheet_name="tau_G_sensitivity", index=False)
        structural_audit.to_excel(writer, sheet_name="structural_audit", index=False)
        energy_pod_selection.to_excel(writer, sheet_name="energy_POD_selection", index=False)
        energy_pod_validation.to_excel(writer, sheet_name="energy_POD_validation", index=False)
        commands.to_excel(writer, sheet_name="commands", index=False)
    (summary_dir / "campaign_config_snapshot.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--stage", choices=("all", "sobol_pod"), default="all")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--geometry-ids", type=int, nargs="*", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse completed geometry runs when assembling a campaign summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    destination = out_root(config, args.out_root)
    geometry_ids = selected_geometry_ids(destination, args.geometry_ids)
    if args.smoke and args.geometry_ids is None:
        geometry_ids = [8]
    validate_geometry_set(destination, geometry_ids)

    base_run_name = str(args.run_name or run_name_now())
    single = len(geometry_ids) == 1
    curves: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    command_rows: list[dict[str, Any]] = []

    for geometry_id in geometry_ids:
        actual_run_name = (
            base_run_name
            if single
            else f"{base_run_name}_geometry_{geometry_id:02d}"
        )
        command = command_for_geometry(
            args=args,
            config=config,
            destination=destination,
            run_name=base_run_name,
            geometry_id=int(geometry_id),
            single_geometry=single,
        )
        existing_summary = (
            destination / "runs" / actual_run_name / "sobol_pod_summary.json"
        )
        reused_existing = bool(args.reuse_existing and existing_summary.exists())
        if reused_existing:
            existing_payload = json.loads(existing_summary.read_text(encoding="utf-8"))
            if str(existing_payload.get("status", "")) != "complete":
                raise RuntimeError(
                    f"Cannot reuse incomplete run: {existing_summary.parent}"
                )
            wall_s = float(existing_payload.get("pipeline_wall_s_before_final_write", 0.0))
            print(f"[STEP3] reuse={existing_summary.parent}", flush=True)
        else:
            wall_s = run(command)
        curve, summary = collect_run_summary(
            destination=destination,
            run_name=actual_run_name,
            geometry_id=int(geometry_id),
            wall_s=wall_s,
        )
        if not curve.empty:
            curves.append(curve)
        summaries.append(summary)
        command_rows.append(
            {
                "geometry_id": int(geometry_id),
                "run_name": actual_run_name,
                "command": (
                    f"REUSED {existing_summary.parent}"
                    if reused_existing
                    else " ".join(command)
                ),
                "wrapper_wall_s": float(wall_s),
            }
        )

    summary_dir = write_campaign_summary(
        destination=destination,
        base_run_name=base_run_name,
        config=config,
        command_rows=command_rows,
        curves=curves,
        summaries=summaries,
    )
    print(f"[STEP3] summary={summary_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
