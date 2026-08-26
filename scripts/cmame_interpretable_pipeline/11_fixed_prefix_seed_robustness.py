#!/usr/bin/env python3
"""Run and analyze fixed-prefix Sobol seed robustness without FOM monitors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

from env_bootstrap import ensure_configured_venv


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_DEFAULT = HERE / "campaign_config.json"
STUDY_CONFIG_DEFAULT = HERE / "fixed_prefix_robustness_config.json"
PIPELINE_SCRIPT = HERE / "04_sobol_pod_pipeline.py"

if __name__ == "__main__":
    ensure_configured_venv(CONFIG_DEFAULT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


for path in (ROOT, ROOT / "scripts", ROOT / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rom_reduced_operator as reduced


TIMING_SUM_COLUMNS = (
    "solve_wall_s",
    "snapshot_step_wall_s",
    "basis_update_wall_s",
    "local_frame_transform_wall_s",
    "operator_assembly_wall_s",
    "affine_stress_wall_s",
    "ritz_contraction_wall_s",
    "gram_product_wall_s",
    "gram_overlap_wait_wall_s",
    "gram_overlap_hidden_wall_s",
    "factorized_upload_wall_s",
    "factorized_host_prepare_wall_s",
    "factorized_kernel_enqueue_wall_s",
)

TIMING_MAX_COLUMNS = (
    "operator_stress_workspace_peak_bytes",
    "operator_contraction_workspace_peak_bytes",
    "local_frame_workspace_peak_bytes",
    "async_pinned_bytes",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--study-config", type=Path, default=STUDY_CONFIG_DEFAULT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--stage", choices=("run", "analyze", "all"), default="all")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--geometry-ids", type=int, nargs="*", default=None)
    parser.add_argument("--training-seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed seed/geometry runs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and print the complete command plan without executing it.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use G03, two seeds, four materials, and two validation points.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def resolved_settings(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = load_json(args.config.resolve())
    study = load_json(args.study_config.resolve())
    if args.run_name is not None:
        study["run_name"] = str(args.run_name)
    if args.geometry_ids is not None:
        study["geometry_ids"] = [int(value) for value in args.geometry_ids]
    if args.training_seeds is not None:
        study["training_seeds"] = [int(value) for value in args.training_seeds]
    if args.smoke:
        study.update(
            {
                "run_name": str(study["run_name"]) + "_smoke",
                "geometry_ids": [3],
                "training_seeds": [
                    int(value) for value in study["training_seeds"][:2]
                ],
                "max_training_materials": 4,
                "checkpoints": [2, 3, 4],
                "candidate_count": 32,
                "validation_pool_count": 32,
                "validation_count": 2,
                "non_owner_validation_count": 1,
                "rom_timing_count": 10,
                "rom_timing_repetitions": 1,
            }
        )
    validate_settings(campaign, study)
    return campaign, study


def validate_settings(campaign: dict[str, Any], study: dict[str, Any]) -> None:
    geometry_ids = [int(value) for value in study["geometry_ids"]]
    seeds = [int(value) for value in study["training_seeds"]]
    checkpoints = [int(value) for value in study["checkpoints"]]
    maximum = int(study["max_training_materials"])
    if not geometry_ids or len(set(geometry_ids)) != len(geometry_ids):
        raise ValueError("geometry_ids must be non-empty and unique.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("training_seeds must be non-empty and unique.")
    if int(study["validation_seed"]) in seeds:
        raise ValueError("validation_seed must differ from every training seed.")
    if not checkpoints or checkpoints != sorted(set(checkpoints)):
        raise ValueError("checkpoints must be sorted and unique.")
    if checkpoints[0] < 1 or checkpoints[-1] != maximum:
        raise ValueError(
            "checkpoints must be positive and end at max_training_materials."
        )
    if int(study["candidate_count"]) < maximum:
        raise ValueError("candidate_count must cover max_training_materials.")
    if int(study["validation_count"]) < 1:
        raise ValueError("validation_count must be positive.")
    if int(study["validation_pool_count"]) < int(study["validation_count"]):
        raise ValueError("validation_pool_count must cover validation_count.")
    if int(study["non_owner_validation_count"]) < 1:
        raise ValueError("non_owner_validation_count must be positive.")
    pipeline = campaign["sobol_pod_pipeline"]
    if not bool(pipeline.get("reference_energy_qr", False)):
        raise ValueError("The study requires reference_energy_qr raw archives.")


def out_root(campaign: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return project_path(campaign["paths"]["out_root"]).resolve()


def run_name(study: dict[str, Any], geometry_id: int, seed: int) -> str:
    return f"{study['run_name']}_g{int(geometry_id):02d}_seed{int(seed)}"


def run_specs(study: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = [int(value) for value in study["training_seeds"]]
    specs: list[dict[str, Any]] = []
    for geometry_id in [int(value) for value in study["geometry_ids"]]:
        for seed_index, seed in enumerate(seeds):
            specs.append(
                {
                    "geometry_id": geometry_id,
                    "training_seed": seed,
                    "seed_index": seed_index,
                    "validation_owner": seed_index == 0,
                    "run_name": run_name(study, geometry_id, seed),
                }
            )
    return specs


def append_bool(command: list[str], enabled: bool, flag: str, no_flag: str) -> None:
    command.append(flag if enabled else no_flag)


def command_for_spec(
    *,
    args: argparse.Namespace,
    campaign: dict[str, Any],
    study: dict[str, Any],
    destination: Path,
    spec: dict[str, Any],
) -> list[str]:
    pipe = campaign["sobol_pod_pipeline"]
    geometry_id = int(spec["geometry_id"])
    validation_count = (
        int(study["validation_count"])
        if bool(spec["validation_owner"])
        else int(study["non_owner_validation_count"])
    )
    training_profile = (
        "rom_floor" if args.smoke else str(pipe.get("training_profile", "snapshot32"))
    )
    validation_profile = (
        "rom_floor" if args.smoke else str(pipe.get("validation_profile", "reference"))
    )
    timing_profile = (
        "rom_floor" if args.smoke else str(pipe.get("timing_profile", "snapshot32"))
    )
    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--config",
        str(args.config.resolve()),
        "--out-root",
        str(destination),
        "--geometry-id",
        str(geometry_id),
        "--geometry-dir",
        str(destination / "geometries" / f"geometry_{geometry_id:02d}"),
        "--run-name",
        str(spec["run_name"]),
        "--candidate-count",
        str(int(study["candidate_count"])),
        "--candidate-seed",
        str(int(spec["training_seed"])),
        "--selection-policy",
        "sobol-prefix",
        "--training-limit",
        str(int(study["max_training_materials"])),
        "--start-materials",
        str(int(study["max_training_materials"])),
        "--fixed-prefix-start",
        str(min(int(value) for value in study["checkpoints"])),
        "--monitor-count",
        "0",
        "--final-validation-count",
        str(validation_count),
        "--final-validation-pool-count",
        str(int(study["validation_pool_count"])),
        "--final-validation-seed",
        str(int(study["validation_seed"])),
        "--training-profile",
        training_profile,
        "--monitor-profile",
        training_profile,
        "--validation-profile",
        validation_profile,
        "--timing-profile",
        timing_profile,
        "--target-error",
        str(float(study["target_error"])),
        "--basis-tolerance",
        str(float(pipe.get("basis_tolerance", 1.0e-12))),
        "--basis-dtype",
        str(pipe.get("basis_dtype", "float32")),
        "--full-rank-basis-mode",
        str(pipe.get("full_rank_basis_mode", "raw-ritz")),
        "--ritz-contraction-dtype",
        str(pipe.get("ritz_contraction_dtype", "float32")),
        "--ritz-gram-compute-dtype",
        str(pipe.get("ritz_gram_compute_dtype", "float64")),
        "--ritz-gram-backend",
        str(pipe.get("ritz_gram_backend", "cpu")),
        "--ritz-gram-rank-rtol",
        str(float(pipe.get("ritz_gram_rank_rtol", 1.0e-15))),
        "--pod-batch-max-gib",
        str(float(pipe.get("pod_batch_max_gib", 8.0))),
        "--affine-stress-max-gib",
        str(float(pipe.get("affine_stress_max_gib", 8.0))),
        "--rom-batch-max-gib",
        str(float(pipe.get("rom_batch_max_gib", 4.0))),
        "--memory-safety-fraction",
        str(float(pipe.get("memory_safety_fraction", 0.8))),
        "--rom-backend",
        str(pipe.get("rom_backend", "gpu")),
        "--fft-backend",
        str(pipe.get("fft_backend", "gpu")),
        "--blas-threads",
        str(pipe.get("blas_threads", "auto")),
        "--rom-timing-count",
        str(int(study["rom_timing_count"])),
        "--rom-timing-seed",
        str(int(pipe.get("rom_timing_seed", 20260823))),
        "--rom-timing-repetitions",
        str(int(study["rom_timing_repetitions"])),
        "--rom-chunk-size",
        str(min(100, int(study["rom_timing_count"]))),
        "--no-adaptive",
        "--record-fixed-prefixes",
        "--no-energy-pod-baseline",
        "--no-structural-audit",
        "--no-tau-sensitivity",
        "--no-experimental-qr-audit",
        "--save-operators",
        "--cleanup-snapshot-fields",
        "--no-write-plot",
    ]
    append_bool(
        command,
        bool(pipe.get("overlap_cpu_gram_gpu", False)),
        "--overlap-cpu-gram-gpu",
        "--no-overlap-cpu-gram-gpu",
    )
    append_bool(
        command,
        bool(pipe.get("reference_energy_qr", True)),
        "--reference-energy-qr",
        "--no-reference-energy-qr",
    )
    append_bool(
        command,
        bool(pipe.get("factorized_ritz", True)),
        "--factorized-ritz",
        "--no-factorized-ritz",
    )
    append_bool(
        command,
        bool(pipe.get("async_ritz", True)),
        "--async-ritz",
        "--no-async-ritz",
    )
    append_bool(
        command,
        bool(pipe.get("gathered_factor_ritz", True)),
        "--gathered-factor-ritz",
        "--no-gathered-factor-ritz",
    )
    append_bool(
        command,
        bool(pipe.get("quiet_solver", True)),
        "--quiet-solver",
        "--no-quiet-solver",
    )
    if args.overwrite:
        command.append("--overwrite")
    return command


def completed_run(run_dir: Path, spec: dict[str, Any], study: dict[str, Any]) -> bool:
    summary_path = run_dir / "sobol_pod_summary.json"
    if not summary_path.is_file():
        return False
    summary = load_json(summary_path)
    return bool(
        summary.get("status") == "complete"
        and not bool(summary.get("adaptive", True))
        and bool(summary.get("record_fixed_prefixes", False))
        and int(summary.get("candidate_seed", -1)) == int(spec["training_seed"])
        and int(summary.get("training_limit", -1))
        == int(study["max_training_materials"])
    )


def executable_command(command: str, run_dir: Path) -> list[str]:
    """Restart an incomplete run directory while preserving completed runs."""
    tokens = shlex.split(command)
    if run_dir.exists() and "--overwrite" not in tokens:
        tokens.append("--overwrite")
    return tokens


def write_run_plan(
    *,
    summary_dir: Path,
    args: argparse.Namespace,
    campaign: dict[str, Any],
    study: dict[str, Any],
    destination: Path,
    specs: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        command = command_for_spec(
            args=args,
            campaign=campaign,
            study=study,
            destination=destination,
            spec=spec,
        )
        rows.append(
            {
                **spec,
                "run_dir": str(destination / "runs" / str(spec["run_name"])),
                "command": shlex.join(command),
                "status": "planned",
                "wall_s": np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    summary_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(summary_dir / "fixed_prefix_run_plan.csv", index=False)
    protocol = {
        "status": "frozen_before_seed_runs",
        "study": study,
        "campaign_config": str(args.config.resolve()),
        "campaign_config_sha256": sha256(args.config.resolve()),
        "study_config": str(args.study_config.resolve()),
        "study_config_sha256": sha256(args.study_config.resolve()),
        "monitor_count": 0,
        "adaptive_stopping": False,
        "training_design": "independently_scrambled_sobol_prefixes",
        "material_load_batch_size": 1,
        "execution_policy": "sequential_single_gpu",
        "shared_validation_owner_seed": int(study["training_seeds"][0]),
        "shared_validation_reused_only_after_training": True,
        "run_count": int(len(frame)),
    }
    write_json(summary_dir / "fixed_prefix_protocol_manifest.json", protocol)
    return frame


def execute_plan(
    *,
    plan: pd.DataFrame,
    summary_dir: Path,
    args: argparse.Namespace,
    study: dict[str, Any],
    destination: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in plan.to_dict(orient="records"):
        run_dir = destination / "runs" / str(row["run_name"])
        spec = {
            "geometry_id": int(row["geometry_id"]),
            "training_seed": int(row["training_seed"]),
            "validation_owner": bool(row["validation_owner"]),
        }
        reused = bool(args.resume and completed_run(run_dir, spec, study))
        command = str(row["command"])
        if args.dry_run:
            status = "dry_run"
            wall_s = 0.0
            print(f"[FIXED-PREFIX] DRY RUN {command}", flush=True)
        elif reused:
            status = "reused"
            wall_s = float(
                load_json(run_dir / "sobol_pod_summary.json").get(
                    "pipeline_wall_s_before_final_write", 0.0
                )
            )
            print(f"[FIXED-PREFIX] reuse {run_dir}", flush=True)
        else:
            tokens = executable_command(command, run_dir)
            print(f"[FIXED-PREFIX] {shlex.join(tokens)}", flush=True)
            started = time.perf_counter()
            subprocess.run(tokens, cwd=str(ROOT), check=True)
            wall_s = float(time.perf_counter() - started)
            status = "complete"
        row["status"] = status
        row["wall_s"] = wall_s
        rows.append(row)
        pd.DataFrame(rows).to_csv(
            summary_dir / "fixed_prefix_run_status.csv", index=False
        )
    return pd.DataFrame(rows)


def prefix_operators(
    archive: dict[str, np.ndarray], training_materials: int
) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    required = {"raw_Kq", "raw_Bq", "Dq", "energy_qr_reference_coefficients"}
    missing = sorted(required.difference(archive))
    if missing:
        raise RuntimeError("Raw prefix archive is missing: " + ", ".join(missing))
    raw_columns = 6 * int(training_materials)
    started = time.perf_counter()
    operators, metadata = reduced._reference_energy_qr_recompile(
        raw_Kq=np.asarray(archive["raw_Kq"][:, :raw_columns, :raw_columns]),
        raw_Bq=np.asarray(archive["raw_Bq"][:, :raw_columns]),
        Dq=np.asarray(archive["Dq"]),
        reference_coefficients=np.asarray(
            archive["energy_qr_reference_coefficients"]
        ),
    )
    return operators, metadata, float(time.perf_counter() - started)


def prefix_timing(
    snapshot_timing: pd.DataFrame,
    training_materials: int,
    run_summary: dict[str, Any],
    recompile_wall_s: float,
    evaluation_wall_s: float,
) -> dict[str, Any]:
    prefix = snapshot_timing.iloc[: int(training_materials)]
    if len(prefix) != int(training_materials):
        raise RuntimeError("Snapshot timing does not cover the requested prefix.")
    row: dict[str, Any] = {}
    for column in TIMING_SUM_COLUMNS:
        if column in prefix:
            row[f"{column}_cumulative"] = float(
                prefix[column].fillna(0.0).sum()
            )
    for column in TIMING_MAX_COLUMNS:
        if column in prefix:
            row[f"{column}_maximum"] = int(
                prefix[column].fillna(0).max()
            )
    row.update(
        {
            "geometry_load_wall_s": float(
                run_summary.get("geometry_load_wall_s", 0.0)
            ),
            "affine_setup_wall_s": float(
                run_summary.get("affine_setup_wall_s", 0.0)
            ),
            "prefix_reference_energy_qr_wall_s": float(recompile_wall_s),
            "prefix_validation_rom_wall_s": float(evaluation_wall_s),
            "prefix_compile_wall_s": float(
                run_summary.get("geometry_load_wall_s", 0.0)
                + run_summary.get("affine_setup_wall_s", 0.0)
                + prefix["snapshot_step_wall_s"].fillna(0.0).sum()
                + recompile_wall_s
            ),
            "full_seed_run_wall_s": float(
                run_summary.get("pipeline_wall_s_before_final_write", np.nan)
            ),
        }
    )
    return row


def summarize_evaluation(
    evaluation: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    errors = evaluation["relative_frobenius_error"].to_numpy(dtype=np.float64)
    finite = errors[np.isfinite(errors)]
    if finite.size != len(errors):
        raise RuntimeError("A prefix ROM produced a non-finite validation error.")
    result: dict[str, Any] = {
        "validation_count": int(len(errors)),
        "error_mean": float(np.mean(errors)),
        "error_median": float(np.median(errors)),
        "error_p95": float(np.quantile(errors, 0.95)),
        "error_max": float(np.max(errors)),
        "below_target_count": int(np.count_nonzero(errors <= threshold)),
        "below_target_fraction": float(np.mean(errors <= threshold)),
        "below_1e3_count": int(np.count_nonzero(errors <= 1.0e-3)),
        "below_1e3_fraction": float(np.mean(errors <= 1.0e-3)),
        "rom_failure_count": int(
            evaluation["rom_numerical_failure"].astype(bool).sum()
        ),
        "rom_min_eig_min": float(evaluation["rom_min_eig"].min()),
        "reduced_K_min_eig_min": float(
            evaluation["reduced_K_min_eig"].min()
        ),
        "reduced_K_spectral_spd_margin_min": float(
            evaluation["reduced_K_spectral_spd_margin"].min()
        ),
        "schur_eta_min": float(evaluation["schur_eta"].min()),
    }
    return result


def analyze(
    *,
    study: dict[str, Any],
    destination: Path,
    summary_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seeds = [int(value) for value in study["training_seeds"]]
    checkpoints = [int(value) for value in study["checkpoints"]]
    threshold = float(study["target_error"])
    summary_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    detail_rows: list[pd.DataFrame] = []

    for geometry_id in [int(value) for value in study["geometry_ids"]]:
        owner_dir = destination / "runs" / run_name(study, geometry_id, seeds[0])
        validation_truth = pd.read_csv(
            owner_dir / "final_validation_truth_results.csv"
        )
        if len(validation_truth) != int(study["validation_count"]):
            raise RuntimeError(
                f"G{geometry_id:02d} shared validation has "
                f"{len(validation_truth)} rows; expected {study['validation_count']}."
            )
        owner_timing = pd.read_csv(owner_dir / "fom_timing_results.csv")
        shared_timing_wall_s = float(owner_timing["solve_wall_s"].sum())
        shared_reference_wall_s = float(validation_truth["solve_wall_s"].sum())

        for seed_index, seed in enumerate(seeds):
            current_dir = destination / "runs" / run_name(
                study, geometry_id, seed
            )
            run_summary = load_json(current_dir / "sobol_pod_summary.json")
            snapshot_timing = pd.read_csv(current_dir / "snapshot_timing.csv")
            with np.load(current_dir / "reduced_operators.npz") as stored:
                archive = {name: stored[name] for name in stored.files}

            for count in checkpoints:
                operators, metadata, recompile_wall_s = prefix_operators(
                    archive, count
                )
                evaluation_started = time.perf_counter()
                evaluation = reduced._evaluate_rom(
                    results_df=validation_truth,
                    Kq=operators["Kq"],
                    Bq=operators["Bq"],
                    Dq=operators["Dq"],
                )
                evaluation_wall_s = float(time.perf_counter() - evaluation_started)
                common = {
                    "geometry_id": geometry_id,
                    "training_seed": seed,
                    "seed_index": seed_index,
                    "training_materials": count,
                    "pod_rank": int(operators["Kq"].shape[1]),
                    "run_name": str(current_dir.name),
                }
                summary_rows.append(
                    {
                        **common,
                        **summarize_evaluation(evaluation, threshold),
                        "reference_energy_condition": float(
                            metadata["energy_qr_reference_condition"]
                        ),
                        "shared_validation_timing_wall_s": shared_timing_wall_s,
                        "shared_validation_reference_wall_s": (
                            shared_reference_wall_s
                        ),
                    }
                )
                timing_rows.append(
                    {
                        **common,
                        **prefix_timing(
                            snapshot_timing,
                            count,
                            run_summary,
                            recompile_wall_s,
                            evaluation_wall_s,
                        ),
                    }
                )
                compact = evaluation[
                    [
                        "material_id",
                        "relative_frobenius_error",
                        "rom_numerical_failure",
                        "rom_min_eig",
                        "reduced_K_min_eig",
                        "reduced_K_spectral_spd_margin",
                        "schur_eta",
                    ]
                ].copy()
                compact.insert(0, "final_validation_id", np.arange(len(compact)))
                for position, (name, value) in enumerate(common.items(), start=0):
                    compact.insert(position, name, value)
                detail_rows.append(compact)
                print(
                    f"[FIXED-PREFIX] analyze G{geometry_id:02d} seed={seed} "
                    f"Ns={count} mean={summary_rows[-1]['error_mean']:.3e}",
                    flush=True,
                )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["geometry_id", "seed_index", "training_materials"]
    )
    timing = pd.DataFrame(timing_rows).sort_values(
        ["geometry_id", "seed_index", "training_materials"]
    )
    details = pd.concat(detail_rows, ignore_index=True).sort_values(
        ["geometry_id", "seed_index", "training_materials", "final_validation_id"]
    )
    summary.to_csv(summary_dir / "fixed_prefix_seed_summary.csv", index=False)
    timing.to_csv(summary_dir / "fixed_prefix_seed_timing.csv", index=False)
    details.to_csv(summary_dir / "fixed_prefix_seed_validation.csv", index=False)
    required = required_materials(summary, checkpoints, threshold)
    required.to_csv(summary_dir / "fixed_prefix_required_materials.csv", index=False)
    write_figure(summary, timing, study, summary_dir / "fixed_prefix_robustness.png")
    return summary, timing, required


def required_materials(
    summary: pd.DataFrame, checkpoints: list[int], threshold: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (geometry_id, seed), frame in summary.groupby(
        ["geometry_id", "training_seed"], sort=True
    ):
        frame = frame.sort_values("training_materials")
        row: dict[str, Any] = {
            "geometry_id": int(geometry_id),
            "training_seed": int(seed),
            "maximum_checkpoint": int(max(checkpoints)),
        }
        for column, label, limit in (
            ("error_p95", "p95_target", threshold),
            ("error_max", "max_target", threshold),
            ("error_p95", "p95_1e3", 1.0e-3),
            ("error_max", "max_1e3", 1.0e-3),
        ):
            passing = frame.loc[frame[column] <= limit, "training_materials"]
            row[f"first_Ns_{label}"] = (
                int(passing.iloc[0]) if len(passing) else np.nan
            )
            row[f"censored_{label}"] = not bool(len(passing))
        rows.append(row)
    return pd.DataFrame(rows)


def _quantile_band(
    frame: pd.DataFrame, value: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped = frame.groupby("training_materials")[value]
    x = np.asarray(sorted(grouped.groups), dtype=int)
    median = grouped.median().reindex(x).to_numpy(dtype=float)
    lower = grouped.quantile(0.25).reindex(x).to_numpy(dtype=float)
    upper = grouped.quantile(0.75).reindex(x).to_numpy(dtype=float)
    return x, median, lower, upper


def write_figure(
    summary: pd.DataFrame,
    timing: pd.DataFrame,
    study: dict[str, Any],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)

    x, median, lower, upper = _quantile_band(summary, "error_p95")
    axes[0, 0].plot(x, median, marker="o", color="#0072B2", label="Median")
    axes[0, 0].fill_between(x, lower, upper, color="#56B4E9", alpha=0.3, label="IQR")
    axes[0, 0].axhline(float(study["target_error"]), color="black", linestyle="--")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Fixed Sobol prefix $N_s$")
    axes[0, 0].set_ylabel("Held-out 95th-percentile error")
    axes[0, 0].legend(frameon=False)

    x, median, lower, upper = _quantile_band(summary, "below_target_fraction")
    axes[0, 1].plot(x, 100.0 * median, marker="o", color="#009E73")
    axes[0, 1].fill_between(
        x, 100.0 * lower, 100.0 * upper, color="#009E73", alpha=0.25
    )
    axes[0, 1].set_ylim(0.0, 100.0)
    axes[0, 1].set_xlabel("Fixed Sobol prefix $N_s$")
    axes[0, 1].set_ylabel("Validation points below $10^{-4}$ [%]")

    final = summary.loc[
        summary["training_materials"] == int(study["max_training_materials"])
    ]
    pivot = final.pivot(
        index="geometry_id", columns="seed_index", values="error_max"
    ).sort_index()
    image = axes[1, 0].imshow(
        np.log10(np.maximum(pivot.to_numpy(dtype=float), np.finfo(float).tiny)),
        aspect="auto",
        cmap="viridis",
    )
    axes[1, 0].set_xlabel("Sobol scramble index")
    axes[1, 0].set_ylabel("Geometry")
    axes[1, 0].set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    axes[1, 0].set_yticks(
        np.arange(len(pivot.index)), [f"G{int(value):02d}" for value in pivot.index]
    )
    colorbar = fig.colorbar(image, ax=axes[1, 0])
    colorbar.set_label("$\\log_{10}$(maximum error)")

    x, median, lower, upper = _quantile_band(timing, "prefix_compile_wall_s")
    axes[1, 1].plot(x, median, marker="o", color="#D55E00")
    axes[1, 1].fill_between(x, lower, upper, color="#E69F00", alpha=0.25)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Fixed Sobol prefix $N_s$")
    axes[1, 1].set_ylabel("Compilation time [s]")

    for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat, strict=True):
        axis.text(0.01, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    args = _parser().parse_args()
    campaign, study = resolved_settings(args)
    destination = out_root(campaign, args.out_root)
    summary_dir = destination / "runs" / f"{study['run_name']}_summary"
    specs = run_specs(study)
    plan = write_run_plan(
        summary_dir=summary_dir,
        args=args,
        campaign=campaign,
        study=study,
        destination=destination,
        specs=specs,
    )
    if args.stage in {"run", "all"}:
        execute_plan(
            plan=plan,
            summary_dir=summary_dir,
            args=args,
            study=study,
            destination=destination,
        )
    if args.stage in {"analyze", "all"} and not args.dry_run:
        analyze(study=study, destination=destination, summary_dir=summary_dir)
    print(f"[FIXED-PREFIX] summary={summary_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
