#!/usr/bin/env python3
"""Step 2: generate binary RVEs from the interpretable design table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from env_bootstrap import ensure_configured_venv


HERE = Path(__file__).resolve().parent
CONFIG_DEFAULT = HERE / "campaign_config.json"
ensure_configured_venv(CONFIG_DEFAULT)

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FFT_ROOT = ROOT / "FFT"
FFTHOMPY_ROOT = FFT_ROOT / "ffthompy_core" / "ffthompy"
for path in (FFT_ROOT, FFTHOMPY_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from pipeline.rve_generator import generate_rve_main


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def out_root(config: dict[str, Any], override: Path | None) -> Path:
    return (override or project_path(config["paths"]["out_root"])).resolve()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attach_geometry_hashes(payload: dict[str, Any], phase_path: Path, ori_path: Path) -> dict[str, Any]:
    payload = dict(payload)
    if "phase_sha256" not in payload:
        payload["phase_sha256"] = file_sha256(phase_path)
    if "ori_sha256" not in payload:
        payload["ori_sha256"] = file_sha256(ori_path)
    payload["phase_size_MB"] = phase_path.stat().st_size / 1024.0**2
    payload["ori_size_MB"] = ori_path.stat().st_size / 1024.0**2
    return payload


def read_design(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="geometry_design") if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    required = {
        "geometry_id",
        "geometry_label",
        "Vf_target",
        "aspect_ratio",
        "A2_11",
        "A2_22",
        "cluster_fraction",
        "seed",
        "fiber_diameter_um",
        "fiber_length_um",
        "box_um",
        "resolution_vox_per_um",
        "grid_size",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Design table is missing columns: {missing}")
    return frame.sort_values("geometry_id").reset_index(drop=True)


def generation_params(row: pd.Series, case_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    generation = config["geometry_generation"]
    cluster_fraction = float(row.get("cluster_fraction", 0.0))
    center_distribution = "gaussian_mixture" if cluster_fraction > 0.0 else "uniform"
    start_scale = float(generation["sam_compaction_start_scale"])
    return {
        "gap_um": 0.0,
        "resol": float(row["resolution_vox_per_um"]),
        "seed": int(row["seed"]),
        "output_dir": str(case_dir),
        "num_cores": int(generation["generator_cores"]),
        "sam_num_threads": int(generation["generator_cores"]),
        "caja_um": float(row["box_um"]),
        "L_um": float(row["fiber_length_um"]),
        "d_um": float(row["fiber_diameter_um"]),
        "Vf_target": float(row["Vf_target"]),
        "a11": float(row["A2_11"]),
        "a22": float(row["A2_22"]),
        "compute_metrics": bool(generation["compute_rve_metrics"]),
        "export_fiber_table": True,
        "save_continuous_geometry": True,
        "sam_batch_vf": float(generation["sam_batch_vf"]),
        "sam_use_voxel_stop_callback": bool(generation["sam_use_voxel_stop_callback"]),
        "sam_voxel_stop_start": float(generation["sam_voxel_stop_start"]),
        "sam_inflation_samples": int(generation["sam_inflation_samples"]),
        "sam_target_safety": float(generation["sam_target_safety"]),
        "sam_max_topups": int(generation["sam_max_topups"]),
        "sam_topup_gain": float(generation["sam_topup_gain"]),
        "sam_continuous_vf_cap_factor": float(generation["sam_continuous_vf_cap_factor"]),
        "sam_vf_tolerance": float(generation["sam_vf_tolerance"]),
        "sam_A2_tolerance": float(generation["sam_A2_tolerance"]),
        "sam_voxel_A2_tolerance": float(generation["sam_voxel_A2_tolerance"]),
        "sam_orientation_max_iter": int(generation["sam_orientation_max_iter"]),
        "sam_geometry_backend": str(generation["geometry_backend"]),
        "sam_cupy_min_candidate_pairs": int(generation["sam_cupy_min_candidate_pairs"]),
        "sam_compaction": bool(generation["sam_compaction"]),
        "sam_compaction_start_scale": start_scale,
        "sam_compaction_topup_scale": start_scale,
        "sam_compaction_stages": int(generation["sam_compaction_stages"]),
        "sam_compaction_max_iter": int(generation["sam_compaction_max_iter"]),
        "sam_compaction_passes": int(generation["sam_compaction_passes"]),
        "sam_collective_fire": bool(generation["sam_collective_fire"]),
        "sam_collective_fire_max_iter": int(generation["sam_collective_fire_max_iter"]),
        "sam_collective_fire_max_restarts": int(generation["sam_collective_fire_max_restarts"]),
        "sam_collective_fire_restart_patience": int(generation["sam_collective_fire_restart_patience"]),
        "sam_collective_fire_pair_rebuild_interval": int(generation["sam_collective_fire_pair_rebuild_interval"]),
        "sam_collective_fire_rescue_passes": int(generation["sam_collective_fire_rescue_passes"]),
        "sam_collective_fire_removal_batch": int(generation["sam_collective_fire_removal_batch"]),
        "sam_overlap_tolerance_relative": float(generation["sam_overlap_tolerance_relative"]),
        "sam_overlap_rescue": bool(generation["sam_overlap_rescue"]),
        "sam_overlap_rescue_passes": int(generation["sam_overlap_rescue_passes"]),
        "sam_overlap_rescue_max_iter": int(generation["sam_overlap_rescue_max_iter"]),
        "sam_overlap_rescue_max_fibers": int(generation["sam_overlap_rescue_max_fibers"]),
        "sam_center_distribution": center_distribution,
        "sam_cluster_fraction": cluster_fraction,
        "sam_cluster_count": int(generation["sam_cluster_count"]),
        "sam_cluster_sigma_rel": float(generation["sam_cluster_sigma_rel"]),
        "sam_verbose": bool(generation["verbose_geometry"]),
    }


def descriptor_from_payload(row: pd.Series, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "geometry_id": int(row["geometry_id"]),
        "grid_size": int(row["grid_size"]),
        "Vf_realized": float(payload.get("Vf", np.nan)),
        "aspect_ratio": float(row["aspect_ratio"]),
        "A2_11_realized": float(payload.get("sam_voxel_A2_11", payload.get("sam_A2_11", np.nan))),
        "A2_22_realized": float(payload.get("sam_voxel_A2_22", payload.get("sam_A2_22", np.nan))),
        "A2_33_realized": float(payload.get("sam_voxel_A2_33", payload.get("sam_A2_33", np.nan))),
        "cluster_fraction_target": float(row.get("cluster_fraction", 0.0)),
        "interface_density": float(payload.get("interface_density", np.nan)),
        "Ripley_peak": float(payload.get("Ripley_peak", np.nan)),
        "D_star": float(payload.get("D_star", np.nan)),
        "n_fibers": int(payload.get("n_fibers", -1)),
        "max_overlap_vox": float(payload.get("sam_final_overlap", np.nan)),
        "max_overlap_relative": float(
            payload.get("sam_final_overlap_relative", np.nan)
        ),
        "overlap_tolerance_relative": float(
            payload.get("sam_overlap_tolerance_relative", np.nan)
        ),
        "generation_strategy": str(
            payload.get("sam_generation_strategy", "")
        ),
        "accepted_local": bool(payload.get("accepted_local", False)),
        "phase_sha256": str(payload.get("phase_sha256", "")),
        "ori_sha256": str(payload.get("ori_sha256", "")),
        "phase_size_MB": float(payload.get("phase_size_MB", np.nan)),
        "ori_size_MB": float(payload.get("ori_size_MB", np.nan)),
    }


def add_descriptor_distances(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        frame["realized_nearest_distance"] = []
        frame["descriptor_separation_ok"] = []
        return frame
    if len(frame) == 1:
        frame = frame.copy()
        frame["realized_nearest_distance"] = np.nan
        frame["descriptor_separation_ok"] = True
        return frame
    columns = [
        "Vf_realized",
        "aspect_ratio",
        "A2_11_realized",
        "A2_22_realized",
        "A2_33_realized",
        "cluster_fraction_target",
    ]
    values = frame[columns].to_numpy(dtype=float)
    scale = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    scaled = (values - np.nanmin(values, axis=0)) / scale
    distances = np.full(len(frame), np.nan)
    for idx in range(len(frame)):
        norms = np.linalg.norm(scaled - scaled[idx], axis=1)
        norms[idx] = np.inf
        distances[idx] = float(np.nanmin(norms))
    frame = frame.copy()
    frame["realized_nearest_distance"] = distances
    frame["descriptor_separation_ok"] = np.isfinite(distances)
    return frame


def generate_one(row: pd.Series, geometry_root: Path, config: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    geometry_id = int(row["geometry_id"])
    case_dir = geometry_root / f"geometry_{geometry_id:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "generation_result.json"
    phase_path = case_dir / "phase.npy"
    ori_path = case_dir / "ori.npy"
    if result_path.is_file() and phase_path.is_file() and ori_path.is_file() and not overwrite:
        expected_shape = (int(row["grid_size"]),) * 3
        stored_shape = tuple(int(value) for value in np.load(phase_path, mmap_mode="r").shape)
        stored_orientation_shape = tuple(
            int(value) for value in np.load(ori_path, mmap_mode="r").shape
        )
        if stored_shape != expected_shape or stored_orientation_shape != expected_shape + (3,):
            raise RuntimeError(
                f"geometry_{geometry_id:02d} has stored phase/orientation shapes "
                f"{stored_shape}/{stored_orientation_shape}, but the design requires "
                f"{expected_shape}/{expected_shape + (3,)}. Use a separate --out-root "
                "or regenerate with --overwrite."
            )
        print(f"[STEP2] reuse geometry_{geometry_id:02d} | {row['geometry_label']}", flush=True)
        payload = attach_geometry_hashes(
            json.loads(result_path.read_text(encoding="utf-8")),
            phase_path,
            ori_path,
        )
        write_json(result_path, payload)
        return payload

    print(
        "[STEP2] generate "
        f"geometry_{geometry_id:02d} | label={row['geometry_label']} | "
        f"Vf={float(row['Vf_target']):.3f} | AR={float(row['aspect_ratio']):.2f} | "
        f"N={int(row['grid_size'])}",
        flush=True,
    )
    started = time.perf_counter()
    gen_info = generate_rve_main(generation_params(row, case_dir, config))
    accepted = bool(
        gen_info.get("sam_vf_ok", False)
        and gen_info.get("sam_A2_ok", False)
        and gen_info.get("sam_overlap_ok", False)
    )
    payload = {
        **gen_info,
        "geometry_id": geometry_id,
        "geometry_label": str(row["geometry_label"]),
        "interpretation": str(row.get("interpretation", "")),
        "generation_wall_s": float(time.perf_counter() - started),
        "accepted_local": accepted,
        "acceptance": {
            "volume_fraction": bool(gen_info.get("sam_vf_ok", False)),
            "orientation": bool(gen_info.get("sam_A2_ok", False)),
            "overlap": bool(gen_info.get("sam_overlap_ok", False)),
        },
        "design": row.to_dict(),
    }
    if not phase_path.is_file() or not ori_path.is_file():
        raise FileNotFoundError(f"Missing phase.npy/ori.npy for geometry_{geometry_id:02d}")
    payload = attach_geometry_hashes(payload, phase_path, ori_path)
    write_json(result_path, payload)
    return payload


def run(config: dict[str, Any], destination: Path, selected_ids: set[int] | None, overwrite: bool) -> None:
    geometry_root = destination / "geometries"
    design = read_design(geometry_root / "geometry_design.csv")
    selected = selected_ids or set(int(value) for value in design["geometry_id"])
    partial_update = selected_ids is not None

    descriptors: list[dict[str, Any]] = []
    generation_records: list[dict[str, Any]] = []
    for _, row in design.iterrows():
        if int(row["geometry_id"]) not in selected:
            continue
        payload = generate_one(row, geometry_root, config, overwrite)
        descriptors.append(descriptor_from_payload(row, payload))
        generation_records.append(
            {
                "geometry_id": int(row["geometry_id"]),
                "geometry_label": str(row["geometry_label"]),
                "grid_size": int(row["grid_size"]),
                "accepted_local": bool(payload.get("accepted_local", False)),
                "Vf_realized": float(payload.get("Vf", np.nan)),
                "n_fibers": int(payload.get("n_fibers", -1)),
                "generation_wall_s": float(payload.get("generation_wall_s", payload.get("t_gen", np.nan))),
                "generation_strategy": str(
                    payload.get("sam_generation_strategy", "")
                ),
                "fire_wall_s": float(
                    payload.get("sam_collective_fire_s", np.nan)
                ),
                "max_overlap_relative": float(
                    payload.get("sam_final_overlap_relative", np.nan)
                ),
                "overlap_tolerance_relative": float(
                    payload.get("sam_overlap_tolerance_relative", np.nan)
                ),
                "removed_fibers": int(
                    payload.get("sam_compaction_removed_fibers", 0)
                ),
                "phase_sha256": str(payload.get("phase_sha256", "")),
                "ori_sha256": str(payload.get("ori_sha256", "")),
                "phase_size_MB": float(payload.get("phase_size_MB", np.nan)),
                "ori_size_MB": float(payload.get("ori_size_MB", np.nan)),
                "case_dir": str(geometry_root / f"geometry_{int(row['geometry_id']):02d}"),
            }
        )

    realized_updates = pd.DataFrame(descriptors)
    generation_updates = pd.DataFrame(generation_records)
    if partial_update:
        descriptors_path = geometry_root / "geometry_realized_descriptors.csv"
        if descriptors_path.is_file():
            previous = pd.read_csv(descriptors_path)
            previous = previous[~previous["geometry_id"].astype(int).isin(selected)]
            realized_updates = pd.concat([previous, realized_updates], ignore_index=True)
        times_path = geometry_root / "geometry_generation_times.csv"
        if times_path.is_file():
            previous_times = pd.read_csv(times_path)
            previous_times = previous_times[
                ~previous_times["geometry_id"].astype(int).isin(selected)
            ]
            generation_updates = pd.concat(
                [previous_times, generation_updates],
                ignore_index=True,
            )

    realized = add_descriptor_distances(
        realized_updates.sort_values("geometry_id").reset_index(drop=True)
    )
    generation_frame = generation_updates.sort_values("geometry_id").reset_index(drop=True)
    realized.to_csv(geometry_root / "geometry_realized_descriptors.csv", index=False)
    generation_frame.to_csv(geometry_root / "geometry_generation_times.csv", index=False)
    with pd.ExcelWriter(geometry_root / "geometry_realized_descriptors.xlsx", engine="openpyxl") as writer:
        design.to_excel(writer, sheet_name="design", index=False)
        realized.to_excel(writer, sheet_name="realized", index=False)
        generation_frame.to_excel(writer, sheet_name="generation", index=False)

    all_accepted = bool(realized["accepted_local"].all()) if not realized.empty else False
    write_json(
        geometry_root / "campaign_manifest.json",
        {
            "status": "complete" if all_accepted else "generated_with_warnings",
            "campaign_id": config["campaign_id"],
            "geometry_count": int(len(realized)),
            "all_accepted": all_accepted,
            "design_space": config["design_space"],
            "geometry_generation": config["geometry_generation"],
        },
    )
    print(
        f"[STEP2] geometries ready | accepted={int(realized['accepted_local'].sum())}/{len(realized)} | dir={geometry_root}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--geometry-ids", type=int, nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    selected = set(args.geometry_ids) if args.geometry_ids is not None else None
    run(config, out_root(config, args.out_root), selected, bool(args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
