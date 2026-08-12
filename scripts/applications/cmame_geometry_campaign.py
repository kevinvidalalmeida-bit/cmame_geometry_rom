#!/usr/bin/env python3
"""Reproducible maximin design and SAM generation for ten CMAME geometries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "FFT") not in sys.path:
    sys.path.insert(0, str(ROOT / "FFT"))

from pipeline.rve_generator import generate_rve_main


OUT_DEFAULT = ROOT / "results" / "cmame_method" / "geometries"
BOX_FACTOR = 2.0
VOXELS_PER_FIBER_DIAMETER = 6.0
NVOX_MULTIPLE = 1
AR_RANGE = (6.0, 12.0)
VF_RANGE = (0.10, 0.24)
BASELINE_AR = 10.0
BASELINE_VF = 0.20


def _round_up_to_multiple(value: float, multiple: int = 1) -> int:
    multiple = max(1, int(multiple))
    return int(np.ceil(float(value) / float(multiple)) * multiple)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _design_features(rows: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        (
            (rows["Vf_target"].to_numpy() - VF_RANGE[0]) / (VF_RANGE[1] - VF_RANGE[0]),
            (rows["aspect_ratio"].to_numpy() - AR_RANGE[0]) / (AR_RANGE[1] - AR_RANGE[0]),
            rows[["A2_11", "A2_22", "A2_33"]].to_numpy() * np.sqrt(3.0 / 2.0),
            rows["cluster_fraction"].to_numpy() / 0.70,
        )
    )


def build_maximin_design(*, candidate_count: int, seed: int) -> pd.DataFrame:
    power = int(np.ceil(np.log2(max(16, int(candidate_count)))))
    unit = qmc.Sobol(d=6, scramble=True, seed=int(seed)).random_base2(power)[:candidate_count]
    orientation_raw = -np.log(np.clip(unit[:, 2:5], 1.0e-12, 1.0))
    orientation = 0.06 + orientation_raw
    orientation /= orientation.sum(axis=1, keepdims=True)
    candidates = pd.DataFrame(
        {
            "Vf_target": VF_RANGE[0] + (VF_RANGE[1] - VF_RANGE[0]) * unit[:, 0],
            "aspect_ratio": AR_RANGE[0] + (AR_RANGE[1] - AR_RANGE[0]) * unit[:, 1],
            "A2_11": orientation[:, 0],
            "A2_22": orientation[:, 1],
            "A2_33": orientation[:, 2],
            "cluster_fraction": 0.70 * unit[:, 5],
        }
    )
    baseline = pd.DataFrame(
        [{
            "Vf_target": BASELINE_VF,
            "aspect_ratio": BASELINE_AR,
            "A2_11": 1.0 / 3.0,
            "A2_22": 1.0 / 3.0,
            "A2_33": 1.0 / 3.0,
            "cluster_fraction": 0.0,
        }]
    )
    features = _design_features(candidates)
    selected_rows = [baseline.iloc[0].to_dict()]
    selected_features = [_design_features(baseline)[0]]
    available = np.ones(len(candidates), dtype=bool)
    while len(selected_rows) < 10:
        distances = np.linalg.norm(
            features[:, None, :] - np.asarray(selected_features)[None, :, :], axis=2
        )
        minimum = distances.min(axis=1)
        minimum[~available] = -np.inf
        index = int(np.argmax(minimum))
        selected_rows.append(candidates.iloc[index].to_dict())
        selected_features.append(features[index])
        available[index] = False
    design = pd.DataFrame(selected_rows)
    design.insert(0, "geometry_id", np.arange(10, dtype=int))
    design["geometry_label"] = ["baseline" if i == 0 else f"maximin_{i:02d}" for i in range(10)]
    design["seed"] = int(seed) * 100 + design["geometry_id"]
    design["fiber_diameter_um"] = 1.0
    design["fiber_length_um"] = design["aspect_ratio"]
    design["box_um"] = BOX_FACTOR * design["fiber_length_um"]
    design["grid_size"] = [
        _round_up_to_multiple(
            VOXELS_PER_FIBER_DIAMETER * box_um / diameter_um,
            NVOX_MULTIPLE,
        )
        for box_um, diameter_um in zip(design["box_um"], design["fiber_diameter_um"])
    ]
    design["resolution_vox_per_um"] = design["grid_size"] / design["box_um"]
    design["df_voxel"] = design["fiber_diameter_um"] * design["resolution_vox_per_um"]
    design["Lf_Ldom"] = design["fiber_length_um"] / design["box_um"]
    design["voxelization"] = "binary"
    design["estimated_fiber_count"] = np.ceil(
        design["Vf_target"]
        * design["box_um"] ** 3
        / (np.pi * design["fiber_diameter_um"] ** 2 * design["fiber_length_um"] / 4.0)
    ).astype(int)
    feature_distances = squareform(pdist(np.asarray(selected_features)))
    np.fill_diagonal(feature_distances, np.inf)
    design["design_nearest_distance"] = feature_distances.min(axis=1)
    return design


def _generate_one(row: pd.Series, out_dir: Path, *, overwrite: bool) -> dict[str, Any]:
    geometry_dir = out_dir / f"geometry_{int(row['geometry_id']):02d}"
    manifest_path = geometry_dir / "generation_result.json"
    if manifest_path.is_file() and not overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    geometry_dir.mkdir(parents=True, exist_ok=True)
    clustered = float(row["cluster_fraction"]) > 0.0
    dense = float(row["Vf_target"]) >= 0.25
    box_um = float(row["box_um"])
    grid_size = int(row["grid_size"])
    result = generate_rve_main(
        {
            "L_um": float(row["fiber_length_um"]),
            "d_um": 1.0,
            "gap_um": 0.0,
            "caja_um": box_um,
            "resol": grid_size / box_um,
            "Vf_target": float(row["Vf_target"]),
            "seed": int(row["seed"]),
            "output_dir": str(geometry_dir),
            "a11": float(row["A2_11"]),
            "a22": float(row["A2_22"]),
            "compute_metrics": True,
            "export_fiber_table": True,
            "save_continuous_geometry": True,
            "sam_geometry_backend": "numba",
            "sam_num_threads": 16,
            "sam_fast_mode": True,
            "sam_verbose": False,
            "sam_center_distribution": "gaussian_mixture" if clustered else "uniform",
            "sam_cluster_fraction": float(row["cluster_fraction"]),
            "sam_cluster_count": 4,
            "sam_cluster_sigma_rel": 0.12,
            "sam_strict_insertion": False,
            "sam_compaction": bool(dense),
            "sam_collective_fire": bool(dense),
            "sam_collective_fire_min_packing_load": 0.0,
            "sam_compaction_max_iter": 500,
            "sam_compaction_passes": 3,
            "sam_overlap_rescue": True,
            "sam_overlap_rescue_passes": 5,
            "sam_overlap_rescue_max_iter": 300,
            "sam_max_iter_relax": 180,
            "sam_soft_insert": bool(dense),
            "sam_soft_insert_start": 0.65,
            "sam_max_topups": 10,
            "sam_A2_tolerance": 0.035,
            "sam_voxel_A2_tolerance": 0.06,
            "sam_vf_tolerance": 0.006,
            "sam_overlap_tolerance": 0.05,
            "sam_tol_overlap": 0.05,
        }
    )
    result.pop("fibers", None)
    acceptance = {
        "volume_fraction": bool(result["sam_vf_ok"]),
        "orientation": bool(result["sam_A2_ok"]),
        "overlap": bool(result["sam_overlap_ok"]),
    }
    result["geometry_id"] = int(row["geometry_id"])
    result["geometry_label"] = str(row["geometry_label"])
    result["acceptance"] = acceptance
    result["accepted_local"] = bool(all(acceptance.values()))
    _write_json(manifest_path, result)
    return result


def _realized_descriptor_table(design: pd.DataFrame, results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for (_, target), result in zip(design.iterrows(), results):
        a2 = np.diag(
            [
                float(result.get("sam_voxel_A2_11", target["A2_11"])),
                float(result.get("sam_voxel_A2_22", target["A2_22"])),
                float(result.get("sam_voxel_A2_33", target["A2_33"])),
            ]
        )
        rows.append(
            {
                "geometry_id": int(target["geometry_id"]),
                "Vf_realized": float(result["Vf"]),
                "aspect_ratio": float(target["aspect_ratio"]),
                "A2_11_realized": float(a2[0, 0]),
                "A2_22_realized": float(a2[1, 1]),
                "A2_33_realized": float(a2[2, 2]),
                "cluster_fraction_target": float(target["cluster_fraction"]),
                "interface_density": float(result["interface_density"]),
                "Ripley_peak": float(result["Ripley_peak"]),
                "D_star": float(result["D_star"]),
                "n_fibers": int(result["n_fibers"]),
                "accepted_local": bool(result["accepted_local"]),
            }
        )
    realized = pd.DataFrame(rows)
    descriptor_columns = [
        "Vf_realized", "aspect_ratio", "A2_11_realized", "A2_22_realized",
        "A2_33_realized", "interface_density", "Ripley_peak", "D_star", "n_fibers",
    ]
    values = realized[descriptor_columns].to_numpy(dtype=float)
    scale = np.ptp(values, axis=0)
    scale[scale <= 1.0e-14] = 1.0
    normalized = (values - values.min(axis=0)) / scale
    distances = squareform(pdist(normalized))
    np.fill_diagonal(distances, np.inf)
    realized["realized_nearest_distance"] = distances.min(axis=1)
    realized["descriptor_separation_ok"] = realized["realized_nearest_distance"] > 0.15
    return realized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--candidate-count", type=int, default=4096)
    parser.add_argument("--design-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--geometry-ids", type=int, nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    design = build_maximin_design(candidate_count=int(args.candidate_count), seed=int(args.seed))
    design.to_csv(out_dir / "geometry_design.csv", index=False)
    design_hash = hashlib.sha256(design.to_csv(index=False).encode("utf-8")).hexdigest()
    if args.design_only:
        _write_json(out_dir / "campaign_manifest.json", {
            "status": "design_only",
            "design_sha256": design_hash,
        "geometry_count": 10,
        "ar_range": list(AR_RANGE),
        "vf_range": list(VF_RANGE),
        "box_factor": float(BOX_FACTOR),
        "voxels_per_fiber_diameter": float(VOXELS_PER_FIBER_DIAMETER),
        "voxelization": "binary",
        "estimated_max_fiber_count": int(design["estimated_fiber_count"].max()),
        "minimum_design_distance": float(design["design_nearest_distance"].min()),
    })
        return 0

    selected = (
        set(int(value) for value in args.geometry_ids)
        if args.geometry_ids is not None
        else set(int(value) for value in design["geometry_id"])
    )
    results = []
    for _, row in design.iterrows():
        geometry_id = int(row["geometry_id"])
        results.append(
            _generate_one(
                row,
                out_dir,
                overwrite=bool(args.overwrite and geometry_id in selected),
            )
            if geometry_id in selected
            else _generate_one(row, out_dir, overwrite=False)
        )
    realized = _realized_descriptor_table(design, results)
    realized.to_csv(out_dir / "geometry_realized_descriptors.csv", index=False)
    campaign = {
        "status": "complete",
        "design_sha256": design_hash,
        "geometry_count": 10,
        "ar_range": list(AR_RANGE),
        "vf_range": list(VF_RANGE),
        "box_factor": float(BOX_FACTOR),
        "voxels_per_fiber_diameter": float(VOXELS_PER_FIBER_DIAMETER),
        "voxelization": "binary",
        "estimated_max_fiber_count": int(design["estimated_fiber_count"].max()),
        "local_acceptance_count": int(realized["accepted_local"].sum()),
        "descriptor_separation_count": int(realized["descriptor_separation_ok"].sum()),
        "minimum_design_distance": float(design["design_nearest_distance"].min()),
        "minimum_realized_descriptor_distance": float(realized["realized_nearest_distance"].min()),
        "all_accepted": bool(realized["accepted_local"].all() and realized["descriptor_separation_ok"].all()),
    }
    _write_json(out_dir / "campaign_manifest.json", campaign)
    print(
        "[GEOMETRIES] listo | "
        f"accepted={campaign['local_acceptance_count']}/10 | "
        f"separated={campaign['descriptor_separation_count']}/10 | "
        f"min_distance={campaign['minimum_realized_descriptor_distance']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
