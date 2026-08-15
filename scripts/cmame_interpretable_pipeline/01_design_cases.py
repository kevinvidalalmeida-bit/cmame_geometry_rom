#!/usr/bin/env python3
"""Step 1: write the interpretable geometry design table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from env_bootstrap import ensure_configured_venv
from design_grid import rounded_grid_size


HERE = Path(__file__).resolve().parent
CONFIG_DEFAULT = HERE / "campaign_config.json"
ensure_configured_venv(CONFIG_DEFAULT)

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


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
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def nominal_fiber_count(vf: float, box_um: float, length_um: float, diameter_um: float) -> int:
    fiber_volume = math.pi * diameter_um**2 * length_um / 4.0
    return max(1, int(round(vf * box_um**3 / fiber_volume)))


def validate_case(case: dict[str, Any], design_space: dict[str, Any]) -> None:
    vf_min, vf_max = (float(value) for value in design_space["Vf_target"])
    ar_min, ar_max = (float(value) for value in design_space["aspect_ratio"])
    vf = float(case["Vf_target"])
    ar = float(case["aspect_ratio"])
    a2 = [float(value) for value in case["A2"]]
    if not vf_min <= vf <= vf_max:
        raise ValueError(f"{case['label']} has Vf={vf}, outside [{vf_min}, {vf_max}].")
    if not ar_min <= ar <= ar_max:
        raise ValueError(f"{case['label']} has AR={ar}, outside [{ar_min}, {ar_max}].")
    if len(a2) != 3 or min(a2) < -1.0e-12 or not math.isclose(sum(a2), 1.0, abs_tol=1.0e-10):
        raise ValueError(f"{case['label']} has invalid diagonal A2={a2}.")


def design_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    design_space = config["design_space"]
    seed_base = int(design_space["seed_base"])
    diameter_um = float(design_space["fiber_diameter_um"])
    box_factor = float(design_space["box_factor"])
    target_resolution = float(design_space["resolution_vox_per_um"])
    nvox_multiple = int(design_space["nvox_multiple"])
    grid_parity = str(design_space.get("grid_parity", "any"))
    campaign_id = str(config["campaign_id"])

    rows: list[dict[str, Any]] = []
    for geometry_id, case in enumerate(config["manual_cases"]):
        validate_case(case, design_space)
        vf = float(case["Vf_target"])
        ar = float(case["aspect_ratio"])
        a11, a22, a33 = (float(value) for value in case["A2"])
        length_um = ar * diameter_um
        box_um = box_factor * length_um
        nominal_grid_size, grid_size = rounded_grid_size(
            box_um * target_resolution,
            nvox_multiple,
            grid_parity,
        )
        resolution = grid_size / box_um
        fiber_count = nominal_fiber_count(vf, box_um, length_um, diameter_um)
        rows.append(
            {
                "geometry_id": geometry_id,
                "design_id": geometry_id,
                "sobol_index": geometry_id,
                "config_id": campaign_id,
                "geometry_label": str(case["label"]),
                "interpretation": str(case["interpretation"]),
                "Vf_target": vf,
                "aspect_ratio": ar,
                "AR": ar,
                "A2_11": a11,
                "A2_22": a22,
                "A2_33": a33,
                "a11": a11,
                "a22": a22,
                "a33": a33,
                "cluster_fraction": float(case["cluster_fraction"]),
                "cluster_fraction_target": float(case["cluster_fraction"]),
                "seed": seed_base + geometry_id,
                "fiber_diameter_um": diameter_um,
                "fiber_length_um": length_um,
                "box_um": box_um,
                "grid_size": grid_size,
                "grid_size_nominal": nominal_grid_size,
                "grid_parity": grid_parity,
                "grid_parity_adjustment": grid_size - nominal_grid_size,
                "resolution_vox_per_um": resolution,
                "df_voxel": diameter_um * resolution,
                "Lf_Ldom": length_um / box_um,
                "voxelization": str(design_space["voxelization"]),
                "estimated_fiber_count": fiber_count,
                "target_fibers_nominal": fiber_count,
                "label": str(case["label"]),
                "is_operable": True,
                "reject_reason": "",
                "BOX_FACTOR": box_factor,
                "DF_VOXEL_TARGET": target_resolution,
                "NVOX_MULTIPLE": nvox_multiple,
                "caja_um": box_um,
                "nvox": grid_size,
                "nvox_ref": grid_size,
                "res": resolution,
                "res_ref": resolution,
                "voxel_um": box_um / grid_size,
                "d_um": diameter_um,
                "L_um": length_um,
                "fiber_length_lf": length_um,
            }
        )
    return rows


def add_design_distances(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    vf_min, vf_max = (float(value) for value in config["design_space"]["Vf_target"])
    ar_min, ar_max = (float(value) for value in config["design_space"]["aspect_ratio"])
    values = np.column_stack(
        [
            (frame["Vf_target"].to_numpy(float) - vf_min) / (vf_max - vf_min),
            (frame["aspect_ratio"].to_numpy(float) - ar_min) / (ar_max - ar_min),
            frame[["A2_11", "A2_22", "A2_33", "cluster_fraction"]].to_numpy(float),
        ]
    )
    distances = np.full(len(frame), np.nan)
    for idx in range(len(frame)):
        norms = np.linalg.norm(values - values[idx], axis=1)
        norms[idx] = np.inf
        distances[idx] = float(np.min(norms))
    frame = frame.copy()
    frame["design_nearest_distance"] = distances
    return frame


def write_design_outputs(config: dict[str, Any], destination: Path) -> None:
    design_dir = destination / "design"
    geometry_dir = destination / "geometries"
    design_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir.mkdir(parents=True, exist_ok=True)
    design = add_design_distances(pd.DataFrame(design_rows(config)), config)
    design_csv = design_dir / "geometry_design.csv"
    design_xlsx = design_dir / "geometry_design.xlsx"
    geometry_csv = geometry_dir / "geometry_design.csv"
    geometry_xlsx = geometry_dir / "geometry_design.xlsx"
    design.to_csv(design_csv, index=False)
    design.to_csv(geometry_csv, index=False)

    design_space = pd.DataFrame(
        [
            {"name": name, "value": json.dumps(value), "section": "design_space"}
            for name, value in config["design_space"].items()
        ]
    )
    seeds = design[["geometry_id", "geometry_label", "seed"]].copy()
    execution = pd.DataFrame(
        [
            {"step": 1, "script": "01_design_cases.py", "does": "write CSV/XLSX design table"},
            {"step": 2, "script": "02_generate_geometries.py", "does": "generate phase.npy and ori.npy"},
            {"step": 3, "script": "03_run_pipeline.py", "does": "run Sobol+POD full-rank ROM validation"},
        ]
    )
    with pd.ExcelWriter(design_xlsx, engine="openpyxl") as writer:
        design.to_excel(writer, sheet_name="geometry_design", index=False)
        design_space.to_excel(writer, sheet_name="design_space", index=False)
        seeds.to_excel(writer, sheet_name="seeds", index=False)
        execution.to_excel(writer, sheet_name="execution", index=False)
    with pd.ExcelWriter(geometry_xlsx, engine="openpyxl") as writer:
        design.to_excel(writer, sheet_name="geometry_design", index=False)
        design_space.to_excel(writer, sheet_name="design_space", index=False)
        seeds.to_excel(writer, sheet_name="seeds", index=False)
        execution.to_excel(writer, sheet_name="execution", index=False)

    write_json(
        design_dir / "design_manifest.json",
        {
            "status": "design_ready",
            "campaign_id": config["campaign_id"],
            "geometry_count": int(len(design)),
            "design_space": config["design_space"],
            "manual_case_labels": design["geometry_label"].tolist(),
            "outputs": {
                "design_csv": design_csv,
                "design_xlsx": design_xlsx,
                "geometry_csv": geometry_csv,
                "geometry_xlsx": geometry_xlsx,
            },
        },
    )
    write_json(
        geometry_dir / "design_manifest.json",
        {
            "status": "design_ready",
            "source": "copied_from_design_folder",
            "design_manifest": design_dir / "design_manifest.json",
            "outputs": {"csv": geometry_csv, "xlsx": geometry_xlsx},
        },
    )
    print(f"[STEP1] design ready | rows={len(design)} | csv={design_csv}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--out-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    write_design_outputs(config, out_root(config, args.out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
