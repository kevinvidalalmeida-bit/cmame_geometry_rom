#!/usr/bin/env python3
"""Generate a nested AR=5 geometry family while varying only fiber volume fraction."""

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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_DEFAULT = HERE / "config.json"
INTERPRETABLE = ROOT / "scripts" / "cmame_interpretable_pipeline"
if str(INTERPRETABLE) not in sys.path:
    sys.path.insert(0, str(INTERPRETABLE))

from env_bootstrap import ensure_configured_venv

ensure_configured_venv(CONFIG_DEFAULT)

sys.path = [value for value in sys.path if value != str(INTERPRETABLE)]
for path in (ROOT / "scripts", ROOT / "FFT"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

from pipeline import rve_generator


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def nested_orientation_order(table: pd.DataFrame, target_diagonal: np.ndarray) -> np.ndarray:
    """Greedily keep every orientation prefix close to the requested second moment."""
    directions = table[["ux", "uy", "uz"]].to_numpy(dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    target = np.diag(np.asarray(target_diagonal, dtype=float))
    remaining = list(range(len(table)))
    selected: list[int] = []
    moment_sum = np.zeros((3, 3), dtype=float)
    while remaining:
        count = len(selected) + 1
        errors = [
            np.linalg.norm(
                (moment_sum + np.outer(directions[index], directions[index])) / count
                - target,
                ord="fro",
            )
            for index in remaining
        ]
        position = int(np.argmin(errors))
        chosen = remaining.pop(position)
        selected.append(chosen)
        moment_sum += np.outer(directions[chosen], directions[chosen])
    return np.asarray(selected, dtype=np.intp)


def natural_grid(case: dict[str, Any]) -> dict[str, float | int]:
    diameter = float(case["fiber_diameter_um"])
    length = float(case["aspect_ratio"]) * diameter
    box = float(case["box_factor"]) * length
    grid_size = max(8, int(round(box * float(case["resolution_vox_per_um"]))))
    return {
        "fiber_length_um": length,
        "box_um": box,
        "grid_size": grid_size,
        "resolution_vox_per_um": grid_size / box,
    }


def rasterize_prefix(
    table: pd.DataFrame,
    *,
    vf_target: float,
    grid: dict[str, float | int],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    resolution = float(grid["resolution_vox_per_um"])
    box = float(grid["box_um"])
    nvox = int(grid["grid_size"])
    centers = np.mod(
        table[["cx_um", "cy_um", "cz_um"]].to_numpy(dtype=float), box
    ) * resolution
    directions = table[["ux", "uy", "uz"]].to_numpy(dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return rve_generator._rasterize_fibers(
        centers,
        directions,
        float(grid["fiber_length_um"]) * resolution,
        0.5 * float(table["d_um"].iloc[0]) * resolution,
        nvox,
        float(vf_target) * nvox**3,
    )


def generate(config: dict[str, Any], out_root: Path, *, overwrite: bool = False) -> Path:
    family = dict(config["geometry_family"])
    values = np.asarray(family["volume_fractions"], dtype=float)
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("volume_fractions must be strictly increasing.")
    held_out = float(family["held_out_volume_fraction"])
    if not np.any(np.isclose(values, held_out, rtol=0.0, atol=1.0e-12)):
        raise ValueError("held_out_volume_fraction must belong to volume_fractions.")

    source = project_path(config["paths"]["source_master_fibers"])
    if not source.is_file():
        raise FileNotFoundError(f"Missing master fiber table: {source}")
    master = pd.read_csv(source)
    required = {"cx_um", "cy_um", "cz_um", "ux", "uy", "uz", "L_um", "d_um"}
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"Master fiber table is missing: {missing}")

    reference_grid = natural_grid(family)
    if not np.allclose(master["L_um"], float(reference_grid["fiber_length_um"])):
        raise ValueError("Master fibers do not match the requested aspect ratio.")
    if not np.allclose(master["d_um"], float(family["fiber_diameter_um"])):
        raise ValueError("Master fibers do not match the requested diameter.")

    order = nested_orientation_order(master, np.asarray(family["target_A2"], dtype=float))
    master = master.iloc[order].reset_index(drop=True)
    geometry_root = out_root / "geometries"
    if geometry_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Geometry output already exists: {geometry_root}. Use --overwrite."
            )
        shutil.rmtree(geometry_root)
    geometry_root.mkdir(parents=True, exist_ok=True)
    master_path = geometry_root / "nested_master_fibers.csv"
    master.to_csv(master_path, index=False)

    rows: list[dict[str, Any]] = []
    previous_phase: np.ndarray | None = None
    mesh_shapes: set[tuple[int, ...]] = set()
    started = time.perf_counter()
    for geometry_id, vf_target in enumerate(values):
        case_started = time.perf_counter()
        case_definition = {**family, "volume_fraction": float(vf_target)}
        grid = natural_grid(case_definition)
        phase, ori, voxel_count, used_fibers = rasterize_prefix(
            master,
            vf_target=float(vf_target),
            grid=grid,
        )
        if previous_phase is not None and np.any(previous_phase > phase):
            raise RuntimeError("The generated volume-fraction family is not nested.")
        previous_phase = phase
        mesh_shapes.add(tuple(int(value) for value in phase.shape))

        case_dir = geometry_root / f"geometry_{geometry_id:02d}"
        case_dir.mkdir(parents=True, exist_ok=False)
        phase_path = case_dir / "phase.npy"
        ori_path = case_dir / "ori.npy"
        np.save(phase_path, phase)
        np.save(ori_path, ori)
        master.iloc[:used_fibers].to_csv(
            case_dir / "continuous_fibers.csv", index=False
        )

        vf_realized = float(voxel_count / phase.size)
        a2 = rve_generator._voxelized_a2(phase, ori)
        target_a2 = np.diag(np.asarray(family["target_A2"], dtype=float))
        a2_error = float(
            np.linalg.norm(a2 - target_a2, ord="fro")
            / max(np.linalg.norm(target_a2, ord="fro"), np.finfo(float).eps)
        )
        vf_ok = abs(vf_realized - float(vf_target)) <= float(family["vf_tolerance"])
        a2_ok = a2_error <= float(family["A2_tolerance"])
        label = f"vf{int(round(100 * vf_target)):02d}_ar05_random3d"
        row = {
            "geometry_id": int(geometry_id),
            "geometry_label": label,
            "Vf_target": float(vf_target),
            "Vf_realized": vf_realized,
            "Vf_error": vf_realized - float(vf_target),
            "aspect_ratio": float(family["aspect_ratio"]),
            "AR": float(family["aspect_ratio"]),
            "A2_11": float(a2[0, 0]),
            "A2_22": float(a2[1, 1]),
            "A2_33": float(a2[2, 2]),
            "A2_error_rel": a2_error,
            "fiber_count": int(used_fibers),
            "fiber_diameter_um": float(family["fiber_diameter_um"]),
            "fiber_length_um": float(grid["fiber_length_um"]),
            "box_um": float(grid["box_um"]),
            "grid_size": int(grid["grid_size"]),
            "resolution_vox_per_um": float(grid["resolution_vox_per_um"]),
            "nvox": int(grid["grid_size"]),
            "res": float(grid["resolution_vox_per_um"]),
            "caja_um": float(grid["box_um"]),
            "L_um": float(grid["fiber_length_um"]),
            "d_um": float(family["fiber_diameter_um"]),
            "a11": float(family["target_A2"][0]),
            "a22": float(family["target_A2"][1]),
            "a33": float(family["target_A2"][2]),
            "cluster_fraction": 0.0,
            "is_held_out": bool(math.isclose(float(vf_target), held_out, abs_tol=1e-12)),
            "accepted_local": bool(vf_ok and a2_ok),
            "phase_sha256": sha256(phase_path),
            "ori_sha256": sha256(ori_path),
            "generation_wall_s": float(time.perf_counter() - case_started),
        }
        write_json(
            case_dir / "generation_result.json",
            {
                **row,
                "status": "geometry_ok" if row["accepted_local"] else "geometry_rejected",
                "nested_family": True,
                "nested_master_fibers": str(master_path),
                "source_master_fibers": str(source),
                "sam_vf_ok": vf_ok,
                "sam_A2_ok": a2_ok,
                "sam_overlap_ok": True,
            },
        )
        rows.append(row)

    compatible_direct_space = len(mesh_shapes) == 1

    frame = pd.DataFrame(rows)
    frame.to_csv(geometry_root / "geometry_family.csv", index=False)
    with pd.ExcelWriter(geometry_root / "geometry_family.xlsx") as writer:
        frame.to_excel(writer, sheet_name="geometry_family", index=False)
    shutil.copy2(source, geometry_root / "source_master_fibers.csv")
    manifest = {
        "campaign_id": config["campaign_id"],
        "status": "geometry_family_ready",
        "geometry_count": int(len(frame)),
        "training_geometry_count": int((~frame["is_held_out"]).sum()),
        "held_out_volume_fraction": held_out,
        "aspect_ratio": float(family["aspect_ratio"]),
        "natural_grid_policy": "round(box_um * resolution_vox_per_um)",
        "mesh_assignment": "independent_per_geometry",
        "native_meshes_directly_compatible": compatible_direct_space,
        "native_mesh_shapes": [list(shape) for shape in sorted(mesh_shapes)],
        "cross_mesh_policy": (
            "native_physical_grid_coordinates"
            if compatible_direct_space
            else "explicit_transfer_required_before_common_POD"
        ),
        "operator_local_ordering": "independent_per_geometry",
        "nested_master_policy": "greedy_isotropic_orientation_prefix",
        "all_accepted": bool(frame["accepted_local"].all()),
        "wall_s": float(time.perf_counter() - started),
        "source_master_fibers": str(source),
        "nested_master_sha256": sha256(master_path),
    }
    write_json(geometry_root / "family_manifest.json", manifest)
    if not manifest["all_accepted"]:
        rejected = frame.loc[~frame["accepted_local"], "geometry_label"].tolist()
        raise RuntimeError(f"Geometry tolerances failed for: {rejected}")
    print(
        f"[VF-FAMILY] ready | geometries={len(frame)} | "
        f"AR={family['aspect_ratio']:.1f} | "
        f"native_grids={sorted({int(row['grid_size']) for row in rows})}",
        flush=True,
    )
    return geometry_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    out_root = (
        args.out_root.resolve()
        if args.out_root is not None
        else project_path(config["paths"]["out_root"]).resolve()
    )
    generate(config, out_root, overwrite=bool(args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
