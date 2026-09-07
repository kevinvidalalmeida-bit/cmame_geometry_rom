#!/usr/bin/env python3
"""Revoxelize three fixed continuous RVEs and quantify FFT resolution convergence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from env_bootstrap import ensure_configured_venv


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_DEFAULT = HERE / "campaign_config.json"
ensure_configured_venv(CONFIG_DEFAULT)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

for path in (ROOT / "FFT", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pipeline.rve_generator import rasterize_continuous_fibers
import cmame_campaign_common as common
import fft_homogenization_solver as sweep


CARBON_FIBER_290_EPOXY: dict[str, Any] = {
    "material_id": 0,
    "material_label": "Carbon Fiber (290 GPa)/Resin Epoxy",
    "Em": 3.78,
    "nu_m": 0.35,
    "Ef_L": 290.0,
    "Ef_T": 23.0,
    "G_LT": 9.0,
    "nu_LT": 0.20,
    "nu_TT": 0.40,
}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    command.add_argument("--geometry-ids", type=int, nargs="+", default=[8, 0, 9])
    command.add_argument("--resolutions", type=float, nargs="+", default=[3, 4, 5, 6])
    command.add_argument("--profile", default="snapshot32", choices=tuple(common.SOLVER_PROFILES))
    command.add_argument("--fft-backend", choices=("cpu", "gpu"), default="gpu")
    command.add_argument("--paper-figure-dir", type=Path, default=ROOT / "paper" / "figures")
    command.add_argument("--overwrite", action="store_true")
    return command


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def assert_material_inside_design_space(material: dict[str, Any]) -> None:
    outside = {
        name: float(material[name])
        for name, (lower, upper) in sweep.MATERIAL_BOUNDS.items()
        if not lower <= float(material[name]) <= upper
    }
    if outside:
        raise ValueError(f"Resolution-study material lies outside Xi: {outside}")


def ceff_from_record(record: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            [float(record[f"Ceff_{ii + 1}{jj + 1}"]) for jj in range(6)]
            for ii in range(6)
        ],
        dtype=np.float64,
    )


def write_convergence_figure(table: pd.DataFrame, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    labels = {
        8: "G08: AR=5, Vf=0.05",
        0: "G00: AR=12.5, Vf=0.15",
        9: "G09: AR=20, Vf=0.25",
    }
    colors = {8: "#0072B2", 0: "#009E73", 9: "#D55E00"}

    figure, error_axis = plt.subplots(
        figsize=(6.2, 3.8),
        constrained_layout=True,
    )
    for geometry_id, group in table.groupby("geometry_id", sort=False):
        ordered = group.sort_values("resolution_vox_per_um")
        error_axis.plot(
            ordered["resolution_vox_per_um"],
            100.0 * ordered["relative_Ceff_error_vs_r6"],
            marker="o",
            linewidth=1.6,
            color=colors.get(int(geometry_id)),
            label=labels.get(int(geometry_id), f"G{int(geometry_id):02d}"),
        )
    error_axis.set_xlabel("Voxels per fiber diameter")
    error_axis.set_ylabel(r"Relative $C^{\mathrm{eff}}$ error vs. resolution 6 [%]")
    error_axis.set_xticks([3, 4, 5, 6])
    error_axis.grid(True, alpha=0.25)
    error_axis.legend(frameon=False, fontsize=8)
    error_axis.set_title("Effective-stiffness convergence", loc="left")

    figure.savefig(
        destination / "numerical_voxel_resolution_convergence.png",
        dpi=240,
    )
    plt.close(figure)


def main() -> int:
    args = parser().parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    root = project_path(config["paths"]["out_root"])
    geometry_root = root / "geometries"
    output_root = root / "voxel_resolution_convergence"
    output_root.mkdir(parents=True, exist_ok=True)
    assert_material_inside_design_space(CARBON_FIBER_290_EPOXY)

    runtime = common.configure_runtime(
        geometry_backend=str(config["geometry_generation"].get("geometry_backend", "numba")),
        generator_cores=common.resolve_cpu_workers(
            config["geometry_generation"].get("generator_cores", "auto")
        ),
        solver_tol=float(common.SOLVER_PROFILES[args.profile]["solver_rtol"]),
        fft_backend=str(args.fft_backend),
        load_batch_size=1,
    )
    records: list[dict[str, Any]] = []
    tensors: dict[tuple[int, float], np.ndarray] = {}
    resolutions = sorted(set(float(value) for value in args.resolutions))
    if 6.0 not in resolutions:
        raise ValueError("The convergence study requires 6 voxels/um as reference.")

    for geometry_id in [int(value) for value in args.geometry_ids]:
        source_dir = geometry_root / f"geometry_{geometry_id:02d}"
        generation = json.loads(
            (source_dir / "generation_result.json").read_text(encoding="utf-8")
        )
        design = dict(generation["design"])
        master = source_dir / "continuous_fibers_master.csv"
        if not master.is_file():
            raise FileNotFoundError(f"Missing continuous master: {master}")
        for resolution in resolutions:
            tag = f"r{resolution:g}".replace(".", "p")
            case_dir = output_root / f"geometry_{geometry_id:02d}" / tag
            raster_dir = case_dir / "geometry"
            solve_dir = case_dir / "fom"
            if args.overwrite:
                for path in (solve_dir / "solve_record.json", solve_dir / "Ceff.npy"):
                    path.unlink(missing_ok=True)
            raster = rasterize_continuous_fibers(
                master,
                caja_um=float(design["box_um"]),
                resolution=resolution,
                output_dir=raster_dir,
            )
            phase = np.asarray(raster["phase"], dtype=np.uint8)
            ori = np.asarray(raster["ori"], dtype=np.float32)
            metadata = dict(raster["metadata"])
            case_design = dict(design)
            case_design.update(
                {
                    "grid_size": int(metadata["grid_size"]),
                    "nvox": int(metadata["grid_size"]),
                    "resolution_vox_per_um": resolution,
                    "res": resolution,
                    "df_voxel": resolution,
                    "Vf": float(metadata["voxel_volume_fraction"]),
                }
            )
            manifest = {
                "study": "voxel-resolution convergence",
                "source_geometry_id": geometry_id,
                "source_continuous_master": str(master),
                "phase_sha256": sha256_array(phase),
                "ori_sha256": sha256_array(ori),
                "design": case_design,
                **metadata,
            }
            common.write_json(raster_dir / "geometry_manifest.json", manifest)
            geometry = common.GeometryData(
                source_run_dir=source_dir,
                geometry_dir=raster_dir,
                design_row=case_design,
                manifest=manifest,
                phase=phase,
                ori=ori,
            )
            print(
                f"[VOXEL] G{geometry_id:02d} | {resolution:g} vox/um | "
                f"shape={phase.shape} | Nvox={phase.size:,}",
                flush=True,
            )
            solved = common.solve_material(
                material_row=CARBON_FIBER_290_EPOXY,
                material_dir=solve_dir,
                geometry=geometry,
                runtime=runtime,
                profile=str(args.profile),
                seed=20260903 + geometry_id,
                save_solution_fields=False,
                persistent_gpu_cache=False,
            )
            ceff = ceff_from_record(solved)
            tensors[(geometry_id, resolution)] = ceff
            A2 = np.asarray(metadata["A2_voxel"], dtype=np.float64)
            record: dict[str, Any] = {
                "geometry_id": geometry_id,
                "geometry_label": str(design["geometry_label"]),
                "aspect_ratio": float(design["AR"]),
                "Vf_target": float(design["Vf_target"]),
                "orientation_target": "random_3D",
                "resolution_vox_per_um": resolution,
                "mesh": "x".join(str(int(value)) for value in phase.shape),
                "Nvox": int(phase.size),
                "Vf_realized": float(metadata["voxel_volume_fraction"]),
                "A2_11_realized": float(A2[0, 0]),
                "A2_22_realized": float(A2[1, 1]),
                "A2_33_realized": float(A2[2, 2]),
                "A2_12_realized": float(A2[0, 1]),
                "A2_13_realized": float(A2[0, 2]),
                "A2_23_realized": float(A2[1, 2]),
                "solver_profile": str(args.profile),
                "solver_real_dtype": str(solved["solver_real_dtype"]),
                "solver_rtol": float(solved["solver_rtol"]),
                "solver_max_relative_residual": float(
                    solved["solver_max_relative_residual"]
                ),
                "solve_wall_s": float(solved["solve_wall_s"]),
                "phase_sha256": manifest["phase_sha256"],
                "ori_sha256": manifest["ori_sha256"],
            }
            for name in sweep.ENGINEERING_COLUMNS:
                record[name] = float(solved[name])
            for ii in range(6):
                for jj in range(6):
                    record[f"Ceff_{ii + 1}{jj + 1}"] = float(ceff[ii, jj])
            records.append(record)

    table = pd.DataFrame(records)
    table["relative_Ceff_error_vs_r6"] = np.nan
    for geometry_id in [int(value) for value in args.geometry_ids]:
        reference = tensors[(geometry_id, 6.0)]
        reference_norm = max(np.linalg.norm(reference, ord="fro"), np.finfo(float).eps)
        mask = table["geometry_id"] == geometry_id
        for index in table.index[mask]:
            resolution = float(table.at[index, "resolution_vox_per_um"])
            table.at[index, "relative_Ceff_error_vs_r6"] = float(
                np.linalg.norm(tensors[(geometry_id, resolution)] - reference, ord="fro")
                / reference_norm
            )
    table = table.sort_values(["geometry_id", "resolution_vox_per_um"]).reset_index(drop=True)
    table.to_csv(output_root / "voxel_resolution_convergence.csv", index=False)
    write_convergence_figure(table, args.paper_figure_dir.resolve())
    summary = {
        "study_name": "voxel-resolution convergence",
        "geometry_ids": [int(value) for value in args.geometry_ids],
        "resolutions_vox_per_um": resolutions,
        "reference_resolution_vox_per_um": 6.0,
        "material": CARBON_FIBER_290_EPOXY,
        "profile": str(args.profile),
        "fft_backend": str(args.fft_backend),
        "same_continuous_master_at_all_resolutions": True,
        "results_csv": str(output_root / "voxel_resolution_convergence.csv"),
    }
    common.write_json(output_root / "voxel_resolution_convergence.json", summary)
    print(table[[
        "geometry_id", "resolution_vox_per_um", "mesh", "Nvox",
        "Vf_realized", "relative_Ceff_error_vs_r6",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
