#!/usr/bin/env python3
"""Same-master voxel refinement at 48^3, 64^3, 91^3, and 128^3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "FFT") not in sys.path:
    sys.path.insert(0, str(ROOT / "FFT"))


def _prepare_cuda_runtime() -> None:
    venv = Path(sys.prefix)
    cuda_dirs = sorted(
        str(path)
        for path in (venv / "lib").glob("python*/site-packages/nvidia/*/lib")
        if path.is_dir()
    )
    if not cuda_dirs:
        return
    marker = ":".join(cuda_dirs)
    if os.environ.get("CMAME_CUDA_RUNTIME_READY") == marker:
        return
    env = os.environ.copy()
    current = [value for value in env.get("LD_LIBRARY_PATH", "").split(":") if value]
    env["LD_LIBRARY_PATH"] = ":".join(
        [*cuda_dirs, *[value for value in current if value not in cuda_dirs]]
    )
    env["CMAME_CUDA_RUNTIME_READY"] = marker
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)


_prepare_cuda_runtime()

from pipeline.fft_solver import solve_homogenization
from pipeline.rve_generator import rasterize_continuous_fibers


OUT_DEFAULT = ROOT / "results" / "cmame_method" / "voxel_scaling"
MASTER_DEFAULT = (
    ROOT
    / "results"
    / "cmame_method"
    / "geometries"
    / "geometry_00"
    / "continuous_fibers_master.csv"
)
BOX_UM = 18.059971
MATERIAL_CENTER = {
    "Em": 2.75,
    "nu_m": 0.385,
    "Ef_L": 233.5,
    "Ef_T": 14.5,
    "G_LT": 19.0,
    "nu_LT": 0.23,
    "nu_TT": 0.375,
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _solver_parameters(
    phase: np.ndarray,
    ori: np.ndarray,
    *,
    output_dir: Path,
    profile: str,
) -> dict:
    return {
        **MATERIAL_CENTER,
        "input_dir": str(output_dir),
        "seed": 20260811,
        "phase_array": phase,
        "ori_array": ori,
        "fft_backend": "cupy",
        "solver_profile": profile,
        "solver_maxiter": 2000,
        "require_convergence": True,
        "solver_fft_form": "r",
        "cfield_storage": "sym21",
        "cfield_indexed": profile == "timing",
        "projection_storage": "direct" if profile == "timing" else "full",
        "projection_backend": "numpy" if profile == "timing" else "cupy",
        "postprocess_assembly": "scalar",
        "load_batch_size": 1,
        "solver_verbose": False,
        "solver_timing_path": str(output_dir / "solver_timing.json"),
        "free_gpu_memory_after_solve": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=MASTER_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--grids", type=int, nargs="*", default=[48, 64, 91, 128])
    parser.add_argument("--profile", choices=("truth", "snapshot", "timing"), default="truth")
    parser.add_argument("--raster-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    master = args.master.resolve()
    if not master.is_file():
        raise FileNotFoundError(master)
    records = []
    for grid in sorted(set(int(value) for value in args.grids)):
        grid_dir = out_dir / f"N{grid:03d}"
        grid_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = grid_dir / "raster_manifest.json"
        if args.overwrite or not (grid_dir / "phase.npy").is_file():
            raster = rasterize_continuous_fibers(
                master,
                caja_um=BOX_UM,
                resolution=grid / BOX_UM,
                output_dir=grid_dir,
            )
            phase = raster["phase"]
            ori = raster["ori"]
            metadata = raster["metadata"]
        else:
            phase = np.load(grid_dir / "phase.npy")
            ori = np.load(grid_dir / "ori.npy")
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = {"grid_size": grid, **metadata}
        ceff_path = grid_dir / f"Ceff_{args.profile}.npy"
        if not args.raster_only and (args.overwrite or not ceff_path.is_file()):
            started = time.perf_counter()
            ceff = solve_homogenization(
                _solver_parameters(
                    phase,
                    ori,
                    output_dir=grid_dir,
                    profile=str(args.profile),
                )
            )
            np.save(ceff_path, ceff)
            record["solve_wall_s"] = float(time.perf_counter() - started)
        if ceff_path.is_file():
            ceff = np.load(ceff_path)
            record["ceff_path"] = str(ceff_path)
            record["ceff_norm"] = float(np.linalg.norm(ceff))
            record["ceff_min_eig"] = float(np.linalg.eigvalsh(ceff).min())
            for ii in range(6):
                for jj in range(6):
                    record[f"Ceff_{ii + 1}{jj + 1}"] = float(ceff[ii, jj])
        records.append(record)

    table = pd.DataFrame(records).sort_values("grid_size")
    ceff_records = [record for record in records if "ceff_path" in record]
    if ceff_records:
        reference_record = max(ceff_records, key=lambda item: item["grid_size"])
        reference = np.load(reference_record["ceff_path"])
        for record in records:
            if "ceff_path" in record:
                ceff = np.load(record["ceff_path"])
                record["relative_discretization_error_vs_finest"] = float(
                    np.linalg.norm(ceff - reference) / np.linalg.norm(reference)
                )
        table = pd.DataFrame(records).sort_values("grid_size")
    table.to_csv(out_dir / "voxel_scaling.csv", index=False)
    _write_json(
        out_dir / "campaign_manifest.json",
        {
            "master_geometry": str(master),
            "all_fibers_required": True,
            "grids": [int(value) for value in table["grid_size"]],
            "profile": str(args.profile),
            "raster_only": bool(args.raster_only),
            "complete_solve_count": int(table.get("ceff_path", pd.Series(dtype=str)).notna().sum()),
        },
    )
    print(f"[VOXEL] listo | grids={list(table['grid_size'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
