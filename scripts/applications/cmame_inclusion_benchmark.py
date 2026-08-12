#!/usr/bin/env python3
"""Refined spherical-inclusion FFT benchmark with classical isotropic bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "FFT") not in sys.path:
    sys.path.insert(0, str(ROOT / "FFT"))

from pipeline.fft_solver import solve_homogenization


OUT_DEFAULT = ROOT / "results" / "cmame_method" / "inclusion_benchmark"
MATERIAL = {
    "Em": 4.0,
    "nu_m": 0.30,
    "Ef_L": 20.0,
    "Ef_T": 20.0,
    "G_LT": 8.0,
    "nu_LT": 0.25,
    "nu_TT": 0.25,
}


def isotropic_moduli(E: float, nu: float) -> tuple[float, float]:
    return E / (3.0 * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def hashin_shtrikman_bounds(
    K1: float,
    G1: float,
    K2: float,
    G2: float,
    volume_fraction_2: float,
) -> tuple[float, float, float, float]:
    if not (K2 > K1 > 0.0 and G2 > G1 > 0.0):
        raise ValueError("Phase 2 must be strictly stiffer than phase 1.")
    vf = float(volume_fraction_2)
    vm = 1.0 - vf
    zeta1 = G1 * (9.0 * K1 + 8.0 * G1) / (6.0 * (K1 + 2.0 * G1))
    zeta2 = G2 * (9.0 * K2 + 8.0 * G2) / (6.0 * (K2 + 2.0 * G2))
    K_lower = K1 + vf / (1.0 / (K2 - K1) + vm / (K1 + 4.0 * G1 / 3.0))
    K_upper = K2 + vm / (1.0 / (K1 - K2) + vf / (K2 + 4.0 * G2 / 3.0))
    G_lower = G1 + vf / (1.0 / (G2 - G1) + vm / (G1 + zeta1))
    G_upper = G2 + vm / (1.0 / (G1 - G2) + vf / (G2 + zeta2))
    return K_lower, K_upper, G_lower, G_upper


def isotropic_mandel(K: float, G: float) -> np.ndarray:
    lam = K - 2.0 * G / 3.0
    matrix = np.zeros((6, 6), dtype=np.float64)
    matrix[:3, :3] = lam
    matrix[np.arange(3), np.arange(3)] += 2.0 * G
    matrix[3:, 3:] = 2.0 * G * np.eye(3)
    return matrix


def sphere_geometry(grid: int, target_vf: float) -> tuple[np.ndarray, np.ndarray]:
    coordinates = (np.arange(grid, dtype=np.float64) + 0.5) / grid - 0.5
    xx, yy, zz = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    radius = (3.0 * float(target_vf) / (4.0 * np.pi)) ** (1.0 / 3.0)
    phase = ((xx * xx + yy * yy + zz * zz) <= radius * radius).astype(np.uint8)
    orientation = np.zeros((grid, grid, grid, 3), dtype=np.float32)
    orientation[..., 0] = 1.0
    return phase, orientation


def _parameters(phase: np.ndarray, orientation: np.ndarray, out_dir: Path) -> dict:
    return {
        **MATERIAL,
        "input_dir": str(out_dir),
        "seed": 20260816,
        "phase_array": phase,
        "ori_array": orientation,
        "fft_backend": "cupy",
        "solver_profile": "truth",
        "solver_maxiter": 2000,
        "require_convergence": True,
        "solver_fft_form": "r",
        "cfield_storage": "sym21",
        "cfield_indexed": False,
        "projection_storage": "full",
        "projection_backend": "cupy",
        "postprocess_assembly": "scalar",
        "load_batch_size": 1,
        "solver_verbose": False,
        "solver_timing_path": str(out_dir / "solver_timing.json"),
        "free_gpu_memory_after_solve": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--grids", type=int, nargs="*", default=[32, 48, 64, 91])
    parser.add_argument("--volume-fraction", type=float, default=0.10)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    Km, Gm = isotropic_moduli(MATERIAL["Em"], MATERIAL["nu_m"])
    Kf, Gf = isotropic_moduli(MATERIAL["Ef_L"], MATERIAL["nu_LT"])
    rows = []
    tensors = {}
    for grid in args.grids:
        grid_dir = out_dir / f"N{int(grid):03d}"
        grid_dir.mkdir(parents=True, exist_ok=True)
        phase, orientation = sphere_geometry(int(grid), float(args.volume_fraction))
        vf = float(np.mean(phase))
        lower_K, upper_K, lower_G, upper_G = hashin_shtrikman_bounds(Km, Gm, Kf, Gf, vf)
        ceff_path = grid_dir / "Ceff_truth.npy"
        if args.reuse_existing and ceff_path.is_file():
            Ceff = np.asarray(np.load(ceff_path), dtype=float)
            wall = float("nan")
        else:
            started = time.perf_counter()
            Ceff = np.asarray(solve_homogenization(_parameters(phase, orientation, grid_dir)), dtype=float)
            wall = float(time.perf_counter() - started)
        Ceff = 0.5 * (Ceff + Ceff.T)
        tensors[int(grid)] = Ceff
        hydro = np.zeros(6)
        hydro[:3] = 1.0
        effective_K = float(hydro @ Ceff @ hydro / 9.0)
        effective_G = float(np.trace(Ceff[3:, 3:]) / 6.0)
        C_voigt = (1.0 - vf) * isotropic_mandel(Km, Gm) + vf * isotropic_mandel(Kf, Gf)
        C_reuss = np.linalg.inv(
            (1.0 - vf) * np.linalg.inv(isotropic_mandel(Km, Gm))
            + vf * np.linalg.inv(isotropic_mandel(Kf, Gf))
        )
        timing = json.loads((grid_dir / "solver_timing.json").read_text())
        load = timing["load_solver_summary"]
        rows.append(
            {
                "grid": int(grid),
                "voxel_volume_fraction": vf,
                "solve_wall_s": wall,
                "effective_bulk": effective_K,
                "effective_shear": effective_G,
                "hs_bulk_lower": lower_K,
                "hs_bulk_upper": upper_K,
                "hs_shear_lower": lower_G,
                "hs_shear_upper": upper_G,
                "bulk_within_hs": bool(lower_K <= effective_K <= upper_K),
                "shear_within_hs": bool(lower_G <= effective_G <= upper_G),
                "min_eig_Ceff_minus_Reuss": float(np.min(np.linalg.eigvalsh(Ceff - C_reuss))),
                "min_eig_Voigt_minus_Ceff": float(np.min(np.linalg.eigvalsh(C_voigt - Ceff))),
                "all_converged": bool(load["all_converged"]),
                "max_relative_residual": float(load["final_norm_res_rel_max"]),
            }
        )
        np.save(ceff_path, Ceff)
        print(f"[INCLUSION] N={grid} | Vf={vf:.5f} | K={effective_K:.6f} | G={effective_G:.6f}", flush=True)
    frame = pd.DataFrame(rows).sort_values("grid")
    finest = tensors[int(max(args.grids))]
    frame["relative_error_vs_finest"] = [
        float(np.linalg.norm(tensors[int(grid)] - finest) / np.linalg.norm(finest))
        for grid in frame["grid"]
    ]
    frame.to_csv(out_dir / "inclusion_refinement.csv", index=False)
    if not frame[["bulk_within_hs", "all_converged"]].all().all():
        raise RuntimeError("The inclusion benchmark failed convergence or the HS bulk bounds.")
    tolerance = 1.0e-9
    if frame["min_eig_Ceff_minus_Reuss"].min() < -tolerance or frame["min_eig_Voigt_minus_Ceff"].min() < -tolerance:
        raise RuntimeError("The inclusion benchmark violated Voigt/Reuss Loewner bounds.")
    manifest = {
        "status": "complete",
        "profile": "truth/float64/rtol=1e-10",
        "grids": [int(value) for value in args.grids],
        "target_volume_fraction": float(args.volume_fraction),
        "all_converged": True,
        "all_hashin_shtrikman_bulk_bounds_pass": True,
        "hashin_shtrikman_shear_is_diagnostic_only": (
            "The periodic sphere array has cubic rather than isotropic effective symmetry."
        ),
        "all_voigt_reuss_bounds_pass": True,
    }
    (out_dir / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
