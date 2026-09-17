#!/usr/bin/env python3
"""Run one real residual enrichment step from a persisted anchor campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cmame_campaign_common as common
import rom_reduced_operator as reduced
from residual_one_mode_fields import residual_stresses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_dir, geometry_dir = args.run_dir.resolve(), args.geometry_dir.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    geometry = common.load_fixed_geometry(geometry_dir)
    phase, ori = geometry.phase, geometry.ori
    order = reduced.phase_orientation_voxel_order(phase, ori)
    affine = reduced.affine_stress_batch_factory(
        phase.reshape(-1)[order], ori.reshape(-1, 3)[order]
    )
    basis_paths = sorted((run_dir / "basis_fields").glob("basis_*.npy"))
    V = np.stack([np.load(path, mmap_mode="r").reshape(6, -1) for path in basis_paths])
    with np.load(run_dir / "reduced_operators.npz") as payload:
        existing = {name: np.asarray(payload[name]) for name in payload.files if name in {
            "Kq", "Bq", "Dq", "raw_Kq", "raw_Bq", "invR", "G"
        }}
    pool = pd.read_csv(run_dir / "final_validation_pool.csv")
    target = pool.iloc[0].to_dict()
    gamma = reduced._material_coefficients(target)
    K = np.einsum("q,qij->ij", gamma, existing["raw_Kq"])
    B = np.einsum("q,qij->ij", gamma, existing["raw_Bq"])
    Y = reduced._solve_spd_reduced(K, B)
    stresses = residual_stresses(affine, V, gamma, Y)

    corrections: list[np.ndarray | None] = [None] * 6
    infos: list[dict] = [{} for _ in range(6)]
    def consume(index: int, field: np.ndarray, meta: dict) -> None:
        # FFT emits physical voxel order; Ritz fields use phase/orientation order.
        corrections[index] = np.asarray(field, dtype=np.float32).reshape(6, -1)[:, order].copy()
        infos[index] = meta

    anchor = json.loads((run_dir / "snapshot_cache/candidate_0000/material.json").read_text())
    runtime = common.configure_runtime(
        geometry_backend="numba", generator_cores=2, solver_tol=1e-5,
        fft_backend="gpu", load_batch_size=1,
    )
    started = time.perf_counter()
    common.solve_material(
        material_row=anchor, material_dir=args.out / "anchor_correction",
        geometry=geometry, runtime=runtime, profile="snapshot", seed=20260821,
        save_solution_fields=False,
        residual_correction_requests=[
            {"residual_stress": np.asarray(stresses[i])[:, np.argsort(order)].reshape((6,) + phase.shape)}
            for i in range(6)
        ],
        residual_correction_consumer=consume,
    )
    Z = np.stack(corrections)  # load, component, voxel
    S = -np.einsum("lcn,mcn->lm", stresses, Z, optimize=True) / phase.size
    S = 0.5 * (S + S.T)
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    direction = eigenvectors[:, -1]
    eta = float(eigenvalues[-1])
    mode = np.einsum("l,lcn->cn", direction, Z, optimize=True)
    mode /= np.sqrt(max(eta, np.finfo(float).eps))
    np.save(args.out / "mode_0006.npy", mode.astype(np.float32))
    result = {
        "geometry_id": int(geometry.manifest.get("geometry_id", -1)),
        "anchor_materials": 1, "rank_before": int(len(V)), "rank_after": int(len(V) + 1),
        "target_material_id": int(target.get("material_id", target.get("final_validation_id", 0))),
        "eta_before": eta, "trace_before": float(np.trace(S)),
        "dominant_load": direction.tolist(),
        "correction_solve_wall_s": float(time.perf_counter() - started),
        "all_corrections_converged": bool(all(x.get("converged", False) for x in infos)),
        "mode_l2_norm": float(np.linalg.norm(mode)),
    }
    (args.out / "one_mode_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
