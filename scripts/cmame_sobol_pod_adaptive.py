#!/usr/bin/env python3
"""Sobol+POD full-rank baseline with adaptive convergence.

Default mode: iteratively add Sobol materials one by one (GPU sequential
to avoid VRAM saturation), build full-rank POD each time, evaluate against
held-out truth, and stop when the maximum relative tensor error drops below
the target tolerance.  The number of materials is geometry-dependent.

Fixed-budget mode is still available via --budgets for reproducibility.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cmame_campaign_common as common
import rom_pod_basis as pod
import rom_reduced_operator as reduced


CAMPAIGN_DEFAULT = ROOT / "results" / "cmame_method" / "causal_comparator"
SOURCE_RUN_DEFAULT = (
    ROOT
    / "results"
    / "fixed_geometry_ffthompy"
    / "fixed_geometry_ar15_vf20_sobol8_center_fields"
)
OUT_DEFAULT = ROOT / "results" / "cmame_method" / "sobol_pod_time_match"
GO_SCT_OFFLINE_S_DEFAULT = 250.71090541232843
DEFAULT_TOLERANCE = 2.0e-5
DEFAULT_MAX_MATERIALS = 60
DEFAULT_MIN_MATERIALS = 6


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_record(campaign_dir: Path, candidate_id: int) -> dict[str, Any]:
    material_dir = common._snapshot_dir(campaign_dir, int(candidate_id))
    if not common._cached_record_is_valid(
        material_dir, profile="snapshot", require_fields=True
    ):
        raise RuntimeError(f"Invalid snapshot cache for Sobol candidate {candidate_id}.")
    record = _read_json(material_dir / "solve_record.json")
    if not bool(record.get("solver_all_converged", False)):
        raise RuntimeError(f"Sobol candidate {candidate_id} did not converge.")
    return record


def _ensure_single_snapshot(
    *,
    candidate_id: int,
    candidates: pd.DataFrame,
    campaign_dir: Path,
    source_run: Path,
    venv_path: Path,
    geometry_backend: str,
    generator_cores: int,
    seed: int,
    _runtime_cache: dict[str, Any],
) -> dict[str, Any]:
    """Solve one snapshot sequentially on GPU to avoid VRAM saturation."""
    if common._cached_record_is_valid(
        common._snapshot_dir(campaign_dir, candidate_id),
        profile="snapshot",
        require_fields=True,
    ):
        return _snapshot_record(campaign_dir, candidate_id)

    # Lazy-init runtime (GPU) only when needed
    if "runtime" not in _runtime_cache:
        common.prepare_runtime(
            venv_path,
            Path(__file__),
            marker_name="CMAME_SOBOL_POD_TIME_MATCH_CUDA_READY",
        )
        _runtime_cache["runtime"] = common.configure_runtime(
            geometry_backend=str(geometry_backend),
            generator_cores=int(generator_cores),
        )
        _runtime_cache["geometry"] = common.load_fixed_geometry(source_run)

    record = common._ensure_snapshot(
        candidate_id=int(candidate_id),
        candidates=candidates,
        out_dir=campaign_dir,
        geometry=_runtime_cache["geometry"],
        runtime=_runtime_cache["runtime"],
        seed=int(seed),
        persistent_gpu_cache=False,  # Free GPU after each solve
    )
    gc.collect()
    return record


def _evaluate_budget(
    *,
    candidate_ids: list[int],
    campaign_dir: Path,
    truth: pd.DataFrame,
    geometry: common.GeometryData,
    rank_cap: int | None,
) -> dict[str, Any]:
    """Build POD from given candidates, assemble reduced ops, evaluate."""
    fields: list[np.ndarray] = []
    load_started = time.perf_counter()
    for candidate_id in candidate_ids:
        fields.extend(
            common.load_snapshot_fields(
                common._snapshot_dir(campaign_dir, candidate_id),
                dtype=np.float32,
            )
        )
    snapshot_shape = tuple(int(value) for value in np.asarray(fields[0]).shape)
    matrix_started = time.perf_counter()
    snapshot_matrix = np.stack(
        [np.asarray(field, dtype=np.float32).reshape(-1) for field in fields], axis=0
    )
    matrix_wall_s = float(time.perf_counter() - matrix_started)
    fields_load_wall_s = float(matrix_started - load_started)
    basis, _, _, _, pod_timings = pod._pod_basis_from_snapshot_matrix(
        snapshot_matrix,
        snapshot_shape,
        eig_tol=1.0e-12,
    )
    del snapshot_matrix, fields
    gc.collect()
    full_rank = len(basis)
    active_rank = full_rank if rank_cap is None else min(int(rank_cap), full_rank)
    basis = basis[:active_rank]

    assembly_started = time.perf_counter()
    Kq, Bq, Dq, assembly = reduced._assemble_reduced_operators(
        phase=geometry.phase,
        ori=geometry.ori.astype(np.float64),
        basis=basis,
    )
    assembly_wall_s = float(time.perf_counter() - assembly_started)
    validation = reduced._evaluate_rom(results_df=truth, Kq=Kq, Bq=Bq, Dq=Dq)
    validation.insert(0, "validation_id", truth["validation_id"].to_numpy(dtype=int))
    stats = common._error_stats(validation)
    records = [_snapshot_record(campaign_dir, candidate_id) for candidate_id in candidate_ids]
    solve_wall_s = float(sum(float(record["solve_wall_s"]) for record in records))
    pod_wall_s = float(sum(float(value) for value in pod_timings.values()))
    offline_wall_s = solve_wall_s + pod_wall_s + assembly_wall_s
    record = {
        "material_budget": int(len(candidate_ids)),
        "full_order_load_budget": int(6 * len(candidate_ids)),
        "basis_rank": int(active_rank),
        "full_pod_rank": int(full_rank),
        "rank_cap": None if rank_cap is None else int(rank_cap),
        "offline_wall_s": offline_wall_s,
        "snapshot_solve_wall_s": solve_wall_s,
        "pod_basis_wall_s": pod_wall_s,
        "reduced_assembly_wall_s": assembly_wall_s,
        "snapshot_field_load_wall_s": fields_load_wall_s,
        "snapshot_matrix_wall_s": matrix_wall_s,
        "operator_memory_bytes": int(Kq.nbytes + Bq.nbytes + Dq.nbytes),
        "basis_memory_bytes": int(sum(np.asarray(field).nbytes for field in basis)),
        "operator_assembly_reported_s": float(assembly["assembly_wall_s"]),
        **stats,
    }
    del validation, Kq, Bq, Dq, basis
    gc.collect()
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=CAMPAIGN_DEFAULT)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument(
        "--budgets", nargs="*", type=int, default=None,
        help="Fixed budget list (legacy mode). If omitted, runs adaptive convergence.",
    )
    parser.add_argument("--rank-cap", type=int, default=0)
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help="Target max relative Frobenius error for adaptive convergence.",
    )
    parser.add_argument(
        "--max-materials", type=int, default=DEFAULT_MAX_MATERIALS,
        help="Maximum number of Sobol materials before stopping.",
    )
    parser.add_argument(
        "--min-materials", type=int, default=DEFAULT_MIN_MATERIALS,
        help="Minimum number of materials before checking convergence.",
    )
    parser.add_argument("--target-offline-s", type=float, default=GO_SCT_OFFLINE_S_DEFAULT)
    parser.add_argument("--geometry-backend", choices=("numba", "cupy", "auto"), default="numba")
    parser.add_argument("--generator-cores", type=int, default=2)
    parser.add_argument("--blas-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--venv-path", type=Path, default=common.DEFAULT_VENV_PATH)
    return parser.parse_args()


def _run_fixed_budgets(args: argparse.Namespace) -> int:
    """Original fixed-budget mode for reproducibility."""
    budgets = sorted(set(int(value) for value in args.budgets))
    if not budgets or budgets[0] < 1:
        raise ValueError("--budgets must contain positive integers.")
    campaign_dir = args.campaign_dir.resolve()
    source_run = args.source_run.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    common._verify_frozen_design(campaign_dir)
    candidates = common._candidate_table(campaign_dir, None)
    if budgets[-1] > len(candidates):
        raise ValueError("The requested Sobol budget exceeds the candidate design.")
    if any(column.startswith("Ceff_") for column in candidates.columns):
        raise RuntimeError("Candidate table exposes FOM responses; aborting causal control.")
    candidate_ids = candidates["candidate_id"].iloc[: budgets[-1]].astype(int).tolist()

    # Solve snapshots one by one (GPU sequential)
    runtime_cache: dict[str, Any] = {}
    for idx, cid in enumerate(candidate_ids, start=1):
        record = _ensure_single_snapshot(
            candidate_id=cid,
            candidates=candidates,
            campaign_dir=campaign_dir,
            source_run=source_run,
            venv_path=args.venv_path,
            geometry_backend=str(args.geometry_backend),
            generator_cores=int(args.generator_cores),
            seed=int(args.seed),
            _runtime_cache=runtime_cache,
        )
        print(
            f"[SOBOL-POD] snapshot {idx}/{len(candidate_ids)} "
            f"candidate={cid} solve={float(record['solve_wall_s']):.3f}s",
            flush=True,
        )

    truth = pd.read_csv(campaign_dir / "held_out_truth_results.csv")
    geometry = runtime_cache.get("geometry") or common.load_fixed_geometry(source_run)
    rank_cap = None if int(args.rank_cap) <= 0 else int(args.rank_cap)
    controller, blas_info = common.configure_blas_threads(int(args.blas_threads))
    curve_path = out_dir / "sobol_pod_time_match_curve.csv"

    rows: list[dict[str, Any]] = []
    for budget in budgets:
        row = _evaluate_budget(
            candidate_ids=candidate_ids[:budget],
            campaign_dir=campaign_dir,
            truth=truth,
            geometry=geometry,
            rank_cap=rank_cap,
        )
        row["target_offline_s"] = float(args.target_offline_s)
        row["within_go_sct_time"] = bool(row["offline_wall_s"] <= float(args.target_offline_s))
        rows.append(row)
        pd.DataFrame(rows).sort_values("material_budget").to_csv(curve_path, index=False)
        print(
            f"[SOBOL-POD] m={budget:02d} r={row['basis_rank']} "
            f"offline={row['offline_wall_s']:.2f}s mean={row['error_mean']:.3e} "
            f"p95={row['error_p95']:.3e} max={row['error_max']:.3e}",
            flush=True,
        )

    _write_manifest(out_dir, rows, args, rank_cap, truth, blas_info)
    del controller
    return 0


def _run_adaptive(args: argparse.Namespace) -> int:
    """Adaptive convergence mode: add Sobol materials until tolerance is met."""
    campaign_dir = args.campaign_dir.resolve()
    source_run = args.source_run.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    common._verify_frozen_design(campaign_dir)
    candidates = common._candidate_table(campaign_dir, None)
    if any(column.startswith("Ceff_") for column in candidates.columns):
        raise RuntimeError("Candidate table exposes FOM responses; aborting causal control.")

    all_candidate_ids = candidates["candidate_id"].astype(int).tolist()
    max_materials = min(int(args.max_materials), len(all_candidate_ids))
    min_materials = max(1, min(int(args.min_materials), max_materials))
    tolerance = float(args.tolerance)
    rank_cap = None if int(args.rank_cap) <= 0 else int(args.rank_cap)

    truth = pd.read_csv(campaign_dir / "held_out_truth_results.csv")
    controller, blas_info = common.configure_blas_threads(int(args.blas_threads))

    runtime_cache: dict[str, Any] = {}
    curve_path = out_dir / "sobol_pod_adaptive_curve.csv"
    rows: list[dict[str, Any]] = []
    converged_at: int | None = None

    print(
        f"[SOBOL-POD-ADAPTIVE] tol={tolerance:.2e} max_materials={max_materials} "
        f"min_materials={min_materials}",
        flush=True,
    )

    for n_materials in range(1, max_materials + 1):
        cid = all_candidate_ids[n_materials - 1]

        # Solve one snapshot on GPU (sequential to avoid OOM)
        record = _ensure_single_snapshot(
            candidate_id=cid,
            candidates=candidates,
            campaign_dir=campaign_dir,
            source_run=source_run,
            venv_path=args.venv_path,
            geometry_backend=str(args.geometry_backend),
            generator_cores=int(args.generator_cores),
            seed=int(args.seed),
            _runtime_cache=runtime_cache,
        )
        print(
            f"[SOBOL-POD-ADAPTIVE] snapshot {n_materials}/{max_materials} "
            f"candidate={cid} solve={float(record['solve_wall_s']):.3f}s",
            flush=True,
        )

        # Only evaluate POD+ROM at meaningful checkpoints to save time
        if n_materials < min_materials:
            continue

        # Evaluate full-rank POD at this budget
        geometry = runtime_cache.get("geometry") or common.load_fixed_geometry(source_run)
        current_ids = all_candidate_ids[:n_materials]
        row = _evaluate_budget(
            candidate_ids=current_ids,
            campaign_dir=campaign_dir,
            truth=truth,
            geometry=geometry,
            rank_cap=rank_cap,
        )
        row["target_offline_s"] = float(args.target_offline_s)
        row["tolerance"] = tolerance
        row["within_go_sct_time"] = bool(row["offline_wall_s"] <= float(args.target_offline_s))
        row["converged"] = bool(row["error_max"] <= tolerance)
        rows.append(row)

        pd.DataFrame(rows).sort_values("material_budget").to_csv(curve_path, index=False)
        status = "CONVERGED" if row["converged"] else "iterating"
        print(
            f"[SOBOL-POD-ADAPTIVE] m={n_materials:02d} r={row['basis_rank']} "
            f"offline={row['offline_wall_s']:.2f}s mean={row['error_mean']:.3e} "
            f"p95={row['error_p95']:.3e} max={row['error_max']:.3e} [{status}]",
            flush=True,
        )

        if row["converged"]:
            converged_at = n_materials
            break

    # Write manifest
    curve = pd.DataFrame(rows).sort_values("material_budget").reset_index(drop=True)
    curve.to_csv(curve_path, index=False)

    best = curve.iloc[-1] if not curve.empty else None
    common.write_json(
        out_dir / "experiment_manifest.json",
        {
            "status": "converged" if converged_at else "max_budget_reached",
            "method": "Sobol+POD adaptive full-rank convergence",
            "mode": "adaptive",
            "selection_rule": "first Sobol points in frozen causal candidate design",
            "selection_uses_candidate_fom_errors": False,
            "tolerance": tolerance,
            "max_materials": max_materials,
            "min_materials": min_materials,
            "converged_at_materials": converged_at,
            "rank_cap": rank_cap,
            "target_offline_s": float(args.target_offline_s),
            "final_budget": None if best is None else int(best["material_budget"]),
            "final_offline_s": None if best is None else float(best["offline_wall_s"]),
            "final_errors": None if best is None else {
                "mean": float(best["error_mean"]),
                "p95": float(best["error_p95"]),
                "max": float(best["error_max"]),
            },
            "validation_truth_count": int(len(truth)),
            "solver_profile": "snapshot",
            "blas_threads": int(args.blas_threads),
            "blas_runtimes": blas_info,
            "files": {"curve": str(curve_path)},
        },
    )
    del controller
    return 0


def _write_manifest(
    out_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    rank_cap: int | None,
    truth: pd.DataFrame,
    blas_info: list[dict[str, Any]],
) -> None:
    """Write manifest for fixed-budget mode."""
    curve = pd.DataFrame(rows).sort_values("material_budget").reset_index(drop=True)
    closest = curve.iloc[
        int(np.argmin(np.abs(curve["offline_wall_s"] - float(args.target_offline_s))))
    ]
    feasible = curve.loc[curve["offline_wall_s"] <= float(args.target_offline_s)]
    strict = None if feasible.empty else feasible.iloc[-1]
    common.write_json(
        out_dir / "experiment_manifest.json",
        {
            "status": "complete",
            "method": "Sobol+POD time-matched control",
            "mode": "fixed_budgets",
            "selection_rule": "first Sobol points in frozen causal candidate design",
            "selection_uses_candidate_fom_errors": False,
            "budgets": sorted(set(int(value) for value in args.budgets)),
            "rank_cap": rank_cap,
            "target_offline_s": float(args.target_offline_s),
            "closest_budget": int(closest["material_budget"]),
            "closest_offline_s": float(closest["offline_wall_s"]),
            "closest_errors": {
                "mean": float(closest["error_mean"]),
                "p95": float(closest["error_p95"]),
                "max": float(closest["error_max"]),
            },
            "strict_time_matched_budget": None if strict is None else int(strict["material_budget"]),
            "strict_time_matched_offline_s": None if strict is None else float(strict["offline_wall_s"]),
            "strict_time_matched_errors": None
            if strict is None
            else {
                "mean": float(strict["error_mean"]),
                "p95": float(strict["error_p95"]),
                "max": float(strict["error_max"]),
            },
            "validation_truth_count": int(len(truth)),
            "solver_profile": "snapshot",
            "blas_threads": int(args.blas_threads),
            "blas_runtimes": blas_info,
            "files": {"curve": str(out_dir / "sobol_pod_time_match_curve.csv")},
        },
    )


def main() -> int:
    args = parse_args()
    if args.budgets is not None:
        return _run_fixed_budgets(args)
    return _run_adaptive(args)


if __name__ == "__main__":
    raise SystemExit(main())
