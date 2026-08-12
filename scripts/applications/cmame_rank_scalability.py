#!/usr/bin/env python3
"""Measure the ROM rank required by voxel grids and realized geometries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for path in (SCRIPTS,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cmame_campaign_common as common
import rom_reduced_operator as reduced


DESIGN_DIR = ROOT / "results" / "cmame_method" / "causal_comparator"
GEOMETRY_DIR = ROOT / "results" / "cmame_method" / "geometries"
VOXEL_DIR = ROOT / "results" / "cmame_method" / "voxel_scaling"
OUT_DEFAULT = ROOT / "results" / "cmame_method" / "rank_scalability"
THRESHOLDS = (1.0e-2, 1.0e-3, 1.0e-4)
DESCRIPTORS = (
    "Vf_realized",
    "aspect_ratio",
    "A2_anisotropy",
    "cluster_fraction_target",
    "interface_density",
    "Ripley_peak",
    "D_star",
    "n_fibers",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _material_dict(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    source = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    return {key: source[key] for key in common.sweep.MATERIAL_COLUMNS if key in source}


def _threshold_tag(threshold: float) -> str:
    exponent = int(round(-math.log10(float(threshold))))
    return f"1e-{exponent}"


def _inner(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(left * right))


def _append_orthonormal(
    basis: list[np.ndarray],
    fields: Iterable[np.ndarray],
    *,
    tolerance: float,
) -> int:
    initial_rank = len(basis)
    appended: list[np.ndarray] = []
    for field in fields:
        vector = np.asarray(field, dtype=np.float64).copy()
        for block in (basis, appended, basis, appended):
            for base in block:
                vector -= _inner(base, vector) * base
        norm = math.sqrt(max(_inner(vector, vector), 0.0))
        if norm > float(tolerance):
            appended.append(vector / norm)
    basis.extend(appended)
    return len(basis) - initial_rank


def _a2_anisotropy(row: pd.Series | dict[str, Any]) -> float:
    values = np.array(
        [row["A2_11_realized"], row["A2_22_realized"], row["A2_33_realized"]],
        dtype=float,
    )
    return float(np.sqrt(1.5) * np.linalg.norm(values - 1.0 / 3.0))


def _geometry_cases(geometry_dir: Path) -> list[dict[str, Any]]:
    descriptors = pd.read_csv(geometry_dir / "geometry_realized_descriptors.csv")
    cases: list[dict[str, Any]] = []
    for _, row in descriptors.sort_values("geometry_id").iterrows():
        geometry_id = int(row["geometry_id"])
        case_dir = geometry_dir / f"geometry_{geometry_id:02d}"
        metadata = row.to_dict()
        metadata["A2_anisotropy"] = _a2_anisotropy(row)
        cases.append(
            {
                "case_kind": "geometry",
                "case_id": f"geometry_{geometry_id:02d}",
                "source_dir": case_dir,
                **metadata,
            }
        )
    return cases


def _voxel_cases(voxel_dir: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case_dir in sorted(voxel_dir.glob("N[0-9][0-9][0-9]")):
        manifest = _read_json(case_dir / "raster_manifest.json")
        a2 = np.asarray(manifest["A2_voxel"], dtype=float)
        cases.append(
            {
                "case_kind": "voxel_grid",
                "case_id": case_dir.name,
                "source_dir": case_dir,
                "grid_size": int(manifest["grid_size"]),
                "Vf_realized": float(manifest["voxel_volume_fraction"]),
                "A2_anisotropy": float(np.sqrt(1.5) * np.linalg.norm(a2 - np.eye(3) / 3.0)),
                "interface_density": float(manifest["interface_density"]),
                "n_fibers": int(manifest["fiber_count"]),
            }
        )
    return cases


def _load_geometry(case: dict[str, Any]) -> common.GeometryData:
    source_dir = Path(case["source_dir"]).resolve()
    phase_path = source_dir / "phase.npy"
    ori_path = source_dir / "ori.npy"
    phase = np.load(phase_path).astype(np.uint8)
    ori = np.load(ori_path).astype(np.float32)
    if tuple(ori.shape[:3]) != tuple(phase.shape) or ori.shape[-1] != 3:
        raise ValueError(f"Geometria incompatible en {source_dir}.")
    manifest = {
        "phase_sha256": _sha256(phase_path),
        "ori_sha256": _sha256(ori_path),
        "grid_shape": list(phase.shape),
        "case_id": str(case["case_id"]),
    }
    return common.GeometryData(
        source_run_dir=source_dir.parent,
        geometry_dir=source_dir,
        design_row={},
        manifest=manifest,
        phase=phase,
        ori=ori,
    )


def _training_sequence(design_dir: Path, max_materials: int) -> tuple[pd.DataFrame, list[int], Path]:
    selection_path = (
        design_dir
        / "methods"
        / "qoi_schur_greedy"
        / "replicate_00"
        / "selection.csv"
    )
    if not selection_path.is_file():
        raise FileNotFoundError(
            "Falta la seleccion QoI/Schur completa. Ejecute primero "
            "scripts/cmame_causal_comparator.py --stage select."
        )
    selection = pd.read_csv(selection_path).sort_values("offline_index")
    if selection["selection_uses_fom_error"].astype(bool).any():
        raise RuntimeError("La secuencia de entrenamiento no es causal.")
    if len(selection) < int(max_materials):
        raise RuntimeError(
            f"La secuencia solo contiene {len(selection)} materiales; "
            f"se solicitaron {max_materials}."
        )
    ids = [int(value) for value in selection["candidate_id"].iloc[:max_materials]]
    candidates = pd.read_csv(design_dir / "candidate_pool.csv")
    if any(column.startswith("Ceff_") for column in candidates.columns):
        raise RuntimeError("El pool de entrenamiento contiene respuestas FOM.")
    indexed = candidates.set_index("candidate_id", drop=False)
    training = indexed.loc[ids].reset_index(drop=True)
    return training, ids, selection_path


def _validation_materials(design_dir: Path, count: int) -> pd.DataFrame:
    validation = pd.read_csv(design_dir / "held_out_validation_pool.csv")
    if len(validation) < int(count):
        raise ValueError(f"Solo existen {len(validation)} validaciones; se solicitaron {count}.")
    return validation.iloc[: int(count)].copy().reset_index(drop=True)


def _solve_snapshots(
    *,
    case_out: Path,
    training: pd.DataFrame,
    geometry: common.GeometryData,
    runtime: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index, row in training.iterrows():
        candidate_id = int(row["candidate_id"])
        material = _material_dict(row)
        material["candidate_id"] = candidate_id
        record = common.solve_material(
            material_row=material,
            material_dir=case_out / "snapshots" / f"candidate_{candidate_id:04d}",
            geometry=geometry,
            runtime=runtime,
            profile="snapshot",
            seed=int(seed) + candidate_id,
            save_solution_fields=True,
        )
        records.append(record)
        pd.DataFrame(records).to_csv(case_out / "snapshot_results.csv", index=False)
        print(
            f"[RANK] {case_out.name} snapshot {index + 1}/{len(training)} "
            f"candidate={candidate_id}",
            flush=True,
        )
    return pd.DataFrame(records)


def _solve_truth(
    *,
    case_out: Path,
    validation: pd.DataFrame,
    geometry: common.GeometryData,
    runtime: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index, row in validation.iterrows():
        validation_id = int(row["validation_id"])
        material = _material_dict(row)
        material["validation_id"] = validation_id
        record = common.solve_material(
            material_row=material,
            material_dir=case_out / "truth" / f"material_{validation_id:04d}",
            geometry=geometry,
            runtime=runtime,
            profile="truth",
            seed=int(seed) + 1_000_000 + validation_id,
            save_solution_fields=False,
        )
        records.append(record)
        pd.DataFrame(records).to_csv(case_out / "truth_results.csv", index=False)
        print(
            f"[RANK] {case_out.name} truth {index + 1}/{len(validation)}",
            flush=True,
        )
    return pd.DataFrame(records)


def _build_basis(
    *,
    case_out: Path,
    candidate_ids: list[int],
    tolerance: float,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    basis: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for offline_index, candidate_id in enumerate(candidate_ids, start=1):
        material_dir = case_out / "snapshots" / f"candidate_{candidate_id:04d}"
        started = time.perf_counter()
        new_directions = _append_orthonormal(
            basis,
            common.load_snapshot_fields(material_dir),
            tolerance=float(tolerance),
        )
        rows.append(
            {
                "offline_index": offline_index,
                "candidate_id": candidate_id,
                "new_directions": new_directions,
                "basis_rank": len(basis),
                "basis_update_wall_s": float(time.perf_counter() - started),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(case_out / "basis_growth.csv", index=False)
    return basis, table


def required_rank(curve: pd.DataFrame, threshold: float) -> dict[str, Any]:
    ordered = curve.sort_values("rank").reset_index(drop=True)
    errors = ordered["error_max"].to_numpy(dtype=float)
    ranks = ordered["rank"].to_numpy(dtype=int)
    suffix_max = np.maximum.accumulate(errors[::-1])[::-1]
    first = np.flatnonzero(errors <= float(threshold))
    stable = np.flatnonzero(suffix_max <= float(threshold))
    return {
        "first_rank": int(ranks[first[0]]) if len(first) else None,
        "stable_rank": int(ranks[stable[0]]) if len(stable) else None,
        "achieved": bool(len(stable)),
        "max_tested_rank": int(ranks[-1]),
        "error_at_max_rank": float(errors[-1]),
    }


def _rank_curve(
    *,
    truth: pd.DataFrame,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank in range(1, Kq.shape[1] + 1):
        validation = reduced._evaluate_rom(
            results_df=truth,
            Kq=Kq[:, :rank, :rank],
            Bq=Bq[:, :rank],
            Dq=Dq,
        )
        errors = validation["relative_frobenius_error"].to_numpy(dtype=float)
        rows.append(
            {
                "rank": rank,
                "error_mean": float(np.mean(errors)),
                "error_median": float(np.median(errors)),
                "error_p95": float(np.quantile(errors, 0.95)),
                "error_max": float(np.max(errors)),
                "worst_material_id": int(
                    validation.iloc[int(np.argmax(errors))]["material_id"]
                ),
                "min_rom_eigenvalue": float(validation["rom_min_eig"].min()),
                "min_loewner_eigenvalue": float(
                    validation["min_eig_Crom_minus_Cfom"].min()
                ),
                "online_median_s": float(validation["rom_online_s"].median()),
            }
        )
    curve = pd.DataFrame(rows)
    curve["error_max_suffix_envelope"] = np.maximum.accumulate(
        curve["error_max"].to_numpy(dtype=float)[::-1]
    )[::-1]
    return curve


def _run_case(
    *,
    case: dict[str, Any],
    out_dir: Path,
    training: pd.DataFrame,
    candidate_ids: list[int],
    validation: pd.DataFrame,
    runtime: dict[str, Any],
    seed: int,
    orthonormal_tolerance: float,
    selection_path: Path,
) -> dict[str, Any]:
    case_out = out_dir / str(case["case_kind"]) / str(case["case_id"])
    case_out.mkdir(parents=True, exist_ok=True)
    complete_path = case_out / "case_summary.json"
    if complete_path.is_file():
        cached = _read_json(complete_path)
        cache_matches = (
            cached.get("status") == "complete"
            and int(cached.get("training_materials", -1)) == len(training)
            and int(cached.get("validation_materials", -1)) == len(validation)
            and cached.get("training_candidate_ids") == candidate_ids
            and cached.get("selection_sha256") == _sha256(selection_path)
        )
        if cache_matches:
            print(f"[RANK] reutilizando {case['case_id']}", flush=True)
            return cached

    geometry = _load_geometry(case)
    _solve_snapshots(
        case_out=case_out,
        training=training,
        geometry=geometry,
        runtime=runtime,
        seed=seed,
    )
    truth = _solve_truth(
        case_out=case_out,
        validation=validation,
        geometry=geometry,
        runtime=runtime,
        seed=seed,
    )
    basis, growth = _build_basis(
        case_out=case_out,
        candidate_ids=candidate_ids,
        tolerance=orthonormal_tolerance,
    )
    started = time.perf_counter()
    Kq, Bq, Dq, assembly = reduced._assemble_reduced_operators(
        phase=geometry.phase,
        ori=geometry.ori.astype(np.float64),
        basis=basis,
    )
    assembly_wall_s = float(time.perf_counter() - started)
    np.savez_compressed(
        case_out / "reduced_operators.npz",
        Kq=Kq,
        Bq=Bq,
        Dq=Dq,
        coefficient_names=np.asarray(reduced.COEFF_NAMES),
    )
    del basis
    curve = _rank_curve(truth=truth, Kq=Kq, Bq=Bq, Dq=Dq)
    curve.to_csv(case_out / "rank_error_curve.csv", index=False)

    summary: dict[str, Any] = {
        "status": "complete",
        "case_kind": str(case["case_kind"]),
        "case_id": str(case["case_id"]),
        "source_dir": str(Path(case["source_dir"]).resolve()),
        "grid_size": int(geometry.phase.shape[0]),
        "voxel_count": int(geometry.phase.size),
        "training_materials": int(len(training)),
        "validation_materials": int(len(validation)),
        "training_candidate_ids": candidate_ids,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": _sha256(selection_path),
        "selection_uses_fom_error": False,
        "basis_rank_max": int(growth["basis_rank"].iloc[-1]),
        "operator_assembly_wall_s": assembly_wall_s,
        "operator_assembly": assembly,
        "truth_all_converged": bool(truth["solver_all_converged"].all()),
        "truth_max_relative_residual": float(truth["solver_max_relative_residual"].max()),
        "snapshot_max_relative_residual": float(
            pd.read_csv(case_out / "snapshot_results.csv")["solver_max_relative_residual"].max()
        ),
    }
    for key, value in case.items():
        if key != "source_dir" and isinstance(value, (str, int, float, bool, np.number)):
            summary[key] = value.item() if isinstance(value, np.number) else value
    for threshold in THRESHOLDS:
        tag = _threshold_tag(threshold)
        result = required_rank(curve, threshold)
        summary[f"r_{tag}_first"] = result["first_rank"]
        summary[f"r_{tag}"] = result["stable_rank"]
        summary[f"r_{tag}_achieved"] = result["achieved"]
        summary[f"error_at_rmax_{tag}"] = result["error_at_max_rank"]
    common.write_json(complete_path, summary)
    print(
        f"[RANK] listo {case['case_id']} | rmax={summary['basis_rank_max']} | "
        f"r_1e-4={summary['r_1e-4']}",
        flush=True,
    )
    return summary


def descriptor_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    geometry = summary.loc[summary["case_kind"] == "geometry"].copy()
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        rank_column = f"r_{_threshold_tag(threshold)}"
        if rank_column not in geometry:
            continue
        for descriptor in DESCRIPTORS:
            if descriptor not in geometry:
                continue
            subset = geometry[[descriptor, rank_column]].dropna()
            x = subset[descriptor].to_numpy(dtype=float)
            y = subset[rank_column].to_numpy(dtype=float)
            if len(subset) >= 3 and np.ptp(x) > 0.0 and np.ptp(y) > 0.0:
                pearson = pearsonr(x, y)
                spearman = spearmanr(x, y)
                pearson_r = float(pearson.statistic)
                pearson_p = float(pearson.pvalue)
                spearman_rho = float(spearman.statistic)
                spearman_p = float(spearman.pvalue)
            else:
                pearson_r = pearson_p = spearman_rho = spearman_p = np.nan
            rows.append(
                {
                    "threshold": threshold,
                    "rank_column": rank_column,
                    "descriptor": descriptor,
                    "geometry_count": int(len(subset)),
                    "censored_count": int(len(geometry) - len(subset)),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                    "interpretation": "exploratory_n10_not_universal_law",
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DESIGN_DIR)
    parser.add_argument("--geometry-dir", type=Path, default=GEOMETRY_DIR)
    parser.add_argument("--voxel-dir", type=Path, default=VOXEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--study", choices=("geometries", "voxel", "all"), default="all")
    parser.add_argument("--max-materials", type=int, default=20)
    parser.add_argument("--validation-count", type=int, default=32)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--orthonormal-tol", type=float, default=1.0e-10)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--geometry-backend", choices=("numba", "cupy", "auto"), default="numba")
    parser.add_argument("--generator-cores", type=int, default=2)
    parser.add_argument("--venv-path", type=Path, default=common.DEFAULT_VENV_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    common.prepare_runtime(
        args.venv_path,
        Path(__file__),
        marker_name="CMAME_RANK_CUDA_READY",
    )
    runtime = common.configure_runtime(
        geometry_backend=args.geometry_backend,
        generator_cores=args.generator_cores,
    )
    design_dir = args.design_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    training, candidate_ids, selection_path = _training_sequence(
        design_dir, int(args.max_materials)
    )
    validation = _validation_materials(design_dir, int(args.validation_count))
    cases: list[dict[str, Any]] = []
    if args.study in {"geometries", "all"}:
        cases.extend(_geometry_cases(args.geometry_dir.resolve()))
    if args.study in {"voxel", "all"}:
        cases.extend(_voxel_cases(args.voxel_dir.resolve()))
    if args.case_ids:
        selected = set(str(value) for value in args.case_ids)
        cases = [case for case in cases if str(case["case_id"]) in selected]
    if not cases:
        raise RuntimeError("No se seleccionaron casos de escalabilidad.")

    summaries: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        summaries.append(
            _run_case(
                case=case,
                out_dir=out_dir,
                training=training,
                candidate_ids=candidate_ids,
                validation=validation,
                runtime=runtime,
                seed=int(args.seed) + 10_000 * case_index,
                orthonormal_tolerance=float(args.orthonormal_tol),
                selection_path=selection_path,
            )
        )
        summary = pd.DataFrame(summaries)
        summary.to_csv(out_dir / "rank_scalability_summary.csv", index=False)
        descriptor_correlations(summary).to_csv(
            out_dir / "descriptor_rank_correlations.csv", index=False
        )

    summary = pd.DataFrame(summaries)
    common.write_json(
        out_dir / "campaign_manifest.json",
        {
            "status": "complete",
            "case_count": int(len(summary)),
            "geometry_case_count": int((summary["case_kind"] == "geometry").sum()),
            "voxel_case_count": int((summary["case_kind"] == "voxel_grid").sum()),
            "training_materials": int(len(training)),
            "validation_materials": int(len(validation)),
            "thresholds": list(THRESHOLDS),
            "principal_rank_definition": (
                "Smallest nested rank whose held-out maximum error and all later "
                "tested ranks remain at or below the threshold."
            ),
            "geometry_correlations_are_exploratory": True,
            "selection_sha256": _sha256(selection_path),
        },
    )
    print(f"[RANK] campana completa | casos={len(summary)} | out={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
