#!/usr/bin/env python3
"""Compare a conventional POD/RB baseline against the tangential ROM hierarchy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
RUN_DEFAULT = (
    PROJECT_ROOT
    / "results"
    / "fixed_geometry_ffthompy"
    / "fixed_geometry_ar15_vf20_sobol8_center_fields"
)
VALIDATION32_DEFAULT = (
    RUN_DEFAULT
    / "independent_validation_r66_sobol32_seed20261111_reuse"
    / "validation_full_order_results.csv"
)
FINAL_ROM_DEFAULT = (
    RUN_DEFAULT
    / "rom_tangential_r66_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v30"
)

TANGENTIAL_ROM_DEFAULTS = [
    RUN_DEFAULT / "rom_tangential_r6_center",
    RUN_DEFAULT / "rom_tangential_r12_center_m1",
    RUN_DEFAULT / "rom_tangential_r18_center_m1_m3",
    RUN_DEFAULT / "rom_tangential_r24_center_m1_m3_m5",
    RUN_DEFAULT / "rom_tangential_r30_center_m1_m3_m5_m7",
    RUN_DEFAULT / "rom_tangential_r36_center_m1_m2_m3_m5_m7",
    RUN_DEFAULT / "rom_tangential_r42_center_m1_m2_m3_m5_m7_v13",
    RUN_DEFAULT / "rom_tangential_r48_center_m1_m2_m3_m5_m7_v13_v3",
    RUN_DEFAULT / "rom_tangential_r54_center_m1_m2_m3_m5_m7_v13_v3_v15",
    RUN_DEFAULT / "rom_tangential_r60_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1",
    FINAL_ROM_DEFAULT,
]

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import rom_reduced_operator as reduced


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
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_out_dir(run_dir: Path, out_name: str, *, overwrite: bool) -> Path:
    out_dir = run_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=overwrite)
    return out_dir


def _stats_from_validation(df: pd.DataFrame) -> dict[str, Any]:
    errors = df["relative_frobenius_error"].to_numpy(dtype=float)
    worst_idx = int(np.argmax(errors))
    return {
        "material_count": int(len(df)),
        "mean_relative_error": float(np.mean(errors)),
        "median_relative_error": float(np.median(errors)),
        "p95_relative_error": float(np.quantile(errors, 0.95)),
        "max_relative_error": float(np.max(errors)),
        "worst_material_id": int(df.iloc[worst_idx]["material_id"]),
        "median_online_s": float(np.median(df["rom_online_s"])),
        "min_rom_eig": float(df["rom_min_eig"].min()),
        "min_eig_Crom_minus_Cfom": float(df["min_eig_Crom_minus_Cfom"].min()),
    }


def _evaluate_rank(
    *,
    results_df: pd.DataFrame,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    rank: int,
) -> pd.DataFrame:
    rank = int(rank)
    return reduced._evaluate_rom(
        results_df=results_df,
        Kq=Kq[:, :rank, :rank],
        Bq=Bq[:, :rank, :],
        Dq=Dq,
    )


def _load_snapshot_matrix(
    run_dir: Path,
    snapshot_ids: list[int],
) -> tuple[np.ndarray, tuple[int, ...], float]:
    t0 = time.perf_counter()
    fields = reduced._load_snapshot_fields(
        run_dir, snapshot_ids, dtype=np.float32
    )
    if not fields:
        raise RuntimeError("No se cargaron campos snapshot.")
    snapshot_shape = tuple(int(value) for value in np.asarray(fields[0]).shape)
    matrix = np.stack(
        [np.asarray(field, dtype=np.float32).reshape(-1) for field in fields],
        axis=0,
    )
    del fields
    return matrix, snapshot_shape, float(time.perf_counter() - t0)


def _pod_basis_from_snapshot_matrix(
    snapshot_matrix: np.ndarray,
    snapshot_shape: tuple[int, ...],
    *,
    eig_tol: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    from scipy.linalg import eigh as scipy_eigh

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    ndof = int(snapshot_matrix.shape[1])
    snapshot_count = int(snapshot_matrix.shape[0])
    if snapshot_matrix.dtype == np.dtype(np.float64):
        gram = (snapshot_matrix @ snapshot_matrix.T) / float(ndof)
    else:
        # Preserve float32 snapshot storage and accumulate the small Gram
        # matrix in float64 through bounded spatial tiles.
        target_bytes = 128 * 1024**2
        chunk_size = max(
            4096,
            min(
                ndof,
                target_bytes
                // max(snapshot_count * np.dtype(np.float64).itemsize, 1),
            ),
        )
        gram = np.zeros((snapshot_count, snapshot_count), dtype=np.float64)
        for start in range(0, ndof, int(chunk_size)):
            end = min(start + int(chunk_size), ndof)
            block = np.asarray(snapshot_matrix[:, start:end], dtype=np.float64)
            gram += block @ block.T
        gram /= float(ndof)
    gram = 0.5 * (gram + gram.T)
    timings["gram_wall_s"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    # scipy eigh returns eigenvalues in ascending order; reverse for descending
    eigvals, eigvecs = scipy_eigh(gram, check_finite=False)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    eigvals = np.maximum(eigvals, 0.0)
    threshold = max(float(eig_tol) * float(eigvals[0]), np.finfo(float).eps)
    keep = eigvals > threshold
    eigvals_kept = eigvals[keep]
    eigvecs_kept = eigvecs[:, keep]
    if eigvals_kept.size == 0:
        raise RuntimeError("El POD no encontro eigenvalores positivos.")
    timings["eigendecomposition_wall_s"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter()
    if snapshot_matrix.dtype == np.dtype(np.float64):
        modes_flat = (
            eigvecs_kept.T @ snapshot_matrix
        ) / np.sqrt(eigvals_kept)[:, None]
    else:
        target_bytes = 128 * 1024**2
        chunk_size = max(
            4096,
            min(
                ndof,
                target_bytes
                // max(len(eigvals_kept) * np.dtype(np.float64).itemsize, 1),
            ),
        )
        modes_flat = np.empty((len(eigvals_kept), ndof), dtype=np.float32)
        scale = np.sqrt(eigvals_kept)[:, None]
        for start in range(0, ndof, int(chunk_size)):
            end = min(start + int(chunk_size), ndof)
            block = (eigvecs_kept.T @ snapshot_matrix[:, start:end]) / scale
            modes_flat[:, start:end] = np.asarray(block, dtype=np.float32)
    basis = [modes_flat[ii].reshape(snapshot_shape).copy() for ii in range(modes_flat.shape[0])]
    timings["basis_reconstruction_wall_s"] = float(time.perf_counter() - t0)

    cumulative_energy = np.cumsum(eigvals_kept) / max(
        float(np.sum(eigvals_kept)),
        np.finfo(float).eps,
    )
    singular_values = np.sqrt(eigvals_kept * float(ndof))
    return basis, eigvals_kept, singular_values, cumulative_energy, timings



def _evaluate_pod_curve(
    *,
    results_df: pd.DataFrame,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    ranks: list[int],
    out_dir: Path,
) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for rank in ranks:
        rank_t0 = time.perf_counter()
        validation_df = _evaluate_rank(
            results_df=results_df,
            Kq=Kq,
            Bq=Bq,
            Dq=Dq,
            rank=int(rank),
        )
        validation_df.insert(0, "rank", int(rank))
        validation_df.to_csv(out_dir / f"pod_validation_r{int(rank):03d}.csv", index=False)
        stats = _stats_from_validation(validation_df)
        rows.append(
            {
                "method": "pod_final_snapshot_svd",
                "rank": int(rank),
                "validation_wall_s": float(time.perf_counter() - rank_t0),
                **stats,
            }
        )
        print(
            "[POD] "
            f"r={int(rank)} | mean={stats['mean_relative_error']:.3e} | "
            f"p95={stats['p95_relative_error']:.3e} | "
            f"max={stats['max_relative_error']:.3e} | "
            f"worst={stats['worst_material_id']}",
            flush=True,
        )
    return pd.DataFrame(rows), float(time.perf_counter() - t0)


def _evaluate_tangential_curve(
    *,
    results_df: pd.DataFrame,
    rom_dirs: list[Path],
    out_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rom_dir in rom_dirs:
        if not rom_dir.is_dir():
            print(f"[POD] saltando ROM tangencial ausente: {rom_dir}", flush=True)
            continue
        manifest = _load_json(rom_dir / "rom_manifest.json")
        operators = np.load(rom_dir / "reduced_operators.npz")
        Kq = np.asarray(operators["Kq"], dtype=float)
        Bq = np.asarray(operators["Bq"], dtype=float)
        Dq = np.asarray(operators["Dq"], dtype=float)
        rank = int(manifest["basis_rank"])
        rank_t0 = time.perf_counter()
        validation_df = reduced._evaluate_rom(
            results_df=results_df,
            Kq=Kq,
            Bq=Bq,
            Dq=Dq,
        )
        validation_df.insert(0, "rank", rank)
        validation_df.insert(1, "rom_dir", str(rom_dir))
        validation_df.to_csv(
            out_dir / f"tangential_validation_r{rank:03d}.csv",
            index=False,
        )
        stats = _stats_from_validation(validation_df)
        rows.append(
            {
                "method": "tangential_adaptive",
                "rank": rank,
                "validation_wall_s": float(time.perf_counter() - rank_t0),
                "rom_dir": str(rom_dir),
                "snapshot_material_ids": manifest.get("snapshot_material_ids", []),
                **stats,
            }
        )
        print(
            "[TANGENTIAL] "
            f"r={rank} | mean={stats['mean_relative_error']:.3e} | "
            f"p95={stats['p95_relative_error']:.3e} | "
            f"max={stats['max_relative_error']:.3e} | "
            f"worst={stats['worst_material_id']}",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye un baseline POD/RB convencional desde los campos "
            "tangenciales FFTHomPy finales y lo compara contra la jerarquia "
            "tangencial adaptativa en la misma geometria fija."
        )
    )
    parser.add_argument("run_dir", type=Path, nargs="?", default=RUN_DEFAULT)
    parser.add_argument("--validation-csv", type=Path, default=VALIDATION32_DEFAULT)
    parser.add_argument("--snapshot-material-ids", type=int, nargs="*", default=None)
    parser.add_argument("--final-rom-dir", type=Path, default=FINAL_ROM_DEFAULT)
    parser.add_argument(
        "--tangential-rom-dirs",
        type=Path,
        nargs="*",
        default=TANGENTIAL_ROM_DEFAULTS,
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="*",
        default=list(range(6, 67, 6)),
    )
    parser.add_argument("--pod-eig-tol", type=float, default=1e-12)
    parser.add_argument("--out-name", default="pod_baseline_final_snapshots_validation32")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    validation_csv = args.validation_csv.resolve()
    final_rom_dir = args.final_rom_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No existe run_dir: {run_dir}")
    if not validation_csv.is_file():
        raise FileNotFoundError(f"No existe validation CSV: {validation_csv}")
    if not final_rom_dir.is_dir():
        raise FileNotFoundError(f"No existe final_rom_dir: {final_rom_dir}")

    out_dir = _make_out_dir(run_dir, str(args.out_name), overwrite=bool(args.overwrite))
    results_df = pd.read_csv(validation_csv)
    geometry = common.load_fixed_geometry(run_dir)
    phase = geometry.phase.astype(np.uint8)
    ori = geometry.ori.astype(np.float64)

    final_manifest = _load_json(final_rom_dir / "rom_manifest.json")
    snapshot_ids = (
        sorted(set(int(value) for value in args.snapshot_material_ids))
        if args.snapshot_material_ids is not None
        else [int(value) for value in final_manifest["snapshot_material_ids"]]
    )
    ranks = sorted(set(int(rank) for rank in args.ranks if int(rank) > 0))
    if not ranks:
        raise ValueError("Debe haber al menos un rank positivo.")

    print(
        "[POD] cargando campos finales | "
        f"snapshots={snapshot_ids} | validation={validation_csv.name}",
        flush=True,
    )
    snapshot_matrix, snapshot_shape, load_s = _load_snapshot_matrix(run_dir, snapshot_ids)
    if max(ranks) > snapshot_matrix.shape[0]:
        raise ValueError(
            f"rank maximo {max(ranks)} supera campos disponibles "
            f"({snapshot_matrix.shape[0]})."
        )

    print(
        "[POD] construyendo POD por snapshots | "
        f"fields={snapshot_matrix.shape[0]} | dof={snapshot_matrix.shape[1]}",
        flush=True,
    )
    basis, eigvals, singular_values, cumulative_energy, pod_timings = (
        _pod_basis_from_snapshot_matrix(
            snapshot_matrix,
            snapshot_shape,
            eig_tol=float(args.pod_eig_tol),
        )
    )
    del snapshot_matrix

    max_rank = max(ranks)
    if len(basis) < max_rank:
        raise RuntimeError(
            f"El POD produjo {len(basis)} modos, menor que rank solicitado {max_rank}."
        )
    basis = basis[:max_rank]

    print(
        "[POD] ensamblando operadores POD | "
        f"max_rank={max_rank} | grid={phase.shape}",
        flush=True,
    )
    Kq, Bq, Dq, assembly_meta = reduced._assemble_reduced_operators(
        phase=phase,
        ori=ori,
        basis=basis,
    )

    np.savez_compressed(
        out_dir / "pod_reduced_operators_full.npz",
        Kq=Kq,
        Bq=Bq,
        Dq=Dq,
        coefficient_names=np.array(reduced.COEFF_NAMES),
        snapshot_material_ids=np.array(snapshot_ids, dtype=np.int64),
        ranks=np.array(ranks, dtype=np.int64),
        pod_eigenvalues=eigvals,
        pod_singular_values=singular_values,
        pod_cumulative_energy=cumulative_energy,
    )

    spectrum_df = pd.DataFrame(
        {
            "mode": np.arange(1, len(eigvals) + 1, dtype=int),
            "eigenvalue": eigvals,
            "singular_value": singular_values,
            "cumulative_energy": cumulative_energy,
        }
    )
    spectrum_df.to_csv(out_dir / "pod_spectrum.csv", index=False)

    print("[POD] evaluando curva POD", flush=True)
    pod_curve, pod_validation_s = _evaluate_pod_curve(
        results_df=results_df,
        Kq=Kq,
        Bq=Bq,
        Dq=Dq,
        ranks=ranks,
        out_dir=out_dir,
    )
    print("[POD] evaluando curva tangencial existente", flush=True)
    tangential_curve = _evaluate_tangential_curve(
        results_df=results_df,
        rom_dirs=[path.resolve() for path in args.tangential_rom_dirs],
        out_dir=out_dir,
    )

    comparison_df = pd.concat([pod_curve, tangential_curve], ignore_index=True, sort=False)
    comparison_df = comparison_df.sort_values(["rank", "method"]).reset_index(drop=True)
    comparison_df.to_csv(out_dir / "pod_vs_tangential_validation_curve.csv", index=False)
    comparison_df.to_excel(out_dir / "pod_vs_tangential_validation_curve.xlsx", index=False)

    pivot = comparison_df.pivot_table(
        index="rank",
        columns="method",
        values="max_relative_error",
        aggfunc="first",
    )
    rank66 = comparison_df[comparison_df["rank"] == 66]
    summary = {
        "run_dir": str(run_dir),
        "validation_csv": str(validation_csv),
        "out_dir": str(out_dir),
        "pod_description": (
            "Strong conventional POD baseline: all final tangential snapshot "
            "fields are visible before truncation, then ranks 6..66 are tested."
        ),
        "snapshot_material_ids": snapshot_ids,
        "snapshot_field_count": int(len(snapshot_ids) * 6),
        "pod_rank_max": int(max_rank),
        "validation_material_count": int(len(results_df)),
        "ranks": ranks,
        "timings": {
            "snapshot_load_wall_s": load_s,
            **pod_timings,
            "operator_assembly_wall_s": assembly_meta["assembly_wall_s"],
            "pod_curve_validation_wall_s": pod_validation_s,
        },
        "assembly": assembly_meta,
        "pod_energy_at_ranks": {
            str(rank): float(cumulative_energy[rank - 1]) for rank in ranks
        },
        "rank66": rank66.to_dict(orient="records"),
    }
    if not pivot.empty and {"pod_final_snapshot_svd", "tangential_adaptive"}.issubset(pivot.columns):
        ratio = pivot["pod_final_snapshot_svd"] / pivot["tangential_adaptive"]
        summary["max_error_ratio_pod_over_tangential_by_rank"] = {
            str(int(rank)): float(value)
            for rank, value in ratio.dropna().items()
        }

    _write_json(out_dir / "pod_baseline_summary.json", summary)

    best_rows = comparison_df.loc[
        comparison_df.groupby("method")["max_relative_error"].idxmin()
    ].sort_values("method")
    text_lines = [
        "# POD/RB Baseline Against Tangential ROM",
        "",
        f"- Source run: `{run_dir}`",
        f"- Validation CSV: `{validation_csv}`",
        f"- Snapshot material IDs: `{snapshot_ids}`",
        f"- Snapshot fields: `{len(snapshot_ids) * 6}`",
        f"- POD max rank: `{max_rank}`",
        "",
        "This is a strong POD baseline: it sees all final snapshot fields before",
        "truncation. It is useful as a conventional RB/POD comparator, but its",
        "low-rank points are not equal-offline-budget adaptive runs.",
        "",
        "## Best Rows",
        "",
        "| Method | Rank | Mean error | P95 error | Max error | Worst ID |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in best_rows.iterrows():
        text_lines.append(
            "| "
            f"{row['method']} | {int(row['rank'])} | "
            f"{float(row['mean_relative_error']):.6e} | "
            f"{float(row['p95_relative_error']):.6e} | "
            f"{float(row['max_relative_error']):.6e} | "
            f"{int(row['worst_material_id'])} |"
        )
    selected_ranks = [rank for rank in ranks if rank in {6, 12, 18, 24, 36, 48, 54, 60, 66}]
    text_lines.extend(
        [
            "",
            "## Selected Rank Curve",
            "",
            "| Rank | POD max error | Tangential max error | POD snapshot energy |",
            "|---:|---:|---:|---:|",
        ]
    )
    curve_pivot = comparison_df.pivot_table(
        index="rank",
        columns="method",
        values="max_relative_error",
        aggfunc="first",
    )
    for rank in selected_ranks:
        pod_max = curve_pivot.loc[rank, "pod_final_snapshot_svd"]
        tan_max = curve_pivot.loc[rank, "tangential_adaptive"]
        energy = cumulative_energy[rank - 1]
        text_lines.append(
            f"| {rank} | {pod_max:.6e} | {tan_max:.6e} | {energy:.6f} |"
        )
    text_lines.extend(
        [
            "",
            "The POD curve is intentionally strong because all final snapshots are",
            "available before truncation. A notable diagnostic is that POD snapshot",
            "energy is not a reliable proxy for the homogenized tensor error: the",
            f"first 6 POD modes already capture {cumulative_energy[5]:.3%} of the",
            "snapshot energy, but still give order-10% maximum tensor error on the",
            "32-point validation set.",
        ]
    )
    text_lines.extend(
        [
            "",
            "## Timing",
            "",
            "| Quantity | Wall time |",
            "|---|---:|",
            f"| Snapshot load | {load_s:.6e} s |",
            f"| POD Gram matrix | {pod_timings['gram_wall_s']:.6e} s |",
            f"| POD eigendecomposition | {pod_timings['eigendecomposition_wall_s']:.6e} s |",
            f"| POD basis reconstruction | {pod_timings['basis_reconstruction_wall_s']:.6e} s |",
            f"| POD operator assembly | {assembly_meta['assembly_wall_s']:.6e} s |",
            "",
            "Detailed curves are in `pod_vs_tangential_validation_curve.csv`.",
        ]
    )
    (out_dir / "pod_baseline_summary.md").write_text(
        "\n".join(text_lines) + "\n",
        encoding="utf-8",
    )

    print(
        "[POD] listo | "
        f"out={out_dir} | max_rank={max_rank} | rows={len(comparison_df)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
