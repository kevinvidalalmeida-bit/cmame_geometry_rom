#!/usr/bin/env python3
"""Benchmark the untruncated snapshot span for G00--G09.

The operators come from a fixed-prefix replay with float32 FFT snapshots and
float64 Gram/Ritz contractions.  Every raw snapshot coordinate is retained.
The independent held-out references are read from the nominal campaign; they
are not used to alter the full-span operators.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import cupy as cp
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rom_reduced_operator as reduced
import rom_validation_utils as validate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=10_000)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--run-prefix", type=str, default="full_span_mixed64_20260821_geometry_"
    )
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _parameters(count: int, seed: int) -> np.ndarray:
    materials = validate._build_independent_materials(count, seed)
    return materials[list(reduced.MATERIAL_PARAMETER_COLUMNS)].to_numpy(
        dtype=np.float64
    )


def _tensor_from_columns(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    tensors = np.empty((len(frame), 6, 6), dtype=np.float64)
    for row in range(6):
        for column in range(6):
            tensors[:, row, column] = frame[
                f"{prefix}_{row + 1}{column + 1}"
            ].to_numpy(dtype=np.float64)
    return 0.5 * (tensors + np.swapaxes(tensors, -1, -2))


def _heldout_metrics(
    run_dir: Path,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> dict[str, float | int]:
    truth = pd.read_csv(run_dir / "final_validation_truth_results.csv")
    parameters = truth[list(reduced.MATERIAL_PARAMETER_COLUMNS)].to_numpy(
        dtype=np.float64
    )
    coefficients = reduced._material_coefficients_batch(parameters)
    C_rom, _, _ = reduced._rom_ceff_batch(coefficients, Kq, Bq, Dq)
    C_fom = _tensor_from_columns(truth, "Ceff")

    differences = C_rom - C_fom
    errors = np.linalg.norm(differences, axis=(1, 2)) / np.linalg.norm(
        C_fom, axis=(1, 2)
    )
    eta = np.empty(len(truth), dtype=np.float64)
    k_min = np.empty(len(truth), dtype=np.float64)
    k_margin = np.empty(len(truth), dtype=np.float64)
    c_min = np.linalg.eigvalsh(C_rom)[:, 0]
    for index, coefficient in enumerate(coefficients):
        stiffness = np.tensordot(coefficient, Kq, axes=(0, 0))
        stiffness = 0.5 * (stiffness + stiffness.T)
        eigenvalues = np.linalg.eigvalsh(stiffness)
        k_min[index] = eigenvalues[0]
        k_margin[index] = eigenvalues[0] / eigenvalues[-1]
        eta[index] = np.linalg.eigvalsh(
            0.5 * (differences[index] + differences[index].T)
        )[0] / np.linalg.norm(C_fom[index], ord=2)

    return {
        "validation_count": int(len(errors)),
        "validation_error_mean": float(np.mean(errors)),
        "validation_error_max": float(np.max(errors)),
        "validation_coverage_1e4_percent": float(
            100.0 * np.mean(errors <= 1.0e-4)
        ),
        "validation_coverage_1e3_percent": float(
            100.0 * np.mean(errors <= 1.0e-3)
        ),
        "validation_schur_eta_min": float(np.min(eta)),
        "validation_K_min_eig_min": float(np.min(k_min)),
        "validation_K_spectral_spd_margin_min": float(np.min(k_margin)),
        "validation_C_min_eig_min": float(np.min(c_min)),
    }


def main() -> None:
    args = _parser().parse_args()
    if min(args.query_count, args.repetitions) < 1:
        raise ValueError("Benchmark counts must be positive.")

    summary_dir = args.summary_dir.resolve()
    campaign = pd.read_csv(summary_dir / "sobol_pod_multigeometry_summary.csv")
    nominal_timing = pd.read_csv(summary_dir / "rom_backend_benchmark.csv").set_index(
        "geometry_id"
    )
    parameters = _parameters(args.query_count, args.seed)
    rows: list[dict[str, Any]] = []

    for campaign_row in campaign.sort_values("geometry_id").itertuples(index=False):
        geometry_id = int(campaign_row.geometry_id)
        nominal_run_dir = Path(str(campaign_row.run_dir))
        run_dir = nominal_run_dir.parent / f"{args.run_prefix}{geometry_id:02d}"
        with np.load(run_dir / "reduced_operators.npz", allow_pickle=False) as data:
            Kq = np.asarray(data["Kq"], dtype=np.float64)
            Bq = np.asarray(data["Bq"], dtype=np.float64)
            G = np.asarray(data["G"], dtype=np.float64)
            Dq = np.asarray(data["Dq"], dtype=np.float64)
            invR = np.asarray(data["invR"], dtype=np.float64)

        run_summary = json.loads(
            (run_dir / "sobol_pod_summary.json").read_text(encoding="utf-8")
        )
        gram_symmetric = 0.5 * (G + G.T)
        gram_eigenvalues = np.linalg.eigvalsh(gram_symmetric)
        raw_rank = int(G.shape[0])
        full_rank = int(Kq.shape[1])
        discarded_rank = raw_rank - full_rank
        if full_rank != raw_rank or discarded_rank != 0:
            raise RuntimeError(
                f"G{geometry_id:02d} did not retain the complete snapshot span: "
                f"{full_rank}/{raw_rank}."
            )

        transform = invR.T
        identity_error = float(
            np.linalg.norm(transform @ G @ transform.T - np.eye(full_rank), ord="fro")
            / np.sqrt(full_rank)
        )
        heldout = _heldout_metrics(nominal_run_dir, Kq, Bq, Dq)
        evaluator = reduced.GpuAffineBatchEvaluator(Kq, Bq, Dq)
        evaluator.evaluate(reduced._material_coefficients_batch(parameters[:1]))
        batch_times: list[float] = []
        cuda_result = np.empty((args.query_count, 6, 6), dtype=np.float64)
        for _ in range(args.repetitions):
            started = time.perf_counter()
            coefficients = reduced._material_coefficients_batch(parameters)
            cuda_result, _ = evaluator.evaluate(coefficients)
            batch_times.append(float(time.perf_counter() - started))

        reference, _, _ = reduced._rom_ceff(
            reduced._material_coefficients_batch(parameters[-1:])[0],
            Kq,
            Bq,
            Dq,
        )
        cuda_difference = float(
            np.linalg.norm(cuda_result[-1] - reference, ord="fro")
            / np.linalg.norm(reference, ord="fro")
        )
        if cuda_difference > 1.0e-10:
            raise RuntimeError(
                f"CPU/CUDA mismatch for full-span G{geometry_id:02d}: "
                f"{cuda_difference:.3e}."
            )

        batch_median = float(np.median(batch_times))
        amortized = batch_median / args.query_count
        nominal = nominal_timing.loc[geometry_id]
        row = {
            "geometry_id": geometry_id,
            "geometry_label": str(campaign_row.geometry_label),
            "full_span_run_dir": str(run_dir.resolve()),
            "stop_materials": int(campaign_row.stop_materials),
            "raw_snapshot_columns": raw_rank,
            "nominal_rank": int(campaign_row.basis_rank),
            "full_span_rank": full_rank,
            "discarded_rank": discarded_rank,
            "configured_rank_rtol": float(run_summary["ritz_gram_rank_rtol"]),
            "gram_relative_min": float(
                gram_eigenvalues[0] / gram_eigenvalues[-1]
            ),
            "gram_condition": float(
                gram_eigenvalues[-1] / gram_eigenvalues[0]
            ),
            "gram_identity_relative_frobenius": identity_error,
            "fixed_prefix_compilation_wall_s": float(
                run_summary["compilation_wall_s"]
            ),
            "snapshot_solve_wall_s": float(
                run_summary["snapshot_total_solve_wall_s"]
            ),
            "ritz_contraction_wall_s": float(
                run_summary["ritz_contraction_total_wall_s"]
            ),
            "cuda_operator_transfer_s": float(evaluator.operator_transfer_wall_s),
            "cuda_batch_query_count": int(args.query_count),
            "cuda_batch_repetition_wall_s": json.dumps(batch_times),
            "cuda_batch_median_s": batch_median,
            "cuda_batch_p95_s": float(np.quantile(batch_times, 0.95)),
            "cuda_batch_amortized_s": amortized,
            "nominal_cuda_batch_amortized_s": float(
                nominal["cuda_batch_amortized_s"]
            ),
            "full_to_nominal_cuda_time_ratio": float(
                amortized / nominal["cuda_batch_amortized_s"]
            ),
            "fom_material_median_s": float(campaign_row.fom_material_median_s),
            "full_span_throughput_speedup": float(
                campaign_row.fom_material_median_s / amortized
            ),
            "cpu_cuda_relative_difference": cuda_difference,
            **heldout,
        }
        rows.append(row)
        print(
            f"G{geometry_id:02d}: r={full_rank}/{raw_rank}, "
            f"CUDA={1.0e3 * batch_median:.3f} ms/10k "
            f"({1.0e6 * amortized:.3f} us/query), "
            f"error_max={heldout['validation_error_max']:.3e}",
            flush=True,
        )
        del evaluator, cuda_result, Kq, Bq, invR
        cp.get_default_memory_pool().free_all_blocks()

    output = pd.DataFrame(rows)
    csv_path = summary_dir / "full_snapshot_span_benchmark.csv"
    output.to_csv(csv_path, index=False)
    metadata = {
        "definition": (
            "Fixed nominal Sobol prefix; all 6*N_s raw snapshot coordinates "
            "retained. The configured 1e-15 machine guard discarded zero "
            "directions, so no energy or numerical-rank truncation occurred."
        ),
        "snapshot_profile": "float32 FFT/CG, rtol=1e-5",
        "gram_ritz_contraction_dtype": "float64",
        "benchmark_fft_solves_performed": 0,
        "adaptive_stopping_repeated": False,
        "query_seed": int(args.seed),
        "cuda_batch_query_count": int(args.query_count),
        "repetitions": int(args.repetitions),
        "cupy_version": str(cp.__version__),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "records": output.to_dict(orient="records"),
    }
    (summary_dir / "full_snapshot_span_benchmark.json").write_text(
        json.dumps(_jsonable(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
