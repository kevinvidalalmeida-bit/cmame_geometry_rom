#!/usr/bin/env python3
"""Benchmark isolated CPU and batched CUDA evaluation of frozen ROMs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DEFAULT = SCRIPT_DIR / "campaign_config.json"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from env_bootstrap import ensure_configured_venv

if __name__ == "__main__":
    ensure_configured_venv(CONFIG_DEFAULT)

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import cupy as cp


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
    parser.add_argument("--single-count", type=int, default=1_000)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parameters(count: int, seed: int) -> np.ndarray:
    materials = validate._build_independent_materials(count, seed)
    return materials[list(reduced.MATERIAL_PARAMETER_COLUMNS)].to_numpy(
        dtype=np.float64
    )


def _cpu_single_times(
    parameters: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    count: int,
) -> list[float]:
    times: list[float] = []
    with threadpool_limits(limits=1, user_api="blas"):
        for query_id in range(min(20, count)):
            coeffs = reduced._material_coefficients_batch(
                parameters[query_id : query_id + 1]
            )
            reduced._rom_ceff(coeffs[0], Kq, Bq, Dq)
        for query_id in range(count):
            started = time.perf_counter()
            coeffs = reduced._material_coefficients_batch(
                parameters[query_id : query_id + 1]
            )
            reduced._rom_ceff(coeffs[0], Kq, Bq, Dq)
            times.append(float(time.perf_counter() - started))
    return times


def _cuda_times(
    single_parameters: np.ndarray,
    batch_parameters: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    single_count: int,
    repetitions: int,
) -> tuple[reduced.GpuAffineBatchEvaluator, list[float], list[float], np.ndarray]:
    evaluator = reduced.GpuAffineBatchEvaluator(Kq, Bq, Dq)
    warm_coeffs = reduced._material_coefficients_batch(single_parameters[:1])
    evaluator.evaluate(warm_coeffs)

    single_times: list[float] = []
    for query_id in range(single_count):
        started = time.perf_counter()
        coeffs = reduced._material_coefficients_batch(
            single_parameters[query_id : query_id + 1]
        )
        evaluator.evaluate(coeffs)
        single_times.append(float(time.perf_counter() - started))

    batch_times: list[float] = []
    result = np.empty((len(batch_parameters), 6, 6), dtype=np.float64)
    for _ in range(repetitions):
        started = time.perf_counter()
        coeffs = reduced._material_coefficients_batch(batch_parameters)
        result, _ = evaluator.evaluate(coeffs)
        batch_times.append(float(time.perf_counter() - started))
    return evaluator, single_times, batch_times, result


def main() -> None:
    args = _parser().parse_args()
    if min(args.query_count, args.single_count, args.repetitions) < 1:
        raise ValueError("All benchmark counts must be positive.")

    summary_dir = args.summary_dir.resolve()
    campaign = pd.read_csv(summary_dir / "sobol_pod_multigeometry_summary.csv")
    parameters = _parameters(max(args.query_count, args.single_count), args.seed)
    batch_parameters = parameters[: args.query_count]
    rows: list[dict[str, Any]] = []

    for campaign_row in campaign.sort_values("geometry_id").itertuples(index=False):
        geometry_id = int(campaign_row.geometry_id)
        run_dir = Path(str(campaign_row.run_dir))
        with np.load(run_dir / "reduced_operators.npz", allow_pickle=False) as operators:
            Kq = np.asarray(operators["Kq"], dtype=np.float64)
            Bq = np.asarray(operators["Bq"], dtype=np.float64)
            Dq = np.asarray(operators["Dq"], dtype=np.float64)

        cpu_times = _cpu_single_times(
            parameters,
            Kq,
            Bq,
            Dq,
            args.single_count,
        )
        evaluator, cuda_single_times, cuda_batch_times, cuda_results = _cuda_times(
            parameters,
            batch_parameters,
            Kq,
            Bq,
            Dq,
            args.single_count,
            args.repetitions,
        )
        reference_coeffs = reduced._material_coefficients_batch(
            batch_parameters[-1:]
        )
        with threadpool_limits(limits=1, user_api="blas"):
            reference, _, _ = reduced._rom_ceff(
                reference_coeffs[0], Kq, Bq, Dq
            )
        relative_difference = float(
            np.linalg.norm(cuda_results[-1] - reference, ord="fro")
            / np.linalg.norm(reference, ord="fro")
        )
        if relative_difference > 1.0e-12:
            raise RuntimeError(
                f"CPU/CUDA ROM mismatch for G{geometry_id:02d}: "
                f"{relative_difference:.3e}"
            )

        timing_path = run_dir / "rom_timing_summary.json"
        old_timing = json.loads(timing_path.read_text(encoding="utf-8"))
        cpu_median = float(np.median(cpu_times))
        cuda_batch_median = float(np.median(cuda_batch_times))
        fom_time = float(campaign_row.fom_material_median_s)
        compilation_time = float(campaign_row.compilation_wall_s)
        rows.append(
            {
                "geometry_id": geometry_id,
                "geometry_label": str(campaign_row.geometry_label),
                "nvox": int(campaign_row.nvox),
                "stop_materials": int(campaign_row.stop_materials),
                "basis_rank": int(campaign_row.basis_rank),
                "fom_material_median_s": fom_time,
                "compilation_wall_s": compilation_time,
                "cuda_cold_start_median_s": float(
                    old_timing["cold_start_median_s"]
                ),
                "cpu_single_thread_count": 1,
                "cpu_hot_single_median_s": cpu_median,
                "cpu_hot_single_p95_s": float(np.quantile(cpu_times, 0.95)),
                "cuda_hot_single_median_s": float(np.median(cuda_single_times)),
                "cuda_hot_single_p95_s": float(
                    np.quantile(cuda_single_times, 0.95)
                ),
                "cuda_operator_transfer_s": float(
                    evaluator.operator_transfer_wall_s
                ),
                "cuda_batch_query_count": int(args.query_count),
                "cuda_batch_median_s": cuda_batch_median,
                "cuda_batch_p95_s": float(
                    np.quantile(cuda_batch_times, 0.95)
                ),
                "cuda_batch_amortized_s": cuda_batch_median / args.query_count,
                "cpu_cuda_relative_difference": relative_difference,
                "latency_speedup": fom_time / cpu_median,
                "throughput_speedup": fom_time
                / (cuda_batch_median / args.query_count),
                "break_even_queries": int(
                    math.ceil(compilation_time / (fom_time - cpu_median))
                ),
            }
        )
        print(
            f"G{geometry_id:02d}: CPU={1.0e6 * cpu_median:.2f} us, "
            f"CUDA single={1.0e6 * np.median(cuda_single_times):.2f} us, "
            f"CUDA batch={1.0e3 * cuda_batch_median:.2f} ms",
            flush=True,
        )

    output = pd.DataFrame(rows)
    output.to_csv(summary_dir / "rom_backend_benchmark.csv", index=False)
    metadata = {
        "seed": int(args.seed),
        "single_query_count": int(args.single_count),
        "cuda_batch_query_count": int(args.query_count),
        "repetitions": int(args.repetitions),
        "cupy_version": str(cp.__version__),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver_api_version": int(cp.cuda.runtime.driverGetVersion()),
        "cuda_device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "single_query_backend": "NumPy/OpenBLAS, one thread",
        "batch_backend": "CuPy/CUDA, operators resident, synchronized end-to-end",
        "timed_operations": (
            "affine coefficient evaluation, reduced assembly, dense solve, "
            "effective-tensor reconstruction, and result return"
        ),
        "records": output.to_dict(orient="records"),
    }
    (summary_dir / "rom_backend_benchmark.json").write_text(
        json.dumps(_jsonable(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
