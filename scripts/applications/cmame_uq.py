#!/usr/bin/env python3
"""Global-uniform and local-truncated-normal UQ with the final 7D ROM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd



ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts", ROOT / "FFT"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fft_homogenization_solver as sweep


OUT_DEFAULT = ROOT / "results" / "cmame_method" / "uq"
PARAMETER_NAMES = tuple(sweep.MATERIAL_BOUNDS)
OUTPUT_NAMES = (
    "E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23"
)


def _material_coefficients_batch(parameters: np.ndarray) -> np.ndarray:
    Em, nu_m, Ef_L, Ef_T, G_LT, nu_LT, nu_TT = parameters.T
    lam_m = Em * nu_m / ((1.0 + nu_m) * (1.0 - 2.0 * nu_m))
    mu_m = Em / (2.0 * (1.0 + nu_m))
    count = len(parameters)
    compliance = np.zeros((count, 6, 6), dtype=np.float64)
    compliance[:, 0, 0] = 1.0 / Ef_L
    compliance[:, 1, 1] = 1.0 / Ef_T
    compliance[:, 2, 2] = 1.0 / Ef_T
    compliance[:, 0, 1] = compliance[:, 1, 0] = -nu_LT / Ef_L
    compliance[:, 0, 2] = compliance[:, 2, 0] = -nu_LT / Ef_L
    compliance[:, 1, 2] = compliance[:, 2, 1] = -nu_TT / Ef_T
    compliance[:, 3, 3] = 2.0 * (1.0 + nu_TT) / Ef_T
    compliance[:, 4, 4] = 1.0 / G_LT
    compliance[:, 5, 5] = 1.0 / G_LT
    voigt = np.linalg.inv(compliance)
    factors = np.array((1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)))
    mandel = voigt * factors[None, :, None] * factors[None, None, :]
    return np.column_stack(
        (
            lam_m,
            mu_m,
            mandel[:, 1, 1],
            mandel[:, 1, 2],
            mandel[:, 0, 1],
            mandel[:, 0, 0],
            mandel[:, 4, 4] / 2.0,
        )
    )


def _engineering_constants_batch(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    compliance = np.linalg.inv(C)
    outputs = np.column_stack(
        (
            1.0 / compliance[:, 0, 0],
            1.0 / compliance[:, 1, 1],
            1.0 / compliance[:, 2, 2],
            1.0 / (2.0 * compliance[:, 3, 3]),
            1.0 / (2.0 * compliance[:, 4, 4]),
            1.0 / (2.0 * compliance[:, 5, 5]),
            -compliance[:, 1, 0] / compliance[:, 0, 0],
            -compliance[:, 2, 0] / compliance[:, 0, 0],
            -compliance[:, 2, 1] / compliance[:, 1, 1],
        )
    )
    return outputs, np.linalg.eigvalsh(C)[:, 0]


def _evaluate_rom_batch(
    parameters: np.ndarray,
    *,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    outputs = np.empty((len(parameters), len(OUTPUT_NAMES)), dtype=np.float64)
    minimum_eigenvalues = np.empty(len(parameters), dtype=np.float64)
    started = time.perf_counter()
    for start in range(0, len(parameters), int(chunk_size)):
        stop = min(start + int(chunk_size), len(parameters))
        coefficients = _material_coefficients_batch(parameters[start:stop])
        K = np.einsum("nq,qij->nij", coefficients, Kq, optimize=True)
        B = np.einsum("nq,qij->nij", coefficients, Bq, optimize=True)
        D = np.einsum("nq,qij->nij", coefficients, Dq, optimize=True)
        amplitudes = np.linalg.solve(K, -B)
        C = D + np.einsum("nri,nrj->nij", B, amplitudes, optimize=True)
        C = 0.5 * (C + np.swapaxes(C, -1, -2))
        batch_outputs, batch_eigenvalues = _engineering_constants_batch(C)
        outputs[start:stop] = batch_outputs
        minimum_eigenvalues[start:stop] = batch_eigenvalues
    return outputs, minimum_eigenvalues, float(time.perf_counter() - started)


def _samples(kind: str, count: int, seed: int) -> np.ndarray:
    lower = np.array([sweep.MATERIAL_BOUNDS[name][0] for name in PARAMETER_NAMES])
    upper = np.array([sweep.MATERIAL_BOUNDS[name][1] for name in PARAMETER_NAMES])
    rng = np.random.default_rng(int(seed))
    if kind == "global_uniform":
        return rng.uniform(lower, upper, size=(count, len(PARAMETER_NAMES)))
    raise ValueError(f"Unknown distribution kind: {kind}")


def _summaries(kind: str, outputs: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for column, name in enumerate(OUTPUT_NAMES):
        values = outputs[:, column]
        rows.append(
            {
                "distribution": kind,
                "output": name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "q01": float(np.quantile(values, 0.01)),
                "q05": float(np.quantile(values, 0.05)),
                "q50": float(np.quantile(values, 0.50)),
                "q95": float(np.quantile(values, 0.95)),
                "q99": float(np.quantile(values, 0.99)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    convergence_rows = []
    checkpoints = sorted(
        {
            min(len(outputs), value)
            for value in (100, 300, 1000, 3000, 10000, 30000, 50000, 100000)
        }
    )
    for count in checkpoints:
        for column, name in enumerate(OUTPUT_NAMES):
            values = outputs[:count, column]
            final_values = outputs[:, column]
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if count > 1 else 0.0
            q05 = float(np.quantile(values, 0.05))
            q95 = float(np.quantile(values, 0.95))
            final_mean = float(np.mean(final_values))
            final_q05 = float(np.quantile(final_values, 0.05))
            final_q95 = float(np.quantile(final_values, 0.95))
            scale = max(abs(final_mean), np.finfo(float).eps)
            convergence_rows.append(
                {
                    "distribution": kind,
                    "sample_count": int(count),
                    "output": name,
                    "mean": mean,
                    "std": std,
                    "mean_standard_error": std / np.sqrt(float(count)),
                    "mean_ci95_low": mean - 1.96 * std / np.sqrt(float(count)),
                    "mean_ci95_high": mean + 1.96 * std / np.sqrt(float(count)),
                    "q05": q05,
                    "q95": q95,
                    "final_sample_count": int(len(outputs)),
                    "final_mean": final_mean,
                    "final_q05": final_q05,
                    "final_q95": final_q95,
                    "mean_relative_change_vs_final": abs(mean - final_mean) / scale,
                    "q05_relative_change_vs_final": abs(q05 - final_q05) / scale,
                    "q95_relative_change_vs_final": abs(q95 - final_q95) / scale,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(convergence_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom-path", type=Path, required=True, help="Path to reduced_operators.npz")
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.rom_path.resolve()) as payload:
        Kq = np.asarray(payload["Kq"], dtype=np.float64)
        Bq = np.asarray(payload["Bq"], dtype=np.float64)
        Dq = np.asarray(payload["Dq"], dtype=np.float64)
    summary_frames = []
    convergence_frames = []
    manifest = {
        "rom_path": str(args.rom_path.resolve()),
        "basis_rank": int(Kq.shape[1]),
        "physical_dimension": 7,
        "sample_count_each": int(args.samples),
        "local_sigma_rule": "(upper-lower)/6",
        "distributions": {},
    }
    for offset, kind in enumerate(["global_uniform"]):
        artifact = out_dir / f"{kind}_samples.npz"
        if artifact.is_file() and not args.overwrite:
            with np.load(artifact) as payload:
                parameters = payload["parameters"]
                outputs = payload["outputs"]
                minimum_eigenvalues = payload["minimum_eigenvalues"]
            wall = float("nan")
        else:
            parameters = _samples(kind, int(args.samples), int(args.seed) + offset)
            outputs, minimum_eigenvalues, wall = _evaluate_rom_batch(
                parameters,
                Kq=Kq,
                Bq=Bq,
                Dq=Dq,
                chunk_size=int(args.chunk_size),
            )
            if np.any(minimum_eigenvalues <= 0.0) or not np.all(np.isfinite(outputs)):
                raise RuntimeError(f"UQ {kind} produjo tensores no SPD o salidas no finitas.")
            np.savez_compressed(
                artifact,
                parameters=parameters,
                outputs=outputs,
                minimum_eigenvalues=minimum_eigenvalues,
                parameter_names=np.asarray(PARAMETER_NAMES),
                output_names=np.asarray(OUTPUT_NAMES),
            )
        summary, convergence = _summaries(kind, outputs)
        summary_frames.append(summary)
        convergence_frames.append(convergence)
        manifest["distributions"][kind] = {
            "artifact": str(artifact),
            "wall_s": wall,
            "minimum_tensor_eigenvalue": float(np.min(minimum_eigenvalues)),
            "spd_failures": int(np.count_nonzero(minimum_eigenvalues <= 0.0)),
        }
    pd.concat(summary_frames, ignore_index=True).to_csv(out_dir / "uq_summary.csv", index=False)
    pd.concat(convergence_frames, ignore_index=True).to_csv(out_dir / "uq_convergence.csv", index=False)
    (out_dir / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[UQ] listo | rank={Kq.shape[1]} | samples={2 * int(args.samples)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
