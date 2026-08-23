#!/usr/bin/env python3
"""Audit full-span reference-energy QR from an existing raw-Ritz archive.

This experimental route performs no FOM solve and no voxel-scale pass.  It
uses the cached affine blocks ``S.T @ K_q @ S`` to construct a weighted QR
coordinate system for exactly the same snapshot span.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DEFAULT = SCRIPT_DIR / "campaign_config.json"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from env_bootstrap import ensure_configured_venv


if __name__ == "__main__":
    ensure_configured_venv(CONFIG_DEFAULT)

import numpy as np
import pandas as pd


ROOT = SCRIPT_DIR.parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rom_reduced_operator as reduced


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_columns() -> list[str]:
    return [f"Crom_{i}{j}" for i in range(1, 7) for j in range(1, 7)]


def _tensor_difference(reference: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, float | int]:
    columns = _tensor_columns()
    left = reference[columns].to_numpy(dtype=np.float64)
    right = comparison[columns].to_numpy(dtype=np.float64)
    relative = np.linalg.norm(right - left, axis=1) / np.maximum(
        np.linalg.norm(left, axis=1), np.finfo(float).tiny
    )
    return {
        "count": int(len(relative)),
        "relative_difference_mean": float(np.mean(relative)),
        "relative_difference_p95": float(np.quantile(relative, 0.95)),
        "relative_difference_max": float(np.max(relative)),
    }


def _rom_stats(frame: pd.DataFrame) -> dict[str, float | int]:
    errors = frame["relative_frobenius_error"].to_numpy(dtype=np.float64)
    return {
        "count": int(len(frame)),
        "numerical_failure_count": int(frame["rom_numerical_failure"].sum()),
        "error_mean": float(np.mean(errors)),
        "error_p95": float(np.quantile(errors, 0.95)),
        "error_max": float(np.max(errors)),
        "reduced_K_spectral_spd_margin_min": float(
            frame["reduced_K_spectral_spd_margin"].min()
        ),
    }


def _condition_stats(coefficients: np.ndarray, Kq: np.ndarray) -> dict[str, float]:
    matrices = np.einsum("nq,qij->nij", coefficients, Kq, optimize=True)
    matrices = 0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    eigenvalues = np.linalg.eigvalsh(matrices)
    condition = eigenvalues[:, -1] / eigenvalues[:, 0]
    return {
        "condition_min": float(np.min(condition)),
        "condition_median": float(np.median(condition)),
        "condition_p95": float(np.quantile(condition, 0.95)),
        "condition_max": float(np.max(condition)),
        "lambda_min_min": float(np.min(eigenvalues[:, 0])),
        "spectral_spd_margin_min": float(np.min(1.0 / condition)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-policy",
        choices=("candidate-mean", "first-candidate"),
        default="candidate-mean",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=0,
        help="Use zero for every row in candidate_pool_used.csv.",
    )
    parser.add_argument("--output-prefix", default="experimental_energy_qr")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    archive_path = run_dir / "reduced_operators.npz"
    candidate_path = run_dir / "candidate_pool_used.csv"
    if not archive_path.is_file() or not candidate_path.is_file():
        raise FileNotFoundError("run directory lacks raw operators or candidate pool")

    summary_path = run_dir / f"{args.output_prefix}_summary.json"
    operators_path = run_dir / f"{args.output_prefix}_operators.npz"
    outputs = [summary_path, operators_path]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("experimental energy-QR outputs already exist")

    with np.load(archive_path) as payload:
        required = ("Kq", "Bq", "Dq", "raw_Kq", "raw_Bq")
        missing = [name for name in required if name not in payload]
        if missing:
            raise KeyError(f"operator archive lacks {missing}")
        nominal_Kq = np.asarray(payload["Kq"], dtype=np.float64)
        nominal_Bq = np.asarray(payload["Bq"], dtype=np.float64)
        Dq = np.asarray(payload["Dq"], dtype=np.float64)
        raw_Kq = np.asarray(payload["raw_Kq"], dtype=np.float64)
        raw_Bq = np.asarray(payload["raw_Bq"], dtype=np.float64)

    candidate_frame = pd.read_csv(candidate_path)
    if int(args.candidate_count) > 0:
        candidate_frame = candidate_frame.iloc[: int(args.candidate_count)].copy()
    parameters = candidate_frame[
        list(reduced.MATERIAL_PARAMETER_COLUMNS)
    ].to_numpy(dtype=np.float64)
    coefficients = reduced._material_coefficients_batch(parameters)
    if args.reference_policy == "candidate-mean":
        reference_coefficients = np.mean(coefficients, axis=0)
    else:
        reference_coefficients = coefficients[0].copy()

    operators, metadata = reduced._experimental_energy_qr_recompile(
        raw_Kq=raw_Kq,
        raw_Bq=raw_Bq,
        Dq=Dq,
        reference_coefficients=reference_coefficients,
    )
    np.savez_compressed(
        operators_path,
        Kq=operators["Kq"],
        Bq=operators["Bq"],
        Dq=operators["Dq"],
        R=operators["R"],
        reference_coefficients=operators["reference_coefficients"],
    )

    validation: dict[str, Any] = {}
    for label, filename in (
        ("monitor", "monitor_truth_results.csv"),
        ("final_validation", "final_validation_truth_results.csv"),
    ):
        truth_path = run_dir / filename
        if not truth_path.is_file():
            continue
        truth = pd.read_csv(truth_path)
        nominal = reduced._evaluate_rom(
            results_df=truth, Kq=nominal_Kq, Bq=nominal_Bq, Dq=Dq
        )
        energy = reduced._evaluate_rom(
            results_df=truth,
            Kq=operators["Kq"],
            Bq=operators["Bq"],
            Dq=operators["Dq"],
        )
        energy.insert(0, f"{label}_id", np.arange(len(energy), dtype=int))
        energy.to_csv(
            run_dir / f"{args.output_prefix}_{label}.csv", index=False
        )
        validation[label] = {
            "energy_qr": _rom_stats(energy),
            "nominal": _rom_stats(nominal),
            "energy_qr_vs_nominal": _tensor_difference(nominal, energy),
        }

    summary = {
        "experimental": True,
        "official_model_modified": False,
        "run_dir": run_dir,
        "source_operator_archive": archive_path,
        "source_operator_sha256": _sha256(archive_path),
        "energy_qr_operator_archive": operators_path,
        "energy_qr_operator_sha256": _sha256(operators_path),
        "reference_policy": str(args.reference_policy),
        "reference_candidate_count": int(len(candidate_frame)),
        "metadata": metadata,
        "candidate_conditioning": {
            "raw_snapshot_coordinates": _condition_stats(coefficients, raw_Kq),
            "nominal_l2_coordinates": _condition_stats(coefficients, nominal_Kq),
            "reference_energy_qr_coordinates": _condition_stats(
                coefficients, operators["Kq"]
            ),
        },
        "validation": validation,
    }
    _write_json(summary_path, summary)
    print(
        "[ENERGY-QR] done | "
        f"run={run_dir.name} | rank={raw_Kq.shape[1]} | "
        f"factor={metadata['energy_qr_factor_wall_s']:.4f}s | "
        "voxel_passes=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
