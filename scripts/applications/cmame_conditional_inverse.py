#!/usr/bin/env python3
"""Noise study for a conditionally identifiable ROM inverse problem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import inverse_identification_utils as inverse


OUT_DEFAULT = ROOT / "results" / "cmame_method" / "conditional_inverse"


def _fit(
    *,
    observed_C: np.ndarray,
    target_C: np.ndarray,
    fixed_unit: np.ndarray,
    unknown_indices: np.ndarray,
    starts: list[np.ndarray],
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    coeff_jac_rel_step: float,
    max_nfev: int,
    domain_eps: float,
) -> tuple[Any, np.ndarray, np.ndarray, float]:
    target_norm = float(np.linalg.norm(target_C))

    def expand(active: np.ndarray) -> np.ndarray:
        unit = fixed_unit.copy()
        unit[unknown_indices] = active
        return unit

    def residual(active: np.ndarray) -> np.ndarray:
        C_rom, _, _ = inverse._rom_model_and_jacobian(
            expand(active), Kq=Kq, Bq=Bq, Dq=Dq,
            coeff_jac_rel_step=coeff_jac_rel_step,
        )
        return inverse._weighted_symmetric_vector(C_rom - observed_C) / target_norm

    def jacobian(active: np.ndarray) -> np.ndarray:
        _, gradients, _ = inverse._rom_model_and_jacobian(
            expand(active), Kq=Kq, Bq=Bq, Dq=Dq,
            coeff_jac_rel_step=coeff_jac_rel_step,
        )
        return np.stack(
            [
                inverse._weighted_symmetric_vector(gradients[index]) / target_norm
                for index in unknown_indices
            ],
            axis=1,
        )

    candidates: list[tuple[Any, float]] = []
    for start in starts:
        started = time.perf_counter()
        result = least_squares(
            residual,
            np.clip(start, domain_eps, 1.0 - domain_eps),
            jac=jacobian,
            bounds=(
                np.full(len(unknown_indices), domain_eps),
                np.full(len(unknown_indices), 1.0 - domain_eps),
            ),
            method="trf",
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
            x_scale="jac",
            max_nfev=int(max_nfev),
        )
        candidates.append((result, float(time.perf_counter() - started)))
    result, wall_s = min(candidates, key=lambda item: float(item[0].cost))
    recovered_unit = expand(result.x)
    recovered_C, _, _ = inverse._rom_model_and_jacobian(
        recovered_unit, Kq=Kq, Bq=Bq, Dq=Dq,
        coeff_jac_rel_step=coeff_jac_rel_step,
    )
    J = jacobian(result.x)
    return result, recovered_unit, recovered_C, wall_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom-dir", type=Path, default=inverse.ROM_DEFAULT)
    parser.add_argument("--target-csv", type=Path, default=inverse.TARGET_CSV_DEFAULT)
    parser.add_argument("--target-material-id", type=int, default=94)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--unknown", nargs="+", default=["Ef_L", "Ef_T"])
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.001, 0.005, 0.01])
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--coeff-jac-rel-step", type=float, default=1.0e-5)
    parser.add_argument("--domain-eps", type=float, default=1.0e-8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    unknown = tuple(str(name) for name in args.unknown)
    invalid = sorted(set(unknown) - set(inverse.PHYSICAL_NAMES))
    if invalid or len(set(unknown)) != len(unknown):
        raise ValueError(f"Invalid or repeated unknown parameters: {invalid or unknown}")
    unknown_indices = np.asarray(
        [inverse.PHYSICAL_NAMES.index(name) for name in unknown], dtype=int
    )
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "conditional_inverse_rows.csv"
    if rows_path.is_file() and not args.overwrite:
        raise FileExistsError(f"Already exists: {rows_path}; use --overwrite.")

    Kq, Bq, Dq, rom_manifest = inverse._load_operators(args.rom_dir.resolve())
    target_row = inverse._load_target_row(
        args.target_csv.resolve(), int(args.target_material_id)
    )
    target_C = inverse._ceff_from_row(target_row)
    target_norm = float(np.linalg.norm(target_C))
    target_unit = inverse._unit_from_physical(inverse._sampled_from_row(target_row))
    starts = [
        target_unit[unknown_indices],
        np.full(len(unknown), 0.5),
        np.full(len(unknown), 0.25),
        np.full(len(unknown), 0.75),
    ]

    rows: list[dict[str, Any]] = []
    for noise_index, noise_level in enumerate(args.noise_levels):
        for replicate in range(int(args.replicates)):
            noise_seed = int(args.seed) + 1000 * noise_index + replicate
            observed_C = target_C + inverse._noise_matrix(
                target_C, float(noise_level), np.random.default_rng(noise_seed)
            )
            result, recovered_unit, recovered_C, wall_s = _fit(
                observed_C=observed_C,
                target_C=target_C,
                fixed_unit=target_unit,
                unknown_indices=unknown_indices,
                starts=starts,
                Kq=Kq,
                Bq=Bq,
                Dq=Dq,
                coeff_jac_rel_step=float(args.coeff_jac_rel_step),
                max_nfev=int(args.max_nfev),
                domain_eps=float(args.domain_eps),
            )
            _, gradients, _ = inverse._rom_model_and_jacobian(
                recovered_unit, Kq=Kq, Bq=Bq, Dq=Dq,
                coeff_jac_rel_step=float(args.coeff_jac_rel_step),
            )
            J = np.stack(
                [
                    inverse._weighted_symmetric_vector(gradients[index]) / target_norm
                    for index in unknown_indices
                ],
                axis=1,
            )
            singular_values = np.linalg.svd(J, compute_uv=False)
            active_error = recovered_unit[unknown_indices] - target_unit[unknown_indices]
            row: dict[str, Any] = {
                "noise_level": float(noise_level),
                "noise_percent": 100.0 * float(noise_level),
                "replicate": int(replicate),
                "noise_seed": int(noise_seed),
                "success": bool(result.success),
                "nfev": int(result.nfev),
                "wall_s": float(wall_s),
                "active_parameter_rmse": float(np.sqrt(np.mean(active_error**2))),
                "active_parameter_max_abs_error": float(np.max(np.abs(active_error))),
                "true_tensor_relative_error": float(
                    np.linalg.norm(recovered_C - target_C) / target_norm
                ),
                "data_fit_relative": float(
                    np.linalg.norm(recovered_C - observed_C) / target_norm
                ),
                "jacobian_condition": float(singular_values[0] / singular_values[-1]),
                "jacobian_min_singular_value": float(singular_values[-1]),
            }
            recovered = inverse._physical_from_unit(recovered_unit)
            target = inverse._physical_from_unit(target_unit)
            for name in unknown:
                row[f"target_{name}"] = float(target[name])
                row[f"recovered_{name}"] = float(recovered[name])
            rows.append(row)
            pd.DataFrame(rows).to_csv(rows_path, index=False)

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["noise_level", "noise_percent"], as_index=False)
        .agg(
            replicate_count=("replicate", "count"),
            success_fraction=("success", "mean"),
            active_parameter_rmse_median=("active_parameter_rmse", "median"),
            active_parameter_rmse_p95=("active_parameter_rmse", lambda x: x.quantile(0.95)),
            active_parameter_max_error=("active_parameter_max_abs_error", "max"),
            true_tensor_error_median=("true_tensor_relative_error", "median"),
            true_tensor_error_p95=("true_tensor_relative_error", lambda x: x.quantile(0.95)),
            data_fit_median=("data_fit_relative", "median"),
            jacobian_condition_median=("jacobian_condition", "median"),
            wall_s_median=("wall_s", "median"),
        )
    )
    summary.to_csv(out_dir / "conditional_inverse_summary.csv", index=False)
    manifest = {
        "status": "complete",
        "rom_dir": str(args.rom_dir.resolve()),
        "rom_rank": int(rom_manifest.get("basis_rank", Kq.shape[1])),
        "target_csv": str(args.target_csv.resolve()),
        "target_material_id": int(args.target_material_id),
        "unknown_parameters": list(unknown),
        "known_parameters": [name for name in inverse.PHYSICAL_NAMES if name not in unknown],
        "known_parameter_policy": "fixed at independently characterized synthetic truth values",
        "noise_levels": [float(value) for value in args.noise_levels],
        "replicates": int(args.replicates),
        "starts": len(starts),
        "full_order_solves_run": 0,
        "selection_policy": "minimum unregularized least-squares cost across four fixed starts",
    }
    (out_dir / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[CONDITIONAL-INVERSE] complete | rows={len(results)} | out={out_dir}")
    return 0


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        raise SystemExit(main())
