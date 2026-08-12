#!/usr/bin/env python3
"""Inverse constituent identification using ROM tensor sensitivities."""

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
from scipy.optimize import least_squares
from scipy.stats import qmc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
RUN_DEFAULT = (
    PROJECT_ROOT
    / "results"
    / "fixed_geometry_ffthompy"
    / "fixed_geometry_ar15_vf20_sobol8_center_fields"
)
ROM_DEFAULT = (
    RUN_DEFAULT
    / "rom_tangential_r168_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_"
    "v1_v6_v7_v8_v9_v10_v11_v12_v13_v14_v16_v17_v22_v23_v24_v25_v30_v31_basis"
)
TARGET_CSV_DEFAULT = (
    RUN_DEFAULT
    / "independent_validation_r168_sobol128_seed20261217_gpu"
    / "validation_full_order_results.csv"
)
ROM_VALIDATION_CSV_DEFAULT = (
    RUN_DEFAULT
    / "independent_validation_r168_sobol128_seed20261217_gpu"
    / "rom_validation_results.csv"
)
OUT_NAME_DEFAULT = "inverse_identification_r168_validation128_m94"

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import fft_homogenization_solver as sweep
import rom_reduced_operator as reduced
import sensitivity_verification as sens


PHYSICAL_NAMES = sens.PHYSICAL_NAMES
UPPER_TRIANGLE = np.triu_indices(6)


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


def _make_out_dir(
    run_dir: Path,
    out_name: str,
    *,
    out_dir: Path | None,
    overwrite: bool,
) -> Path:
    out_dir = out_dir.resolve() if out_dir is not None else run_dir / out_name
    if out_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Ya existe {out_dir}. Usa --overwrite para regenerar la demo."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _weighted_symmetric_vector(C: np.ndarray) -> np.ndarray:
    """Return Mandel 21-vector for 6x6 matrix C."""
    C_sym = 0.5 * (C + C.T)
    diag = np.diag(C_sym)
    off = np.array([
        C_sym[0, 1], C_sym[0, 2], C_sym[0, 3], C_sym[0, 4], C_sym[0, 5],
        C_sym[1, 2], C_sym[1, 3], C_sym[1, 4], C_sym[1, 5],
        C_sym[2, 3], C_sym[2, 4], C_sym[2, 5],
        C_sym[3, 4], C_sym[3, 5],
        C_sym[4, 5],
    ])
    return np.concatenate([diag, np.sqrt(2.0) * off])


def _noise_matrix(scale_C: np.ndarray, noise_percent: float, rng: np.random.Generator) -> np.ndarray:
    """Symmetric 6x6 Gaussian noise matrix with relative Frobenius norm = noise_percent."""
    raw = rng.normal(0.0, 1.0, size=(6, 6))
    sym = 0.5 * (raw + raw.T)
    norm_sym = float(np.linalg.norm(sym))
    if norm_sym > 0:
        sym = sym / norm_sym
    return float(noise_percent) * float(np.linalg.norm(scale_C)) * sym


def _ceff_from_row(row: pd.Series | dict[str, Any], *, prefix: str = "Ceff") -> np.ndarray:
    matrix = np.zeros((6, 6), dtype=float)
    for ii in range(6):
        for jj in range(6):
            column = f"{prefix}_{ii + 1}{jj + 1}"
            if column not in row:
                raise KeyError(f"Falta columna {column}.")
            matrix[ii, jj] = float(row[column])
    return 0.5 * (matrix + matrix.T)


def _sampled_from_row(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    return {name: float(row[name]) for name in PHYSICAL_NAMES}


def _bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([sweep.MATERIAL_BOUNDS[name][0] for name in PHYSICAL_NAMES], dtype=float)
    upper = np.array([sweep.MATERIAL_BOUNDS[name][1] for name in PHYSICAL_NAMES], dtype=float)
    return lower, upper


def _physical_from_unit(unit: np.ndarray) -> dict[str, float]:
    lower, upper = _bounds_arrays()
    values = lower + np.asarray(unit, dtype=float) * (upper - lower)
    return {name: float(value) for name, value in zip(PHYSICAL_NAMES, values)}


def _unit_from_physical(sampled: dict[str, float]) -> np.ndarray:
    lower, upper = _bounds_arrays()
    values = np.array([sampled[name] for name in PHYSICAL_NAMES], dtype=float)
    return (values - lower) / (upper - lower)


def _material_row_from_unit(unit: np.ndarray) -> dict[str, Any]:
    sampled = _physical_from_unit(unit)
    material = sweep._material_derived(sampled)
    sweep._validate_material(material)
    return {
        "material_id": 0,
        "material_label": "inverse_trial",
        **material,
    }


def _load_target_row(target_csv: Path, target_material_id: int) -> pd.Series:
    df = pd.read_csv(target_csv)
    if "material_id" not in df.columns:
        raise KeyError(f"{target_csv} no contiene material_id.")
    matches = df[df["material_id"].astype(int) == int(target_material_id)]
    if matches.empty:
        raise ValueError(
            f"No existe material_id={target_material_id} en {target_csv}."
        )
    return matches.iloc[0]


def _load_operators(rom_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = _load_json(rom_dir / "rom_manifest.json")
    with np.load(rom_dir / "reduced_operators.npz") as payload:
        Kq = np.asarray(payload["Kq"], dtype=float)
        Bq = np.asarray(payload["Bq"], dtype=float)
        Dq = np.asarray(payload["Dq"], dtype=float)
    return Kq, Bq, Dq, manifest


def _rom_model_and_jacobian(
    unit: np.ndarray,
    *,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    coeff_jac_rel_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    material = _material_row_from_unit(unit)
    coeffs = reduced._material_coefficients(material)
    C_rom, affine_gradients = sens._rom_affine_sensitivities(coeffs, Kq, Bq, Dq)
    sampled = _sampled_from_row(material)
    coeff_jac_phys, coeff_steps = sens._coefficient_jacobian_physical(
        sampled,
        rel_step=float(coeff_jac_rel_step),
    )
    # affine_gradients[q] = dC/dxi_q, coeff_jac_phys[q,j] = dxi_q/dp_j.
    physical_gradients = np.einsum(
        "qab,qj->jab",
        affine_gradients,
        coeff_jac_phys,
        optimize=True,
    )
    lower, upper = _bounds_arrays()
    unit_gradients = physical_gradients * (upper - lower)[:, None, None]
    return C_rom, unit_gradients, coeff_steps


def _build_starts(n_starts: int, seed: int) -> list[tuple[str, np.ndarray]]:
    starts: list[tuple[str, np.ndarray]] = [("center", np.full(len(PHYSICAL_NAMES), 0.5))]
    if n_starts <= 1:
        return starts
    sampler = qmc.Sobol(d=len(PHYSICAL_NAMES), scramble=True, seed=int(seed))
    power = int(math.ceil(math.log2(max(1, n_starts - 1))))
    unit = sampler.random_base2(m=power)[: n_starts - 1]
    for idx, row in enumerate(unit, start=1):
        starts.append((f"sobol_{idx:02d}", np.asarray(row, dtype=float)))
    return starts


def _fit_from_start(
    *,
    start_label: str,
    x0: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    target_C: np.ndarray,
    target_norm: float,
    coeff_jac_rel_step: float,
    max_nfev: int,
    domain_eps: float,
) -> tuple[dict[str, Any], least_squares]:
    def residual(unit: np.ndarray) -> np.ndarray:
        C_rom, _, _ = _rom_model_and_jacobian(
            unit,
            Kq=Kq,
            Bq=Bq,
            Dq=Dq,
            coeff_jac_rel_step=coeff_jac_rel_step,
        )
        return (C_rom - target_C)[UPPER_TRIANGLE] / target_norm

    def jacobian(unit: np.ndarray) -> np.ndarray:
        _, unit_gradients, _ = _rom_model_and_jacobian(
            unit,
            Kq=Kq,
            Bq=Bq,
            Dq=Dq,
            coeff_jac_rel_step=coeff_jac_rel_step,
        )
        return np.stack(
            [unit_gradients[j][UPPER_TRIANGLE] / target_norm for j in range(len(PHYSICAL_NAMES))],
            axis=1,
        )

    start_residual_norm = float(np.linalg.norm(residual(x0)))
    t0 = time.perf_counter()
    result = least_squares(
        residual,
        np.clip(x0, domain_eps, 1.0 - domain_eps),
        jac=jacobian,
        bounds=(np.full(len(PHYSICAL_NAMES), domain_eps), np.full(len(PHYSICAL_NAMES), 1.0 - domain_eps)),
        method="trf",
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        x_scale="jac",
        max_nfev=int(max_nfev),
    )
    elapsed = float(time.perf_counter() - t0)
    C_final, _, coeff_steps = _rom_model_and_jacobian(
        result.x,
        Kq=Kq,
        Bq=Bq,
        Dq=Dq,
        coeff_jac_rel_step=coeff_jac_rel_step,
    )
    fit_rel = float(np.linalg.norm(C_final - target_C) / target_norm)
    row: dict[str, Any] = {
        "start_label": start_label,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "njev": int(result.njev) if result.njev is not None else 0,
        "wall_s": elapsed,
        "initial_residual_norm": start_residual_norm,
        "final_upper_residual_norm": float(np.linalg.norm(result.fun)),
        "final_tensor_relative_error": fit_rel,
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "active_mask_count": int(np.count_nonzero(result.active_mask)),
    }
    for name, start_value, final_value, step in zip(
        PHYSICAL_NAMES,
        _physical_from_unit(x0).values(),
        _physical_from_unit(result.x).values(),
        coeff_steps,
    ):
        row[f"start_{name}"] = float(start_value)
        row[f"final_{name}"] = float(final_value)
        row[f"coeff_fd_step_{name}"] = float(step)
    return row, result


def _target_rom_error(
    *,
    rom_validation_csv: Path,
    target_material_id: int,
    fallback_error: float,
) -> float:
    if not rom_validation_csv.is_file():
        return fallback_error
    df = pd.read_csv(rom_validation_csv)
    if "material_id" not in df.columns or "relative_frobenius_error" not in df.columns:
        return fallback_error
    matches = df[df["material_id"].astype(int) == int(target_material_id)]
    if matches.empty:
        return fallback_error
    return float(matches.iloc[0]["relative_frobenius_error"])


def _markdown(summary: dict[str, Any], parameter_rows: pd.DataFrame) -> str:
    lines = [
        "# ROM Sensitivity Inverse Identification Demo",
        "",
        f"- ROM: `{summary['rom_dir']}`",
        f"- Target CSV: `{summary['target_csv']}`",
        f"- Target material ID: `{summary['target_material_id']}`",
        f"- Multistarts: `{summary['n_starts']}`",
        f"- Full-order solves run: `0`",
        "",
        "## Fit",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Best start | `{summary['best_start_label']}` |",
        f"| Initial residual norm | {summary['best_initial_residual_norm']:.6e} |",
        f"| Final tensor fit relative error | {summary['best_final_tensor_relative_error']:.6e} |",
        f"| ROM model mismatch at true target | {summary['target_rom_vs_fom_relative_error']:.6e} |",
        f"| Objective cost | {summary['best_cost']:.6e} |",
        f"| Function evaluations | {summary['best_nfev']} |",
        f"| Wall time | {summary['best_wall_s']:.3f} s |",
        f"| Normalized Jacobian rank | {summary['normalized_jacobian_rank']} / 7 |",
        f"| Normalized Jacobian condition | {summary['normalized_jacobian_condition']:.6e} |",
        "",
        "## Recovered Parameters",
        "",
        "| Parameter | Target | Recovered | Abs. error | Rel. error |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in parameter_rows.iterrows():
        lines.append(
            f"| {row['parameter']} | "
            f"{row['target']:.6g} | "
            f"{row['recovered']:.6g} | "
            f"{row['absolute_error']:.6e} | "
            f"{row['relative_error']:.6e} |"
        )
    lines.extend(
        [
            "",
            "This is an inverse identification demo, not a uniqueness proof. A small "
            "tensor residual with larger parameter spread indicates the homogenized "
            "tensor alone may be locally ill-conditioned for some constituent "
            "directions.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inverse constituent identification using ROM sensitivities."
    )
    parser.add_argument("--run-dir", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--rom-dir", type=Path, default=ROM_DEFAULT)
    parser.add_argument("--target-csv", type=Path, default=TARGET_CSV_DEFAULT)
    parser.add_argument("--rom-validation-csv", type=Path, default=ROM_VALIDATION_CSV_DEFAULT)
    parser.add_argument("--target-material-id", type=int, default=94)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--start-seed", type=int, default=20270203)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--coeff-jac-rel-step", type=float, default=1e-5)
    parser.add_argument("--domain-eps", type=float, default=1e-8)
    parser.add_argument("--out-name", default=OUT_NAME_DEFAULT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Explicit output directory; overrides --run-dir/--out-name.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rom_dir = args.rom_dir.resolve()
    target_csv = args.target_csv.resolve()
    out_dir = _make_out_dir(
        run_dir,
        str(args.out_name),
        out_dir=args.out_dir,
        overwrite=bool(args.overwrite),
    )

    Kq, Bq, Dq, rom_manifest = _load_operators(rom_dir)
    target_row = _load_target_row(target_csv, int(args.target_material_id))
    target_C = _ceff_from_row(target_row, prefix="Ceff")
    target_norm = float(np.linalg.norm(target_C))
    target_sampled = _sampled_from_row(target_row)
    target_unit = _unit_from_physical(target_sampled)

    target_coeffs = reduced._material_coefficients(target_row.to_dict())
    target_C_rom, _, _ = reduced._rom_ceff(target_coeffs, Kq, Bq, Dq)
    fallback_rom_error = float(np.linalg.norm(target_C_rom - target_C) / target_norm)
    target_rom_error = _target_rom_error(
        rom_validation_csv=args.rom_validation_csv.resolve(),
        target_material_id=int(args.target_material_id),
        fallback_error=fallback_rom_error,
    )

    starts = _build_starts(int(args.starts), int(args.start_seed))
    rows: list[dict[str, Any]] = []
    for label, x0 in starts:
        row, _ = _fit_from_start(
            start_label=label,
            x0=x0,
            Kq=Kq,
            Bq=Bq,
            Dq=Dq,
            target_C=target_C,
            target_norm=target_norm,
            coeff_jac_rel_step=float(args.coeff_jac_rel_step),
            max_nfev=int(args.max_nfev),
            domain_eps=float(args.domain_eps),
        )
        rows.append(row)
        print(
            "[INVERSE] "
            f"{label} | fit={row['final_tensor_relative_error']:.3e} | "
            f"nfev={row['nfev']} | success={row['success']}",
            flush=True,
        )

    results_df = pd.DataFrame(rows).sort_values("final_tensor_relative_error").reset_index(drop=True)
    best = results_df.iloc[0].to_dict()
    recovered = {name: float(best[f"final_{name}"]) for name in PHYSICAL_NAMES}
    recovered_unit = _unit_from_physical(recovered)
    _, unit_gradients, _ = _rom_model_and_jacobian(
        recovered_unit,
        Kq=Kq,
        Bq=Bq,
        Dq=Dq,
        coeff_jac_rel_step=float(args.coeff_jac_rel_step),
    )
    normalized_jacobian = np.stack(
        [unit_gradients[j][UPPER_TRIANGLE] / target_norm for j in range(len(PHYSICAL_NAMES))],
        axis=1,
    )
    singular_values = np.linalg.svd(normalized_jacobian, compute_uv=False)
    svd_tolerance = float(
        max(normalized_jacobian.shape)
        * np.finfo(float).eps
        * singular_values[0]
    )
    jacobian_rank = int(np.count_nonzero(singular_values > svd_tolerance))
    jacobian_condition = float(
        singular_values[0] / singular_values[-1]
        if singular_values[-1] > svd_tolerance
        else math.inf
    )
    parameter_rows = []
    for name in PHYSICAL_NAMES:
        target_value = float(target_sampled[name])
        recovered_value = float(recovered[name])
        absolute = abs(recovered_value - target_value)
        relative = absolute / max(abs(target_value), np.finfo(float).eps)
        parameter_rows.append(
            {
                "parameter": name,
                "target": target_value,
                "recovered": recovered_value,
                "absolute_error": absolute,
                "relative_error": relative,
                "target_unit": float(target_unit[PHYSICAL_NAMES.index(name)]),
                "recovered_unit": float(recovered_unit[PHYSICAL_NAMES.index(name)]),
            }
        )
    parameter_df = pd.DataFrame(parameter_rows)

    summary = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "rom_dir": str(rom_dir),
        "rom_rank": int(rom_manifest.get("basis_rank", Kq.shape[1])),
        "target_csv": str(target_csv),
        "target_material_id": int(args.target_material_id),
        "target_material_label": str(target_row.get("material_label", "")),
        "n_starts": int(len(starts)),
        "start_seed": int(args.start_seed),
        "max_nfev": int(args.max_nfev),
        "coeff_jac_rel_step": float(args.coeff_jac_rel_step),
        "domain_eps": float(args.domain_eps),
        "target_rom_vs_fom_relative_error": float(target_rom_error),
        "fallback_rom_error_direct": float(fallback_rom_error),
        "best_start_label": str(best["start_label"]),
        "best_initial_residual_norm": float(best["initial_residual_norm"]),
        "best_final_upper_residual_norm": float(best["final_upper_residual_norm"]),
        "best_final_tensor_relative_error": float(best["final_tensor_relative_error"]),
        "best_cost": float(best["cost"]),
        "best_nfev": int(best["nfev"]),
        "best_wall_s": float(best["wall_s"]),
        "max_parameter_relative_error": float(parameter_df["relative_error"].max()),
        "mean_parameter_relative_error": float(parameter_df["relative_error"].mean()),
        "normalized_jacobian_shape": list(normalized_jacobian.shape),
        "normalized_jacobian_rank": jacobian_rank,
        "normalized_jacobian_singular_values": singular_values.tolist(),
        "normalized_jacobian_condition": jacobian_condition,
        "normalized_jacobian_rank_tolerance": svd_tolerance,
        "full_order_solves_run": 0,
        "note": (
            "The optimizer uses the affine ROM derivative dC/dxi and a finite-"
            "difference Jacobian of the physical-to-affine coefficient map."
        ),
    }

    results_df.to_csv(out_dir / "inverse_multistart_results.csv", index=False)
    results_df.to_excel(out_dir / "inverse_multistart_results.xlsx", index=False)
    parameter_df.to_csv(out_dir / "inverse_parameter_recovery.csv", index=False)
    parameter_df.to_excel(out_dir / "inverse_parameter_recovery.xlsx", index=False)
    pd.DataFrame(
        normalized_jacobian,
        columns=PHYSICAL_NAMES,
    ).to_csv(out_dir / "inverse_normalized_jacobian.csv", index=False)
    pd.DataFrame(
        {
            "index": np.arange(1, singular_values.size + 1),
            "singular_value": singular_values,
        }
    ).to_csv(out_dir / "inverse_jacobian_singular_values.csv", index=False)
    _write_json(out_dir / "inverse_identification_summary.json", summary)
    (out_dir / "inverse_identification_summary.md").write_text(
        _markdown(summary, parameter_df),
        encoding="utf-8",
    )

    print(
        "[INVERSE] listo | "
        f"out={out_dir} | target={int(args.target_material_id)} | "
        f"fit={summary['best_final_tensor_relative_error']:.3e} | "
        f"rom_mismatch={summary['target_rom_vs_fom_relative_error']:.3e} | "
        f"max_param_rel={summary['max_parameter_relative_error']:.3e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
