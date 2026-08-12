#!/usr/bin/env python3
"""High-precision Schur-bound, effectivity, and Loewner hierarchy study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rom_reduced_operator as reduced
from constitutive_transfer.schur_estimator import (
    TwoKernelEstimator,
    hierarchical_matrix_upper_bound,
    isotropic_reference_upper_bounds,
    optimize_convex_reference_envelope,
)


RUN = ROOT / "results" / "fixed_geometry_ffthompy" / "fixed_geometry_ar15_vf20_sobol8_center_fields"
ROM_R48 = RUN / "rom_tangential_r48_center_m1_m2_m3_m5_m7_v13_v3_basis"
ROM_R168 = RUN / (
    "rom_tangential_r168_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_"
    "v6_v7_v8_v9_v10_v11_v12_v13_v14_v16_v17_v22_v23_v24_v25_v30_v31_basis"
)
TRUTH_DEFAULT = ROOT / "results" / "cmame_method" / "bound_truth_r48_v3"
OUT_DEFAULT = ROOT / "results" / "cmame_method" / "bound_study"
BENCHMARK_DEFAULT = (
    ROOT / "results" / "cmame_method" / "causal_comparator" / "candidate_pool.csv"
)
RANK_DIRS = {
    6: "rom_tangential_r6_center",
    12: "rom_tangential_r12_center_m1",
    18: "rom_tangential_r18_center_m1_m3",
    24: "rom_tangential_r24_center_m1_m3_m5",
    30: "rom_tangential_r30_center_m1_m3_m5_m7",
    36: "rom_tangential_r36_center_m1_m2_m3_m5_m7",
    42: "rom_tangential_r42_center_m1_m2_m3_m5_m7_v13",
    48: "rom_tangential_r48_center_m1_m2_m3_m5_m7_v13_v3_basis",
    168: ROM_R168.name,
}


def _matrix_from_row(row: pd.Series, prefix: str) -> np.ndarray:
    matrix = np.array(
        [[float(row[f"{prefix}_{ii + 1}{jj + 1}"]) for jj in range(6)] for ii in range(6)],
        dtype=np.float64,
    )
    return 0.5 * (matrix + matrix.T)


def _phase_stiffnesses(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix_bases = reduced._isotropic_bases()
    matrix = coefficients[0] * matrix_bases[0] + coefficients[1] * matrix_bases[1]
    fiber = sum(
        coefficients[q + 2] * basis
        for q, basis in enumerate(reduced._fiber_local_bases_axis0())
    )
    return 0.5 * (matrix + matrix.T), 0.5 * (fiber + fiber.T)


def _load_rom(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path / "reduced_operators.npz") as payload:
        return tuple(np.asarray(payload[name], dtype=np.float64) for name in ("Kq", "Bq", "Dq"))


def _add_matrix(row: dict[str, object], prefix: str, matrix: np.ndarray) -> None:
    for ii in range(6):
        for jj in range(6):
            row[f"{prefix}_{ii + 1}{jj + 1}"] = float(matrix[ii, jj])


def _plot_loewner(rows: pd.DataFrame, hard_material: int, out_dir: Path) -> None:
    selected = rows.loc[rows["material_id"].astype(int) == int(hard_material)].sort_values("rank")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, 6))
    for eigen_index, color in enumerate(colors, start=1):
        axes[0].plot(
            selected["rank"],
            selected[f"error_eig_{eigen_index}"],
            marker="o",
            color=color,
            label=rf"$\lambda_{eigen_index}$",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Reduced rank $r$")
    axes[0].set_ylabel(r"Eigenvalues of $C_r-C_{\mathrm{FOM}}$")
    axes[0].grid(True, which="both", alpha=0.25)

    transitions = selected.dropna(subset=["previous_rank"])
    x = np.arange(len(transitions))
    for eigen_index, color in enumerate(colors, start=1):
        axes[1].plot(
            x,
            transitions[f"nested_eig_{eigen_index}"],
            marker="s",
            color=color,
        )
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, [f"{int(a)}-{int(b)}" for a, b in zip(transitions.previous_rank, transitions["rank"])], rotation=35)
    axes[1].set_xlabel("Nested rank transition")
    axes[1].set_ylabel(r"Eigenvalues of $C_{r^-}-C_r$")
    axes[1].grid(True, which="both", alpha=0.25)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.035), ncol=6, frameon=False)
    fig.savefig(out_dir / "loewner_eigenvalues.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "loewner_eigenvalues.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _benchmark_hierarchical_online(
    *,
    materials: pd.DataFrame,
    coarse_operators: tuple[np.ndarray, np.ndarray, np.ndarray],
    enriched_operators: tuple[np.ndarray, np.ndarray, np.ndarray],
    enriched_estimator: TwoKernelEstimator,
    beta_margin: float,
    loewner_tolerance: float,
    records_path: Path | None = None,
) -> dict[str, float | int]:
    timings: list[float] = []
    relative_widths: list[float] = []
    records: list[dict[str, float | int]] = []
    for row_index, material in materials.iterrows():
        started = time.perf_counter()
        coefficients = reduced._material_coefficients(material.to_dict())
        C_coarse, _, _ = reduced._rom_ceff(coefficients, *coarse_operators)
        C_enriched, amplitudes, _ = reduced._rom_ceff(
            coefficients, *enriched_operators
        )
        family = isotropic_reference_upper_bounds(
            enriched_estimator.kernel_energy_matrices(coefficients, amplitudes),
            _phase_stiffnesses(coefficients),
            beta_margin=float(beta_margin),
        )
        envelope = optimize_convex_reference_envelope(
            family["upper_bound_matrices"], relative_tolerance=1.0e-12
        )
        upper = hierarchical_matrix_upper_bound(
            C_coarse,
            C_enriched,
            envelope.upper_bound_matrix,
            loewner_tolerance=float(loewner_tolerance),
        )
        width = float(np.linalg.norm(upper) / np.linalg.norm(C_coarse))
        elapsed = float(time.perf_counter() - started)
        relative_widths.append(width)
        timings.append(elapsed)
        records.append(
            {
                "candidate_id": int(material.get("candidate_id", row_index)),
                "relative_certificate_width": width,
                "implied_relative_error_bound": width / max(
                    1.0 - width, np.finfo(float).tiny
                ),
                "certified_query_online_s": elapsed,
            }
        )
    if records_path is not None:
        pd.DataFrame(records).to_csv(records_path, index=False)
    values = np.asarray(timings, dtype=np.float64)
    widths = np.asarray(relative_widths, dtype=np.float64)
    implied_relative_errors = widths / np.maximum(1.0 - widths, np.finfo(float).tiny)
    return {
        "query_count": int(len(values)),
        "total_s": float(np.sum(values)),
        "median_ms": float(1000.0 * np.median(values)),
        "p95_ms": float(1000.0 * np.quantile(values, 0.95)),
        "max_ms": float(1000.0 * np.max(values)),
        "queries_per_s": float(len(values) / np.sum(values)),
        "relative_width_median": float(np.median(widths)),
        "relative_width_p95": float(np.quantile(widths, 0.95)),
        "relative_width_max": float(np.max(widths)),
        "relative_width_below_1e-4_count": int(
            np.sum(implied_relative_errors <= 1.0e-4)
        ),
        "implied_relative_error_bound_median": float(
            np.median(implied_relative_errors)
        ),
        "implied_relative_error_bound_p95": float(
            np.quantile(implied_relative_errors, 0.95)
        ),
        "implied_relative_error_bound_max": float(
            np.max(implied_relative_errors)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-dir", type=Path, default=TRUTH_DEFAULT)
    parser.add_argument("--rom-dir", type=Path, default=ROM_R48)
    parser.add_argument("--enriched-rom-dir", type=Path, default=ROM_R168)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--beta-margin", type=float, default=1.0e-10)
    parser.add_argument("--truth-uncertainty-multiplier", type=float, default=100.0)
    parser.add_argument("--loewner-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--benchmark-csv", type=Path, default=BENCHMARK_DEFAULT)
    parser.add_argument("--benchmark-limit", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    truth_dir = args.truth_dir.resolve()
    rom_dir = args.rom_dir.resolve()
    enriched_rom_dir = args.enriched_rom_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(truth_dir / "validation_full_order_results.csv")
    if not truth["solver_all_converged"].astype(bool).all():
        raise RuntimeError("Every FOM truth solve must declare convergence.")
    if set(truth["solver_real_dtype"].astype(str)) != {"float64"}:
        raise RuntimeError("The strict bound study requires float64 FOM truth.")

    Kq, Bq, Dq = _load_rom(rom_dir)
    estimator = TwoKernelEstimator.load(rom_dir / "online_estimator.npz")
    Kq_enriched, Bq_enriched, Dq_enriched = _load_rom(enriched_rom_dir)
    enriched_estimator = TwoKernelEstimator.load(
        enriched_rom_dir / "online_estimator.npz"
    )
    if enriched_estimator.rank <= estimator.rank:
        raise ValueError("The enriched certification space must have larger rank.")
    bound_rows: list[dict[str, object]] = []
    for _, material in truth.iterrows():
        coefficients = reduced._material_coefficients(material.to_dict())
        C_rom, amplitudes, rom_s = reduced._rom_ceff(coefficients, Kq, Bq, Dq)
        C_enriched, enriched_amplitudes, enriched_rom_s = reduced._rom_ceff(
            coefficients,
            Kq_enriched,
            Bq_enriched,
            Dq_enriched,
        )
        C_fom = _matrix_from_row(material, "Ceff")
        error_matrix = 0.5 * (C_rom - C_fom + (C_rom - C_fom).T)
        enriched_error_matrix = 0.5 * (
            C_enriched - C_fom + (C_enriched - C_fom).T
        )
        started = time.perf_counter()
        kernels = estimator.kernel_energy_matrices(coefficients, amplitudes)
        family = isotropic_reference_upper_bounds(
            kernels,
            _phase_stiffnesses(coefficients),
            beta_margin=float(args.beta_margin),
        )
        single_index = int(np.argmin(family["objectives"]))
        envelope = optimize_convex_reference_envelope(
            family["upper_bound_matrices"],
            relative_tolerance=1.0e-12,
        )
        direct_estimator_s = float(time.perf_counter() - started)
        started = time.perf_counter()
        enriched_kernels = enriched_estimator.kernel_energy_matrices(
            coefficients,
            enriched_amplitudes,
        )
        enriched_family = isotropic_reference_upper_bounds(
            enriched_kernels,
            _phase_stiffnesses(coefficients),
            beta_margin=float(args.beta_margin),
        )
        enriched_single_index = int(np.argmin(enriched_family["objectives"]))
        enriched_envelope = optimize_convex_reference_envelope(
            enriched_family["upper_bound_matrices"],
            relative_tolerance=1.0e-12,
        )
        estimator_s = float(time.perf_counter() - started)
        residual_rel = float(material["solver_max_relative_residual"])
        truth_uncertainty = float(
            args.truth_uncertainty_multiplier * residual_rel * np.linalg.norm(C_fom)
        )
        upper_single = (
            family["upper_bound_matrices"][single_index]
            + truth_uncertainty * np.eye(6)
        )
        direct_upper = envelope.upper_bound_matrix + truth_uncertainty * np.eye(6)
        enriched_tail_upper = (
            enriched_envelope.upper_bound_matrix
            + truth_uncertainty * np.eye(6)
        )
        upper = hierarchical_matrix_upper_bound(
            C_rom,
            C_enriched,
            enriched_tail_upper,
            loewner_tolerance=float(args.loewner_tolerance),
        )
        upper_gap = 0.5 * (upper - error_matrix + (upper - error_matrix).T)
        direct_upper_gap = 0.5 * (
            direct_upper - error_matrix + (direct_upper - error_matrix).T
        )
        enriched_tail_gap = 0.5 * (
            enriched_tail_upper
            - enriched_error_matrix
            + (enriched_tail_upper - enriched_error_matrix).T
        )
        min_eig_error = float(np.min(np.linalg.eigvalsh(error_matrix)))
        min_eig_enriched_error = float(
            np.min(np.linalg.eigvalsh(enriched_error_matrix))
        )
        true_abs = float(np.linalg.norm(error_matrix))
        bound_abs = float(np.linalg.norm(upper))
        direct_bound_abs = float(np.linalg.norm(direct_upper))
        single_bound_abs = float(np.linalg.norm(upper_single))
        bridge_drop = 0.5 * (
            C_rom - C_enriched + (C_rom - C_enriched).T
        )
        row: dict[str, object] = {
            "material_id": int(material["material_id"]),
            "true_error_abs": true_abs,
            "true_error_rel": true_abs / float(np.linalg.norm(C_fom)),
            "bound_abs": bound_abs,
            "bound_rel": bound_abs / float(np.linalg.norm(C_fom)),
            "effectivity": bound_abs / max(true_abs, np.finfo(float).tiny),
            "direct_bound_abs": direct_bound_abs,
            "direct_bound_rel": direct_bound_abs / float(np.linalg.norm(C_fom)),
            "direct_effectivity": direct_bound_abs / max(true_abs, np.finfo(float).tiny),
            "hierarchical_improvement_ratio": direct_bound_abs
            / max(bound_abs, np.finfo(float).tiny),
            "bridge_drop_abs": float(np.linalg.norm(bridge_drop)),
            "bridge_drop_min_eig": float(np.min(np.linalg.eigvalsh(bridge_drop))),
            "enriched_true_error_abs": float(np.linalg.norm(enriched_error_matrix)),
            "enriched_true_error_ratio": float(np.linalg.norm(enriched_error_matrix))
            / max(true_abs, np.finfo(float).tiny),
            "enriched_tail_bound_abs": float(np.linalg.norm(enriched_tail_upper)),
            "enriched_tail_effectivity": float(np.linalg.norm(enriched_tail_upper))
            / max(float(np.linalg.norm(enriched_error_matrix)), np.finfo(float).tiny),
            "single_bound_abs": single_bound_abs,
            "single_bound_rel": single_bound_abs / float(np.linalg.norm(C_fom)),
            "single_effectivity": single_bound_abs / max(true_abs, np.finfo(float).tiny),
            "envelope_improvement_ratio": single_bound_abs
            / max(direct_bound_abs, np.finfo(float).tiny),
            "envelope_active_references": int(len(envelope.active_indices)),
            "envelope_iterations": int(envelope.iterations),
            "envelope_relative_duality_gap": float(envelope.relative_duality_gap),
            "enriched_envelope_active_references": int(
                len(enriched_envelope.active_indices)
            ),
            "enriched_envelope_iterations": int(enriched_envelope.iterations),
            "enriched_envelope_relative_duality_gap": float(
                enriched_envelope.relative_duality_gap
            ),
            "beta": float(family["beta"][single_index]),
            "beta_safe": float(family["beta_safe"][single_index]),
            "reference_index": single_index,
            "reference_poisson": float(family["family"][single_index, 2]),
            "enriched_beta": float(enriched_family["beta"][enriched_single_index]),
            "enriched_beta_safe": float(
                enriched_family["beta_safe"][enriched_single_index]
            ),
            "enriched_reference_index": enriched_single_index,
            "enriched_reference_poisson": float(
                enriched_family["family"][enriched_single_index, 2]
            ),
            "truth_uncertainty_abs": truth_uncertainty,
            "truth_relative_residual": residual_rel,
            "rom_online_s": rom_s,
            "enriched_rom_online_s": enriched_rom_s,
            "direct_estimator_online_s": direct_estimator_s,
            "estimator_online_s": estimator_s,
            "certified_query_online_s": rom_s + enriched_rom_s + estimator_s,
            "min_eig_error": min_eig_error,
            "min_eig_enriched_error": min_eig_enriched_error,
            "min_eig_upper_minus_error": float(np.min(np.linalg.eigvalsh(upper_gap))),
            "direct_min_eig_upper_minus_error": float(
                np.min(np.linalg.eigvalsh(direct_upper_gap))
            ),
            "enriched_tail_min_eig_upper_minus_error": float(
                np.min(np.linalg.eigvalsh(enriched_tail_gap))
            ),
            "strict_violation": bool(
                np.min(np.linalg.eigvalsh(upper_gap)) < -float(args.loewner_tolerance)
            ),
            "direct_strict_violation": bool(
                np.min(np.linalg.eigvalsh(direct_upper_gap))
                < -float(args.loewner_tolerance)
            ),
            "enriched_tail_strict_violation": bool(
                np.min(np.linalg.eigvalsh(enriched_tail_gap))
                < -float(args.loewner_tolerance)
            ),
            "lower_hierarchy_strict_violation": bool(
                min_eig_error < -float(args.loewner_tolerance)
            ),
            "enriched_lower_hierarchy_strict_violation": bool(
                min_eig_enriched_error < -float(args.loewner_tolerance)
            ),
        }
        _add_matrix(row, "error_matrix", error_matrix)
        _add_matrix(row, "upper_bound_matrix", upper)
        _add_matrix(row, "direct_upper_bound_matrix", direct_upper)
        _add_matrix(row, "enriched_tail_upper_bound_matrix", enriched_tail_upper)
        _add_matrix(row, "single_upper_bound_matrix", upper_single)
        bound_rows.append(row)
    bound = pd.DataFrame(bound_rows)
    bound.to_csv(out_dir / "bound_rows.csv", index=False)

    hierarchy_rows: list[dict[str, object]] = []
    operators = {rank: _load_rom(RUN / directory) for rank, directory in RANK_DIRS.items()}
    for _, material in truth.iterrows():
        coefficients = reduced._material_coefficients(material.to_dict())
        C_fom = _matrix_from_row(material, "Ceff")
        previous_rank: int | None = None
        previous_C: np.ndarray | None = None
        for rank in sorted(operators):
            C_rank, _, _ = reduced._rom_ceff(coefficients, *operators[rank])
            error_eigenvalues = np.linalg.eigvalsh(0.5 * (C_rank - C_fom + (C_rank - C_fom).T))
            nested_eigenvalues = (
                np.full(6, np.nan)
                if previous_C is None
                else np.linalg.eigvalsh(0.5 * (previous_C - C_rank + (previous_C - C_rank).T))
            )
            row = {
                "material_id": int(material["material_id"]),
                "rank": rank,
                "previous_rank": previous_rank,
                "relative_error": float(np.linalg.norm(C_rank - C_fom) / np.linalg.norm(C_fom)),
            }
            for index, value in enumerate(error_eigenvalues, start=1):
                row[f"error_eig_{index}"] = float(value)
            for index, value in enumerate(nested_eigenvalues, start=1):
                row[f"nested_eig_{index}"] = float(value)
            hierarchy_rows.append(row)
            previous_rank, previous_C = rank, C_rank
    hierarchy = pd.DataFrame(hierarchy_rows)
    hierarchy.to_csv(out_dir / "loewner_hierarchy.csv", index=False)
    hard_material = int(bound.loc[bound["true_error_rel"].idxmax(), "material_id"])
    _plot_loewner(hierarchy, hard_material, out_dir)

    nested_columns = [f"nested_eig_{index}" for index in range(1, 7)]
    benchmark_materials = pd.read_csv(args.benchmark_csv.resolve())
    benchmark_materials = benchmark_materials.iloc[: int(args.benchmark_limit)]
    if len(benchmark_materials) == 0:
        raise ValueError("The online benchmark material table is empty.")
    online_benchmark = _benchmark_hierarchical_online(
        materials=benchmark_materials,
        coarse_operators=(Kq, Bq, Dq),
        enriched_operators=(Kq_enriched, Bq_enriched, Dq_enriched),
        enriched_estimator=enriched_estimator,
        beta_margin=float(args.beta_margin),
        loewner_tolerance=float(args.loewner_tolerance),
        records_path=out_dir / "online_certificate_widths.csv",
    )
    (out_dir / "online_benchmark.json").write_text(
        json.dumps(online_benchmark, indent=2, sort_keys=True), encoding="utf-8"
    )
    lower_violation_count = int(bound["lower_hierarchy_strict_violation"].sum())
    enriched_lower_violation_count = int(
        bound["enriched_lower_hierarchy_strict_violation"].sum()
    )
    upper_violation_count = int(bound["strict_violation"].sum())
    truth_rtols = sorted(set(float(value) for value in truth["solver_rtol"]))
    summary = {
        "truth_dir": str(truth_dir),
        "rom_dir": str(rom_dir),
        "enriched_rom_dir": str(enriched_rom_dir),
        "coarse_rank": int(estimator.rank),
        "enriched_rank": int(enriched_estimator.rank),
        "material_count": int(len(bound)),
        "truth_profile": "float64/rtol=" + ",".join(f"{value:.0e}" for value in truth_rtols),
        "reference_count": 129,
        "beta_margin": float(args.beta_margin),
        "truth_uncertainty_multiplier": float(args.truth_uncertainty_multiplier),
        "strict_violation_count": upper_violation_count,
        "lower_hierarchy_strict_violation_count": lower_violation_count,
        "enriched_lower_hierarchy_strict_violation_count": enriched_lower_violation_count,
        "direct_strict_violation_count": int(
            bound["direct_strict_violation"].sum()
        ),
        "enriched_tail_strict_violation_count": int(
            bound["enriched_tail_strict_violation"].sum()
        ),
        "minimum_upper_gap_eigenvalue": float(bound["min_eig_upper_minus_error"].min()),
        "minimum_error_eigenvalue": float(bound["min_eig_error"].min()),
        "minimum_enriched_error_eigenvalue": float(
            bound["min_eig_enriched_error"].min()
        ),
        "effectivity_min": float(bound["effectivity"].min()),
        "effectivity_median": float(bound["effectivity"].median()),
        "effectivity_p95": float(bound["effectivity"].quantile(0.95)),
        "effectivity_max": float(bound["effectivity"].max()),
        "direct_effectivity_min": float(bound["direct_effectivity"].min()),
        "direct_effectivity_median": float(bound["direct_effectivity"].median()),
        "direct_effectivity_p95": float(bound["direct_effectivity"].quantile(0.95)),
        "direct_effectivity_max": float(bound["direct_effectivity"].max()),
        "hierarchical_improvement_ratio_median": float(
            bound["hierarchical_improvement_ratio"].median()
        ),
        "enriched_true_error_ratio_median": float(
            bound["enriched_true_error_ratio"].median()
        ),
        "single_effectivity_median": float(bound["single_effectivity"].median()),
        "single_effectivity_p95": float(bound["single_effectivity"].quantile(0.95)),
        "single_effectivity_max": float(bound["single_effectivity"].max()),
        "envelope_improvement_ratio_median": float(bound["envelope_improvement_ratio"].median()),
        "envelope_active_references_median": float(bound["envelope_active_references"].median()),
        "envelope_max_relative_duality_gap": float(bound["envelope_relative_duality_gap"].max()),
        "median_estimator_online_ms": float(1000.0 * bound["estimator_online_s"].median()),
        "median_certified_query_online_ms": float(
            1000.0 * bound["certified_query_online_s"].median()
        ),
        "online_benchmark_query_count": int(online_benchmark["query_count"]),
        "online_benchmark_total_s": float(online_benchmark["total_s"]),
        "online_benchmark_median_ms": float(online_benchmark["median_ms"]),
        "online_benchmark_p95_ms": float(online_benchmark["p95_ms"]),
        "online_benchmark_relative_width_median": float(
            online_benchmark["relative_width_median"]
        ),
        "online_benchmark_relative_width_p95": float(
            online_benchmark["relative_width_p95"]
        ),
        "online_benchmark_relative_width_max": float(
            online_benchmark["relative_width_max"]
        ),
        "online_benchmark_relative_width_below_1e-4_count": int(
            online_benchmark["relative_width_below_1e-4_count"]
        ),
        "online_benchmark_implied_relative_error_bound_median": float(
            online_benchmark["implied_relative_error_bound_median"]
        ),
        "online_benchmark_implied_relative_error_bound_p95": float(
            online_benchmark["implied_relative_error_bound_p95"]
        ),
        "online_benchmark_implied_relative_error_bound_max": float(
            online_benchmark["implied_relative_error_bound_max"]
        ),
        "hard_material_id": hard_material,
        "hierarchy_minimum_error_eigenvalue": float(hierarchy[[f"error_eig_{i}" for i in range(1, 7)]].min().min()),
        "hierarchy_minimum_nested_eigenvalue": float(hierarchy[nested_columns].min().min()),
        "certified_label_allowed": bool(
            upper_violation_count == 0
            and lower_violation_count == 0
            and enriched_lower_violation_count == 0
        ),
        "note": (
            "The hierarchical bound adds the exact reduced drop C_coarse-C_enriched "
            "to a certified two-kernel bound on the enriched tail. The iterative-truth "
            "guard is a conservative numerical allowance, not a separate analytical "
            "discretization-error certificate. The no-FEM project exception remains active."
        ),
    }
    (out_dir / "bound_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "[BOUND] listo | "
        f"upper_violations={summary['strict_violation_count']} | "
        f"lower_violations={summary['lower_hierarchy_strict_violation_count']} | "
        f"median_effectivity={summary['effectivity_median']:.3f} | "
        f"min_gap={summary['minimum_upper_gap_eigenvalue']:.3e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        raise SystemExit(main())
