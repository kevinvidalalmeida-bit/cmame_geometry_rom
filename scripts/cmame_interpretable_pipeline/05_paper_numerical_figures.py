#!/usr/bin/env python3
"""Generate the numerical-example figures used by paper/main.tex."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--paper-figure-dir", type=Path, required=True)
    return parser


def _adaptive_validation(summary_dir: Path, figure_dir: Path) -> None:
    curves = pd.read_csv(summary_dir / "sobol_pod_multigeometry_curve.csv")
    validation = pd.read_csv(
        summary_dir / "sobol_pod_multigeometry_validation.csv"
    )

    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, 10))
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), constrained_layout=True)

    for geometry_id, data in curves.groupby("geometry_id"):
        data = data.sort_values("training_materials")
        axes[0].plot(
            data["training_materials"],
            data["monitor_error_max"],
            marker="o",
            markersize=2.8,
            linewidth=1.1,
            color=colors[int(geometry_id)],
            label=f"G{int(geometry_id):02d}",
        )
    axes[0].axhline(1.0e-4, color="black", linestyle="--", linewidth=0.9)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Sobol-prefix length $\\ell$")
    axes[0].set_ylabel("Maximum monitor error")
    axes[0].grid(True, which="both", linewidth=0.35, alpha=0.45)
    axes[0].legend(ncol=2, fontsize=6.4, frameon=False)
    axes[0].text(
        -0.10,
        1.02,
        "(a)",
        transform=axes[0].transAxes,
        fontweight="bold",
        va="bottom",
    )

    for geometry_id, data in validation.groupby("geometry_id"):
        x = np.full(len(data), int(geometry_id), dtype=float)
        offsets = np.linspace(-0.16, 0.16, len(data))
        axes[1].scatter(
            x + offsets,
            data["relative_frobenius_error"],
            s=9,
            color=colors[int(geometry_id)],
            alpha=0.8,
            linewidths=0,
        )
        mean = float(data["relative_frobenius_error"].mean())
        axes[1].plot(
            [geometry_id - 0.28, geometry_id + 0.28],
            [mean, mean],
            color="black",
            linewidth=1.2,
        )
    axes[1].axhline(1.0e-4, color="black", linestyle="--", linewidth=0.9)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(10), [f"G{i:02d}" for i in range(10)], rotation=45)
    axes[1].set_xlabel("Fixed microstructure")
    axes[1].set_ylabel("Held-out relative tensor error")
    axes[1].grid(True, which="both", linewidth=0.35, alpha=0.45)
    axes[1].text(
        -0.10,
        1.02,
        "(b)",
        transform=axes[1].transAxes,
        fontweight="bold",
        va="bottom",
    )

    for suffix in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"numerical_adaptive_validation.{suffix}",
            dpi=300,
        )
    plt.close(fig)


def _performance_scaling(summary_dir: Path, figure_dir: Path) -> None:
    benchmark_path = summary_dir / "rom_backend_benchmark.csv"
    if not benchmark_path.exists():
        raise FileNotFoundError(
            "Run 09_rom_backend_benchmark.py before generating paper figures."
        )
    data = pd.read_csv(benchmark_path)
    data = data.sort_values("geometry_id")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), constrained_layout=True)

    axes[0].scatter(data["nvox"], data["fom_material_median_s"], s=28)
    fom_offsets = {
        0: (8, 12), 1: (8, -2), 2: (8, 22), 3: (7, 8), 4: (7, 8),
        5: (8, -12), 6: (8, -22), 7: (8, -32), 8: (7, 8), 9: (7, -9),
    }
    for row in data.itertuples():
        axes[0].annotate(
            f"G{int(row.geometry_id):02d}",
            (row.nvox, row.fom_material_median_s),
            xytext=fom_offsets[int(row.geometry_id)],
            textcoords="offset points",
            fontsize=6.7,
        )
    axes[0].set_xlabel("FFT voxel count $N_{\\mathrm{vox}}$")
    axes[0].set_ylabel("Median FOM time $t_{\\mathrm{FOM}}$ [s]")
    axes[0].set_xlim(-0.02 * data["nvox"].max(), 1.08 * data["nvox"].max())
    axes[0].set_ylim(-0.05, 1.10 * data["fom_material_median_s"].max())
    axes[0].grid(True, linewidth=0.35, alpha=0.45)
    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, va="top")

    batch_us = 1.0e6 * data["cuda_batch_amortized_s"]
    axes[1].scatter(
        data["basis_rank"],
        batch_us,
        s=28,
        marker="s",
        zorder=3,
    )
    axes[1].set_xlabel("Reduced rank $r$")
    axes[1].set_ylabel("Amortized CUDA time per query [$\\mu$s]")
    axes[1].set_xlim(data["basis_rank"].min() - 3, data["basis_rank"].max() + 6)
    axes[1].set_ylim(0.90 * batch_us.min(), 1.08 * batch_us.max())
    axes[1].grid(True, linewidth=0.35, alpha=0.45)
    axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, va="top")

    for suffix in ("pdf", "png"):
        fig.savefig(
            figure_dir / f"numerical_performance_scaling.{suffix}",
            dpi=300,
        )
    plt.close(fig)


def main() -> None:
    args = _parser().parse_args()
    summary_dir = args.summary_dir.resolve()
    figure_dir = args.paper_figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)

    _adaptive_validation(summary_dir, figure_dir)
    _performance_scaling(summary_dir, figure_dir)


if __name__ == "__main__":
    main()
