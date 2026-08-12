#!/usr/bin/env python3
"""Regenerate CMAME manuscript figures and tables from campaign manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib import cm, colors, patches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "cmame_method"
FIGURES = ROOT / "paper" / "figures"
TABLES = ROOT / "paper" / "tables"
FIXED_STUDY = (
    ROOT
    / "results"
    / "fixed_geometry_ffthompy"
    / "fixed_geometry_ar15_vf20_sobol8_center_fields"
)
FIXED_GEOMETRY = FIXED_STUDY / "_fixed_geometry"


def _save(fig: plt.Figure, name: str) -> list[Path]:
    paths = [FIGURES / f"{name}.pdf", FIGURES / f"{name}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=220, bbox_inches="tight")
    plt.close(fig)
    return paths


def _periodic_centerline_segments(
    center: np.ndarray,
    direction: np.ndarray,
    length: float,
    box_size: float,
) -> list[np.ndarray]:
    """Split a wrapped fiber centerline at periodic-boundary jumps."""
    points_per_fiber = max(80, int(np.ceil(12 * length / box_size)))
    coordinate = np.linspace(-0.5 * length, 0.5 * length, points_per_fiber)
    points = (center[None, :] + coordinate[:, None] * direction[None, :]) % box_size
    jumps = np.flatnonzero(np.any(np.abs(np.diff(points, axis=0)) > 0.5 * box_size, axis=1))
    bounds = np.concatenate(([0], jumps + 1, [points.shape[0]]))
    return [points[start:stop] for start, stop in zip(bounds[:-1], bounds[1:]) if stop - start > 1]


def _draw_periodic_cube(axis: plt.Axes, box_size: float) -> None:
    corners = np.array(
        [
            [0, 0, 0],
            [box_size, 0, 0],
            [0, box_size, 0],
            [0, 0, box_size],
            [box_size, box_size, 0],
            [box_size, 0, box_size],
            [0, box_size, box_size],
            [box_size, box_size, box_size],
        ],
        dtype=float,
    )
    for i, j in (
        (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
        (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
    ):
        axis.plot(*np.vstack((corners[i], corners[j])).T, color="0.28", linewidth=0.55, alpha=0.8)


def _fixed_geometry_figure() -> list[Path]:
    fibers = pd.read_csv(FIXED_GEOMETRY / "continuous_fibers_master.csv")
    phase = np.load(FIXED_GEOMETRY / "phase.npy", mmap_mode="r")
    manifest = json.loads((FIXED_GEOMETRY / "geometry_manifest.json").read_text())
    box_size = float(manifest["design"]["caja_um"])

    fig = plt.figure(figsize=(10.0, 3.8))
    outer = fig.add_gridspec(
        1, 3, width_ratios=(1.45, 1.35, 1.0),
        left=0.035, right=0.985, bottom=0.12, top=0.88, wspace=0.28,
    )
    axis_3d = fig.add_subplot(outer[0], projection="3d")
    directions = fibers[["ux", "uy", "uz"]].to_numpy(dtype=float)
    normalization = colors.Normalize(vmin=0.0, vmax=1.0)
    color_map = cm.get_cmap("viridis")
    for row in fibers.itertuples(index=False):
        center = np.array([row.cx_um, row.cy_um, row.cz_um], dtype=float)
        direction = np.array([row.ux, row.uy, row.uz], dtype=float)
        for segment in _periodic_centerline_segments(center, direction, row.L_um, box_size):
            axis_3d.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                color=color_map(normalization(abs(row.uz))),
                linewidth=1.35,
                alpha=0.88,
                solid_capstyle="round",
            )
    _draw_periodic_cube(axis_3d, box_size)
    axis_3d.set(xlim=(0, box_size), ylim=(0, box_size), zlim=(0, box_size))
    axis_3d.set_box_aspect((1, 1, 1))
    axis_3d.view_init(elev=24, azim=-56)
    ticks = [0.0, 0.5 * box_size, box_size]
    axis_3d.set_xticks(ticks, labels=["0", "", f"{box_size:.1f}"])
    axis_3d.set_yticks(ticks, labels=["0", "", f"{box_size:.1f}"])
    axis_3d.set_zticks(ticks, labels=["0", "", f"{box_size:.1f}"])
    axis_3d.tick_params(labelsize=6, pad=-2)
    axis_3d.set_xlabel(r"$x_1$ [$\mu$m]", fontsize=7, labelpad=-1)
    axis_3d.set_ylabel(r"$x_2$ [$\mu$m]", fontsize=7, labelpad=-1)
    axis_3d.set_title(r"(a) Continuous periodic fibers ($n_f=96$)", fontsize=9, pad=1)
    axis_3d.grid(False)
    for pane in (axis_3d.xaxis.pane, axis_3d.yaxis.pane, axis_3d.zaxis.pane):
        pane.set_alpha(0.0)
    scalar_map = cm.ScalarMappable(norm=normalization, cmap=color_map)
    color_bar = fig.colorbar(scalar_map, ax=axis_3d, fraction=0.034, pad=0.025, shrink=0.68)
    color_bar.set_label(r"$|u_3|$", fontsize=7)
    color_bar.ax.tick_params(labelsize=6)

    slice_grid = outer[1].subgridspec(1, 3, wspace=0.04)
    midpoint = phase.shape[0] // 2
    slices = (
        (phase[midpoint, :, :].T, r"$x_1=L/2$"),
        (phase[:, midpoint, :].T, r"$x_2=L/2$"),
        (phase[:, :, midpoint].T, r"$x_3=L/2$"),
    )
    for index, (image, label) in enumerate(slices):
        axis = fig.add_subplot(slice_grid[index])
        axis.imshow(image, origin="lower", cmap="gray_r", interpolation="nearest")
        axis.set_title(label, fontsize=7, pad=2)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color("0.35")
    slice_title = fig.add_subplot(outer[1], frameon=False)
    slice_title.set_title(r"(b) Rasterized phase at $91^3$", fontsize=9, pad=14)
    slice_title.set_xticks([])
    slice_title.set_yticks([])

    orientation_grid = outer[2].subgridspec(2, 1, hspace=0.35)
    axis_distribution = fig.add_subplot(orientation_grid[0])
    axis_distribution.hist(
        np.abs(directions[:, 2]), bins=np.linspace(0, 1, 9), density=True,
        color="#2a7f9e", edgecolor="white", linewidth=0.6, alpha=0.9,
    )
    axis_distribution.axhline(1.0, color="#b44a44", linestyle="--", linewidth=1.1, label="isotropic target")
    axis_distribution.set_xlabel(r"$|u_3|$", fontsize=8)
    axis_distribution.set_ylabel("Density", fontsize=8)
    axis_distribution.set_title("(c) Realized orientations", fontsize=9, pad=2)
    axis_distribution.legend(frameon=False, fontsize=6, loc="upper right")
    axis_distribution.tick_params(labelsize=7)
    axis_distribution.grid(True, axis="y", alpha=0.2)

    axis_tensor = fig.add_subplot(orientation_grid[1])
    realized_a2 = np.mean(directions**2, axis=0)
    axis_tensor.bar(np.arange(3), realized_a2, color=("#4c78a8", "#72a05a", "#b65b6a"), width=0.62)
    axis_tensor.axhline(1.0 / 3.0, color="0.2", linestyle="--", linewidth=1.0, label=r"target $1/3$")
    axis_tensor.set_xticks(np.arange(3), labels=(r"$A_{11}$", r"$A_{22}$", r"$A_{33}$"))
    axis_tensor.set_ylim(0.0, max(0.42, 1.15 * realized_a2.max()))
    axis_tensor.set_ylabel("Second moment", fontsize=8)
    axis_tensor.legend(frameon=False, fontsize=6, loc="upper right")
    axis_tensor.tick_params(labelsize=7)
    axis_tensor.grid(True, axis="y", alpha=0.2)
    return _save(fig, "cmame_fixed_rve_morphology")


def _architecture_assets() -> list[Path]:
    phase = np.load(FIXED_GEOMETRY / "phase.npy", mmap_mode="r")
    result = pd.read_csv(FIXED_STUDY / "fixed_geometry_ffthompy_results.csv", nrows=1).iloc[0]
    ceff = np.array(
        [[result[f"Ceff_{i}{j}"] for j in range(1, 7)] for i in range(1, 7)],
        dtype=float,
    )
    reduced_path = next(FIXED_STUDY.glob("rom_tangential_r168*/reduced_operators.npz"))
    estimator_path = next(FIXED_STUDY.glob("rom_tangential_r168*/online_estimator.npz"))
    with np.load(reduced_path) as reduced, np.load(estimator_path) as estimator:
        rank = int(reduced["Kq"].shape[1])
        material_count = int(reduced["snapshot_material_ids"].size)
        bb_shape = tuple(int(value) for value in estimator["BB"].shape)
        bk_shape = tuple(int(value) for value in estimator["BK"].shape)
        kk_shape = tuple(int(value) for value in estimator["KK"].shape)

    fig = plt.figure(figsize=(10.0, 4.35))
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def block(x: float, y: float, width: float, height: float, color: str, title: str, body: str = "") -> None:
        axis.add_patch(
            patches.FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor=color, edgecolor="0.28", linewidth=0.8,
            )
        )
        axis.text(x + width / 2, y + height - 0.045, title, ha="center", va="top", fontsize=9, fontweight="bold")
        if body:
            axis.text(x + width / 2, y + 0.075, body, ha="center", va="bottom", fontsize=7.3, linespacing=1.35)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        axis.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": "0.25"})

    axis.add_patch(patches.Rectangle((0.018, 0.535), 0.964, 0.425, facecolor="#eef3f5", edgecolor="none"))
    axis.add_patch(patches.Rectangle((0.018, 0.055), 0.964, 0.405, facecolor="#f7f3ea", edgecolor="none"))
    axis.text(0.03, 0.928, "OFFLINE COMPILATION", fontsize=9, fontweight="bold", color="#24566b")
    axis.text(0.03, 0.428, "ONLINE QUERY", fontsize=9, fontweight="bold", color="#7a4a20")

    top_y, top_h, top_w = 0.595, 0.285, 0.19
    top_x = (0.045, 0.285, 0.525, 0.765)
    block(top_x[0], top_y, top_w, top_h, "#d9e8ed", "Periodic RVE")
    block(top_x[1], top_y, top_w, top_h, "#dce9d7", "Affine geometry atoms", r"$\{K_q,B_q,D_q\}_{q=1}^{7}$")
    block(top_x[2], top_y, top_w, top_h, "#eadfe8", "Tangential responses", rf"{material_count} materials $\times$ 6 loads" + "\n" + rf"$V_r\in\mathbb{{R}}^{{n\times {rank}}}$")
    block(
        top_x[3], top_y, top_w, top_h, "#eadfcd", "Two-kernel contractions",
        rf"$BB$: {bb_shape}" + "\n" + rf"$BK$: {bk_shape}" + "\n" + rf"$KK$: {kk_shape}",
    )
    for left, right in zip(top_x[:-1], top_x[1:]):
        arrow(left + top_w + 0.008, top_y + top_h / 2, right - 0.008, top_y + top_h / 2)

    phase_axis = fig.add_axes((top_x[0] + 0.055, top_y + 0.102, 0.08, 0.105))
    phase_axis.imshow(phase[:, :, phase.shape[2] // 2].T, origin="lower", cmap="gray_r", interpolation="nearest")
    phase_axis.set_xticks([])
    phase_axis.set_yticks([])
    for spine in phase_axis.spines.values():
        spine.set_linewidth(0.5)
    axis.text(top_x[0] + top_w / 2, top_y + 0.042, r"$91^3$ voxels; $n_f=96$", ha="center", va="bottom", fontsize=7.3)

    bottom_y, bottom_h, bottom_w = 0.105, 0.255, 0.19
    bottom_x = top_x
    block(bottom_x[0], bottom_y, bottom_w, bottom_h, "#f0dfc5", "Physical input", r"$\mathbf{\xi}\in\mathbb{R}^{7}$")
    block(bottom_x[1], bottom_y, bottom_w, bottom_h, "#dce9d7", "Affine assembly", r"$\mathbf{\gamma}(\mathbf{\xi})$" + "\n" + rf"$K_r\in\mathbb{{R}}^{{{rank}\times {rank}}}$")
    block(bottom_x[2], bottom_y, bottom_w, bottom_h, "#d9e8ed", "Small dense solve", r"$K_rY_r=B_r$" + "\n" + "no voxel fields / no FFT")
    block(bottom_x[3], bottom_y, bottom_w, bottom_h, "#eadfe8", "Constitutive outputs")
    for left, right in zip(bottom_x[:-1], bottom_x[1:]):
        arrow(left + bottom_w + 0.008, bottom_y + bottom_h / 2, right - 0.008, bottom_y + bottom_h / 2)
    arrow(top_x[2] + top_w / 2, top_y - 0.005, bottom_x[1] + bottom_w / 2, bottom_y + bottom_h + 0.005)
    arrow(top_x[3] + top_w / 2, top_y - 0.005, bottom_x[3] + bottom_w / 2, bottom_y + bottom_h + 0.005)

    ceff_axis = fig.add_axes((bottom_x[3] + 0.014, bottom_y + 0.060, 0.052, 0.080))
    ceff_axis.imshow(ceff, cmap="RdBu_r", norm=colors.TwoSlopeNorm(vcenter=0.0, vmin=ceff.min(), vmax=ceff.max()))
    ceff_axis.set_xticks([])
    ceff_axis.set_yticks([])
    axis.text(
        bottom_x[3] + 0.132, bottom_y + 0.075,
        r"$C_r^{\rm eff},\ \nabla_{\xi}C_r^{\rm eff}$" + "\n" + r"$U_r$ and effectivity",
        ha="center", va="bottom", fontsize=7.3, linespacing=1.35,
    )
    return _save(fig, "cmame_offline_online_architecture")


def _voxel_assets() -> list[Path]:
    data = pd.read_csv(RESULTS / "voxel_scaling" / "voxel_scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    axes[0].plot(data["grid_size"], data["relative_discretization_error_vs_finest"], "o-", color="#147d75")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Grid points per direction")
    axes[0].set_ylabel(r"$\|C_N-C_{128}\|_F/\|C_{128}\|_F$")
    axes[1].plot(data["grid_size"], data["solve_wall_s"], "s-", color="#c44e52")
    axes[1].set_xlabel("Grid points per direction")
    axes[1].set_ylabel("Truth solve wall time [s]")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    outputs = _save(fig, "cmame_voxel_scaling")
    rows = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$N$ & $N^3$ & $V_f$ & rel. error vs. $128^3$ & solve [s] \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        rows.append(
            f"{int(row['grid_size'])} & {int(row['voxel_count']):d} & {row['voxel_volume_fraction']:.5f} & "
            f"{row['relative_discretization_error_vs_finest']:.3e} & {row['solve_wall_s']:.2f} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    path = TABLES / "cmame_voxel_scaling.tex"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return [*outputs, path]


def _inclusion_assets() -> list[Path]:
    data = pd.read_csv(RESULTS / "inclusion_benchmark" / "inclusion_refinement.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True)
    axes[0].plot(data["grid"], data["relative_error_vs_finest"], "o-", color="#4c6a92")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Grid points per direction")
    axes[0].set_ylabel("Relative tensor error vs. $91^3$")
    axes[1].plot(data["grid"], data["effective_bulk"], "o-", color="#9c3d54", label=r"$K_{\mathrm{eff}}$")
    axes[1].fill_between(data["grid"], data["hs_bulk_lower"], data["hs_bulk_upper"], color="#9c3d54", alpha=0.18, label="HS bulk interval")
    axes[1].set_xlabel("Grid points per direction")
    axes[1].set_ylabel("Bulk modulus")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(True, which="both", alpha=0.22)
    outputs = _save(fig, "cmame_inclusion_benchmark")
    rows = [r"\begin{tabular}{rrrrr}", r"\toprule", r"$N$ & $V_f$ & rel. error & min eig. above Reuss & min eig. below Voigt \\", r"\midrule"]
    for _, row in data.iterrows():
        rows.append(
            f"{int(row['grid'])} & {row['voxel_volume_fraction']:.5f} & "
            f"{row['relative_error_vs_finest']:.3e} & {row['min_eig_Ceff_minus_Reuss']:.3e} & "
            f"{row['min_eig_Voigt_minus_Ceff']:.3e} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    path = TABLES / "cmame_inclusion_benchmark.tex"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return [*outputs, path]


def _geometry_assets() -> list[Path]:
    design = pd.read_csv(RESULTS / "geometries" / "geometry_design.csv")
    realized = pd.read_csv(RESULTS / "geometries" / "geometry_realized_descriptors.csv")
    data = design.merge(realized, on="geometry_id", suffixes=("_target", "_realized"))
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.5), constrained_layout=True)
    pairs = [
        ("Vf_target", "Vf_realized", "$V_f$"),
        ("aspect_ratio_target", "aspect_ratio_realized", "Aspect ratio"),
        ("A2_11", "A2_11_realized", "$A_{11}$"),
        ("A2_22", "A2_22_realized", "$A_{22}$"),
        ("A2_33", "A2_33_realized", "$A_{33}$"),
        ("cluster_fraction", "Ripley_peak", "clustering/Ripley"),
    ]
    for axis, (target, realized, label) in zip(axes.flat, pairs):
        axis.scatter(data[target], data[realized], c=data["geometry_id"], cmap="viridis", s=34)
        if label != "clustering/Ripley":
            low = min(data[target].min(), data[realized].min())
            high = max(data[target].max(), data[realized].max())
            axis.plot([low, high], [low, high], "--", color="0.35", linewidth=1)
        axis.set_xlabel(f"Target {label}")
        axis.set_ylabel(f"Realized {label}")
        axis.grid(True, alpha=0.2)
    outputs = _save(fig, "cmame_geometry_descriptors")

    fig, axes = plt.subplots(2, 5, figsize=(10.0, 4.2), constrained_layout=True)
    for geometry_id, axis in enumerate(axes.flat):
        phase = np.load(RESULTS / "geometries" / f"geometry_{geometry_id:02d}" / "phase.npy")
        axis.imshow(phase[:, :, phase.shape[2] // 2].T, origin="lower", cmap="gray_r", interpolation="nearest")
        axis.set_title(f"G{geometry_id}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    outputs.extend(_save(fig, "cmame_geometry_slices"))
    summary = json.loads((RESULTS / "geometries" / "campaign_manifest.json").read_text())
    path = TABLES / "cmame_geometry_summary.tex"
    path.write_text(
        "\n".join(
            [
                r"\begin{tabular}{lr}", r"\toprule", r"Quantity & Value \\", r"\midrule",
                f"Accepted geometries & {summary['local_acceptance_count']} / {summary['geometry_count']} \\\\ ",
                f"Minimum design distance & {summary['minimum_design_distance']:.3f} \\\\ ",
                f"Minimum realized-descriptor distance & {summary['minimum_realized_descriptor_distance']:.3f} \\\\ ",
                r"Fixed grid & $91^3$ \\", r"\bottomrule", r"\end{tabular}",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return [*outputs, path]


def _uq_assets() -> list[Path]:
    artifacts = {}
    for kind in ("global_uniform", "local_truncated_normal"):
        with np.load(RESULTS / "uq" / f"{kind}_samples.npz") as payload:
            artifacts[kind] = np.asarray(payload["outputs"], dtype=np.float64)
            output_names = [str(value) for value in payload["output_names"]]
    indices = [output_names.index(name) for name in ("E3", "G12", "nu12")]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), constrained_layout=True)
    for axis, index, name in zip(axes, indices, ("$E_3$", "$G_{12}$", r"$\nu_{12}$")):
        for kind, color, label in (
            ("global_uniform", "#2b6cb0", "global uniform"),
            ("local_truncated_normal", "#c05621", "local truncated normal"),
        ):
            axis.hist(artifacts[kind][:, index], bins=70, density=True, histtype="step", linewidth=1.4, color=color, label=label)
        axis.set_xlabel(name)
        axis.set_ylabel("Density")
        axis.grid(True, alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    outputs = _save(fig, "cmame_uq_distributions")
    summary = pd.read_csv(RESULTS / "uq" / "uq_summary.csv")
    selected = summary.loc[summary["output"].isin(["E3", "G12", "nu12"])]
    rows = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Distribution & output & mean & std. & P05 & P95 \\", r"\midrule"]
    for _, row in selected.iterrows():
        label = "global" if row["distribution"] == "global_uniform" else "local"
        rows.append(f"{label} & {row['output']} & {row['mean']:.4f} & {row['std']:.4f} & {row['q05']:.4f} & {row['q95']:.4f} \\\\ ")
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    path = TABLES / "cmame_uq_summary.tex"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return [*outputs, path]


def _bound_assets() -> list[Path]:
    data = pd.read_csv(RESULTS / "bound_study" / "bound_rows.csv")
    order = np.argsort(data["effectivity"].to_numpy())
    ordered = data.iloc[order].reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.25), constrained_layout=True)

    scatter = axes[0].scatter(
        data["true_error_abs"], data["bound_abs"], c=data["effectivity"],
        cmap="viridis", s=38, edgecolor="white", linewidth=0.45, zorder=3,
    )
    lower = 0.82 * min(data["true_error_abs"].min(), data["bound_abs"].min())
    upper = 1.22 * max(data["true_error_abs"].max(), data["bound_abs"].max())
    axes[0].plot([lower, upper], [lower, upper], "--", color="#b44a44", linewidth=1.1, label="exact limit")
    axes[0].set(xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper))
    axes[0].set_xlabel(r"True error $\|C_r-C_{\rm FOM}\|_F$")
    axes[0].set_ylabel(r"Upper bound $\|U_r\|_F$")
    axes[0].set_title("(a) Reliability over 16 materials", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    color_bar = fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.025)
    color_bar.set_label("Effectivity", fontsize=8)
    color_bar.ax.tick_params(labelsize=7)

    x_values = np.arange(1, len(ordered) + 1)
    axes[1].plot(
        x_values, ordered["direct_effectivity"], "o-", color="#8a8a8a",
        markersize=3.6, linewidth=1.0, label="direct residual bound",
    )
    axes[1].plot(
        x_values, ordered["effectivity"], "s-", color="#4c78a8",
        markersize=4.0, linewidth=1.2, label="hierarchical bound",
    )
    median_effectivity = float(data["effectivity"].median())
    direct_median = float(data["direct_effectivity"].median())
    axes[1].axhline(5.0, color="#b44a44", linestyle="--", linewidth=1.1, label="quality target 5")
    axes[1].axhline(
        median_effectivity, color="#4f7a4b", linestyle=":", linewidth=1.3,
        label=rf"hierarchical median {median_effectivity:.2f}",
    )
    axes[1].axhline(
        direct_median, color="#666666", linestyle=":", linewidth=1.0,
        label=rf"direct median {direct_median:.2f}",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Materials sorted by effectivity")
    axes[1].set_ylabel(r"$\|U_r\|_F/\|C_r-C_{\rm FOM}\|_F$")
    axes[1].set_title("(b) Bound sharpness", fontsize=9)
    axes[1].set_xticks((1, 4, 8, 12, 16))
    axes[1].legend(frameon=False, fontsize=6.4, loc="upper right")

    axes[2].semilogy(
        x_values, ordered["min_eig_error"], "o-", color="#2a7f9e",
        markersize=4, linewidth=1.1, label=r"$\lambda_{\min}(C_r-C_{\rm FOM})$",
    )
    axes[2].semilogy(
        x_values, ordered["min_eig_upper_minus_error"], "s-", color="#b65b6a",
        markersize=4, linewidth=1.1, label=r"$\lambda_{\min}(U_r-C_r+C_{\rm FOM})$",
    )
    axes[2].set_xlabel("Same material order")
    axes[2].set_ylabel("Minimum eigenvalue")
    axes[2].set_title("(c) Strict Loewner margins", fontsize=9)
    axes[2].set_xticks((1, 4, 8, 12, 16))
    axes[2].legend(frameon=False, fontsize=6.5, loc="best")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.2)
        axis.tick_params(labelsize=7)
    outputs = _save(fig, "cmame_bound_reliability")

    source_pdf = RESULTS / "bound_study" / "loewner_eigenvalues.pdf"
    source_png = RESULTS / "bound_study" / "loewner_eigenvalues.png"
    targets = [FIGURES / "cmame_loewner_eigenvalues.pdf", FIGURES / "cmame_loewner_eigenvalues.png"]
    shutil.copy2(source_pdf, targets[0])
    shutil.copy2(source_png, targets[1])
    summary = json.loads((RESULTS / "bound_study" / "bound_summary.json").read_text())
    final_summary_path = RESULTS / "final_bound_study_r168_m20" / "bound_summary.json"
    final_summary = (
        json.loads(final_summary_path.read_text())
        if final_summary_path.is_file()
        else None
    )
    path = TABLES / "cmame_bound_summary.tex"
    rows = [
        r"\begin{tabular}{lr}", r"\toprule", r"Quantity & Value \\", r"\midrule",
        f"High-precision materials & {summary['material_count']} \\\\ ",
        f"Coarse/enriched ranks & {summary['coarse_rank']}/{summary['enriched_rank']} \\\\ ",
        f"Strict violations & {summary['strict_violation_count']} \\\\ ",
        f"Direct median effectivity & {summary['direct_effectivity_median']:.3f} \\\\ ",
        f"Hierarchical median effectivity & {summary['effectivity_median']:.3f} \\\\ ",
        f"Hierarchical P95 effectivity & {summary['effectivity_p95']:.3f} \\\\ ",
        f"Median estimator [ms] & {summary['median_estimator_online_ms']:.3f} \\\\ ",
        f"4096-query bounded sweep [s] & {summary['online_benchmark_total_s']:.3f} \\\\ ",
        f"Minimum Loewner gap & {summary['minimum_upper_gap_eigenvalue']:.3e} \\\\ ",
    ]
    if final_summary is not None:
        rows.extend(
            [
                r"\midrule",
                f"Production/auxiliary ranks & {final_summary['coarse_rank']}/"
                f"{final_summary['enriched_rank']} \\\\ ",
                f"Final upper violations & {final_summary['strict_violation_count']} \\\\ ",
                "Implied bounds $\\leq 10^{-4}$ & "
                f"{final_summary['online_benchmark_relative_width_below_1e-4_count']}/"
                f"{final_summary['online_benchmark_query_count']} \\\\ ",
                f"Final width median/P95 & "
                f"{final_summary['online_benchmark_relative_width_median']:.2e}/"
                f"{final_summary['online_benchmark_relative_width_p95']:.2e} \\\\ ",
                f"Final maximum width & "
                f"{final_summary['online_benchmark_relative_width_max']:.2e} \\\\ ",
                f"Final 4096-query sweep [s] & "
                f"{final_summary['online_benchmark_total_s']:.3f} \\\\ ",
            ]
        )
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return [*outputs, *targets, path]


def _noisy_inverse_assets() -> list[Path]:
    data = pd.read_csv(RESULTS / "noisy_inverse" / "noisy_inverse_summary.csv")
    regularizations = sorted(data["regularization"].unique())
    x_values = np.arange(len(regularizations))
    palette = ("#2a7f9e", "#b65b6a", "#4f7a4b")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), constrained_layout=True)
    for color, (noise, group) in zip(palette, data.groupby("noise_percent")):
        ordered = group.set_index("regularization").loc[regularizations]
        label = f"{noise:g}% noise"
        axes[0].plot(
            x_values, ordered["unit_parameter_rmse_median"], "o-",
            color=color, linewidth=1.25, markersize=4.2, label=label,
        )
        axes[0].fill_between(
            x_values,
            ordered["unit_parameter_rmse_median"],
            ordered["unit_parameter_rmse_p95"],
            color=color, alpha=0.14,
        )
        axes[1].plot(
            x_values, ordered["true_tensor_error_median"], "s-",
            color=color, linewidth=1.25, markersize=4.2, label=label,
        )
    labels = ["0" if value == 0 else f"{value:.0e}" for value in regularizations]
    axes[0].set_ylabel("Normalized parameter RMSE")
    axes[0].set_title("(a) Physical-parameter recovery", fontsize=9)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("True tensor relative error")
    axes[1].set_title("(b) Constitutive recovery", fontsize=9)
    for axis in axes:
        axis.set_xticks(x_values, labels)
        axis.set_xlabel(r"Tikhonov weight $\lambda_{\rm reg}$")
        axis.grid(True, which="both", alpha=0.22)
        axis.tick_params(labelsize=7)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    outputs = _save(fig, "cmame_noisy_inverse")

    best = data.loc[
        data.groupby("noise_percent")["unit_parameter_rmse_median"].idxmin()
    ].sort_values("noise_percent")
    path = TABLES / "cmame_noisy_inverse_summary.tex"
    rows = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"Noise [\%] & $\lambda_{\rm reg}$ & Parameter RMSE & Tensor error \\",
        r"\midrule",
    ]
    for _, row in best.iterrows():
        rows.append(
            f"{row['noise_percent']:.1f} & {row['regularization']:.0e} & "
            f"{row['unit_parameter_rmse_median']:.3f} & "
            f"{row['true_tensor_error_median']:.3e} \\\\"
        )
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return [*outputs, path]


def _conditional_inverse_assets() -> list[Path]:
    conditional = pd.read_csv(
        RESULTS / "conditional_inverse" / "conditional_inverse_summary.csv"
    ).sort_values("noise_percent")
    x = conditional["noise_percent"].to_numpy(dtype=float)
    median = conditional["active_parameter_rmse_median"].to_numpy(dtype=float)
    p95 = conditional["active_parameter_rmse_p95"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), constrained_layout=True)
    axes[0].plot(x, median, "o-", color="#2a7f9e", linewidth=1.4, label="median")
    axes[0].fill_between(x, median, p95, color="#2a7f9e", alpha=0.18, label="median--P95")
    axes[0].set_xlabel("Relative tensor-noise level [%]")
    axes[0].set_ylabel(r"Normalized RMSE in $(E_L,E_T)$")
    axes[0].set_title("(a) Conditional parameter recovery", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)

    full_condition = 4.5819584e4
    conditional_condition = float(conditional["jacobian_condition_median"].median())
    axes[1].bar(
        [0, 1], [full_condition, conditional_condition],
        color=["#b65b6a", "#4f7a4b"], width=0.62,
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks([0, 1], ["Seven unknowns", r"$E_L,E_T$ unknown"])
    axes[1].set_ylabel("Jacobian condition number")
    axes[1].set_title("(b) Rank-revealing reformulation", fontsize=9)
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.22)
        axis.tick_params(labelsize=7)
    outputs = _save(fig, "cmame_conditional_inverse")

    path = TABLES / "cmame_conditional_inverse_summary.tex"
    rows = [
        r"\begin{tabular}{rrrr}", r"\toprule",
        r"Noise [\%] & Parameter RMSE & P95 & Tensor error \\", r"\midrule",
    ]
    for _, row in conditional.iterrows():
        rows.append(
            f"{row['noise_percent']:.1f} & "
            f"{row['active_parameter_rmse_median']:.3f} & "
            f"{row['active_parameter_rmse_p95']:.3f} & "
            f"{row['true_tensor_error_median']:.2e} \\\\" 
        )
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return [*outputs, path]


def _adaptive_convergence_assets() -> list[Path]:
    import matplotlib.pyplot as plt

    src = RESULTS / "sobol_pod_time_match" / "sobol_pod_time_match_curve.csv"
    if not src.is_file():
        return []
    df = pd.read_csv(src)

    fig, ax = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
    ax.plot(
        df["material_budget"], df["error_max"],
        marker="o", markersize=4, linestyle="-",
        color="#D95F02", linewidth=1.5, label="Max error"
    )
    if "tolerance" in df.columns:
        ax.axhline(
            float(df["tolerance"].iloc[0]), color="#7570B3", linestyle="--", linewidth=1.5, zorder=0,
            label="Tolerance"
        )

    ax.set_yscale("log")
    ax.set_xlabel("Sobol snapshots (budget)")
    ax.set_ylabel("Relative tensor error")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)

    return _save(fig, "cmame_adaptive_convergence")


def _rank_scalability_assets() -> list[Path]:
    campaign = RESULTS / "rank_scalability"
    summary = pd.read_csv(campaign / "rank_scalability_summary.csv")
    correlations = pd.read_csv(campaign / "descriptor_rank_correlations.csv")
    voxel = summary.loc[summary["case_kind"] == "voxel_grid"].sort_values("grid_size")
    geometry = summary.loc[summary["case_kind"] == "geometry"].sort_values("interface_density")
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.35), constrained_layout=True)
    threshold_columns = (("r_1e-2", r"$10^{-2}$"), ("r_1e-3", r"$10^{-3}$"), ("r_1e-4", r"$10^{-4}$"))
    for color, (column, label) in zip(("#6f6f6f", "#2a7f9e", "#b65b6a"), threshold_columns):
        axes[0].plot(voxel["grid_size"], voxel[column], "o-", color=color, label=label)
    axes[0].set_xlabel("Grid size $N$")
    axes[0].set_ylabel("Stable required rank")
    axes[0].set_title("(a) Same geometry, four grids", fontsize=9)
    axes[0].legend(title="Error target", frameon=False, fontsize=7, title_fontsize=7)

    censored = geometry.loc[geometry["r_1e-4"].isna()]
    scatter = axes[1].scatter(
        geometry["interface_density"], geometry["r_1e-3"],
        c=geometry["cluster_fraction_target"], cmap="viridis", s=42, edgecolor="0.2", linewidth=0.4,
    )
    if not censored.empty:
        axes[1].scatter(
            censored["interface_density"], censored["r_1e-3"],
            marker="x", color="#b65b6a", s=70, linewidth=1.2,
            label=r"$10^{-4}$ not reached by $r=120$",
        )
        axes[1].legend(frameon=False, fontsize=6.5)
    axes[1].set_xlabel("Realized interface density")
    axes[1].set_ylabel(r"$r_{10^{-3}}$")
    axes[1].set_title("(b) Ten realized geometries", fontsize=9)
    colorbar = fig.colorbar(scatter, ax=axes[1], fraction=0.05, pad=0.03)
    colorbar.set_label("Clustering target", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)

    selected = correlations.loc[
        (correlations["rank_column"] == "r_1e-3")
        & correlations["spearman_rho"].notna()
    ].copy().sort_values("spearman_rho")
    axes[2].barh(
        np.arange(len(selected)), selected["spearman_rho"],
        color=np.where(selected["spearman_rho"] >= 0.0, "#2a7f9e", "#b65b6a"),
    )
    axes[2].axvline(0.0, color="0.25", linewidth=0.8)
    axes[2].set_yticks(np.arange(len(selected)), selected["descriptor"])
    axes[2].set_xlim(-1.05, 1.05)
    axes[2].set_xlabel("Spearman $\rho$")
    axes[2].set_title(r"(c) Descriptor association with $r_{10^{-3}}$", fontsize=9)
    for axis in axes:
        axis.grid(True, alpha=0.2)
        axis.tick_params(labelsize=7)
    outputs = _save(fig, "cmame_rank_scalability")

    table_path = TABLES / "cmame_rank_scalability.tex"
    rows = [
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Case & $r_{10^{-2}}$ & $r_{10^{-3}}$ & $r_{10^{-4}}$ \\", r"\midrule",
    ]
    for _, row in voxel.iterrows():
        values = ["--" if pd.isna(row[column]) else str(int(row[column])) for column, _ in threshold_columns]
        rows.append(f"${int(row['grid_size'])}^3$ & " + " & ".join(values) + r" \\")
    for column, label in threshold_columns:
        values = geometry[column].dropna().to_numpy(dtype=float)
        censored_count = int(geometry[column].isna().sum())
        if len(values) == 0:
            value = rf">{int(geometry['basis_rank_max'].max())} ({censored_count} cens.)"
        else:
            value = (
                f"{int(values.min())}--{int(values.max())}"
                + (rf"; >{int(geometry['basis_rank_max'].max())} for {censored_count}" if censored_count else "")
            )
        if column == "r_1e-2":
            geometry_values = [value]
        else:
            geometry_values.append(value)
    rows.append("Ten geometries (range) & " + " & ".join(geometry_values) + r" \\")
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    table_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return [*outputs, table_path]


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    outputs = [
        *_architecture_assets(),
        *_fixed_geometry_figure(),
        *_voxel_assets(),
        *_inclusion_assets(),
        *_geometry_assets(),
        *_uq_assets(),
        *_bound_assets(),
        *_bound_assets(),
        *_conditional_inverse_assets(),
        *_adaptive_convergence_assets(),
    ]
    manifest = {
        "source_root": str(RESULTS),
        "outputs": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }
    manifest_path = ROOT / "paper" / "cmame_assets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[ASSETS] listo | outputs={len(outputs)} | manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
