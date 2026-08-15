from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cmame_campaign_common as rank_scaling


def test_required_rank_reports_first_and_stable_crossings() -> None:
    curve = pd.DataFrame(
        {
            "rank": [1, 2, 3, 4, 5],
            "error_max": [2.0e-2, 8.0e-3, 1.2e-2, 7.0e-3, 6.0e-3],
        }
    )

    result = rank_scaling.required_rank(curve, 1.0e-2)

    assert result["first_rank"] == 2
    assert result["stable_rank"] == 4
    assert result["achieved"] is True


def test_required_rank_marks_censored_threshold() -> None:
    curve = pd.DataFrame(
        {"rank": [1, 2, 3], "error_max": [1.0e-2, 2.0e-3, 4.0e-4]}
    )

    result = rank_scaling.required_rank(curve, 1.0e-4)

    assert result["first_rank"] is None
    assert result["stable_rank"] is None
    assert result["achieved"] is False
    assert result["max_tested_rank"] == 3
    assert np.isclose(result["error_at_max_rank"], 4.0e-4)


def test_descriptor_correlations_keep_censoring_explicit() -> None:
    summary = pd.DataFrame(
        {
            "case_kind": ["geometry"] * 4 + ["voxel_grid"],
            "interface_density": [0.1, 0.2, 0.3, 0.4, 0.5],
            "r_1e-4": [10.0, 20.0, 30.0, np.nan, 99.0],
        }
    )

    correlations = rank_scaling.descriptor_correlations(summary)
    row = correlations.loc[
        (correlations["rank_column"] == "r_1e-4")
        & (correlations["descriptor"] == "interface_density")
    ].iloc[0]

    assert int(row["geometry_count"]) == 3
    assert int(row["censored_count"]) == 1
    assert np.isclose(float(row["spearman_rho"]), 1.0)
    assert row["interpretation"] == "exploratory_n10_not_universal_law"


def test_full_rank_memory_plan_bounds_240_cubed_workspaces() -> None:
    gib = 1024**3
    plan = rank_scaling.full_rank_memory_plan(
        nvox=240**3,
        max_rank=192,
        basis_dtype="float32",
        pod_batch_max_gib=8.0,
        affine_stress_max_gib=8.0,
        memory_safety_fraction=0.8,
        available_bytes=118 * gib,
    )

    assert plan["pod_batch_material_limit"] == 4
    assert plan["pod_projection_row_block_size"] == 6
    assert plan["pod_projection_workspace_bytes"] < 2 * gib
    assert plan["affine_q_block_size"] == 4
    assert plan["basis_bytes"] < 60 * gib
    assert plan["estimated_peak_bytes"] < plan["safe_memory_bytes"]
    assert 288 <= plan["max_safe_rank"] < 384


def test_full_rank_memory_plan_exposes_unbounded_rank_request() -> None:
    gib = 1024**3
    plan = rank_scaling.full_rank_memory_plan(
        nvox=240**3,
        max_rank=384,
        basis_dtype="float32",
        pod_batch_max_gib=8.0,
        affine_stress_max_gib=8.0,
        memory_safety_fraction=0.8,
        available_bytes=118 * gib,
    )

    assert plan["estimated_peak_bytes"] > plan["safe_memory_bytes"]


def test_full_rank_memory_plan_shrinks_batches_before_rejecting_rank() -> None:
    gib = 1024**3
    plan = rank_scaling.full_rank_memory_plan(
        nvox=240**3,
        max_rank=288,
        basis_dtype="float32",
        pod_batch_max_gib=8.0,
        affine_stress_max_gib=8.0,
        memory_safety_fraction=0.8,
        max_material_batch=16,
        available_bytes=118 * gib,
    )

    assert plan["pod_batch_material_limit"] == 1
    assert plan["affine_q_block_size"] == 2
    assert plan["estimated_peak_bytes"] <= plan["safe_memory_bytes"]


def test_rom_chunk_memory_decreases_quadratically_with_rank() -> None:
    low = rank_scaling.rom_chunk_memory_plan(
        rank=192,
        requested_chunk_size=640,
        memory_max_gib=1.0,
    )
    high = rank_scaling.rom_chunk_memory_plan(
        rank=768,
        requested_chunk_size=640,
        memory_max_gib=1.0,
    )

    assert low["effective_chunk_size"] == 640
    assert high["effective_chunk_size"] < low["effective_chunk_size"]
    assert high["workspace_bytes_estimate"] <= 1024**3


def test_memory_plan_uses_actual_checkpoint_increment() -> None:
    plan = rank_scaling.full_rank_memory_plan(
        nvox=60**3,
        max_rank=24,
        basis_dtype="float32",
        pod_batch_max_gib=8.0,
        affine_stress_max_gib=8.0,
        memory_safety_fraction=0.8,
        max_material_batch=2,
        available_bytes=118 * 1024**3,
    )

    assert plan["pod_batch_material_limit"] == 2
    assert plan["pod_requested_material_batch"] == 2
    assert plan["pod_workspace_bytes"] == 2 * plan["material_snapshot_bytes"]
