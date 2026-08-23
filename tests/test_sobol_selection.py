"""Deterministic design checks for the full-rank Sobol pipeline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "scripts" / "cmame_interpretable_pipeline"
for path in (ROOT, ROOT / "scripts", ROOT / "src", PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

spec = importlib.util.spec_from_file_location(
    "cmame_sobol_pod_pipeline",
    PIPELINE_DIR / "04_sobol_pod_pipeline.py",
)
assert spec is not None and spec.loader is not None
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_affine_maximin_and_warm_route_preserve_fixed_design():
    candidates = pipeline.validate._build_independent_materials(64, 20260821).copy()
    candidates.insert(0, "candidate_id", np.arange(len(candidates), dtype=int))

    selected_a = pipeline.affine_maximin_sequence(candidates, 10)
    selected_b = pipeline.affine_maximin_sequence(candidates, 10)
    assert selected_a["candidate_id"].is_unique
    assert selected_a["candidate_id"].tolist() == selected_b["candidate_id"].tolist()

    routed = pipeline.fixed_warm_start_route(selected_a)
    assert set(routed["candidate_id"]) == set(selected_a["candidate_id"])
    assert routed["solve_position"].tolist() == list(range(10))
    assert sorted(routed["sobol_set_position"].tolist()) == list(range(10))


def test_default_protocol_uses_adaptive_training_and_post_freeze_validation():
    config = json.loads(
        (PIPELINE_DIR / "campaign_config.json").read_text(encoding="utf-8")
    )["sobol_pod_pipeline"]

    assert pipeline.resolve_training_protocol(
        config["adaptive"], config["monitor_count"], config["adaptive_training_limit"]
    ) == "adaptive"
    assert config["training_limit"] == 14
    assert config["adaptive_training_limit"] == 0
    assert config["adaptive"] is True
    assert config["monitor_count"] == 5
    assert config["final_validation_pool_count"] == 4096
    assert config["final_validation_count"] == 20
    assert config["training_profile"] == "snapshot32"
    assert config["monitor_profile"] == "snapshot32"
    assert config["validation_profile"] == "reference"
    assert config["timing_profile"] == "snapshot32"
    assert config["audit_profile"] == "truth"
    profiles = pipeline.common.SOLVER_PROFILES
    assert profiles[config["training_profile"]] == {
        "solver_real_dtype": "float32",
        "solver_rtol": 1.0e-5,
        "solver_atol": 0.0,
        "solver_maxiter": 1000,
    }
    assert profiles[config["monitor_profile"]] == profiles["snapshot32"]
    assert profiles[config["validation_profile"]] == {
        "solver_real_dtype": "float64",
        "solver_rtol": 1.0e-6,
        "solver_atol": 0.0,
        "solver_maxiter": 5000,
    }
    assert profiles[config["timing_profile"]] == profiles["snapshot32"]
    assert profiles[config["timing_profile"]]["solver_real_dtype"] == "float32"
    assert profiles[config["timing_profile"]]["solver_rtol"] == pytest.approx(1.0e-5)
    assert profiles[config["audit_profile"]] == {
        "solver_real_dtype": "float64",
        "solver_rtol": 1.0e-10,
        "solver_atol": 0.0,
        "solver_maxiter": 5000,
    }
    assert "spd_probe_count" not in config
    assert "ritz_precision_fallback" not in config
    assert config["basis_dtype"] == "float32"
    assert config["ritz_contraction_dtype"] == "float32"
    assert config["ritz_gram_compute_dtype"] == "float64"
    assert config["ritz_gram_backend"] == "cpu"
    assert config["ritz_gram_rank_rtol"] == pytest.approx(1.0e-15)
    assert config["tau_sensitivity_geometry_ids"] == []
    assert config["final_validation_seed"] == 20260901
    assert config["structural_audit_geometry_ids"] == []


def test_frozen_model_hash_precedes_final_validation_design():
    source = (PIPELINE_DIR / "04_sobol_pod_pipeline.py").read_text(encoding="utf-8")
    freeze_position = source.index("frozen_model_hash = sha256(frozen_model_path)")
    validation_position = source.index(
        "validation_candidates, final_validation = maximin_validation_pool(",
        freeze_position,
    )

    assert freeze_position < validation_position
    pod_selection_position = source.index(
        "energy_pod_summary, energy_pod_operators = select_energy_pod_baseline(",
        freeze_position,
    )
    assert freeze_position < pod_selection_position < validation_position


def test_maximin_validation_design_is_deterministic_and_held_out_sized():
    pool_a, selected_a = pipeline.maximin_validation_pool(
        pool_count=128, count=20, seed=20260901
    )
    pool_b, selected_b = pipeline.maximin_validation_pool(
        pool_count=128, count=20, seed=20260901
    )

    assert len(pool_a) == 128
    assert len(selected_a) == 20
    assert selected_a["final_validation_id"].tolist() == list(range(20))
    assert selected_a["validation_pool_id"].is_unique
    assert selected_a["validation_pool_id"].tolist() == selected_b[
        "validation_pool_id"
    ].tolist()
    assert pool_a["validation_pool_id"].tolist() == pool_b[
        "validation_pool_id"
    ].tolist()


def test_empirical_coverage_counts_reduced_numerical_failures():
    summary = pipeline.empirical_coverage([5.0e-5, np.nan, 2.0e-4], 1.0e-4)

    assert summary["observed_count"] == 3
    assert summary["finite_error_count"] == 2
    assert summary["numerical_failure_count"] == 1
    assert summary["below_target_count"] == 1
    assert summary["above_target_count"] == 2
    assert summary["below_target_fraction"] == pytest.approx(1.0 / 3.0)


def test_shared_affine_constitutive_route_reconstructs_every_voxel_group():
    phase = np.zeros((4, 3, 2), dtype=np.uint8)
    phase.reshape(-1)[::2] = 1
    ori = np.zeros(phase.shape + (3,), dtype=np.float64)
    ori[..., 0] = 1.0
    ori.reshape(-1, 3)[1::4] = np.array([0.3, 0.4, np.sqrt(0.75)])
    material = {
        "Em": 3.78,
        "nu_m": 0.35,
        "Ef_L": 290.0,
        "Ef_T": 23.0,
        "G_LT": 9.0,
        "nu_LT": 0.20,
        "nu_TT": 0.40,
    }

    result = pipeline.reduced.affine_constitutive_reconstruction_error(
        phase, ori, material
    )

    assert result["voxel_count"] == phase.size
    assert result["orientation_group_count"] >= 1
    assert result["relative_frobenius_error"] < 1.0e-12
    assert result["maximum_voxel_group_relative_error"] < 1.0e-12


def test_fixed_prefix_tau_sensitivity_changes_only_rank_threshold():
    rng = np.random.default_rng(20260827)
    raw_rank = 12
    raw_Kq = np.zeros((7, raw_rank, raw_rank), dtype=np.float64)
    raw_Bq = np.zeros((7, raw_rank, 6), dtype=np.float64)
    Dq = np.zeros((7, 6, 6), dtype=np.float64)
    raw_Kq[0] = np.eye(raw_rank)
    raw_Bq[0] = 0.01 * rng.standard_normal((raw_rank, 6))
    Dq[0] = np.eye(6)
    G = np.diag(np.geomspace(1.0, 1.0e-8, raw_rank))
    nominal_Kq, nominal_Bq, invR, _ = (
        pipeline.reduced._transform_raw_operators_with_rank_policy(
            raw_Kq,
            raw_Bq,
            G,
            allow_rank_reveal=True,
            rank_rtol=1.0e-6,
        )
    )
    materials = pipeline.validate._build_independent_materials(3, 20260828).copy()
    coefficients = np.stack(
        [pipeline.reduced._material_coefficients(row) for row in materials.to_dict("records")]
    )
    C_batch, _, _ = pipeline.reduced._rom_ceff_batch(
        coefficients, nominal_Kq, nominal_Bq, Dq
    )
    for ii in range(6):
        for jj in range(6):
            materials[f"Ceff_{ii + 1}{jj + 1}"] = C_batch[:, ii, jj]

    summary, details = pipeline.tau_sensitivity_study(
        validation_truth=materials,
        operators={
            "Kq": nominal_Kq,
            "Bq": nominal_Bq,
            "Dq": Dq,
            "raw_Kq": raw_Kq,
            "raw_Bq": raw_Bq,
            "G": G,
            "invR": invR,
        },
        thresholds=[1.0e-4, 1.0e-6, 1.0e-7],
        training_materials=2,
    )

    assert summary["training_materials"].tolist() == [2, 2, 2]
    assert summary["raw_snapshot_columns"].tolist() == [12, 12, 12]
    assert summary["pod_rank"].is_monotonic_increasing
    assert details.groupby("tau_G").size().tolist() == [3, 3, 3]
    assert {
        "schur_eta",
        "reduced_K_min_eig",
        "reduced_K_spectral_spd_margin",
    }.issubset(details.columns)


def test_ritz_schur_difference_matches_microscopic_error_energy():
    rng = np.random.default_rng(20260904)
    microscopic_size = 18
    reduced_size = 7
    loading_count = 6
    factor = rng.standard_normal((microscopic_size, microscopic_size))
    K = factor.T @ factor + 0.75 * np.eye(microscopic_size)
    B = rng.standard_normal((microscopic_size, loading_count))
    D = 4.0 * np.eye(loading_count)
    basis, _ = np.linalg.qr(
        rng.standard_normal((microscopic_size, reduced_size)), mode="reduced"
    )

    X = np.linalg.solve(K, B)
    coordinates = np.linalg.solve(basis.T @ K @ basis, basis.T @ B)
    Xr = basis @ coordinates
    full_schur = D - B.T @ X
    reduced_schur = D - B.T @ Xr
    error = X - Xr
    error_energy = error.T @ K @ error

    np.testing.assert_allclose(
        reduced_schur - full_schur,
        error_energy,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert np.linalg.eigvalsh(0.5 * (error_energy + error_energy.T))[0] > -1.0e-12


def test_conventional_pod_energy_retention_uses_the_frozen_gram_only():
    raw_rank = 6
    raw_Kq = np.zeros((7, raw_rank, raw_rank), dtype=np.float64)
    raw_Bq = np.zeros((7, raw_rank, 6), dtype=np.float64)
    raw_Kq[0] = np.eye(raw_rank)
    raw_Bq[0] = np.eye(raw_rank)
    gram = np.diag([1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001])

    K99, B99, _, meta99 = (
        pipeline.reduced._transform_raw_operators_with_energy_retention(
            raw_Kq, raw_Bq, gram, retention=0.99
        )
    )
    K9999, B9999, _, meta9999 = (
        pipeline.reduced._transform_raw_operators_with_energy_retention(
            raw_Kq, raw_Bq, gram, retention=0.9999
        )
    )

    assert K99.shape[1] == B99.shape[1] == meta99["effective_rank"]
    assert K9999.shape[1] == B9999.shape[1] == meta9999["effective_rank"]
    assert K99.shape[1] < K9999.shape[1]
    assert meta99["pod_energy_retention_realized"] >= 0.99
    assert meta9999["pod_energy_retention_realized"] >= 0.9999


def test_fixed_protocol_requires_an_explicit_training_budget():
    with pytest.raises(ValueError, match="positive training_limit"):
        pipeline.resolve_training_protocol(False, 5, 0)


def test_adaptive_protocol_uses_a_positive_monitor_pool():
    assert pipeline.resolve_training_protocol(True, 5, 0) == "adaptive"
    with pytest.raises(ValueError, match="positive monitor_count"):
        pipeline.resolve_training_protocol(True, 0, 14)
