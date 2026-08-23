"""Checks for the post-freeze RBF and Kriging benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "scripts" / "cmame_interpretable_pipeline"
for path in (ROOT, ROOT / "scripts", ROOT / "src", PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

spec = importlib.util.spec_from_file_location(
    "cmame_surrogate_baselines",
    PIPELINE_DIR / "06_surrogate_baselines.py",
)
assert spec is not None and spec.loader is not None
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_independent_tensor_coordinates_round_trip():
    rng = np.random.default_rng(20260830)
    raw = rng.standard_normal((4, 6, 6))
    tensors = 0.5 * (raw + np.swapaxes(raw, -1, -2))

    restored = baseline.independent_to_tensors(
        baseline.tensors_to_independent(tensors)
    )

    np.testing.assert_allclose(restored, tensors)


def test_linear_rbf_interpolates_training_outputs():
    train_x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float
    )
    train_y = np.column_stack(
        [train_x[:, 0] + 2.0 * train_x[:, 1], train_x[:, 0] - train_x[:, 1]]
    )

    prediction = baseline.rbf_predict(train_x, train_y, train_x)

    np.testing.assert_allclose(prediction, train_y, atol=1.0e-12)


def test_rbf_search_space_covers_kernels_shape_and_smoothing():
    train_x = np.linspace(0.0, 1.0, 5)[:, None]
    configurations = baseline.rbf_configurations(train_x)

    assert {str(item["kernel"]) for item in configurations} == {
        *baseline.RBF_KERNELS_WITHOUT_SHAPE,
        *baseline.RBF_KERNELS_WITH_SHAPE,
    }
    assert {float(item["smoothing"]) for item in configurations} == set(
        baseline.RBF_SMOOTHING_VALUES
    )
    gaussian = [item for item in configurations if item["kernel"] == "gaussian"]
    assert {float(item["epsilon_multiplier"]) for item in gaussian} == set(
        baseline.RBF_EPSILON_MULTIPLIERS
    )


def test_rbf_hyperparameter_selection_is_training_only_and_deterministic():
    train_x = np.linspace(0.0, 1.0, 5)[:, None]
    scalar = 2.0 + np.sin(2.0 * np.pi * train_x[:, 0])
    train_tensors = np.stack([np.eye(6) * value for value in scalar])

    selected_a = baseline.select_rbf_configuration(train_x, train_tensors)
    selected_b = baseline.select_rbf_configuration(train_x, train_tensors)

    assert selected_a == selected_b
    assert np.isfinite(selected_a[1])
    assert np.isfinite(selected_a[2])


def test_kriging_prediction_is_deterministic():
    train_x = np.linspace(0.0, 1.0, 6)[:, None]
    train_y = np.column_stack((np.sin(train_x[:, 0]), np.cos(train_x[:, 0])))
    query_x = np.array([[0.15], [0.55], [0.95]])

    result_a = baseline.select_kriging_and_predict(
        train_x,
        train_y,
        query_x,
        selection_restarts=0,
        final_restarts=1,
        random_seed=20260830,
    )
    result_b = baseline.select_kriging_and_predict(
        train_x,
        train_y,
        query_x,
        selection_restarts=0,
        final_restarts=1,
        random_seed=20260830,
    )

    np.testing.assert_allclose(result_a[0], result_b[0])
    assert result_a[1:4] == result_b[1:4]


def test_kriging_selection_compares_all_declared_covariance_families():
    train_x = np.linspace(0.0, 1.0, 6)[:, None]
    train_y = np.column_stack((np.sin(train_x[:, 0]), np.cos(train_x[:, 0])))
    query_x = np.array([[0.2], [0.8]])

    prediction, _, selected, score, candidates = baseline.select_kriging_and_predict(
        train_x,
        train_y,
        query_x,
        selection_restarts=0,
        final_restarts=0,
        random_seed=20260830,
    )

    declared = {name for name, _ in baseline.kriging_kernel_candidates(1)}
    assert {str(item["family"]) for item in candidates} == declared
    assert selected in declared
    assert prediction.shape == (2, 2)
    assert np.isfinite(score)


def test_final_prefix_selection_keeps_all_methods_and_heldout_rows():
    rows = []
    for geometry_id, final_count in ((3, 4), (9, 5)):
        for count in range(2, final_count + 1):
            for method in baseline.METHOD_LABELS:
                for validation_id in range(3):
                    rows.append(
                        {
                            "geometry_id": geometry_id,
                            "training_materials": count,
                            "method": method,
                            "final_validation_id": validation_id,
                        }
                    )
    selected = baseline.final_prefix_rows(pd.DataFrame(rows))

    assert len(selected) == 2 * 3 * 3
    assert set(selected["method"]) == set(baseline.METHOD_LABELS)
    assert selected.groupby("geometry_id")["training_materials"].unique().map(
        len
    ).eq(1).all()
