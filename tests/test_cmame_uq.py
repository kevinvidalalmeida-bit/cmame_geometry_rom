"""Vectorized UQ algebra must match the established scalar ROM implementation."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "FFT"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cmame_uq
import rom_reduced_operator as reduced


def test_vectorized_material_coefficients_match_scalar_definition():
    parameters = cmame_uq._samples("global_uniform", 8, 13)
    vectorized = cmame_uq._material_coefficients_batch(parameters)
    for index, values in enumerate(parameters):
        row = dict(zip(cmame_uq.PARAMETER_NAMES, values))
        scalar = reduced._material_coefficients(row)
        np.testing.assert_allclose(vectorized[index], scalar, rtol=2.0e-13, atol=2.0e-13)


def test_truncated_normal_samples_remain_inside_domain():
    values = cmame_uq._samples("local_truncated_normal", 1000, 19)
    lower = np.array([cmame_uq.sweep.MATERIAL_BOUNDS[name][0] for name in cmame_uq.PARAMETER_NAMES])
    upper = np.array([cmame_uq.sweep.MATERIAL_BOUNDS[name][1] for name in cmame_uq.PARAMETER_NAMES])
    assert np.all(values >= lower)
    assert np.all(values <= upper)


def test_uq_convergence_is_measured_against_final_sample():
    rng = np.random.default_rng(23)
    outputs = rng.normal(size=(1000, len(cmame_uq.OUTPUT_NAMES)))

    _, convergence = cmame_uq._summaries("test", outputs)
    final = convergence[convergence["sample_count"] == 1000]

    assert len(final) == len(cmame_uq.OUTPUT_NAMES)
    np.testing.assert_allclose(final["mean_relative_change_vs_final"], 0.0, atol=0.0)
    np.testing.assert_allclose(final["q05_relative_change_vs_final"], 0.0, atol=0.0)
    np.testing.assert_allclose(final["q95_relative_change_vs_final"], 0.0, atol=0.0)
    assert np.all(final["mean_ci95_low"] <= final["mean"])
    assert np.all(final["mean_ci95_high"] >= final["mean"])
