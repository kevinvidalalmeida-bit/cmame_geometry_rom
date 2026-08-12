"""Regression tests for the strict relative-residual CG contract."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FFTHOMPY = ROOT / "FFT" / "ffthompy_core" / "ffthompy"
if str(FFTHOMPY) not in sys.path:
    sys.path.insert(0, str(FFTHOMPY))

from ffthompy.general.solver import CG


def _diagonal_problem(scale: float = 1.0):
    diagonal = np.array([1.0, 2.0, 5.0, 11.0], dtype=np.float64)
    rhs = scale * np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float64)
    return lambda value: diagonal * value, rhs, np.zeros_like(rhs), diagonal


def test_cg_relative_tolerance_is_scale_invariant():
    results = []
    for scale in (1.0, 1.0e9):
        operator, rhs, x0, diagonal = _diagonal_problem(scale)
        solution, info = CG(
            operator,
            rhs,
            x0,
            par={"rtol": 1.0e-12, "atol": 0.0, "maxiter": 20},
        )
        np.testing.assert_allclose(solution, rhs / diagonal, rtol=1.0e-11, atol=1.0e-11)
        assert info["converged"]
        assert info["norm_res_rel"] <= 1.0e-12
        results.append(info)

    assert results[0]["kit"] == results[1]["kit"]
    assert max(item["norm_res_rel"] for item in results) < 1.0e-12


def test_cg_legacy_tol_is_relative_alias():
    operator, rhs, x0, _ = _diagonal_problem()
    _, info = CG(operator, rhs, x0, par={"tol": 1.0e-10, "maxiter": 20})
    assert info["converged"]
    assert info["threshold"] == 1.0e-10 * info["rhs_norm"]


def test_cg_reports_nonconvergence_at_maxiter():
    operator, rhs, x0, _ = _diagonal_problem()
    _, info = CG(
        operator,
        rhs,
        x0,
        par={"rtol": 1.0e-14, "atol": 0.0, "maxiter": 1},
    )
    assert not info["converged"]
    assert info["hit_maxiter"]
    assert info["kit"] == 1
