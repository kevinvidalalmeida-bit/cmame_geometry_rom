"""Session-cache checks for fixed-grid FFT projection operators."""

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "FFT") not in sys.path:
    sys.path.insert(0, str(ROOT / "FFT"))
FFTHOMPY = ROOT / "FFT" / "ffthompy_core" / "ffthompy"
if str(FFTHOMPY) not in sys.path:
    sys.path.insert(0, str(FFTHOMPY))

from ffthompy.applications import clear_elasticity_projection_cache
from pipeline.fft_solver import solve_homogenization


def test_second_fixed_grid_solve_reuses_projection(tmp_path: Path):
    clear_elasticity_projection_cache()
    n = 7
    phase = np.zeros((n, n, n), dtype=np.uint8)
    phase[:3] = 1
    ori = np.zeros((n, n, n, 3), dtype=np.float64)
    ori[..., 0] = 1.0
    base = {
        "input_dir": str(tmp_path),
        "seed": 0,
        "phase_array": phase,
        "ori_array": ori,
        "Em": 3.0,
        "nu_m": 0.31,
        "Ef_L": 11.0,
        "Ef_T": 11.0,
        "nu_LT": 0.23,
        "nu_TT": 0.23,
        "G_LT": 11.0 / (2.0 * 1.23),
        "fft_backend": "scipy",
        "solver_profile": "truth",
        "solver_maxiter": 500,
        "projection_storage": "full",
        "projection_backend": "numpy",
        "cache_projection": True,
    }
    first_path = tmp_path / "first_timing.json"
    second_path = tmp_path / "second_timing.json"
    first = solve_homogenization({**base, "solver_timing_path": str(first_path)})
    second = solve_homogenization({**base, "solver_timing_path": str(second_path)})
    first_timing = json.loads(first_path.read_text(encoding="utf-8"))
    second_timing = json.loads(second_path.read_text(encoding="utf-8"))
    first_app = first_timing["ffthompy_application_timing"]
    second_app = second_timing["ffthompy_application_timing"]

    np.testing.assert_allclose(second, first, rtol=2.0e-13, atol=2.0e-13)
    assert first_app["projection_cache_enabled"] is True
    assert first_app["projection_cache_hit"] is False
    assert second_app["projection_cache_hit"] is True
    assert second_app["projection_s"] < first_app["projection_s"]


def test_float64_sym21_projection_matches_full_projection(tmp_path: Path):
    clear_elasticity_projection_cache()
    n = 7
    phase = np.zeros((n, n, n), dtype=np.uint8)
    phase[:3] = 1
    ori = np.zeros((n, n, n, 3), dtype=np.float64)
    ori[..., 0] = 1.0
    base = {
        "input_dir": str(tmp_path),
        "seed": 1,
        "phase_array": phase,
        "ori_array": ori,
        "Em": 3.0,
        "nu_m": 0.31,
        "Ef_L": 17.0,
        "Ef_T": 9.0,
        "nu_LT": 0.23,
        "nu_TT": 0.27,
        "G_LT": 4.2,
        "fft_backend": "scipy",
        "solver_profile": "reference",
        "solver_maxiter": 500,
        "projection_backend": "numpy",
        "cache_projection": False,
    }
    full = solve_homogenization({**base, "projection_storage": "full"})
    packed = solve_homogenization({**base, "projection_storage": "sym21"})

    np.testing.assert_allclose(packed, full, rtol=2.0e-12, atol=2.0e-12)
