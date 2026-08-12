"""Tests for clustered SAM sampling and same-master voxel refinement."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "FFT") not in sys.path:
    sys.path.insert(0, str(ROOT / "FFT"))

from pipeline.rve_generator import rasterize_continuous_fibers
from pipeline.sam_generator import SAMLiteGenerator


def _clustered_generator(seed: int) -> SAMLiteGenerator:
    return SAMLiteGenerator(
        A2=np.eye(3) / 3.0,
        fiber_length=3.0,
        fiber_diameter=0.4,
        vf_target=0.1,
        cell_size=(10.0, 10.0, 10.0),
        seed=seed,
        center_distribution="gaussian_mixture",
        cluster_fraction=0.7,
        cluster_count=3,
        cluster_sigma_rel=0.1,
    )


def test_periodic_gaussian_mixture_is_reproducible_and_wrapped():
    first = _clustered_generator(44)
    second = _clustered_generator(44)
    points_a = first._sample_candidate_centers(128, progress=0.0, hard_d=0.4)
    points_b = second._sample_candidate_centers(128, progress=0.0, hard_d=0.4)
    np.testing.assert_allclose(points_a, points_b)
    assert np.all(points_a >= 0.0)
    assert np.all(points_a < 10.0)


def test_rasterization_uses_every_master_fiber(tmp_path: Path):
    fibers = pd.DataFrame(
        {
            "cx_um": [1.0, 3.0],
            "cy_um": [2.0, 2.5],
            "cz_um": [2.0, 3.0],
            "ux": [1.0, 0.0],
            "uy": [0.0, 1.0],
            "uz": [0.0, 0.0],
            "L_um": [2.0, 2.0],
            "d_um": [0.5, 0.5],
        }
    )
    result = rasterize_continuous_fibers(
        fibers,
        caja_um=4.0,
        resolution=4.0,
        output_dir=tmp_path,
    )
    assert result["metadata"]["fiber_count"] == 2
    assert result["metadata"]["grid_size"] == 16
    assert result["metadata"]["voxel_count"] > 0
    assert (tmp_path / "phase.npy").is_file()
    assert (tmp_path / "raster_manifest.json").is_file()
