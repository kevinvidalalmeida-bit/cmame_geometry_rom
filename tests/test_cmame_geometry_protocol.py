"""Tests for clustered SAM sampling and same-master voxel refinement."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "FFT") not in sys.path:
    sys.path.insert(0, str(ROOT / "FFT"))

from pipeline.rve_generator import (
    rasterize_continuous_fibers,
    resolve_overlap_tolerance,
)
from pipeline.sam_generator import (
    SAMLiteGenerator,
    _complete_neighbor_csr_numba,
)


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


def test_overlap_tolerance_scales_with_voxelized_diameter():
    parameters = {"sam_overlap_tolerance_relative": 1.0 / 120.0}
    tolerance_6, relative_6 = resolve_overlap_tolerance(parameters, 6.0)
    tolerance_12, relative_12 = resolve_overlap_tolerance(parameters, 12.0)
    assert np.isclose(tolerance_6, 0.05)
    assert np.isclose(tolerance_12, 0.10)
    assert np.isclose(relative_6, 1.0 / 120.0)
    assert np.isclose(relative_12, relative_6)


def test_complete_neighbor_csr_contains_every_other_fiber_once():
    offsets, neighbors = _complete_neighbor_csr_numba(7)
    np.testing.assert_array_equal(offsets, np.arange(8, dtype=np.int32) * 6)
    for fiber_index in range(7):
        row = neighbors[offsets[fiber_index] : offsets[fiber_index + 1]]
        np.testing.assert_array_equal(
            np.sort(row),
            np.delete(np.arange(7, dtype=np.int32), fiber_index),
        )


def test_collective_fire_uses_bounded_static_graph_for_dense_neighbors():
    generator = SAMLiteGenerator(
        A2=np.eye(3) / 3.0,
        fiber_length=5.0,
        fiber_diameter=0.5,
        vf_target=0.01,
        cell_size=(10.0, 10.0, 10.0),
        tol_A=1.0,
        tol_overlap=10.0,
        seed=91,
    )
    generator._ensure_capacity(12)
    generator._centers[:12] = generator.rng.random((12, 3)) * 10.0
    directions = generator.rng.standard_normal((12, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    generator._directions[:12] = directions
    generator._n_fibers = 12

    info = generator.relax_collective_fire(
        max_iter=1,
        target_overlap=10.0,
        static_neighbor_min_density=0.0,
        static_neighbor_max_mib=1.0,
    )
    assert info["converged"] == 1.0
    assert info["static_neighbor_graph"] == 1.0
    assert info["neighbor_builds"] == 1.0


def test_collective_fire_confirms_contacts_with_exact_final_check(monkeypatch):
    generator = SAMLiteGenerator(
        A2=np.diag([1.0, 0.0, 0.0]),
        fiber_length=2.0,
        fiber_diameter=0.5,
        vf_target=0.01,
        cell_size=(5.0, 5.0, 5.0),
        tol_A=1.0,
        tol_overlap=0.01,
        seed=5,
    )
    generator._ensure_capacity(2)
    generator._centers[:2] = np.array([[2.5, 2.5, 2.5]] * 2)
    generator._directions[:2] = np.array([[1.0, 0.0, 0.0]] * 2)
    generator._n_fibers = 2
    real_pair_builder = generator.build_neighbor_pairs_vec
    build_calls = 0

    def omit_only_cached_pairs():
        nonlocal build_calls
        build_calls += 1
        if build_calls == 1:
            return np.empty((0, 2), dtype=np.int32)
        return real_pair_builder()

    monkeypatch.setattr(
        generator,
        "build_neighbor_pairs_vec",
        omit_only_cached_pairs,
    )

    info = generator.relax_collective_fire(
        max_iter=1,
        target_overlap=0.01,
        static_neighbor_min_density=1.1,
    )
    assert info["converged"] == 0.0
    assert info["final_overlap"] > 0.01
