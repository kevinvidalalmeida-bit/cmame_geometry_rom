import numpy as np

from cmame_interpretable_pipeline.constitutive_neighbors import (
    nearest_constitutive_snapshot,
)


def test_constitutive_nearest_neighbor_is_scale_rotation_invariant():
    current = np.stack([np.diag(np.arange(1.0, 7.0)), np.diag(np.arange(5.0, 11.0))])
    previous = np.stack([3.0 * current, current + 2.0 * np.eye(6)])
    source, distance = nearest_constitutive_snapshot(current, previous)
    assert source == 0
    assert distance < 1.0e-14

    rotation, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(6, 6)))
    rotated_source, rotated_distance = nearest_constitutive_snapshot(
        rotation @ current @ rotation.T,
        rotation @ previous @ rotation.T,
    )
    assert rotated_source == source
    assert abs(rotated_distance - distance) < 1.0e-14


def test_constitutive_nearest_neighbor_prefers_latest_equal_snapshot():
    current = np.stack([np.diag(np.arange(1.0, 7.0)), np.diag(np.arange(5.0, 11.0))])
    source, distance = nearest_constitutive_snapshot(current, [current, current])
    assert source == 1
    assert distance < 1.0e-14
