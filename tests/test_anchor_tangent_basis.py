"""Accuracy checks for streamed anchor-tangent basis enrichment."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "scripts" / "cmame_interpretable_pipeline" / "experiments"
for path in (ROOT, ROOT / "scripts", ROOT / "src", EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cmame_anchor_tangent_adaptive as adaptive


def test_fused_gpu_cgs2_preserves_reference_subspace():
    rng = np.random.default_rng(20260818)
    nvox = 24
    dimension = 6 * nvox

    old_q, _ = np.linalg.qr(rng.standard_normal((dimension, 7)))
    old_matrix = old_q.T.astype(np.float32)
    existing = [old_matrix[index].reshape(6, nvox) for index in range(7)]

    raw = rng.standard_normal((6, dimension))
    raw += 5.0 * rng.standard_normal((6, 7)) @ old_q.T
    residual = raw - (raw @ old_q) @ old_q.T
    reference_q, _ = np.linalg.qr(residual.T)

    appended = adaptive.append_block_orthonormal(
        existing,
        [raw.astype(np.float32).reshape(6, 6, nvox)],
        tolerance=1.0e-12,
        verbose=False,
        chunk_size=65536,
    )

    assert len(appended) == 6
    appended_matrix = np.stack(appended).reshape(6, dimension).astype(np.float64)
    combined = np.concatenate((old_matrix.astype(np.float64), appended_matrix))
    np.testing.assert_allclose(
        combined @ combined.T,
        np.eye(len(combined)),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    singular_values = np.linalg.svd(appended_matrix @ reference_q, compute_uv=False)
    np.testing.assert_allclose(singular_values, 1.0, rtol=2.0e-5, atol=2.0e-5)
