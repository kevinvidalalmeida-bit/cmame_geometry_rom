from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'scripts/cmame_interpretable_pipeline')]
import pod_baseline as pod


@pytest.mark.parametrize('weighted', [False, True])
def test_snapshot_pod_matches_independent_field_svd_and_affine_outputs(weighted):
    rng = np.random.default_rng(5246)
    snapshots = rng.normal(size=(80, 9))
    weights = np.exp(rng.normal(size=80)) if weighted else np.ones(80)
    gram = snapshots.T @ (weights[:, None] * snapshots)
    c = rng.normal(size=(3, 80, 80))
    c = c.swapaxes(-1, -2) @ c + np.eye(80)
    forcing = rng.normal(size=(3, 80, 6))
    raw_k = snapshots.T @ c @ snapshots
    raw_b = snapshots.T @ forcing
    model, meta = pod.project_snapshot_pod(raw_k, raw_b, gram)
    u, _, _ = np.linalg.svd(np.sqrt(weights)[:, None] * snapshots, full_matrices=False)
    field_modes = u / np.sqrt(weights)[:, None]
    assert meta['numerical_rank'] == 9
    assert meta['metric_orthogonality_error'] < 1e-12
    for rank in [1, 4, 9]:
        q = field_modes[:, :rank]
        # SVD and eigenvectors can differ in sign; compare physical operators.
        k = (q.T @ c @ q).sum(axis=0)
        b = (q.T @ forcing).sum(axis=0)
        reduced_k = model['Kq'][:, :rank, :rank].sum(axis=0)
        reduced_b = model['Bq'][:, :rank].sum(axis=0)
        np.testing.assert_allclose(b.T @ np.linalg.solve(k, b),
                                   reduced_b.T @ np.linalg.solve(reduced_k, reduced_b),
                                   rtol=2e-12, atol=2e-12)


def test_pod_rejects_indefinite_gram_and_removes_only_null_modes():
    k, b = np.eye(3)[None], np.ones((1, 3, 6))
    with pytest.raises(np.linalg.LinAlgError, match='indefinite'):
        pod.project_snapshot_pod(k, b, np.diag([1., .1, -.01]))
    model, meta = pod.project_snapshot_pod(k, b, np.diag([1., .1, 0.]))
    assert model['Kq'].shape == (1, 2, 2)
    assert meta['discarded_null_modes'] == 1


def test_all_rank_selection_chooses_smallest_passing_rank_from_monitors(monkeypatch):
    monkeypatch.setattr(pod.reduced, '_material_coefficients', lambda row: np.ones(1))
    k, b, d = np.eye(6)[None], np.zeros((1, 6, 6)), (10 * np.eye(6))[None]
    b[0, :3, :3] = np.eye(3)
    truth = 10 * np.eye(6) - b[0].T @ b[0]
    monitors = pd.DataFrame([dict(material_id=9,
        **{f'Ceff_{i+1}{j+1}': truth[i, j] for i in range(6) for j in range(6)})])
    rows, selected, _ = pod.select_all_ranks(
        raw_k=k, raw_b=b, dq=d, gram=np.diag([6., 5., 4., 3., 2., 1.]),
        monitors=monitors, target_error=1e-10)
    assert rows['pod_rank'].tolist() == list(range(1, 7))
    assert rows.loc[rows.selected_without_heldout, 'pod_rank'].tolist() == [3]
    assert selected['Kq'].shape == (1, 3, 3)
