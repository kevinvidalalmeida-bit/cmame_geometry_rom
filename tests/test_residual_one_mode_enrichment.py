import numpy as np

from scripts.residual_one_mode_enrichment import ResidualOneModeGreedy


def test_residual_gram_and_one_mode_audits():
    rng = np.random.default_rng(31)
    n, q, m = 14, 3, 2
    # A uniformly SPD affine family on a small, deterministic test problem.
    Kq = np.empty((q, n, n))
    Kq[0] = np.eye(n) * 4.0
    for i in range(1, q):
        a = rng.normal(size=(n, n))
        Kq[i] = (a + a.T) / 20.0
    Bq = rng.normal(size=(q, n, m))
    anchor_gamma = np.array([1.0, .2, .2])
    M = np.einsum("q,qij->ij", anchor_gamma, Kq)
    B = np.einsum("q,qij->ij", anchor_gamma, Bq)
    X = np.linalg.solve(M, B)
    V, _ = np.linalg.qr(X)
    # QR is Euclidean: explicitly whiten in the anchor metric.
    L = np.linalg.cholesky(V.T @ M @ V)
    V = V @ np.linalg.inv(L.T)
    greedy = ResidualOneModeGreedy(Kq, Bq, M, V)
    pool = np.array([[1., .1, .1], [1., -.1, .2], [1., .2, -.1]])
    index, values, _ = greedy.select(pool)
    assert index == int(np.argmax(values))
    gamma = pool[index]
    S, C, y = greedy.indicator(gamma)
    R = np.einsum("q,qij->ij", gamma, Bq) - np.einsum("q,qij->ij", gamma, Kq) @ greedy.V @ y
    assert np.allclose(R, greedy.A @ C, atol=1e-11)
    assert np.linalg.norm(greedy.V.T @ R) < 1e-10
    old = greedy.V.copy()
    step = greedy.append_mode(gamma)
    assert greedy.rank == old.shape[1] + 1
    assert np.allclose(greedy.V[:, :-1], old)
    assert step.energy_orthogonality < 1e-9
    # The M-residual indicator is diagnostic only.  Unlike the true
    # K(gamma)-energy Ritz defect, it need not be monotone under enrichment.
    assert np.isfinite(step.eta_after)
    # Incremental Gram update must equal a fresh anchor-metric product.
    assert np.allclose(greedy.G, greedy.A.T @ np.linalg.solve(M, greedy.A), atol=1e-11)
