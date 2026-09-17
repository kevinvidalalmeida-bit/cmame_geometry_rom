"""One-mode residual greedy enrichment for affine Ritz ROMs.

This module is deliberately independent of the Sobol--POD campaign.  It is a
diagnostic building block: callers provide already assembled affine operators
and may therefore run it on a frozen G00/G09 geometry without changing the
production compiler.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GreedyStep:
    candidate_index: int
    eta_before: float
    eta_after: float
    chi: float
    dominant_load: np.ndarray
    galerkin_residual: float
    energy_orthogonality: float


class ResidualOneModeGreedy:
    """Exact anchor-metric residual estimator with incremental Gram updates.

    ``Kq`` has shape ``(q,n,n)``, ``Bq`` ``(q,n,m)``, and every candidate is a
    row of affine coefficients.  The anchor matrix must be SPD.  Only the
    newly appended ``K_q v`` columns are solved with the anchor on extension;
    old Gram entries are retained unchanged.
    """

    def __init__(self, Kq: np.ndarray, Bq: np.ndarray, anchor: np.ndarray, V: np.ndarray):
        self.Kq = np.asarray(Kq, dtype=float)
        self.Bq = np.asarray(Bq, dtype=float)
        self.M = np.asarray(anchor, dtype=float)
        self.V = np.asarray(V, dtype=float).copy()
        q, n, n2 = self.Kq.shape
        if n != n2 or self.Bq.shape[:2] != (q, n) or self.M.shape != (n, n):
            raise ValueError("incompatible affine operator shapes")
        self.q, self.n, self.m = q, n, self.Bq.shape[2]
        self._rebuild_initial_gram()

    @property
    def rank(self) -> int:
        return self.V.shape[1]

    def _rebuild_initial_gram(self) -> None:
        columns = [self.Bq[q] for q in range(self.q)]
        columns += [self.Kq[q] @ self.V for q in range(self.q)]
        self.A = np.concatenate(columns, axis=1)
        self.MinvA = np.linalg.solve(self.M, self.A)
        self.G = self.A.T @ self.MinvA
        self.G = 0.5 * (self.G + self.G.T)

    def _coefficients(self, gamma: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
        """Return C satisfying R=A C for one candidate and all macro loads."""
        gamma = np.asarray(gamma, dtype=float)
        if gamma.shape != (self.q,) or amplitudes.shape != (self.rank, self.m):
            raise ValueError("invalid coefficients or reduced amplitudes")
        b = np.concatenate([gamma[q] * np.eye(self.m) for q in range(self.q)], axis=0)
        kv = np.concatenate([-gamma[q] * amplitudes for q in range(self.q)], axis=0)
        return np.concatenate((b, kv), axis=0)

    def reduced_amplitudes(self, gamma: np.ndarray) -> np.ndarray:
        K = np.einsum("q,qij->ij", gamma, self.Kq)
        B = np.einsum("q,qij->ij", gamma, self.Bq)
        Kr = self.V.T @ K @ self.V
        Br = self.V.T @ B
        return np.linalg.solve(Kr, Br)

    def indicator(self, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = self.reduced_amplitudes(gamma)
        C = self._coefficients(gamma, y)
        S = C.T @ self.G @ C
        return 0.5 * (S + S.T), C, y

    def select(self, pool: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        values = []
        details = []
        for gamma in np.asarray(pool, dtype=float):
            S, C, y = self.indicator(gamma)
            values.append(float(np.linalg.eigvalsh(S)[-1]))
            details.append((S, C, y))
        idx = int(np.argmax(values))
        return idx, np.asarray(values), details[idx][0]

    def append_mode(self, gamma: np.ndarray) -> GreedyStep:
        S, C, _ = self.indicator(gamma)
        eig, vectors = np.linalg.eigh(S)
        eta_before = float(eig[-1])
        direction = vectors[:, -1]
        residual = self.A @ C @ direction
        mode = np.linalg.solve(self.M, residual)
        # One M-orthogonalization is sufficient in exact arithmetic; repeat
        # once only when roundoff leaves a visible component.
        coeff = self.V.T @ self.M @ mode
        mode -= self.V @ coeff
        if np.linalg.norm(self.V.T @ self.M @ mode) > 1e-10 * max(np.linalg.norm(mode), 1.0):
            mode -= self.V @ (self.V.T @ self.M @ mode)
        mode /= np.sqrt(mode @ self.M @ mode)
        orth = float(np.linalg.norm(self.V.T @ self.M @ mode))
        self._append_gram_columns(mode)
        self.V = np.column_stack((self.V, mode))
        eta_after = float(np.linalg.eigvalsh(self.indicator(gamma)[0])[-1])
        return GreedyStep(-1, eta_before, eta_after, 1.0 - eta_after / max(eta_before, np.finfo(float).eps), direction,
                          float(np.linalg.norm(self.V[:, :-1].T @ residual)), orth)

    def _append_gram_columns(self, mode: np.ndarray) -> None:
        new = np.concatenate([self.Kq[q] @ mode[:, None] for q in range(self.q)], axis=1)
        minv_new = np.linalg.solve(self.M, new)
        cross = self.A.T @ minv_new
        diagonal = new.T @ minv_new
        self.G = np.block([[self.G, cross], [cross.T, diagonal]])
        self.A = np.column_stack((self.A, new))
        self.MinvA = np.column_stack((self.MinvA, minv_new))
