"""Snapshot POD with affine output operators and monitor-only rank selection.

The field matrix is never formed online. For S containing the snapshot columns,
G = S.T W S = V diag(lambda) V.T and T = V / sqrt(lambda), the POD basis is
S T. Its operators are T.T Kq T and T.T Bq. W may be the voxel L2 metric or
a fixed constitutive energy metric. Both are ordinary weighted snapshot POD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, eigh

import rom_reduced_operator as reduced


def project_snapshot_pod(raw_k, raw_b, gram, *, rank_rtol=1e-15):
    """Diagonalize once, project once, and expose nested leading POD blocks.

    Significant negative eigenvalues are an error, not silently clipped. Only
    numerical null modes are removed. Input contractions remain the caller's
    responsibility; a small positive eigenvalue alone does not certify them.
    """
    gram = np.asarray(gram, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("POD Gram must be square")
    if not np.isfinite(rank_rtol) or rank_rtol <= 0:
        raise ValueError("POD rank tolerance must be positive and finite")
    eigenvalues, vectors = eigh((gram + gram.T) * 0.5, check_finite=True)
    scale = eigenvalues[-1]
    rounding = max(rank_rtol, 10 * len(gram) * np.finfo(float).eps)
    if scale <= 0 or eigenvalues[0] < -rounding * scale:
        raise np.linalg.LinAlgError("POD Gram has no positive energy or is indefinite")
    keep = eigenvalues > rank_rtol * scale
    values = eigenvalues[keep][::-1]
    transform = vectors[:, keep][:, ::-1] / np.sqrt(values)
    kq = np.asarray([transform.T @ k @ transform for k in raw_k])
    kq = (kq + kq.swapaxes(-1, -2)) * 0.5
    bq = np.asarray([transform.T @ b for b in raw_b])
    return dict(Kq=kq, Bq=bq, invR=transform), dict(
        eigenvalues=values,
        gram_min_eigenvalue=float(eigenvalues[0]),
        gram_max_eigenvalue=float(scale),
        numerical_rank=int(keep.sum()),
        discarded_null_modes=int(len(gram) - keep.sum()),
        metric_orthogonality_error=float(np.linalg.norm(
            transform.T @ gram @ transform - np.eye(keep.sum()), ord=2)),
    )


def select_all_ranks(*, raw_k, raw_b, dq, gram, monitors, target_error,
                     rank_rtol=1e-15):
    """Choose the smallest passing rank, without any held-out data argument."""
    if monitors.empty:
        raise ValueError("POD rank selection requires independent monitors")
    if not np.isfinite(target_error) or target_error <= 0:
        raise ValueError("POD target error must be positive and finite")
    projected, metadata = project_snapshot_pod(raw_k, raw_b, gram, rank_rtol=rank_rtol)
    coefficients = np.asarray([
        reduced._material_coefficients(row.to_dict()) for _, row in monitors.iterrows()
    ])
    truth = np.asarray([reduced._full_ceff_from_row(row) for _, row in monitors.iterrows()])
    truth = (truth + truth.swapaxes(-1, -2)) * 0.5
    norms = np.linalg.norm(truth, axis=(-1, -2))
    if np.any(norms <= 0) or not np.all(np.isfinite(truth)):
        raise ValueError("Monitor tensors must have finite positive norms")
    ks, bs, ds = (np.einsum('nq,qij->nij', coefficients, value)
                  for value in (projected['Kq'], projected['Bq'], dq))
    energies = metadata['eigenvalues']
    cumulative = np.cumsum(energies) / energies.sum()
    rows, selected = [], None
    for rank in range(1, len(energies) + 1):
        predictions = []
        failure = None
        for k, b, d in zip(ks, bs, ds):
            try:
                factor = cho_factor(k[:rank, :rank], lower=True, check_finite=False)
                br = b[:rank]
                c = d - br.T @ cho_solve(factor, br, check_finite=False)
                c = (c + c.T) * 0.5
                np.linalg.cholesky(c)
                predictions.append(c)
            except np.linalg.LinAlgError as exc:
                failure = str(exc)
                break
        errors = (np.linalg.norm(np.asarray(predictions) - truth, axis=(-1, -2)) / norms
                  if failure is None else np.full(len(truth), np.inf))
        passes = bool(np.all(np.isfinite(errors)) and errors.max() <= target_error)
        chosen = bool(passes and selected is None)
        rows.append(dict(
            pod_rank=rank, pod_energy_retention=float(cumulative[rank-1]),
            pod_energy_retention_realized=float(cumulative[rank-1]),
            monitor_error_mean=float(errors.mean()), monitor_error_max=float(errors.max()),
            passes_monitor_target=passes, selected_without_heldout=chosen,
            numerical_failure=failure,
        ))
        if chosen:
            selected = dict(Kq=projected['Kq'][:, :rank, :rank].copy(),
                            Bq=projected['Bq'][:, :rank].copy(), Dq=np.asarray(dq).copy(),
                            invR=projected['invR'][:, :rank].copy())
    return pd.DataFrame(rows), selected, metadata
