"""Genera RVEs voxelizados con SAMLite para el pipeline FFT."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

try:
    from .sam_generator import SAMLiteGenerator
except ImportError:
    from pipeline.sam_generator import SAMLiteGenerator

try:
    import numba

    NUMBA_AVAILABLE = True
except Exception:
    numba = None
    NUMBA_AVAILABLE = False


_INFLATION_CACHE: Dict[Tuple[Any, ...], float] = {}
_INFLATION_CACHE_LOCK = threading.Lock()


def metric_cv_local_vf(phase, n_div=10):
    N = phase.shape[0]
    step = N // n_div
    if step == 0:
        return 0.0
    ph = phase[: step * n_div, : step * n_div, : step * n_div]
    blk = ph.reshape(n_div, step, n_div, step, n_div, step)
    vfs = blk.mean(axis=(1, 3, 5)).ravel()
    mu = vfs.mean()
    return 0.0 if mu < 1e-12 else float(vfs.std() / mu)


def interface_density_periodic(phase: np.ndarray) -> float:
    """Fraction of periodic nearest-neighbor faces crossing a phase boundary."""
    values = np.asarray(phase, dtype=np.uint8)
    crossings = sum(
        int(np.count_nonzero(values != np.roll(values, shift=1, axis=axis)))
        for axis in range(3)
    )
    return float(crossings / max(3 * values.size, 1))


def _pairwise_distances_periodic_fast(centers, box_size):
    n = len(centers)
    if n < 2:
        return np.array([], dtype=np.float64)
    diff = centers[:, None, :] - centers[None, :, :]
    diff -= box_size * np.round(diff / box_size)
    dist_mat = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    idx = np.triu_indices(n, k=1)
    return dist_mat[idx]


def ripley_k_3d(points, box_size, radii):
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 2:
        return np.zeros_like(radii, dtype=np.float64)
    V   = float(box_size**3)
    lam = n / V
    dists = np.sort(_pairwise_distances_periodic_fast(points, box_size))
    counts = np.searchsorted(dists, radii, side='right')
    K = (2.0 * counts) / (lam * n)
    return K


def pair_correlation_3d(points, box_size, r_max, dr=1.0):
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 2:
        return np.zeros(1), np.zeros(1)
    V   = float(box_size**3)
    rho = n / V
    bins = np.arange(0, r_max + dr, dr)
    r_centers = 0.5 * (bins[1:] + bins[:-1])
    dists = _pairwise_distances_periodic_fast(points, box_size)
    hist, _ = np.histogram(dists, bins=bins)
    shell_volume = (4.0/3.0) * np.pi * (bins[1:]**3 - bins[:-1]**3)
    g_r = (2.0 * hist.astype(np.float64)) / (rho * n * shell_volume)
    return r_centers, g_r


def compute_star_discrepancy(centers, N, n_samples=4000):
    if len(centers) == 0:
        return 0.0
    C = np.mod(centers, N) / N
    rng = np.random.default_rng(0)
    U = rng.uniform(0., 1., (n_samples, 3))
    dstar = 0.0
    for s in range(0, n_samples, 500):
        Ub = U[s:s+500]
        inside = np.all(C[:, None, :] < Ub[None, :, :], axis=2)
        A = inside.mean(axis=0)
        Vol = Ub[:, 0] * Ub[:, 1] * Ub[:, 2]
        d_ = np.abs(A - Vol).max()
        if d_ > dstar:
            dstar = d_
    return float(dstar)


def _empty_index_triplet() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
    )


def _fiber_voxel_indices_python(
    center: np.ndarray,
    direction: np.ndarray,
    fiber_length: float,
    fiber_radius: float,
    N: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    P0 = center - 0.5 * fiber_length * direction
    P1 = center + 0.5 * fiber_length * direction
    u_arr = P1 - P0
    uu = float(u_arr @ u_arr)
    if uu <= 1e-12:
        return _empty_index_triplet()

    lo = np.floor(np.minimum(P0, P1) - (fiber_radius + 1.0)).astype(int)
    hi = np.ceil(np.maximum(P0, P1) + (fiber_radius + 1.0)).astype(int)
    xs = np.arange(lo[0], hi[0] + 1, dtype=np.float64)
    ys = np.arange(lo[1], hi[1] + 1, dtype=np.float64)
    zs = np.arange(lo[2], hi[2] + 1, dtype=np.float64)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    t_r = np.clip(
        ((gx - P0[0]) * u_arr[0] + (gy - P0[1]) * u_arr[1] + (gz - P0[2]) * u_arr[2]) / uu,
        0.0,
        1.0,
    )
    dist2_r = (
        (gx - (P0[0] + t_r * u_arr[0])) ** 2
        + (gy - (P0[1] + t_r * u_arr[1])) ** 2
        + (gz - (P0[2] + t_r * u_arr[2])) ** 2
    )
    mask_r = dist2_r <= fiber_radius * fiber_radius
    if not np.any(mask_r):
        return _empty_index_triplet()

    ix_v = gx[mask_r].astype(int) % N
    iy_v = gy[mask_r].astype(int) % N
    iz_v = gz[mask_r].astype(int) % N
    return ix_v, iy_v, iz_v


if NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def _fiber_voxel_indices(  # type: ignore[misc]
        center: np.ndarray,
        direction: np.ndarray,
        fiber_length: float,
        fiber_radius: float,
        N: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p0x = center[0] - 0.5 * fiber_length * direction[0]
        p0y = center[1] - 0.5 * fiber_length * direction[1]
        p0z = center[2] - 0.5 * fiber_length * direction[2]
        p1x = center[0] + 0.5 * fiber_length * direction[0]
        p1y = center[1] + 0.5 * fiber_length * direction[1]
        p1z = center[2] + 0.5 * fiber_length * direction[2]

        ux = p1x - p0x
        uy = p1y - p0y
        uz = p1z - p0z
        uu = ux * ux + uy * uy + uz * uz
        if uu <= 1e-12:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
            )

        lo0 = int(np.floor(min(p0x, p1x) - (fiber_radius + 1.0)))
        lo1 = int(np.floor(min(p0y, p1y) - (fiber_radius + 1.0)))
        lo2 = int(np.floor(min(p0z, p1z) - (fiber_radius + 1.0)))
        hi0 = int(np.ceil(max(p0x, p1x) + (fiber_radius + 1.0)))
        hi1 = int(np.ceil(max(p0y, p1y) + (fiber_radius + 1.0)))
        hi2 = int(np.ceil(max(p0z, p1z) + (fiber_radius + 1.0)))

        r2 = fiber_radius * fiber_radius
        count = 0
        for ix in range(lo0, hi0 + 1):
            fx = float(ix)
            for iy in range(lo1, hi1 + 1):
                fy = float(iy)
                for iz in range(lo2, hi2 + 1):
                    fz = float(iz)
                    t = ((fx - p0x) * ux + (fy - p0y) * uy + (fz - p0z) * uz) / uu
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    cx = p0x + t * ux
                    cy = p0y + t * uy
                    cz = p0z + t * uz
                    dx = fx - cx
                    dy = fy - cy
                    dz = fz - cz
                    if dx * dx + dy * dy + dz * dz <= r2:
                        count += 1

        if count == 0:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
            )

        ix_v = np.empty(count, dtype=np.int64)
        iy_v = np.empty(count, dtype=np.int64)
        iz_v = np.empty(count, dtype=np.int64)
        pos = 0
        for ix in range(lo0, hi0 + 1):
            fx = float(ix)
            for iy in range(lo1, hi1 + 1):
                fy = float(iy)
                for iz in range(lo2, hi2 + 1):
                    fz = float(iz)
                    t = ((fx - p0x) * ux + (fy - p0y) * uy + (fz - p0z) * uz) / uu
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    cx = p0x + t * ux
                    cy = p0y + t * uy
                    cz = p0z + t * uz
                    dx = fx - cx
                    dy = fy - cy
                    dz = fz - cz
                    if dx * dx + dy * dy + dz * dz <= r2:
                        ix_v[pos] = ix % N
                        iy_v[pos] = iy % N
                        iz_v[pos] = iz % N
                        pos += 1
        return ix_v, iy_v, iz_v

    @numba.njit(parallel=True, cache=True)
    def _count_and_fill_fiber_indices_parallel(
        centers: np.ndarray,
        directions: np.ndarray,
        fiber_length: float,
        fiber_radius: float,
        N: int,
        max_vox_per_fiber: int,
        ix_out: np.ndarray,
        iy_out: np.ndarray,
        iz_out: np.ndarray,
        counts_out: np.ndarray,
    ) -> None:
        """Pasada única: cuenta y llena índices de vóxel para cada fibra."""
        n_fibers = centers.shape[0]
        for i in numba.prange(n_fibers):
            center = centers[i]
            direction = directions[i]

            p0x = center[0] - 0.5 * fiber_length * direction[0]
            p0y = center[1] - 0.5 * fiber_length * direction[1]
            p0z = center[2] - 0.5 * fiber_length * direction[2]
            p1x = center[0] + 0.5 * fiber_length * direction[0]
            p1y = center[1] + 0.5 * fiber_length * direction[1]
            p1z = center[2] + 0.5 * fiber_length * direction[2]

            ux = p1x - p0x
            uy = p1y - p0y
            uz = p1z - p0z
            uu = ux * ux + uy * uy + uz * uz

            if uu <= 1e-12:
                counts_out[i] = 0
                continue

            lo0 = int(np.floor(min(p0x, p1x) - (fiber_radius + 1.0)))
            lo1 = int(np.floor(min(p0y, p1y) - (fiber_radius + 1.0)))
            lo2 = int(np.floor(min(p0z, p1z) - (fiber_radius + 1.0)))
            hi0 = int(np.ceil(max(p0x, p1x) + (fiber_radius + 1.0)))
            hi1 = int(np.ceil(max(p0y, p1y) + (fiber_radius + 1.0)))
            hi2 = int(np.ceil(max(p0z, p1z) + (fiber_radius + 1.0)))

            r2 = fiber_radius * fiber_radius
            base = i * max_vox_per_fiber
            pos = 0
            for ix in range(lo0, hi0 + 1):
                fx = float(ix)
                for iy in range(lo1, hi1 + 1):
                    fy = float(iy)
                    for iz in range(lo2, hi2 + 1):
                        fz = float(iz)
                        t = ((fx - p0x) * ux + (fy - p0y) * uy + (fz - p0z) * uz) / uu
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0
                        cx = p0x + t * ux
                        cy = p0y + t * uy
                        cz = p0z + t * uz
                        dx = fx - cx
                        dy = fy - cy
                        dz = fz - cz
                        if dx * dx + dy * dy + dz * dz <= r2:
                            if pos < max_vox_per_fiber:
                                ix_out[base + pos] = ix % N
                                iy_out[base + pos] = iy % N
                                iz_out[base + pos] = iz % N
                                pos += 1
            counts_out[i] = pos

else:

    def _fiber_voxel_indices(
        center: np.ndarray,
        direction: np.ndarray,
        fiber_length: float,
        fiber_radius: float,
        N: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _fiber_voxel_indices_python(center, direction, fiber_length, fiber_radius, N)


def _rasterize_fibers_python(
    centers: np.ndarray,
    directions: np.ndarray,
    fiber_length: float,
    fiber_radius: float,
    N: int,
    voxel_target: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    phase = np.zeros((N, N, N), dtype=np.uint8)
    ori = np.zeros((N, N, N, 3), dtype=np.float32)
    voxel_count = 0
    used_fibers = 0

    for c, a in zip(centers, directions):
        ix_v, iy_v, iz_v = _fiber_voxel_indices(
            c,
            a,
            fiber_length=fiber_length,
            fiber_radius=fiber_radius,
            N=N,
        )
        if ix_v.size == 0:
            continue

        new_voxels_mask = phase[ix_v, iy_v, iz_v] == 0
        candidate_count = voxel_count + int(np.count_nonzero(new_voxels_mask))
        if (
            used_fibers > 0
            and voxel_count < voxel_target <= candidate_count
            and voxel_target - voxel_count
            <= candidate_count - voxel_target
        ):
            break

        voxel_count = candidate_count
        phase[ix_v, iy_v, iz_v] = 1
        ori[ix_v, iy_v, iz_v] = a.astype(np.float32)
        used_fibers += 1

        if voxel_count >= voxel_target:
            break

    return phase, ori, voxel_count, used_fibers


def _rasterize_fibers_numba(
    centers: np.ndarray,
    directions: np.ndarray,
    fiber_length: float,
    fiber_radius: float,
    N: int,
    voxel_target: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    phase = np.zeros((N, N, N), dtype=np.uint8)
    ori = np.zeros((N, N, N, 3), dtype=np.float32)
    voxel_count = 0
    used_fibers = 0

    if len(centers) == 0:
        return phase, ori, voxel_count, used_fibers

    centers64 = np.ascontiguousarray(centers, dtype=np.float64)
    directions64 = np.ascontiguousarray(directions, dtype=np.float64)
    n_fibers = len(centers64)

    # --- Pasada única paralela: count + fill fusionados ---
    # Estimar capacidad máxima por fibra para pre-allocar.
    # Bounding box de una fibra: (L + 2*(R+1))^2 * (2*(R+1))
    bbox_side = fiber_length + 2.0 * (fiber_radius + 1.0)
    bbox_cross = 2.0 * (fiber_radius + 1.0)
    max_vox_per_fiber = int(np.ceil(bbox_side * bbox_cross * bbox_cross)) + 64
    # Limitar para no abusar de RAM en fibras muy largas
    max_vox_per_fiber = min(max_vox_per_fiber, 500_000)

    ix_all = np.empty(n_fibers * max_vox_per_fiber, dtype=np.int64)
    iy_all = np.empty(n_fibers * max_vox_per_fiber, dtype=np.int64)
    iz_all = np.empty(n_fibers * max_vox_per_fiber, dtype=np.int64)
    actual_counts = np.empty(n_fibers, dtype=np.int64)

    _count_and_fill_fiber_indices_parallel(
        centers64,
        directions64,
        float(fiber_length),
        float(fiber_radius),
        int(N),
        int(max_vox_per_fiber),
        ix_all,
        iy_all,
        iz_all,
        actual_counts,
    )

    voxel_target_int = int(np.ceil(float(voxel_target)))
    for i in range(n_fibers):
        count_i = int(actual_counts[i])
        if count_i <= 0:
            continue

        base = i * max_vox_per_fiber
        ix_v = ix_all[base:base + count_i]
        iy_v = iy_all[base:base + count_i]
        iz_v = iz_all[base:base + count_i]

        new_voxels_mask = phase[ix_v, iy_v, iz_v] == 0
        candidate_count = voxel_count + int(np.count_nonzero(new_voxels_mask))
        if (
            used_fibers > 0
            and voxel_count < voxel_target_int <= candidate_count
            and voxel_target_int - voxel_count
            <= candidate_count - voxel_target_int
        ):
            break

        voxel_count = candidate_count
        phase[ix_v, iy_v, iz_v] = 1
        ori[ix_v, iy_v, iz_v] = directions64[i].astype(np.float32)
        used_fibers += 1

        if voxel_count >= voxel_target_int:
            break

    return phase, ori, voxel_count, used_fibers


def _rasterize_fibers(
    centers: np.ndarray,
    directions: np.ndarray,
    fiber_length: float,
    fiber_radius: float,
    N: int,
    voxel_target: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if NUMBA_AVAILABLE:
        return _rasterize_fibers_numba(
            centers=centers,
            directions=directions,
            fiber_length=fiber_length,
            fiber_radius=fiber_radius,
            N=N,
            voxel_target=voxel_target,
        )
    return _rasterize_fibers_python(
        centers=centers,
        directions=directions,
        fiber_length=fiber_length,
        fiber_radius=fiber_radius,
        N=N,
        voxel_target=voxel_target,
    )


def rasterize_continuous_fibers(
    fibers: pd.DataFrame | str | Path,
    *,
    caja_um: float,
    resolution: float,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Rasterize every fiber from one continuous master geometry."""
    table = pd.read_csv(fibers) if isinstance(fibers, (str, Path)) else fibers.copy()
    required = {"cx_um", "cy_um", "cz_um", "ux", "uy", "uz", "L_um", "d_um"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Faltan columnas en la geometria continua: {missing}")
    if table.empty:
        raise ValueError("La geometria continua no contiene fibras.")
    lengths = table["L_um"].to_numpy(dtype=float)
    diameters = table["d_um"].to_numpy(dtype=float)
    if not np.allclose(lengths, lengths[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("El rasterizador actual requiere una longitud comun.")
    if not np.allclose(diameters, diameters[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("El rasterizador actual requiere un diametro comun.")

    N = int(round(float(caja_um) * float(resolution)))
    if N < 2:
        raise ValueError("La resolucion produce una malla invalida.")
    centers = table[["cx_um", "cy_um", "cz_um"]].to_numpy(dtype=float)
    centers = np.mod(centers, float(caja_um)) * float(resolution)
    directions = table[["ux", "uy", "uz"]].to_numpy(dtype=float).copy()
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("Todas las direcciones deben ser no nulas.")
    directions /= norms[:, None]
    phase, ori, voxel_count, used_fibers = _rasterize_fibers(
        centers,
        directions,
        float(lengths[0]) * float(resolution),
        0.5 * float(diameters[0]) * float(resolution),
        N,
        float(N**3 + 1),
    )
    if used_fibers != len(table):
        raise RuntimeError(
            f"Rasterizacion incompleta: se usaron {used_fibers} de {len(table)} fibras."
        )
    metadata: Dict[str, Any] = {
        "grid_size": int(N),
        "grid_shape": [int(N), int(N), int(N)],
        "resolution_vox_per_um": float(resolution),
        "box_um": float(caja_um),
        "fiber_count": int(used_fibers),
        "voxel_count": int(voxel_count),
        "voxel_volume_fraction": float(voxel_count / N**3),
        "interface_density": float(interface_density_periodic(phase)),
        "A2_voxel": _voxelized_a2(phase, ori).tolist(),
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "phase.npy", phase)
        np.save(destination / "ori.npy", ori)
        (destination / "raster_manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return {"phase": phase, "ori": ori, "metadata": metadata}


def _directional_a2(directions: np.ndarray) -> np.ndarray:
    if len(directions) == 0:
        return np.zeros((3, 3), dtype=float)
    return np.einsum("ni,nj->ij", directions, directions) / len(directions)


def _voxelized_a2(phase: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    phase_flat = np.asarray(phase).reshape(-1)
    orientation_flat = np.asarray(orientations).reshape(-1, 3)
    moment_sum = np.zeros((3, 3), dtype=float)
    valid_count = 0
    chunk_size = 1_000_000
    for start in range(0, len(phase_flat), chunk_size):
        stop = min(len(phase_flat), start + chunk_size)
        mask = phase_flat[start:stop] != 0
        if not np.any(mask):
            continue
        directions = np.asarray(
            orientation_flat[start:stop][mask],
            dtype=float,
        )
        norms = np.linalg.norm(directions, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            continue
        directions = directions[valid] / norms[valid, None]
        moment_sum += np.einsum("ni,nj->ij", directions, directions)
        valid_count += len(directions)
    if valid_count == 0:
        return np.zeros((3, 3), dtype=float)
    return moment_sum / valid_count


def _count_periodic_crossing_fibers(
    centers: np.ndarray,
    directions: np.ndarray,
    fiber_length: float,
    N: int,
) -> int:
    if len(centers) == 0:
        return 0
    p0 = centers - 0.5 * float(fiber_length) * directions
    p1 = centers + 0.5 * float(fiber_length) * directions
    crosses = np.any((p0 < 0.0) | (p0 >= N) | (p1 < 0.0) | (p1 >= N), axis=1)
    return int(np.count_nonzero(crosses))


def _estimate_voxel_count(
    centers: np.ndarray,
    directions: np.ndarray,
    fiber_length: float,
    fiber_radius: float,
    N: int,
    voxel_target: float,
) -> Tuple[int, int]:
    phase = np.zeros((N, N, N), dtype=np.uint8)
    voxel_count = 0
    used_fibers = 0

    for c, a in zip(centers, directions):
        ix_v, iy_v, iz_v = _fiber_voxel_indices(
            c,
            a,
            fiber_length=fiber_length,
            fiber_radius=fiber_radius,
            N=N,
        )
        if ix_v.size == 0:
            continue

        new_voxels_mask = phase[ix_v, iy_v, iz_v] == 0
        voxel_count += int(np.count_nonzero(new_voxels_mask))
        phase[ix_v, iy_v, iz_v] = 1
        used_fibers += 1

        if voxel_count >= voxel_target:
            break

    return voxel_count, used_fibers


def _sample_direction_from_A2_target(
    rng: np.random.Generator,
    A2_target: np.ndarray,
) -> np.ndarray:
    return SAMLiteGenerator.sample_directions_for_A2(
        A2_target,
        rng,
        1,
    )[0]


def _estimate_voxel_inflation(
    A2_target: np.ndarray,
    fiber_length: float,
    fiber_radius: float,
    N: int,
    seed: int,
    samples: int,
) -> float:
    if samples < 1:
        return 1.0

    cache_key = (
        round(float(fiber_length), 4),
        round(float(fiber_radius), 4),
        int(N),
        int(samples),
        int(seed),
        round(float(A2_target[0, 0]), 4),
        round(float(A2_target[1, 1]), 4),
        round(float(A2_target[2, 2]), 4),
    )
    with _INFLATION_CACHE_LOCK:
        cached = _INFLATION_CACHE.get(cache_key)
    if cached is not None:
        return float(cached)

    rng = np.random.default_rng(seed + 99173)
    counts = []
    continuous_voxels = max(np.pi * fiber_radius * fiber_radius * fiber_length, 1e-12)

    for _ in range(samples):
        center = rng.uniform(0.0, N, size=3)
        direction = _sample_direction_from_A2_target(rng, A2_target)
        voxel_count, _ = _estimate_voxel_count(
            np.asarray([center], dtype=np.float64),
            np.asarray([direction], dtype=np.float64),
            fiber_length,
            fiber_radius,
            N,
            float("inf"),
        )
        counts.append(voxel_count)

    if not counts:
        inflation = 1.0
    else:
        avg_voxels = float(np.mean(counts))
        inflation = max(1.0, avg_voxels / continuous_voxels)

    with _INFLATION_CACHE_LOCK:
        _INFLATION_CACHE[cache_key] = float(inflation)
    return float(inflation)


def _export_continuous_fibers_master(
    centers_vox: np.ndarray,
    directions: np.ndarray,
    resol: float,
    L_um: float,
    d_um: float,
    caja_um: float,
    out_dir: str,
) -> Tuple[pd.DataFrame, str]:
    centers_vox = np.asarray(centers_vox, dtype=float)
    directions = np.asarray(directions, dtype=float)

    if centers_vox.ndim != 2 or centers_vox.shape[1] != 3:
        raise ValueError("centers_vox debe tener forma (N, 3).")
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions debe tener forma (N, 3).")
    if len(centers_vox) != len(directions):
        raise ValueError("centers_vox y directions deben tener el mismo numero de fibras.")

    n = len(centers_vox)
    out_path = os.path.join(out_dir, "continuous_fibers_master.csv")

    if n == 0:
        df = pd.DataFrame(columns=["cx_um", "cy_um", "cz_um", "ux", "uy", "uz", "L_um", "d_um"])
        df.to_csv(out_path, index=False)
        return df, out_path

    dirs = directions.copy()
    norm = np.linalg.norm(dirs, axis=1)
    norm = np.maximum(norm, 1e-12)
    dirs = dirs / norm[:, None]

    centers_um = centers_vox / float(resol)
    centers_um = np.mod(centers_um, float(caja_um))

    df = pd.DataFrame(
        {
            "cx_um": centers_um[:, 0],
            "cy_um": centers_um[:, 1],
            "cz_um": centers_um[:, 2],
            "ux": dirs[:, 0],
            "uy": dirs[:, 1],
            "uz": dirs[:, 2],
            "L_um": np.full(n, float(L_um), dtype=float),
            "d_um": np.full(n, float(d_um), dtype=float),
        }
    )

    df.to_csv(out_path, index=False)

    manifest_path = os.path.join(out_dir, "continuous_fibers_master_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "fiber_table": out_path,
                "n_fibers": int(n),
                "units": "um",
                "columns": list(df.columns),
                "source": "pipeline/rve_generator.py",
                "resol_internal_vox_per_um": float(resol),
                "L_um": float(L_um),
                "d_um": float(d_um),
                "caja_um": float(caja_um),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    return df, out_path


def generate_rve_main(p: Dict[str, Any]) -> Dict[str, Any]:
    L_um = p["L_um"]
    d_um = p["d_um"]
    gap_um = p["gap_um"]
    caja_um = p["caja_um"]
    resol = p["resol"]
    Vf_target = p["Vf_target"]
    seed = p["seed"]
    out_dir = p["output_dir"]

    a11 = p.get("a11", 1.0 / 3.0)
    a22 = p.get("a22", 1.0 / 3.0)
    a33 = 1.0 - a11 - a22
    a11, a22, a33 = max(0.0, a11), max(0.0, a22), max(0.0, a33)

    compute_metrics = p.get("compute_metrics", True)

    sam_batch_vf = float(p.get("sam_batch_vf", 0.01))
    sam_tau_translate = float(p.get("sam_tau_translate", 0.25))
    sam_tau_rotate = float(p.get("sam_tau_rotate", 0.01))
    sam_w_orient = float(p.get("sam_w_orient", 0.005))
    sam_tol_A = float(
        p.get("sam_A2_tolerance", p.get("sam_tol_A", 1e-2))
    )
    sam_voxel_A2_tolerance = float(
        p.get("sam_voxel_A2_tolerance", max(1e-2, sam_tol_A))
    )
    sam_geometry_backend = str(
        p.get("sam_geometry_backend", "numba")
    ).strip().lower()
    sam_cupy_min_candidate_pairs = int(
        p.get("sam_cupy_min_candidate_pairs", 50_000)
    )
    sam_tol_overlap = float(p.get("sam_tol_overlap", 0.05))
    sam_max_iter_relax = int(p.get("sam_max_iter_relax", 80))
    sam_fast_mode = bool(p.get("sam_fast_mode", True))
    sam_verbose = bool(p.get("sam_verbose", True))
    sam_soft_insert = bool(p.get("sam_soft_insert", False))
    default_soft_insert_start = 1.10
    if Vf_target >= 0.35:
        default_soft_insert_start = 0.60
    elif Vf_target >= 0.30:
        default_soft_insert_start = 0.70
    sam_soft_insert_start = float(
        p.get("sam_soft_insert_start", default_soft_insert_start)
    )
    sam_target_mode = str(p.get("sam_target_mode", "calibrated"))
    sam_inflation_samples = int(p.get("sam_inflation_samples", 2))
    sam_target_safety = float(p.get("sam_target_safety", 1.03))
    sam_max_topups = int(p.get("sam_max_topups", 6))
    sam_topup_gain = float(p.get("sam_topup_gain", 1.35))
    sam_continuous_vf_cap_factor = float(
        p.get("sam_continuous_vf_cap_factor", 1.25)
    )
    sam_vf_tolerance_configured = float(p.get("sam_vf_tolerance", 0.005))
    sam_vf_tolerance = sam_vf_tolerance_configured
    sam_voxel_stop_start = float(p.get("sam_voxel_stop_start", 0.60))
    sam_num_threads = int(p.get("sam_num_threads", p.get("num_cores", 1)))
    sam_compaction = bool(p.get("sam_compaction", False))
    aspect_ratio = float(L_um) / max(float(d_um), 1e-12)
    default_compaction_scale = 1.0
    if Vf_target >= 0.30:
        if aspect_ratio >= 24.0:
            default_compaction_scale = 0.45
        elif aspect_ratio >= 18.0:
            default_compaction_scale = 0.50
        elif aspect_ratio >= 12.0:
            default_compaction_scale = 0.65
        else:
            default_compaction_scale = 0.75
    sam_compaction_start_scale = float(
        p.get("sam_compaction_start_scale", default_compaction_scale)
    )
    sam_compaction_topup_scale = float(
        p.get("sam_compaction_topup_scale", sam_compaction_start_scale)
    )
    sam_compaction_stages = int(p.get("sam_compaction_stages", 6))
    sam_compaction_max_iter = int(
        p.get("sam_compaction_max_iter", max(120, sam_max_iter_relax))
    )
    sam_compaction_passes = int(p.get("sam_compaction_passes", 2))
    packing_load = float(p.get(
        "sam_packing_load",
        aspect_ratio * Vf_target,
    ))
    sam_collective_fire = bool(p.get("sam_collective_fire", True))
    sam_collective_fire_min_packing_load = float(
        p.get("sam_collective_fire_min_packing_load", 3.0)
    )
    use_collective_fire = (
        sam_compaction
        and sam_collective_fire
        and packing_load >= sam_collective_fire_min_packing_load
    )
    sam_collective_fire_max_iter = int(
        p.get("sam_collective_fire_max_iter", 2500)
    )
    sam_collective_fire_max_restarts = int(
        p.get("sam_collective_fire_max_restarts", 3)
    )
    sam_collective_fire_restart_patience = int(
        p.get("sam_collective_fire_restart_patience", 250)
    )
    sam_overlap_tolerance = float(p.get("sam_overlap_tolerance", sam_tol_overlap))
    sam_overlap_rescue = bool(p.get("sam_overlap_rescue", False))
    sam_overlap_rescue_passes = int(p.get("sam_overlap_rescue_passes", 3))
    sam_overlap_rescue_max_iter = int(
        p.get("sam_overlap_rescue_max_iter", max(160, sam_max_iter_relax))
    )
    sam_orientation_max_iter = int(
        p.get("sam_orientation_max_iter", max(120, sam_max_iter_relax))
    )
    sam_overlap_rescue_max_fibers = int(
        p.get("sam_overlap_rescue_max_fibers", 48)
    )
    sam_long_fiber_vf_fallback = bool(
        p.get(
            "sam_long_fiber_vf_fallback",
            sam_compaction and Vf_target >= 0.35 and aspect_ratio >= 24.0,
        )
    )
    if sam_long_fiber_vf_fallback:
        sam_compaction = False
        use_collective_fire = False
        sam_max_topups = max(sam_max_topups, 8)

    sam_use_local_relax = bool(p.get("sam_use_local_relax", True))
    sam_strict_insertion = bool(
        p.get(
            "sam_strict_insertion",
            str(p.get("sam_center_distribution", "uniform")).strip().lower()
            == "gaussian_mixture",
        )
    )

    if Vf_target >= 0.35:
        sam_max_iter_relax = max(sam_max_iter_relax, 120)
        sam_compaction_max_iter = max(sam_compaction_max_iter, 160)
        sam_soft_insert_start = min(sam_soft_insert_start, 0.60)
        sam_max_topups = max(sam_max_topups, 4)
        sam_topup_gain = max(sam_topup_gain, 1.25)
        sam_continuous_vf_cap_factor = max(sam_continuous_vf_cap_factor, 1.18)
    if sam_compaction or sam_long_fiber_vf_fallback:
        sam_target_safety = min(sam_target_safety, 1.0)

    L = float(L_um * resol)
    d = float(d_um * resol)
    gap = float(gap_um * resol)
    N = int(round(caja_um * resol))
    R = d / 2.0
    V_total = N**3
    V_target = Vf_target * V_total
    V_target_int = int(math.ceil(V_target))
    single_fiber_vf_quantum = float((np.pi * d * d * L / 4.0) / max(V_total, 1))
    sam_vf_tolerance = max(sam_vf_tolerance, single_fiber_vf_quantum)

    if sam_verbose:
        print(
            f"[SAM] seed={seed} | N={N} | Vf_obj={Vf_target}",
            flush=True,
        )
    t0 = time.time()

    A2_target = np.diag([a11, a22, a33]).astype(float)
    voxel_stop_checks = []
    voxel_callback_s = 0.0
    inflation_factor = 1.0
    effective_vf_target = Vf_target

    def stop_callback(gen: SAMLiteGenerator) -> bool:
        nonlocal voxel_callback_s
        current_cont_vf = gen.current_vf()
        if current_cont_vf < sam_voxel_stop_start * Vf_target:
            return False
        callback_t0 = time.perf_counter()
        voxel_count_est, used_fibers_est = _estimate_voxel_count(
            gen.centers,
            gen.directions,
            L,
            R,
            N,
            V_target,
        )
        voxel_stop_checks.append(
            {
                "cont_vf": float(current_cont_vf),
                "vox_vf_est": float(voxel_count_est / V_total),
                "used_fibers_est": int(used_fibers_est),
            }
        )
        voxel_callback_s += time.perf_counter() - callback_t0
        return voxel_count_est >= V_target_int

    calibration_t0 = time.perf_counter()
    if sam_target_mode == "calibrated":
        inflation_factor = _estimate_voxel_inflation(
            A2_target=A2_target,
            fiber_length=L,
            fiber_radius=R,
            N=N,
            seed=seed,
            samples=sam_inflation_samples,
        )
        calibrated_target = (Vf_target / max(inflation_factor, 1e-12)) * sam_target_safety
        effective_vf_target = min(
            0.95,
            max(Vf_target, effective_vf_target)
            * max(1.0, sam_continuous_vf_cap_factor),
            calibrated_target,
        )
    elif sam_target_mode != "raw":
        raise ValueError("'sam_target_mode' debe ser 'calibrated' o 'raw'.")
    calibration_s = time.perf_counter() - calibration_t0

    generator = SAMLiteGenerator(
        A2=A2_target,
        fiber_length=L,
        fiber_diameter=d,
        vf_target=effective_vf_target,
        cell_size=(float(N), float(N), float(N)),
        batch_vf=sam_batch_vf,
        tau_translate=sam_tau_translate,
        tau_rotate=sam_tau_rotate,
        w_orient=sam_w_orient,
        tol_A=sam_tol_A,
        tol_overlap=sam_tol_overlap,
        max_iter_relax=sam_max_iter_relax,
        min_gap=gap,
        fast_mode=sam_fast_mode,
        use_local_relax=sam_use_local_relax,
        strict_insertion=(
            sam_strict_insertion or sam_compaction or sam_long_fiber_vf_fallback
        ),
        soft_insert=sam_soft_insert,
        soft_insert_start=sam_soft_insert_start,
        seed=seed,
        num_threads=sam_num_threads,
        geometry_backend=sam_geometry_backend,
        cupy_min_candidate_pairs=sam_cupy_min_candidate_pairs,
        center_distribution=str(p.get("sam_center_distribution", "uniform")),
        cluster_fraction=float(p.get("sam_cluster_fraction", 0.0)),
        cluster_count=int(p.get("sam_cluster_count", 4)),
        cluster_sigma_rel=float(p.get("sam_cluster_sigma_rel", 0.12)),
    )
    use_callback = bool(p.get("sam_use_voxel_stop_callback", False)) or sam_compaction
    sam_max_fiber_removals = int(
        math.floor(
            sam_vf_tolerance
            * generator.V_cell
            / max(generator.V_fiber, 1e-12)
        )
    )
    sam_overlap_rescue_max_removals = int(
        p.get(
            "sam_overlap_rescue_max_removals",
            max(1, sam_max_fiber_removals // 4),
        )
    )
    run_stats_history = []
    compaction_history = []
    collective_fire_history = []
    sam_run_s = 0.0
    sam_compaction_s = 0.0
    sam_collective_fire_s = 0.0
    sam_overlap_rescue_s = 0.0
    sam_overlap_rescue_attempts = 0
    sam_overlap_rescue_reinserted = 0
    sam_overlap_rescue_shaken = 0
    sam_overlap_rescue_iters = 0
    sam_overlap_rescue_removed = 0
    sam_orientation_finalize_s = 0.0
    sam_orientation_finalize_iters = 0
    sam_orientation_optimizer_accepted_steps = 0
    sam_orientation_optimizer_line_search_evaluations = 0
    sam_orientation_optimizer_contact_projections = 0

    def finalize_orientation() -> Dict[str, float]:
        nonlocal sam_orientation_finalize_s, sam_orientation_finalize_iters
        nonlocal sam_orientation_optimizer_accepted_steps
        nonlocal sam_orientation_optimizer_line_search_evaluations
        nonlocal sam_orientation_optimizer_contact_projections
        if not hasattr(generator, "relax"):
            return {
                "iters": 0.0,
                "max_overlap": float(generator.measure_max_overlap()),
                "A_err": float("nan"),
            }
        orientation_t0 = time.perf_counter()
        orientation_info = generator.relax(
            verbose=False,
            max_iter=sam_orientation_max_iter,
            target_overlap=sam_overlap_tolerance,
            apply_orientation=True,
            pair_rebuild_interval=3,
            parallel_per_fiber=(
                sam_num_threads > 1 and len(generator.centers) >= 512
            ),
        )
        if hasattr(generator, "optimize_A2_riemannian"):
            optimizer_info = generator.optimize_A2_riemannian(
                max_iter=max(32, sam_orientation_max_iter // 2),
                tolerance=max(1e-4, 0.25 * sam_tol_A),
                overlap_limit=sam_overlap_tolerance,
            )
            orientation_info["iters"] = int(orientation_info["iters"]) + int(
                optimizer_info["iterations"]
            )
            orientation_info["A_err"] = float(
                optimizer_info["final_error"]
            )
            orientation_info["max_overlap"] = float(
                optimizer_info["final_overlap"]
            )
            sam_orientation_optimizer_accepted_steps += int(
                optimizer_info["accepted_steps"]
            )
            sam_orientation_optimizer_line_search_evaluations += int(
                optimizer_info["line_search_evaluations"]
            )
            sam_orientation_optimizer_contact_projections += int(
                optimizer_info["contact_projections"]
            )
        sam_orientation_finalize_s += time.perf_counter() - orientation_t0
        sam_orientation_finalize_iters += int(orientation_info["iters"])
        return orientation_info

    def run_and_compact(*, insertion_scale: float, verbose_run: bool) -> Dict[str, float]:
        nonlocal sam_run_s, sam_compaction_s, sam_collective_fire_s
        nonlocal sam_overlap_rescue_s
        nonlocal sam_overlap_rescue_attempts, sam_overlap_rescue_reinserted
        nonlocal sam_overlap_rescue_shaken, sam_overlap_rescue_iters
        nonlocal sam_overlap_rescue_removed
        if use_collective_fire:
            generator.set_collision_diameter_scale(1.0)
        elif sam_compaction or sam_long_fiber_vf_fallback:
            generator.set_collision_diameter_scale(insertion_scale)

        run_t0 = time.perf_counter()
        if use_collective_fire:
            generator.run_collective_sam(
                batch_vf=min(sam_batch_vf, 0.0125),
                max_iter_per_batch=max(
                    160,
                    min(240, sam_collective_fire_max_iter // 8),
                ),
                target_overlap=sam_overlap_tolerance,
                max_restarts=sam_collective_fire_max_restarts,
                restart_patience=sam_collective_fire_restart_patience,
                stop_callback=stop_callback if use_callback else None,
                verbose=verbose_run,
            )
        else:
            generator.run(
                verbose=verbose_run,
                stop_callback=stop_callback if use_callback else None,
            )
        sam_run_s += time.perf_counter() - run_t0
        run_stats = dict(generator.last_run_stats)

        if (
            bool(run_stats.get("stopped_by_callback", 0.0))
            and voxel_stop_checks
        ):
            callback_used_fibers = int(voxel_stop_checks[-1]["used_fibers_est"])
            if 0 < callback_used_fibers < len(generator.centers):
                generator.retain_first_fibers(callback_used_fibers)
                run_stats["n_fibers"] = float(callback_used_fibers)
                run_stats["final_vf"] = float(generator.current_vf())

        if use_collective_fire:
            fire_t0 = time.perf_counter()
            fire_info = {
                "iters": float(
                    run_stats.get("collective_fire_iters", 0.0)
                ),
                "restarts": float(
                    run_stats.get("collective_fire_restarts", 0.0)
                ),
                "force_evaluations": float(
                    run_stats.get(
                        "collective_fire_force_evaluations",
                        0.0,
                    )
                ),
                "pair_build_s": float(
                    run_stats.get("collective_fire_pair_build_s", 0.0)
                ),
                "pair_forces_s": float(
                    run_stats.get("collective_fire_pair_forces_s", 0.0)
                ),
                "apply_updates_s": float(
                    run_stats.get(
                        "collective_fire_apply_updates_s",
                        0.0,
                    )
                ),
                "wall_s": float(
                    run_stats.get("collective_fire_s", 0.0)
                ),
                "final_overlap": float(
                    run_stats.get("final_overlap", np.inf)
                ),
                "final_A2_error_rel": float(
                    run_stats.get("final_A2_error_rel", np.inf)
                ),
                "converged": float(
                    bool(run_stats.get("target_reached", 0.0))
                    and not bool(run_stats.get("stopped_by_stall", 0.0))
                    and float(run_stats.get("final_overlap", np.inf))
                    <= sam_overlap_tolerance
                    and float(
                        run_stats.get("final_A2_error_rel", np.inf)
                    )
                    <= sam_tol_A
                ),
                "successful_stages": float(
                    run_stats.get(
                        "collective_fire_successful_batches",
                        0.0,
                    )
                ),
                "stage_failures": float(
                    run_stats.get(
                        "collective_fire_failed_batches",
                        0.0,
                    )
                ),
            }
            fire_wall_s = time.perf_counter() - fire_t0
            sam_collective_fire_s += float(fire_info["wall_s"])
            collective_fire_history.append(dict(fire_info))
            compact_info = {
                "initial_scale": 1.0,
                "final_scale": 1.0,
                "stages": float(
                    fire_info.get("successful_stages", 0.0)
                    + fire_info.get("stage_failures", 0.0)
                ),
                "successful_stages": float(
                    fire_info.get("successful_stages", 0.0)
                ),
                "subdivisions": float(
                    fire_info.get("stage_failures", 0.0)
                ),
                "initial_relax_passes": 0.0,
                "shake_attempts": float(fire_info.get("restarts", 0.0)),
                "shaken_fibers": 0.0,
                "reinsert_candidates": 0.0,
                "reinserted_fibers": 0.0,
                "reinsert_failures": 0.0,
                "removed_fibers": 0.0,
                "relax_iters": float(fire_info.get("iters", 0.0)),
                "relax_s": float(fire_info.get("wall_s", fire_wall_s)),
                "pair_build_s": float(
                    fire_info.get("pair_build_s", 0.0)
                ),
                "pair_forces_s": float(
                    fire_info.get("pair_forces_s", 0.0)
                ),
                "final_overlap": float(
                    fire_info.get("final_overlap", np.inf)
                ),
                "final_A2_error_rel": float(
                    fire_info.get("final_A2_error_rel", np.inf)
                ),
                "converged": float(fire_info.get("converged", 0.0)),
                "wall_s": float(fire_info.get("wall_s", fire_wall_s)),
                "collective_fire": 1.0,
            }
        elif sam_compaction:
            compact_t0 = time.perf_counter()
            compact_info = generator.compact_collision_diameter(
                n_stages=sam_compaction_stages,
                max_iter_per_stage=sam_compaction_max_iter,
                target_overlap=sam_overlap_tolerance,
                max_passes_per_stage=sam_compaction_passes,
                max_fiber_removals=sam_max_fiber_removals,
                verbose=False,
            )
            sam_compaction_s += time.perf_counter() - compact_t0
        else:
            generator.set_collision_diameter_scale(1.0)
            final_overlap = generator.measure_max_overlap()
            rescue_t0 = time.perf_counter()
            allow_overlap_rescue = (
                sam_overlap_rescue
                and not sam_long_fiber_vf_fallback
            )
            if allow_overlap_rescue and final_overlap > sam_overlap_tolerance:
                for rescue_pass in range(max(0, sam_overlap_rescue_passes)):
                    sam_overlap_rescue_attempts += 1
                    reinsertion = generator.reinsert_overlapping_fibers(
                        target_overlap=sam_overlap_tolerance,
                        max_attempts_per_fiber=200,
                        hard_diameter_factor=0.80,
                        max_fibers=sam_overlap_rescue_max_fibers,
                    )
                    sam_overlap_rescue_reinserted += int(
                        reinsertion["reinserted"]
                    )
                    shaken = generator.shake_overlapping_fibers(
                        target_overlap=sam_overlap_tolerance,
                        translate_fraction=0.06 + 0.02 * rescue_pass,
                        rotate_radians=0.01,
                    )
                    sam_overlap_rescue_shaken += int(shaken)
                    rescue_info = generator.relax(
                        verbose=False,
                        max_iter=sam_overlap_rescue_max_iter,
                        target_overlap=sam_overlap_tolerance,
                        apply_orientation=False,
                        pair_rebuild_interval=3,
                        parallel_per_fiber=(
                            sam_num_threads > 1 and len(generator.centers) >= 512
                        ),
                    )
                    sam_overlap_rescue_iters += int(rescue_info["iters"])
                    final_overlap = generator.measure_max_overlap()
                    if final_overlap <= sam_overlap_tolerance:
                        break
                    removal_budget = 0
                    if generator.current_vf() + 1e-12 >= Vf_target:
                        removal_budget = max(
                            0,
                            min(
                                sam_max_fiber_removals,
                                sam_overlap_rescue_max_removals,
                            )
                            - sam_overlap_rescue_removed,
                        )
                    if removal_budget > 0:
                        removed = generator.drop_worst_overlapping_fibers(
                            target_overlap=sam_overlap_tolerance,
                            max_remove=min(8, removal_budget),
                        )
                        sam_overlap_rescue_removed += int(removed)
                        final_overlap = generator.measure_max_overlap()
                        if final_overlap <= sam_overlap_tolerance:
                            break
            sam_overlap_rescue_s += time.perf_counter() - rescue_t0
            compact_info = {
                "initial_scale": 1.0,
                "final_scale": 1.0,
                "stages": 0.0,
                "successful_stages": 0.0,
                "subdivisions": 0.0,
                "initial_relax_passes": 0.0,
                "shake_attempts": 0.0,
                "shaken_fibers": 0.0,
                "reinsert_candidates": 0.0,
                "reinserted_fibers": 0.0,
                "reinsert_failures": 0.0,
                "removed_fibers": 0.0,
                "relax_iters": 0.0,
                "relax_s": 0.0,
                "pair_build_s": 0.0,
                "pair_forces_s": 0.0,
                "final_overlap": float(final_overlap),
                "converged": float(final_overlap <= sam_overlap_tolerance),
                "wall_s": 0.0,
            }

        orientation_info = finalize_orientation()
        compact_info["final_overlap"] = float(generator.measure_max_overlap())
        compact_info["final_A2_error_rel"] = float(
            generator.orientation_error()
            if hasattr(generator, "orientation_error")
            else orientation_info.get("A_err", np.nan)
        )
        compact_info["converged"] = float(
            compact_info["final_overlap"] <= sam_overlap_tolerance
            and compact_info["final_A2_error_rel"] <= sam_tol_A
        )

        run_stats["final_overlap"] = float(compact_info["final_overlap"])
        run_stats["final_A_err"] = float(
            compact_info["final_A2_error_rel"]
        )
        run_stats["final_A2_error_rel"] = float(
            compact_info["final_A2_error_rel"]
        )
        run_stats["collision_diameter_scale"] = float(
            generator.collision_diameter_scale
        )
        generator.last_run_stats.update(run_stats)
        run_stats_history.append(run_stats)
        compaction_history.append(dict(compact_info))
        return compact_info

    compaction_info = run_and_compact(
        insertion_scale=(
            sam_compaction_start_scale
            if sam_compaction or sam_long_fiber_vf_fallback
            else 1.0
        ),
        verbose_run=sam_verbose,
    )

    topup_count = 0
    phase = None
    ori = None
    voxel_count = 0
    used_fibers = 0
    total_raster = 0.0
    post_raster_compaction_retries = 0
    post_raster_orientation_retries = 0

    while True:
        centers, directions, _ = generator.export_arrays()
        t_ras0 = time.time()
        phase, ori, voxel_count, used_fibers = _rasterize_fibers(centers, directions, L, R, N, V_target)
        t_raster = time.time() - t_ras0
        total_raster += t_raster

        if used_fibers < len(centers):
            if hasattr(generator, "retain_best_oriented_fibers"):
                generator.retain_best_oriented_fibers(used_fibers)
            else:
                generator.retain_first_fibers(used_fibers)
            centers, directions, _ = generator.export_arrays()
            if (
                hasattr(generator, "relax")
                and post_raster_orientation_retries < 3
                and len(centers) > 0
            ):
                finalize_orientation()
                post_raster_orientation_retries += 1
                continue

        current_overlap = generator.measure_max_overlap()
        needs_post_raster_compaction = (
            sam_compaction
            and current_overlap > sam_overlap_tolerance
            and post_raster_compaction_retries < 1
        )
        if needs_post_raster_compaction:
            retry_t0 = time.perf_counter()
            if use_collective_fire:
                compaction_info = generator.relax_collective_fire(
                    max_iter=sam_collective_fire_max_iter,
                    target_overlap=sam_overlap_tolerance,
                    apply_orientation=True,
                    pair_rebuild_interval=1,
                    max_restarts=sam_collective_fire_max_restarts,
                    restart_patience=sam_collective_fire_restart_patience,
                    verbose=False,
                )
                retry_wall_s = time.perf_counter() - retry_t0
                sam_collective_fire_s += retry_wall_s
                collective_fire_history.append(dict(compaction_info))
                compaction_info = {
                    "initial_scale": 1.0,
                    "final_scale": 1.0,
                    "stages": 0.0,
                    "successful_stages": 0.0,
                    "subdivisions": 0.0,
                    "initial_relax_passes": 0.0,
                    "shake_attempts": float(
                        compaction_info.get("restarts", 0.0)
                    ),
                    "shaken_fibers": 0.0,
                    "reinsert_candidates": 0.0,
                    "reinserted_fibers": 0.0,
                    "reinsert_failures": 0.0,
                    "removed_fibers": 0.0,
                    "relax_iters": float(
                        compaction_info.get("iters", 0.0)
                    ),
                    "relax_s": float(
                        compaction_info.get("wall_s", retry_wall_s)
                    ),
                    "pair_build_s": float(
                        compaction_info.get("pair_build_s", 0.0)
                    ),
                    "pair_forces_s": float(
                        compaction_info.get("pair_forces_s", 0.0)
                    ),
                    "final_overlap": float(
                        compaction_info.get("final_overlap", np.inf)
                    ),
                    "final_A2_error_rel": float(
                        compaction_info.get(
                            "final_A2_error_rel",
                            np.inf,
                        )
                    ),
                    "converged": float(
                        compaction_info.get("converged", 0.0)
                    ),
                    "wall_s": float(
                        compaction_info.get("wall_s", retry_wall_s)
                    ),
                    "collective_fire": 1.0,
                }
            else:
                retry_scale = sam_compaction_start_scale
                generator.set_collision_diameter_scale(retry_scale)
                compaction_info = generator.compact_collision_diameter(
                    n_stages=max(sam_compaction_stages, 8),
                    max_iter_per_stage=sam_compaction_max_iter,
                    target_overlap=sam_overlap_tolerance,
                    max_passes_per_stage=sam_compaction_passes,
                    max_fiber_removals=sam_max_fiber_removals,
                    verbose=False,
                )
                sam_compaction_s += time.perf_counter() - retry_t0
            compaction_history.append(dict(compaction_info))
            post_raster_compaction_retries += 1
            continue

        voxel_vf_error = voxel_count / V_total - Vf_target
        if (
            abs(voxel_vf_error) <= sam_vf_tolerance
            or voxel_count >= V_target_int
            or topup_count >= sam_max_topups
        ):
            break
        deficit_vf = (V_target_int - voxel_count) / V_total
        extra_cont_vf = max(deficit_vf / max(inflation_factor, 1e-12), 0.25 * sam_batch_vf)
        continuous_vf_cap = min(
            0.95,
            max(Vf_target, effective_vf_target)
            * max(1.0, sam_continuous_vf_cap_factor),
        )
        generator.vf_target = min(
            continuous_vf_cap,
            generator.vf_target + sam_topup_gain * extra_cont_vf,
        )
        continuous_vf_before = generator.current_vf()
        compaction_info = run_and_compact(
            insertion_scale=(
                sam_compaction_topup_scale
                if sam_compaction or sam_long_fiber_vf_fallback
                else 1.0
            ),
            verbose_run=False,
        )
        topup_count += 1
        min_progress_vf = 0.5 * generator.V_fiber / max(generator.V_cell, 1e-12)
        if (
            bool(generator.last_run_stats.get("stopped_by_stall", 0.0))
            and generator.current_vf() <= continuous_vf_before + min_progress_vf
        ):
            break

    t_construct = time.time() - t0 - total_raster
    t_fill = t_construct + t_raster

    centers = centers[:used_fibers]
    directions = directions[:used_fibers]
    used_continuous_vf = float(
        len(centers) * (np.pi * d * d * L / 4.0) / max(V_total, 1)
    )
    periodic_crossing_fibers = _count_periodic_crossing_fibers(
        centers,
        directions,
        L,
        N,
    )
    a2_real = _directional_a2(directions)
    a2_voxel = _voxelized_a2(phase, ori)
    a2_err = SAMLiteGenerator.relative_tensor_error(a2_real, A2_target)
    a2_voxel_err = SAMLiteGenerator.relative_tensor_error(
        a2_voxel,
        A2_target,
    )
    a2_ok = bool(
        a2_err <= sam_tol_A
        and a2_voxel_err <= sam_voxel_A2_tolerance
    )

    os.makedirs(out_dir, exist_ok=True)
    save_t0 = time.perf_counter()
    np.save(os.path.join(out_dir, "phase.npy"), phase)
    np.save(os.path.join(out_dir, "ori.npy"), ori)
    save_s = time.perf_counter() - save_t0

    Vf_real = voxel_count / V_total
    vf_error = float(Vf_real - Vf_target)
    vf_deficit = float(max(0.0, -vf_error))
    vf_ok = bool(abs(vf_error) <= sam_vf_tolerance)
    final_overlap = float(generator.measure_max_overlap_prefix(used_fibers))
    overlap_ok = bool(final_overlap <= sam_overlap_tolerance)

    if compute_metrics:
        t_metrics0 = time.time()
        cv = metric_cv_local_vf(phase)
        if len(centers) > 0:
            dstar = compute_star_discrepancy(centers, N)
            radii = np.linspace(2.0, N * 0.25, 40)
            K_r = ripley_k_3d(centers, N, radii)
            L_r = ((3.0 * K_r) / (4.0 * np.pi)) ** (1.0 / 3.0)
            ripley_peak = float(np.max(L_r - radii)) if len(L_r) > 0 else 0.0
            _, g_values = pair_correlation_3d(centers, box_size=N, r_max=N * 0.25, dr=1.0)
            g_peak = float(np.max(g_values)) if len(g_values) > 0 else 0.0
        else:
            dstar = 1.0
            ripley_peak = 0.0
            g_peak = 0.0
        interface_density = interface_density_periodic(phase)
        t_metrics = time.time() - t_metrics0
    else:
        cv = 0.0
        dstar = 0.0
        ripley_peak = 0.0
        g_peak = 0.0
        interface_density = interface_density_periodic(phase)
        t_metrics = 0.0

    t_gen = time.time() - t0
    run_stats = generator.last_run_stats
    additive_stat_keys = (
        "batches",
        "attempts",
        "relax_iters",
        "add_batch_s",
        "sample_centers_s",
        "sample_dirs_s",
        "candidate_checks_s",
        "relax_s",
        "relax_local_s",
        "pair_build_s",
        "pair_forces_s",
        "orientation_corr_s",
        "compute_a4_s",
        "apply_updates_s",
        "cleanup_passes",
        "soft_inserted_batches",
        "orientation_optimizer_iterations",
        "orientation_optimizer_accepted_steps",
        "orientation_optimizer_line_search_evaluations",
        "orientation_optimizer_contact_projections",
        "orientation_optimizer_s",
        "cupy_candidate_calls",
        "cupy_candidate_pairs",
        "cupy_candidate_s",
        "cupy_candidate_fallbacks",
        "cupy_force_calls",
        "cupy_force_pairs",
        "cupy_force_s",
        "cupy_force_fallbacks",
        "collective_fire_successful_batches",
        "collective_fire_failed_batches",
    )
    aggregate_run_stats = {
        key: float(sum(stats.get(key, 0.0) for stats in run_stats_history))
        for key in additive_stat_keys
    }
    aggregate_compaction_stats = {
        key: float(sum(stats.get(key, 0.0) for stats in compaction_history))
        for key in (
            "stages",
            "successful_stages",
            "subdivisions",
            "initial_relax_passes",
            "shake_attempts",
            "shaken_fibers",
            "reinsert_candidates",
            "reinserted_fibers",
            "reinsert_failures",
            "removed_fibers",
            "relax_iters",
            "relax_s",
            "pair_build_s",
            "pair_forces_s",
            "wall_s",
        )
    }
    aggregate_collective_fire_stats = {
        key: float(sum(stats.get(key, 0.0) for stats in collective_fire_history))
        for key in (
            "iters",
            "restarts",
            "force_evaluations",
            "pair_build_s",
            "pair_forces_s",
            "apply_updates_s",
            "wall_s",
        )
    }
    collective_batch_sizes = [
        float(stats["collective_fire_smallest_batch"])
        for stats in run_stats_history
        if "collective_fire_smallest_batch" in stats
    ]

    if sam_verbose:
        print(
            f"[SAM] listo | t={t_gen:.1f}s | raster={t_raster:.1f}s | "
            f"Vf={Vf_real:.4f} | fibras={len(centers)}",
            flush=True,
        )

    continuous_fibers_df = None
    continuous_fiber_table_path = ""
    if (
        bool(p.get("export_fiber_table", False))
        or bool(p.get("return_fiber_table", False))
        or bool(p.get("save_continuous_geometry", False))
    ):
        continuous_fibers_df, continuous_fiber_table_path = _export_continuous_fibers_master(
            centers_vox=centers,
            directions=directions,
            resol=float(resol),
            L_um=float(L_um),
            d_um=float(d_um),
            caja_um=float(caja_um),
            out_dir=out_dir,
        )

    return {
        "fibers": continuous_fibers_df if bool(p.get("return_fiber_table", False)) else None,
        "fiber_table_path": continuous_fiber_table_path,
        "continuous_fibers_master_csv": continuous_fiber_table_path,
        "Vf": Vf_real,
        "Vf_target": float(Vf_target),
        "sam_vf_target": float(Vf_target),
        "sam_vf_error": float(vf_error),
        "sam_vf_error_percent_points": float(100.0 * vf_error),
        "sam_vf_deficit": float(vf_deficit),
        "sam_vf_tolerance": float(sam_vf_tolerance),
        "sam_vf_tolerance_configured": float(sam_vf_tolerance_configured),
        "sam_single_fiber_vf_quantum": float(single_fiber_vf_quantum),
        "sam_vf_ok": bool(vf_ok),
        "sam_overlap_tolerance": float(sam_overlap_tolerance),
        "sam_overlap_ok": bool(overlap_ok),
        "sam_A2_tolerance": float(sam_tol_A),
        "sam_voxel_A2_tolerance": float(sam_voxel_A2_tolerance),
        "sam_A2_ok": bool(a2_ok),
        "CV": cv,
        "D_star": dstar,
        "Ripley_peak": ripley_peak,
        "g_peak": g_peak,
        "interface_density": float(interface_density),
        "n_fibers": int(len(centers)),
        "t_collision": np.nan,
        "t_raster": t_raster,
        "t_fill": t_fill,
        "t_metrics": t_metrics,
        "t_gen": t_gen,
        "t_construct": t_construct,
        "sam_batches": int(aggregate_run_stats.get("batches", 0.0)),
        "sam_attempts": int(aggregate_run_stats.get("attempts", 0.0)),
        "sam_relax_iters": int(aggregate_run_stats.get("relax_iters", 0.0)),
        "sam_final_overlap": float(final_overlap),
        "sam_final_A_err": float(a2_err),
        "sam_center_distribution": str(generator.center_distribution),
        "sam_cluster_fraction": float(generator.cluster_fraction),
        "sam_cluster_count": int(generator.cluster_count),
        "sam_cluster_sigma_rel": float(generator.cluster_sigma_rel),
        "sam_strict_insertion": bool(sam_strict_insertion),
        "sam_stopped_by_callback": bool(run_stats.get("stopped_by_callback", 0.0)),
        "sam_voxel_stop_checks": int(len(voxel_stop_checks)),
        "sam_calibration_s": float(calibration_s),
        "sam_run_s": float(sam_run_s),
        "sam_compaction_s": float(sam_compaction_s),
        "sam_collective_fire_configured": bool(sam_collective_fire),
        "sam_collective_fire_used": bool(collective_fire_history),
        "sam_collective_fire_mode": (
            "sequential_soft_batch"
            if collective_fire_history
            else "disabled"
        ),
        "sam_collective_fire_min_packing_load": float(
            sam_collective_fire_min_packing_load
        ),
        "sam_collective_fire_calls": int(len(collective_fire_history)),
        "sam_collective_fire_s": float(sam_collective_fire_s),
        "sam_collective_fire_iters": int(
            aggregate_collective_fire_stats.get("iters", 0.0)
        ),
        "sam_collective_fire_restarts": int(
            aggregate_collective_fire_stats.get("restarts", 0.0)
        ),
        "sam_collective_fire_force_evaluations": int(
            aggregate_collective_fire_stats.get(
                "force_evaluations",
                0.0,
            )
        ),
        "sam_collective_fire_pair_build_s": float(
            aggregate_collective_fire_stats.get("pair_build_s", 0.0)
        ),
        "sam_collective_fire_pair_forces_s": float(
            aggregate_collective_fire_stats.get("pair_forces_s", 0.0)
        ),
        "sam_collective_fire_apply_updates_s": float(
            aggregate_collective_fire_stats.get("apply_updates_s", 0.0)
        ),
        "sam_collective_fire_successful_batches": int(
            aggregate_run_stats.get(
                "collective_fire_successful_batches",
                0.0,
            )
        ),
        "sam_collective_fire_failed_batches": int(
            aggregate_run_stats.get(
                "collective_fire_failed_batches",
                0.0,
            )
        ),
        "sam_collective_fire_smallest_batch": int(
            min(collective_batch_sizes)
            if collective_batch_sizes
            else 0
        ),
        "sam_overlap_rescue": bool(sam_overlap_rescue),
        "sam_overlap_rescue_s": float(sam_overlap_rescue_s),
        "sam_overlap_rescue_attempts": int(sam_overlap_rescue_attempts),
        "sam_overlap_rescue_reinserted": int(sam_overlap_rescue_reinserted),
        "sam_overlap_rescue_shaken": int(sam_overlap_rescue_shaken),
        "sam_overlap_rescue_iters": int(sam_overlap_rescue_iters),
        "sam_overlap_rescue_removed": int(sam_overlap_rescue_removed),
        "sam_orientation_finalize_s": float(sam_orientation_finalize_s),
        "sam_orientation_finalize_iters": int(sam_orientation_finalize_iters),
        "sam_orientation_optimizer_accepted_steps": int(
            aggregate_run_stats.get(
                "orientation_optimizer_accepted_steps",
                0.0,
            )
            + sam_orientation_optimizer_accepted_steps
        ),
        "sam_orientation_optimizer_line_search_evaluations": int(
            aggregate_run_stats.get(
                "orientation_optimizer_line_search_evaluations",
                0.0,
            )
            + sam_orientation_optimizer_line_search_evaluations
        ),
        "sam_orientation_optimizer_contact_projections": int(
            aggregate_run_stats.get(
                "orientation_optimizer_contact_projections",
                0.0,
            )
            + sam_orientation_optimizer_contact_projections
        ),
        "sam_orientation_optimizer_s": float(
            aggregate_run_stats.get("orientation_optimizer_s", 0.0)
        ),
        "sam_voxel_callback_s": float(voxel_callback_s),
        "sam_raster_total_s": float(total_raster),
        "sam_save_s": float(save_s),
        "sam_inflation_factor": float(inflation_factor),
        "sam_effective_vf_target": float(effective_vf_target),
        "sam_generator_target_vf": float(generator.vf_target),
        "sam_generator_target_n_fibers": int(generator.target_fiber_count())
        if hasattr(generator, "target_fiber_count")
        else int(len(centers)),
        "sam_geometry_backend_requested": str(
            getattr(generator, "geometry_backend_requested", "numba")
        ),
        "sam_geometry_backend_effective": str(
            getattr(generator, "geometry_backend_effective", "numba")
        ),
        "sam_cupy_min_candidate_pairs": int(sam_cupy_min_candidate_pairs),
        "sam_cupy_candidate_calls": int(
            getattr(generator, "_cupy_candidate_calls", 0)
        ),
        "sam_cupy_candidate_pairs": int(
            getattr(generator, "_cupy_candidate_pairs", 0)
        ),
        "sam_cupy_candidate_s": float(
            getattr(generator, "_cupy_candidate_s", 0.0)
        ),
        "sam_cupy_candidate_fallbacks": int(
            getattr(generator, "_cupy_candidate_fallbacks", 0)
        ),
        "sam_cupy_force_calls": int(
            getattr(generator, "_cupy_force_calls", 0)
        ),
        "sam_cupy_force_pairs": int(
            getattr(generator, "_cupy_force_pairs", 0)
        ),
        "sam_cupy_force_s": float(
            getattr(generator, "_cupy_force_s", 0.0)
        ),
        "sam_cupy_force_fallbacks": int(
            getattr(generator, "_cupy_force_fallbacks", 0)
        ),
        "sam_final_continuous_vf": float(generator.current_vf()),
        "sam_used_continuous_vf": float(used_continuous_vf),
        "sam_periodic_boundary_mode": "wrapped_voxel_indices_mod_N",
        "sam_periodic_crossing_fibers": int(periodic_crossing_fibers),
        "sam_periodic_crossing_fraction": float(
            periodic_crossing_fibers / max(len(centers), 1)
        ),
        "sam_soft_insert": bool(sam_soft_insert),
        "sam_soft_insert_start": float(sam_soft_insert_start),
        "sam_soft_inserted_batches": int(
            aggregate_run_stats.get("soft_inserted_batches", 0.0)
        ),
        "sam_use_local_relax": bool(sam_use_local_relax),
        "sam_compaction": bool(sam_compaction),
        "sam_long_fiber_vf_fallback": bool(sam_long_fiber_vf_fallback),
        "sam_generation_strategy": (
            "reduced_diameter_vf_fallback"
            if sam_long_fiber_vf_fallback
            else "collective_fire"
            if collective_fire_history
            else "staged_compaction"
            if sam_compaction
            else "sam_lite_rescue"
            if sam_overlap_rescue_attempts > 0
            else "sam_lite"
        ),
        "sam_compaction_start_scale": float(sam_compaction_start_scale),
        "sam_compaction_topup_scale": float(sam_compaction_topup_scale),
        "sam_compaction_stages": int(aggregate_compaction_stats.get("stages", 0.0)),
        "sam_compaction_successful_stages": int(
            aggregate_compaction_stats.get("successful_stages", 0.0)
        ),
        "sam_compaction_subdivisions": int(
            aggregate_compaction_stats.get("subdivisions", 0.0)
        ),
        "sam_compaction_initial_relax_passes": int(
            aggregate_compaction_stats.get("initial_relax_passes", 0.0)
        ),
        "sam_compaction_shake_attempts": int(
            aggregate_compaction_stats.get("shake_attempts", 0.0)
        ),
        "sam_compaction_shaken_fibers": int(
            aggregate_compaction_stats.get("shaken_fibers", 0.0)
        ),
        "sam_compaction_reinsert_candidates": int(
            aggregate_compaction_stats.get("reinsert_candidates", 0.0)
        ),
        "sam_compaction_reinserted_fibers": int(
            aggregate_compaction_stats.get("reinserted_fibers", 0.0)
        ),
        "sam_compaction_reinsert_failures": int(
            aggregate_compaction_stats.get("reinsert_failures", 0.0)
        ),
        "sam_compaction_removed_fibers": int(
            aggregate_compaction_stats.get("removed_fibers", 0.0)
        ),
        "sam_compaction_max_fiber_removals": int(sam_max_fiber_removals),
        "sam_compaction_relax_iters": int(
            aggregate_compaction_stats.get("relax_iters", 0.0)
        ),
        "sam_compaction_relax_s": float(
            aggregate_compaction_stats.get("relax_s", 0.0)
        ),
        "sam_compaction_pair_build_s": float(
            aggregate_compaction_stats.get("pair_build_s", 0.0)
        ),
        "sam_compaction_pair_forces_s": float(
            aggregate_compaction_stats.get("pair_forces_s", 0.0)
        ),
        "sam_compaction_converged": bool(overlap_ok),
        "sam_post_raster_compaction_retries": int(post_raster_compaction_retries),
        "sam_post_raster_orientation_retries": int(
            post_raster_orientation_retries
        ),
        "sam_topup_count": int(topup_count),
        "sam_A2_11": float(a2_real[0, 0]),
        "sam_A2_22": float(a2_real[1, 1]),
        "sam_A2_33": float(a2_real[2, 2]),
        "sam_A2_error_rel": float(a2_err),
        "sam_voxel_A2_11": float(a2_voxel[0, 0]),
        "sam_voxel_A2_22": float(a2_voxel[1, 1]),
        "sam_voxel_A2_33": float(a2_voxel[2, 2]),
        "sam_voxel_A2_error_rel": float(a2_voxel_err),
        "sam_num_threads": int(run_stats.get("numba_threads", sam_num_threads)),
        "sam_numba_available": bool(run_stats.get("numba_available", 0.0)),
        "sam_add_batch_s": float(aggregate_run_stats.get("add_batch_s", 0.0)),
        "sam_sample_centers_s": float(aggregate_run_stats.get("sample_centers_s", 0.0)),
        "sam_sample_dirs_s": float(aggregate_run_stats.get("sample_dirs_s", 0.0)),
        "sam_candidate_checks_s": float(aggregate_run_stats.get("candidate_checks_s", 0.0)),
        "sam_relax_s_time": float(aggregate_run_stats.get("relax_s", 0.0)),
        "sam_relax_local_s_time": float(aggregate_run_stats.get("relax_local_s", 0.0)),
        "sam_pair_build_s": float(aggregate_run_stats.get("pair_build_s", 0.0)),
        "sam_pair_forces_s": float(aggregate_run_stats.get("pair_forces_s", 0.0)),
        "sam_orientation_corr_s": float(aggregate_run_stats.get("orientation_corr_s", 0.0)),
        "sam_compute_a4_s": float(aggregate_run_stats.get("compute_a4_s", 0.0)),
        "sam_apply_updates_s": float(aggregate_run_stats.get("apply_updates_s", 0.0)),
        "sam_cleanup_passes": int(aggregate_run_stats.get("cleanup_passes", 0.0)),
    }
