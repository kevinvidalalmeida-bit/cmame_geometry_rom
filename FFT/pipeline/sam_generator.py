"""Motor continuo SAM usado por ``pipeline.rve_generator``."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List, Tuple, Dict, Optional
import math
import numpy as np
import time

try:
    from numba import get_num_threads as numba_get_num_threads
    from numba import njit, prange, set_num_threads as numba_set_num_threads
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(f):
            return f

        return decorator

    def prange(*args):
        return range(*args)

    def numba_set_num_threads(n: int) -> None:
        return None

    def numba_get_num_threads() -> int:
        return 1

Array = np.ndarray


@lru_cache(maxsize=512)
def _angular_gaussian_transform(a2_key: Tuple[float, ...]) -> Array:
    """Fit an angular-Gaussian sampler whose second moment matches ``A2``."""
    A2 = np.asarray(a2_key, dtype=float).reshape(3, 3)
    vals, vecs = np.linalg.eigh(0.5 * (A2 + A2.T))
    vals = np.clip(vals, 1e-8, None)
    vals /= np.sum(vals)

    # A fixed quadrature keeps the calibration deterministic and cacheable.
    probe_rng = np.random.default_rng(104729)
    probes = probe_rng.standard_normal((8192, 3))
    probes /= np.maximum(np.linalg.norm(probes, axis=1)[:, None], 1e-12)

    log_shape = np.log(vals)
    log_shape -= np.mean(log_shape)
    for _ in range(40):
        shape = np.exp(np.clip(log_shape, -30.0, 30.0))
        sampled = probes * np.sqrt(shape)
        sampled /= np.maximum(np.linalg.norm(sampled, axis=1)[:, None], 1e-12)
        moment = np.mean(sampled * sampled, axis=0)
        if np.max(np.abs(moment - vals)) <= 2e-7:
            break
        log_shape += 1.5 * (
            np.log(vals) - np.log(np.maximum(moment, 1e-12))
        )
        log_shape -= np.mean(log_shape)

    principal_scale = np.sqrt(np.exp(np.clip(log_shape, -30.0, 30.0)))
    return np.ascontiguousarray(vecs @ np.diag(principal_scale), dtype=float)


@njit(cache=True)
def _check_candidate_serial_numba(
    cand_cx: float, cand_cy: float, cand_cz: float,
    cand_dx: float, cand_dy: float, cand_dz: float,
    centers: Array, directions: Array, n: int,
    Lx: float, Ly: float, Lz: float,
    L: float, hard_d: float, cull_r_sq: float,
    skip_idx: int = -1,
) -> bool:
    SMALL = 1e-12
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz
    ux = cand_dx * L
    uy = cand_dy * L
    uz = cand_dz * L

    for i in range(n):
        if i == skip_idx:
            continue
        ddx = centers[i, 0] - cand_cx
        ddy = centers[i, 1] - cand_cy
        ddz = centers[i, 2] - cand_cz
        ddx -= round(ddx * inv_Lx) * Lx
        ddy -= round(ddy * inv_Ly) * Ly
        ddz -= round(ddz * inv_Lz) * Lz
        if ddx*ddx + ddy*ddy + ddz*ddz > cull_r_sq:
            continue

        vx = directions[i, 0] * L
        vy = directions[i, 1] * L
        vz = directions[i, 2] * L

        ox = cand_cx + ddx
        oy = cand_cy + ddy
        oz = cand_cz + ddz

        w0x = (cand_cx - 0.5*ux) - (ox - 0.5*vx)
        w0y = (cand_cy - 0.5*uy) - (oy - 0.5*vy)
        w0z = (cand_cz - 0.5*uz) - (oz - 0.5*vz)

        a = ux*ux + uy*uy + uz*uz
        b = ux*vx + uy*vy + uz*vz
        c = vx*vx + vy*vy + vz*vz
        d = ux*w0x + uy*w0y + uz*w0z
        e = vx*w0x + vy*w0y + vz*w0z
        Det = a*c - b*b

        if Det < SMALL:
            sN = 0.0
            sD = 1.0
            tN = e
            tD = c
        else:
            sN = b*e - c*d
            tN = a*e - b*d
            sD = Det
            tD = Det
            if sN < 0.0:
                sN = 0.0
                tN = e
                tD = c
            elif sN > sD:
                sN = sD
                tN = e + b
                tD = c

        if tN < 0.0:
            tN = 0.0
            sN = max(0.0, min(-d, a))
            sD = a
        elif tN > tD:
            tN = tD
            sN = max(0.0, min(-d + b, a))
            sD = a

        sc = 0.0 if abs(sN) < SMALL else sN / sD
        tc = 0.0 if abs(tN) < SMALL else tN / tD

        dfx = w0x + sc*ux - tc*vx
        dfy = w0y + sc*uy - tc*vy
        dfz = w0z + sc*uz - tc*vz
        dist = math.sqrt(dfx*dfx + dfy*dfy + dfz*dfz)

        if dist < hard_d:
            return True
    return False

@njit(parallel=True, cache=True)
def _check_candidates_against_existing_numba(
    candidate_centers: Array,
    candidate_directions: Array,
    centers: Array,
    directions: Array,
    n_existing: int,
    Lx: float,
    Ly: float,
    Lz: float,
    L: float,
    hard_d: float,
    cull_r_sq: float,
) -> Array:
    n_candidates = candidate_centers.shape[0]
    overlaps = np.zeros(n_candidates, dtype=np.uint8)
    for k in prange(n_candidates):
        cc = candidate_centers[k]
        dd = candidate_directions[k]
        overlap = _check_candidate_serial_numba(
            cc[0], cc[1], cc[2],
            dd[0], dd[1], dd[2],
            centers, directions, n_existing,
            Lx, Ly, Lz,
            L, hard_d, cull_r_sq, -1,
        )
        overlaps[k] = 1 if overlap else 0
    return overlaps


_CUPY_CANDIDATE_OVERLAP_KERNEL = None


def _check_candidates_against_existing_cupy(
    candidate_centers: Array,
    candidate_directions: Array,
    centers: Array,
    directions: Array,
    n_existing: int,
    Lx: float,
    Ly: float,
    Lz: float,
    L: float,
    hard_d: float,
    cull_r_sq: float,
) -> Array:
    """Run the exact periodic segment-distance candidate audit on CUDA."""
    import cupy as cp

    global _CUPY_CANDIDATE_OVERLAP_KERNEL
    if _CUPY_CANDIDATE_OVERLAP_KERNEL is None:
        source = r"""
        extern "C" __global__
        void candidate_overlap(
            const double* candidate_centers,
            const double* candidate_directions,
            const double* centers,
            const double* directions,
            const int n_candidates,
            const int n_existing,
            const double Lx,
            const double Ly,
            const double Lz,
            const double L,
            const double hard_d,
            const double cull_r_sq,
            unsigned char* overlaps)
        {
            const int k = blockDim.x * blockIdx.x + threadIdx.x;
            if (k >= n_candidates) return;
            const double SMALL = 1.0e-12;
            const double ccx = candidate_centers[3*k];
            const double ccy = candidate_centers[3*k + 1];
            const double ccz = candidate_centers[3*k + 2];
            const double ddx0 = candidate_directions[3*k];
            const double ddy0 = candidate_directions[3*k + 1];
            const double ddz0 = candidate_directions[3*k + 2];
            const double ux = ddx0 * L;
            const double uy = ddy0 * L;
            const double uz = ddz0 * L;

            overlaps[k] = 0;
            for (int i = 0; i < n_existing; ++i) {
                double dcx = centers[3*i] - ccx;
                double dcy = centers[3*i + 1] - ccy;
                double dcz = centers[3*i + 2] - ccz;
                dcx -= nearbyint(dcx / Lx) * Lx;
                dcy -= nearbyint(dcy / Ly) * Ly;
                dcz -= nearbyint(dcz / Lz) * Lz;
                if (dcx*dcx + dcy*dcy + dcz*dcz > cull_r_sq) continue;

                const double vx = directions[3*i] * L;
                const double vy = directions[3*i + 1] * L;
                const double vz = directions[3*i + 2] * L;
                const double ox = ccx + dcx;
                const double oy = ccy + dcy;
                const double oz = ccz + dcz;
                const double w0x = (ccx - 0.5*ux) - (ox - 0.5*vx);
                const double w0y = (ccy - 0.5*uy) - (oy - 0.5*vy);
                const double w0z = (ccz - 0.5*uz) - (oz - 0.5*vz);

                const double a = ux*ux + uy*uy + uz*uz;
                const double b = ux*vx + uy*vy + uz*vz;
                const double c = vx*vx + vy*vy + vz*vz;
                const double d = ux*w0x + uy*w0y + uz*w0z;
                const double e = vx*w0x + vy*w0y + vz*w0z;
                const double det = a*c - b*b;
                double sN, sD, tN, tD;

                if (det < SMALL) {
                    sN = 0.0;
                    sD = 1.0;
                    tN = e;
                    tD = c;
                } else {
                    sN = b*e - c*d;
                    tN = a*e - b*d;
                    sD = det;
                    tD = det;
                    if (sN < 0.0) {
                        sN = 0.0;
                        tN = e;
                        tD = c;
                    } else if (sN > sD) {
                        sN = sD;
                        tN = e + b;
                        tD = c;
                    }
                }

                if (tN < 0.0) {
                    tN = 0.0;
                    sN = fmax(0.0, fmin(-d, a));
                    sD = a;
                } else if (tN > tD) {
                    tN = tD;
                    sN = fmax(0.0, fmin(-d + b, a));
                    sD = a;
                }

                const double sc = fabs(sN) < SMALL ? 0.0 : sN / sD;
                const double tc = fabs(tN) < SMALL ? 0.0 : tN / tD;
                const double dfx = w0x + sc*ux - tc*vx;
                const double dfy = w0y + sc*uy - tc*vy;
                const double dfz = w0z + sc*uz - tc*vz;
                if (dfx*dfx + dfy*dfy + dfz*dfz < hard_d*hard_d) {
                    overlaps[k] = 1;
                    return;
                }
            }
        }
        """
        _CUPY_CANDIDATE_OVERLAP_KERNEL = cp.RawKernel(
            source,
            "candidate_overlap",
            options=("--std=c++11",),
        )

    candidate_centers_gpu = cp.asarray(
        np.ascontiguousarray(candidate_centers, dtype=np.float64)
    )
    candidate_directions_gpu = cp.asarray(
        np.ascontiguousarray(candidate_directions, dtype=np.float64)
    )
    centers_gpu = cp.asarray(
        np.ascontiguousarray(centers[:n_existing], dtype=np.float64)
    )
    directions_gpu = cp.asarray(
        np.ascontiguousarray(directions[:n_existing], dtype=np.float64)
    )
    overlaps_gpu = cp.zeros(len(candidate_centers), dtype=cp.uint8)
    threads = 128
    blocks = (len(candidate_centers) + threads - 1) // threads
    _CUPY_CANDIDATE_OVERLAP_KERNEL(
        (blocks,),
        (threads,),
        (
            candidate_centers_gpu,
            candidate_directions_gpu,
            centers_gpu,
            directions_gpu,
            np.int32(len(candidate_centers)),
            np.int32(n_existing),
            np.float64(Lx),
            np.float64(Ly),
            np.float64(Lz),
            np.float64(L),
            np.float64(hard_d),
            np.float64(cull_r_sq),
            overlaps_gpu,
        ),
    )
    return cp.asnumpy(overlaps_gpu)


@njit(cache=True)
def _occupancy_counts_numba(
    centers: Array,
    nx: int,
    ny: int,
    nz: int,
    Lx: float,
    Ly: float,
    Lz: float,
) -> Array:
    total_cells = nx * ny * nz
    occ = np.zeros(total_cells, dtype=np.float64)
    n = centers.shape[0]
    for i in range(n):
        ix = min(nx - 1, int(math.floor(centers[i, 0] / Lx * nx)))
        iy = min(ny - 1, int(math.floor(centers[i, 1] / Ly * ny)))
        iz = min(nz - 1, int(math.floor(centers[i, 2] / Lz * nz)))
        key = ix * (ny * nz) + iy * nz + iz
        occ[key] += 1.0
    return occ


@njit(parallel=True, cache=True)
def _nearest_periodic_sq_numba(
    grid_centers: Array,
    existing: Array,
    Lx: float,
    Ly: float,
    Lz: float,
) -> Array:
    m = grid_centers.shape[0]
    n = existing.shape[0]
    nearest_sq = np.empty(m, dtype=np.float64)
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz
    for i in prange(m):
        gx = grid_centers[i, 0]
        gy = grid_centers[i, 1]
        gz = grid_centers[i, 2]
        best = np.inf
        for j in range(n):
            dx = gx - existing[j, 0]
            dy = gy - existing[j, 1]
            dz = gz - existing[j, 2]
            dx -= round(dx * inv_Lx) * Lx
            dy -= round(dy * inv_Ly) * Ly
            dz -= round(dz * inv_Lz) * Lz
            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq < best:
                best = dist_sq
        nearest_sq[i] = best
    return nearest_sq

@njit(cache=True)
def _build_pairs_numba(centers: Array, Lx: float, Ly: float, Lz: float, cell_w: float) -> Array:
    n = len(centers)
    if n == 0:
        return np.empty((0, 2), dtype=np.int32)

    nx = max(1, int(math.floor(Lx / cell_w)))
    ny = max(1, int(math.floor(Ly / cell_w)))
    nz = max(1, int(math.floor(Lz / cell_w)))
    total_cells = nx * ny * nz

    keys = np.empty(n, dtype=np.int32)
    for i in range(n):
        kx = min(nx - 1, int(math.floor(centers[i, 0] / Lx * nx)))
        ky = min(ny - 1, int(math.floor(centers[i, 1] / Ly * ny)))
        kz = min(nz - 1, int(math.floor(centers[i, 2] / Lz * nz)))
        keys[i] = kx * (ny * nz) + ky * nz + kz

    head = np.full(total_cells, -1, dtype=np.int32)
    nxt = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        k = keys[i]
        nxt[i] = head[k]
        head[k] = i

    max_pairs = min(n * n, max(1000, n * 500))
    pairs = np.empty((max_pairs, 2), dtype=np.int32)
    p_count = 0

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                cell_id = ix * (ny * nz) + iy * nz + iz
                i = head[cell_id]
                while i != -1:
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            for dz in (-1, 0, 1):
                                nx_c = ix + dx
                                ny_c = iy + dy
                                nz_c = iz + dz

                                if 0 <= nx_c < nx and 0 <= ny_c < ny and 0 <= nz_c < nz:
                                    n_cell = nx_c * (ny * nz) + ny_c * nz + nz_c
                                else:
                                    n_cell = (nx_c % nx) * (ny * nz) + (ny_c % ny) * nz + (nz_c % nz)

                                j = head[n_cell]
                                while j != -1:
                                    if i < j:
                                        if p_count < max_pairs:
                                            pairs[p_count, 0] = i
                                            pairs[p_count, 1] = j
                                            p_count += 1
                                    j = nxt[j]
                    i = nxt[i]

    if p_count == 0:
        return np.empty((0, 2), dtype=np.int32)

    raw_pairs = pairs[:p_count]
    pair_ids = np.empty(p_count, dtype=np.int64)
    for k in range(p_count):
        pair_ids[k] = raw_pairs[k, 0] * n + raw_pairs[k, 1]

    pair_ids.sort()

    unique_count = 1
    for k in range(1, p_count):
        if pair_ids[k] != pair_ids[k-1]:
            unique_count += 1

    final_pairs = np.empty((unique_count, 2), dtype=np.int32)
    final_pairs[0, 0] = pair_ids[0] // n
    final_pairs[0, 1] = pair_ids[0] % n

    idx = 1
    for k in range(1, p_count):
        if pair_ids[k] != pair_ids[k-1]:
            final_pairs[idx, 0] = pair_ids[k] // n
            final_pairs[idx, 1] = pair_ids[k] % n
            idx += 1

    return final_pairs


@njit(cache=True)
def _build_pairs_center_cull_numba(
    centers: Array,
    Lx: float,
    Ly: float,
    Lz: float,
    cull_r_sq: float,
) -> Array:
    n = len(centers)
    if n < 2:
        return np.empty((0, 2), dtype=np.int32)

    max_pairs = n * (n - 1) // 2
    pairs = np.empty((max_pairs, 2), dtype=np.int32)
    pos = 0
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz
    for i in range(n - 1):
        cix = centers[i, 0]
        ciy = centers[i, 1]
        ciz = centers[i, 2]
        for j in range(i + 1, n):
            dx = centers[j, 0] - cix
            dy = centers[j, 1] - ciy
            dz = centers[j, 2] - ciz
            dx -= round(dx * inv_Lx) * Lx
            dy -= round(dy * inv_Ly) * Ly
            dz -= round(dz * inv_Lz) * Lz
            if dx * dx + dy * dy + dz * dz <= cull_r_sq:
                pairs[pos, 0] = i
                pairs[pos, 1] = j
                pos += 1

    if pos == 0:
        return np.empty((0, 2), dtype=np.int32)
    return pairs[:pos]


@njit(cache=True)
def _apply_pair_forces_numba(
    pairs: Array, centers: Array, directions: Array,
    center_updates: Array, rot_updates: Array,
    Lx: float, Ly: float, Lz: float,
    L: float, d_eff: float,
    pair_cull_sq: float,
    effective_tau: float, tol_overlap: float,
    tau_rotate: float, eps_inv: float
) -> float:
    M = len(pairs)
    max_overlap = 0.0
    SMALL = 1e-12
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz

    for k in range(M):
        i = pairs[k, 0]
        j = pairs[k, 1]

        c1x, c1y, c1z = centers[i, 0], centers[i, 1], centers[i, 2]
        c2x, c2y, c2z = centers[j, 0], centers[j, 1], centers[j, 2]

        dx = c2x - c1x
        dy = c2y - c1y
        dz = c2z - c1z

        # Periodic wrap
        dx -= round(dx * inv_Lx) * Lx
        dy -= round(dy * inv_Ly) * Ly
        dz -= round(dz * inv_Lz) * Lz
        if dx*dx + dy*dy + dz*dz > pair_cull_sq:
            continue

        c2img_x = c1x + dx
        c2img_y = c1y + dy
        c2img_z = c1z + dz

        d1x, d1y, d1z = directions[i, 0], directions[i, 1], directions[i, 2]
        d2x, d2y, d2z = directions[j, 0], directions[j, 1], directions[j, 2]

        ux, uy, uz = d1x * L, d1y * L, d1z * L
        vx, vy, vz = d2x * L, d2y * L, d2z * L

        p0x = c1x - 0.5 * ux
        p0y = c1y - 0.5 * uy
        p0z = c1z - 0.5 * uz

        q0x = c2img_x - 0.5 * vx
        q0y = c2img_y - 0.5 * vy
        q0z = c2img_z - 0.5 * vz

        w0x = p0x - q0x
        w0y = p0y - q0y
        w0z = p0z - q0z

        a = ux*ux + uy*uy + uz*uz
        b = ux*vx + uy*vy + uz*vz
        c = vx*vx + vy*vy + vz*vz
        d = ux*w0x + uy*w0y + uz*w0z
        e = vx*w0x + vy*w0y + vz*w0z

        Det = a*c - b*b
        if Det < SMALL:
            sN = 0.0
            sD = 1.0
            tN = e
            tD = c
        else:
            sN = b*e - c*d
            tN = a*e - b*d
            sD = Det
            tD = Det
            if sN < 0.0:
                sN = 0.0
                tN = e
                tD = c
            elif sN > sD:
                sN = sD
                tN = e + b
                tD = c

        if tN < 0.0:
            tN = 0.0
            sN = max(0.0, min(-d, a))
            sD = a
        elif tN > tD:
            tN = tD
            md = -d + b
            sN = max(0.0, min(md, a))
            sD = a

        sc = 0.0 if abs(sN) < SMALL else sN / sD
        tc = 0.0 if abs(tN) < SMALL else tN / tD

        diffx = w0x + sc*ux - tc*vx
        diffy = w0y + sc*uy - tc*vy
        diffz = w0z + sc*uz - tc*vz

        dist = math.sqrt(diffx*diffx + diffy*diffy + diffz*diffz)
        delta = d_eff - dist

        if delta > 0.0:
            if delta > max_overlap:
                max_overlap = delta

            xix = -diffx
            xiy = -diffy
            xiz = -diffz
            norm_xi = max(dist, SMALL)

            uvec_x = xix / norm_xi
            uvec_y = xiy / norm_xi
            uvec_z = xiz / norm_xi

            pmag = max(
                min(max(effective_tau, 1e-6), 0.501) * delta,
                min(0.5 * tol_overlap, 0.45 * delta),
            )

            pushx = pmag * uvec_x
            pushy = pmag * uvec_y
            pushz = pmag * uvec_z

            center_updates[i, 0] -= pushx
            center_updates[i, 1] -= pushy
            center_updates[i, 2] -= pushz

            center_updates[j, 0] += pushx
            center_updates[j, 1] += pushy
            center_updates[j, 2] += pushz

            si = 2.0 * sc - 1.0
            sj = 2.0 * tc - 1.0

            raw_ix = (L * si / 2.0) * delta * (-uvec_x)
            raw_iy = (L * si / 2.0) * delta * (-uvec_y)
            raw_iz = (L * si / 2.0) * delta * (-uvec_z)

            dot_i = d1x*raw_ix + d1y*raw_iy + d1z*raw_iz
            proj_ix = raw_ix - dot_i * d1x
            proj_iy = raw_iy - dot_i * d1y
            proj_iz = raw_iz - dot_i * d1z

            rot_updates[i, 0] += tau_rotate * proj_ix * eps_inv
            rot_updates[i, 1] += tau_rotate * proj_iy * eps_inv
            rot_updates[i, 2] += tau_rotate * proj_iz * eps_inv

            raw_jx = (L * sj / 2.0) * delta * uvec_x
            raw_jy = (L * sj / 2.0) * delta * uvec_y
            raw_jz = (L * sj / 2.0) * delta * uvec_z

            dot_j = d2x*raw_jx + d2y*raw_jy + d2z*raw_jz
            proj_jx = raw_jx - dot_j * d2x
            proj_jy = raw_jy - dot_j * d2y
            proj_jz = raw_jz - dot_j * d2z

            rot_updates[j, 0] += tau_rotate * proj_jx * eps_inv
            rot_updates[j, 1] += tau_rotate * proj_jy * eps_inv
            rot_updates[j, 2] += tau_rotate * proj_jz * eps_inv

    return max_overlap


@njit(cache=True)
def _pairs_to_csr_numba(pairs: Array, n_fibers: int) -> Tuple[Array, Array]:
    counts = np.zeros(n_fibers, dtype=np.int32)
    for pair_idx in range(len(pairs)):
        counts[pairs[pair_idx, 0]] += 1
        counts[pairs[pair_idx, 1]] += 1

    offsets = np.empty(n_fibers + 1, dtype=np.int32)
    offsets[0] = 0
    for fiber_idx in range(n_fibers):
        offsets[fiber_idx + 1] = offsets[fiber_idx] + counts[fiber_idx]

    neighbors = np.empty(offsets[-1], dtype=np.int32)
    cursor = offsets[:-1].copy()
    for pair_idx in range(len(pairs)):
        i = pairs[pair_idx, 0]
        j = pairs[pair_idx, 1]
        neighbors[cursor[i]] = j
        cursor[i] += 1
        neighbors[cursor[j]] = i
        cursor[j] += 1
    return offsets, neighbors


@njit(cache=True)
def _complete_neighbor_csr_numba(n_fibers: int) -> Tuple[Array, Array]:
    """Build a compact all-to-all CSR graph without materializing pair rows."""
    offsets = np.empty(n_fibers + 1, dtype=np.int32)
    for fiber_idx in range(n_fibers + 1):
        offsets[fiber_idx] = fiber_idx * max(0, n_fibers - 1)

    neighbors = np.empty(max(0, n_fibers * (n_fibers - 1)), dtype=np.int32)
    cursor = 0
    for fiber_idx in range(n_fibers):
        for neighbor_idx in range(n_fibers):
            if neighbor_idx != fiber_idx:
                neighbors[cursor] = neighbor_idx
                cursor += 1
    return offsets, neighbors


@njit(parallel=True, cache=True)
def _apply_forces_per_fiber_numba(
    centers: Array,
    directions: Array,
    neighbor_offsets: Array,
    neighbor_indices: Array,
    center_updates: Array,
    rot_updates: Array,
    max_overlap_per_fiber: Array,
    Lx: float,
    Ly: float,
    Lz: float,
    L: float,
    d_eff: float,
    pair_cull_sq: float,
    effective_tau: float,
    tol_overlap: float,
    tau_rotate: float,
    eps_inv: float,
) -> None:
    """Jacobi contact migration with race-free parallel accumulation by fiber."""
    n = centers.shape[0]
    SMALL = 1e-12
    move_gain = min(max(effective_tau, 1e-6), 0.501)
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz

    for i in prange(n):
        c1x = centers[i, 0]
        c1y = centers[i, 1]
        c1z = centers[i, 2]
        d1x = directions[i, 0]
        d1y = directions[i, 1]
        d1z = directions[i, 2]
        ux = d1x * L
        uy = d1y * L
        uz = d1z * L
        p0x = c1x - 0.5 * ux
        p0y = c1y - 0.5 * uy
        p0z = c1z - 0.5 * uz

        update_x = 0.0
        update_y = 0.0
        update_z = 0.0
        rotate_x = 0.0
        rotate_y = 0.0
        rotate_z = 0.0
        local_max_overlap = 0.0

        for neighbor_pos in range(neighbor_offsets[i], neighbor_offsets[i + 1]):
            j = neighbor_indices[neighbor_pos]

            dx = centers[j, 0] - c1x
            dy = centers[j, 1] - c1y
            dz = centers[j, 2] - c1z
            dx -= round(dx * inv_Lx) * Lx
            dy -= round(dy * inv_Ly) * Ly
            dz -= round(dz * inv_Lz) * Lz
            if dx * dx + dy * dy + dz * dz > pair_cull_sq:
                continue

            d2x = directions[j, 0]
            d2y = directions[j, 1]
            d2z = directions[j, 2]
            vx = d2x * L
            vy = d2y * L
            vz = d2z * L
            q0x = c1x + dx - 0.5 * vx
            q0y = c1y + dy - 0.5 * vy
            q0z = c1z + dz - 0.5 * vz

            w0x = p0x - q0x
            w0y = p0y - q0y
            w0z = p0z - q0z
            a = ux * ux + uy * uy + uz * uz
            b = ux * vx + uy * vy + uz * vz
            c = vx * vx + vy * vy + vz * vz
            dd = ux * w0x + uy * w0y + uz * w0z
            e = vx * w0x + vy * w0y + vz * w0z
            det = a * c - b * b

            if det < SMALL:
                sN = 0.0
                sD = 1.0
                tN = e
                tD = c
            else:
                sN = b * e - c * dd
                tN = a * e - b * dd
                sD = det
                tD = det
                if sN < 0.0:
                    sN = 0.0
                    tN = e
                    tD = c
                elif sN > sD:
                    sN = sD
                    tN = e + b
                    tD = c

            if tN < 0.0:
                tN = 0.0
                sN = max(0.0, min(-dd, a))
                sD = a
            elif tN > tD:
                tN = tD
                sN = max(0.0, min(-dd + b, a))
                sD = a

            sc = 0.0 if abs(sN) < SMALL else sN / sD
            tc = 0.0 if abs(tN) < SMALL else tN / tD
            diffx = w0x + sc * ux - tc * vx
            diffy = w0y + sc * uy - tc * vy
            diffz = w0z + sc * uz - tc * vz
            dist = math.sqrt(diffx * diffx + diffy * diffy + diffz * diffz)
            delta = d_eff - dist
            if delta <= 0.0:
                continue

            if delta > local_max_overlap:
                local_max_overlap = delta

            if dist > 1e-10:
                uvec_x = -diffx / dist
                uvec_y = -diffy / dist
                uvec_z = -diffz / dist
            else:
                uvec_x = d1y * d2z - d1z * d2y
                uvec_y = d1z * d2x - d1x * d2z
                uvec_z = d1x * d2y - d1y * d2x
                unorm = math.sqrt(
                    uvec_x * uvec_x + uvec_y * uvec_y + uvec_z * uvec_z
                )
                if unorm < 1e-10:
                    if abs(d1x) <= abs(d1y) and abs(d1x) <= abs(d1z):
                        uvec_x, uvec_y, uvec_z = 0.0, -d1z, d1y
                    elif abs(d1y) <= abs(d1z):
                        uvec_x, uvec_y, uvec_z = -d1z, 0.0, d1x
                    else:
                        uvec_x, uvec_y, uvec_z = -d1y, d1x, 0.0
                    unorm = math.sqrt(
                        uvec_x * uvec_x + uvec_y * uvec_y + uvec_z * uvec_z
                    )
                if i > j:
                    unorm = -unorm
                uvec_x /= unorm
                uvec_y /= unorm
                uvec_z /= unorm

            pmag = max(
                move_gain * delta,
                min(0.5 * tol_overlap, 0.45 * delta),
            )
            update_x -= pmag * uvec_x
            update_y -= pmag * uvec_y
            update_z -= pmag * uvec_z

            si = 2.0 * sc - 1.0
            raw_x = (L * si / 2.0) * delta * (-uvec_x)
            raw_y = (L * si / 2.0) * delta * (-uvec_y)
            raw_z = (L * si / 2.0) * delta * (-uvec_z)
            dot_i = d1x * raw_x + d1y * raw_y + d1z * raw_z
            rotate_x += tau_rotate * (raw_x - dot_i * d1x) * eps_inv
            rotate_y += tau_rotate * (raw_y - dot_i * d1y) * eps_inv
            rotate_z += tau_rotate * (raw_z - dot_i * d1z) * eps_inv

        center_updates[i, 0] = update_x
        center_updates[i, 1] = update_y
        center_updates[i, 2] = update_z
        rot_updates[i, 0] = rotate_x
        rot_updates[i, 1] = rotate_y
        rot_updates[i, 2] = rotate_z
        max_overlap_per_fiber[i] = local_max_overlap


_CUPY_CONTACT_FORCE_KERNEL = None


def _apply_forces_per_fiber_cupy(
    centers: Array,
    directions: Array,
    neighbor_offsets: Array,
    neighbor_indices: Array,
    Lx: float,
    Ly: float,
    Lz: float,
    L: float,
    d_eff: float,
    pair_cull_sq: float,
    effective_tau: float,
    tol_overlap: float,
    tau_rotate: float,
    eps_inv: float,
    all_pairs: bool = False,
) -> Tuple[Array, Array, Array]:
    """Evaluate race-free per-fiber contact migration on CUDA."""
    import cupy as cp

    global _CUPY_CONTACT_FORCE_KERNEL
    if _CUPY_CONTACT_FORCE_KERNEL is None:
        source = r"""
        extern "C" __global__
        void contact_forces(
            const double* centers,
            const double* directions,
            const int* offsets,
            const int* neighbors,
            const int n,
            const double Lx,
            const double Ly,
            const double Lz,
            const double L,
            const double d_eff,
            const double pair_cull_sq,
            const double effective_tau,
            const double tol_overlap,
            const double tau_rotate,
            const double eps_inv,
            const int all_pairs,
            double* center_updates,
            double* rot_updates,
            double* max_overlap)
        {
            const int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= n) return;
            const double SMALL = 1.0e-12;
            const double move_gain = fmin(fmax(effective_tau, 1.0e-6), 0.501);
            const double c1x = centers[3*i];
            const double c1y = centers[3*i + 1];
            const double c1z = centers[3*i + 2];
            const double d1x = directions[3*i];
            const double d1y = directions[3*i + 1];
            const double d1z = directions[3*i + 2];
            const double ux = d1x * L;
            const double uy = d1y * L;
            const double uz = d1z * L;
            const double p0x = c1x - 0.5 * ux;
            const double p0y = c1y - 0.5 * uy;
            const double p0z = c1z - 0.5 * uz;
            double update_x = 0.0;
            double update_y = 0.0;
            double update_z = 0.0;
            double rotate_x = 0.0;
            double rotate_y = 0.0;
            double rotate_z = 0.0;
            double local_max = 0.0;

            const int pair_start = all_pairs ? 0 : offsets[i];
            const int pair_stop = all_pairs ? n : offsets[i + 1];
            for (int pos = pair_start; pos < pair_stop; ++pos) {
                const int j = all_pairs ? pos : neighbors[pos];
                if (j == i) continue;
                double dx = centers[3*j] - c1x;
                double dy = centers[3*j + 1] - c1y;
                double dz = centers[3*j + 2] - c1z;
                dx -= nearbyint(dx / Lx) * Lx;
                dy -= nearbyint(dy / Ly) * Ly;
                dz -= nearbyint(dz / Lz) * Lz;
                if (dx*dx + dy*dy + dz*dz > pair_cull_sq) continue;

                const double d2x = directions[3*j];
                const double d2y = directions[3*j + 1];
                const double d2z = directions[3*j + 2];
                const double vx = d2x * L;
                const double vy = d2y * L;
                const double vz = d2z * L;
                const double q0x = c1x + dx - 0.5 * vx;
                const double q0y = c1y + dy - 0.5 * vy;
                const double q0z = c1z + dz - 0.5 * vz;
                const double w0x = p0x - q0x;
                const double w0y = p0y - q0y;
                const double w0z = p0z - q0z;
                const double a = ux*ux + uy*uy + uz*uz;
                const double b = ux*vx + uy*vy + uz*vz;
                const double c = vx*vx + vy*vy + vz*vz;
                const double dd = ux*w0x + uy*w0y + uz*w0z;
                const double e = vx*w0x + vy*w0y + vz*w0z;
                const double det = a*c - b*b;
                double sN, sD, tN, tD;

                if (det < SMALL) {
                    sN = 0.0; sD = 1.0; tN = e; tD = c;
                } else {
                    sN = b*e - c*dd;
                    tN = a*e - b*dd;
                    sD = det;
                    tD = det;
                    if (sN < 0.0) {
                        sN = 0.0; tN = e; tD = c;
                    } else if (sN > sD) {
                        sN = sD; tN = e + b; tD = c;
                    }
                }
                if (tN < 0.0) {
                    tN = 0.0;
                    sN = fmax(0.0, fmin(-dd, a));
                    sD = a;
                } else if (tN > tD) {
                    tN = tD;
                    sN = fmax(0.0, fmin(-dd + b, a));
                    sD = a;
                }

                const double sc = fabs(sN) < SMALL ? 0.0 : sN / sD;
                const double tc = fabs(tN) < SMALL ? 0.0 : tN / tD;
                const double diffx = w0x + sc*ux - tc*vx;
                const double diffy = w0y + sc*uy - tc*vy;
                const double diffz = w0z + sc*uz - tc*vz;
                const double dist = sqrt(
                    diffx*diffx + diffy*diffy + diffz*diffz
                );
                const double delta = d_eff - dist;
                if (delta <= 0.0) continue;
                local_max = fmax(local_max, delta);

                double uvec_x, uvec_y, uvec_z;
                if (dist > 1.0e-10) {
                    uvec_x = -diffx / dist;
                    uvec_y = -diffy / dist;
                    uvec_z = -diffz / dist;
                } else {
                    uvec_x = d1y*d2z - d1z*d2y;
                    uvec_y = d1z*d2x - d1x*d2z;
                    uvec_z = d1x*d2y - d1y*d2x;
                    double unorm = sqrt(
                        uvec_x*uvec_x + uvec_y*uvec_y + uvec_z*uvec_z
                    );
                    if (unorm < 1.0e-10) {
                        if (fabs(d1x) <= fabs(d1y) && fabs(d1x) <= fabs(d1z)) {
                            uvec_x = 0.0; uvec_y = -d1z; uvec_z = d1y;
                        } else if (fabs(d1y) <= fabs(d1z)) {
                            uvec_x = -d1z; uvec_y = 0.0; uvec_z = d1x;
                        } else {
                            uvec_x = -d1y; uvec_y = d1x; uvec_z = 0.0;
                        }
                        unorm = sqrt(
                            uvec_x*uvec_x + uvec_y*uvec_y + uvec_z*uvec_z
                        );
                    }
                    if (i > j) unorm = -unorm;
                    uvec_x /= unorm;
                    uvec_y /= unorm;
                    uvec_z /= unorm;
                }

                const double pmag = fmax(
                    move_gain * delta,
                    fmin(0.5 * tol_overlap, 0.45 * delta)
                );
                update_x -= pmag * uvec_x;
                update_y -= pmag * uvec_y;
                update_z -= pmag * uvec_z;
                const double si = 2.0 * sc - 1.0;
                const double raw_x = 0.5 * L * si * delta * (-uvec_x);
                const double raw_y = 0.5 * L * si * delta * (-uvec_y);
                const double raw_z = 0.5 * L * si * delta * (-uvec_z);
                const double dot_i = d1x*raw_x + d1y*raw_y + d1z*raw_z;
                rotate_x += tau_rotate * (raw_x - dot_i*d1x) * eps_inv;
                rotate_y += tau_rotate * (raw_y - dot_i*d1y) * eps_inv;
                rotate_z += tau_rotate * (raw_z - dot_i*d1z) * eps_inv;
            }

            center_updates[3*i] = update_x;
            center_updates[3*i + 1] = update_y;
            center_updates[3*i + 2] = update_z;
            rot_updates[3*i] = rotate_x;
            rot_updates[3*i + 1] = rotate_y;
            rot_updates[3*i + 2] = rotate_z;
            max_overlap[i] = local_max;
        }
        """
        _CUPY_CONTACT_FORCE_KERNEL = cp.RawKernel(
            source,
            "contact_forces",
            options=("--std=c++11",),
        )

    centers_gpu = cp.asarray(np.ascontiguousarray(centers, dtype=np.float64))
    directions_gpu = cp.asarray(
        np.ascontiguousarray(directions, dtype=np.float64)
    )
    offsets_gpu = cp.asarray(
        np.ascontiguousarray(neighbor_offsets, dtype=np.int32)
    )
    neighbors_gpu = cp.asarray(
        np.ascontiguousarray(neighbor_indices, dtype=np.int32)
    )
    center_updates_gpu = cp.zeros_like(centers_gpu)
    rot_updates_gpu = cp.zeros_like(directions_gpu)
    max_overlap_gpu = cp.zeros(len(centers), dtype=cp.float64)
    threads = 128
    blocks = (len(centers) + threads - 1) // threads
    _CUPY_CONTACT_FORCE_KERNEL(
        (blocks,),
        (threads,),
        (
            centers_gpu,
            directions_gpu,
            offsets_gpu,
            neighbors_gpu,
            np.int32(len(centers)),
            np.float64(Lx),
            np.float64(Ly),
            np.float64(Lz),
            np.float64(L),
            np.float64(d_eff),
            np.float64(pair_cull_sq),
            np.float64(effective_tau),
            np.float64(tol_overlap),
            np.float64(tau_rotate),
            np.float64(eps_inv),
            np.int32(bool(all_pairs)),
            center_updates_gpu,
            rot_updates_gpu,
            max_overlap_gpu,
        ),
    )
    return (
        cp.asnumpy(center_updates_gpu),
        cp.asnumpy(rot_updates_gpu),
        cp.asnumpy(max_overlap_gpu),
    )


@njit(parallel=True, cache=True)
def _apply_updates_numba(
    centers: Array,
    directions: Array,
    center_updates: Array,
    rot_updates: Array,
    Lx: float,
    Ly: float,
    Lz: float,
) -> None:
    n = centers.shape[0]
    eps = 1e-16

    for i in prange(n):
        cx = centers[i, 0] + center_updates[i, 0]
        cy = centers[i, 1] + center_updates[i, 1]
        cz = centers[i, 2] + center_updates[i, 2]

        if Lx > 0.0:
            cx -= math.floor(cx / Lx) * Lx
        if Ly > 0.0:
            cy -= math.floor(cy / Ly) * Ly
        if Lz > 0.0:
            cz -= math.floor(cz / Lz) * Lz

        centers[i, 0] = cx
        centers[i, 1] = cy
        centers[i, 2] = cz

        px = directions[i, 0]
        py = directions[i, 1]
        pz = directions[i, 2]
        dpx = rot_updates[i, 0]
        dpy = rot_updates[i, 1]
        dpz = rot_updates[i, 2]

        theta = math.sqrt(dpx*dpx + dpy*dpy + dpz*dpz)
        if theta < 1e-14:
            pn = math.sqrt(px*px + py*py + pz*pz)
            invn = 1.0 / max(pn, eps)
            directions[i, 0] = px * invn
            directions[i, 1] = py * invn
            directions[i, 2] = pz * invn
            continue

        ax = py*dpz - pz*dpy
        ay = pz*dpx - px*dpz
        az = px*dpy - py*dpx
        axis_norm = math.sqrt(ax*ax + ay*ay + az*az)

        if axis_norm < 1e-14:
            pn = math.sqrt(px*px + py*py + pz*pz)
            invn = 1.0 / max(pn, eps)
            directions[i, 0] = px * invn
            directions[i, 1] = py * invn
            directions[i, 2] = pz * invn
            continue

        ax /= axis_norm
        ay /= axis_norm
        az /= axis_norm

        sin_t = math.sin(theta)
        cos_t = math.cos(theta)

        cross_x = ay*pz - az*py
        cross_y = az*px - ax*pz
        cross_z = ax*py - ay*px
        axis_dot_p = ax*px + ay*py + az*pz

        pnx = px * cos_t + cross_x * sin_t + ax * axis_dot_p * (1.0 - cos_t)
        pny = py * cos_t + cross_y * sin_t + ay * axis_dot_p * (1.0 - cos_t)
        pnz = pz * cos_t + cross_z * sin_t + az * axis_dot_p * (1.0 - cos_t)

        pn = math.sqrt(pnx*pnx + pny*pny + pnz*pnz)
        invn = 1.0 / max(pn, eps)
        directions[i, 0] = pnx * invn
        directions[i, 1] = pny * invn
        directions[i, 2] = pnz * invn


def warmup_numba_geometry_kernels() -> float:
    """Compile/cache the CPU geometry signatures once before spawning workers."""
    if not NUMBA_AVAILABLE:
        return 0.0

    t0 = time.perf_counter()
    previous_threads = numba_get_num_threads()
    numba_set_num_threads(1)
    try:
        centers = np.array(
            [[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [6.0, 6.0, 6.0]],
            dtype=np.float64,
        )
        directions = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        candidate_centers = np.array([[2.0, 2.0, 2.0]], dtype=np.float64)
        candidate_directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        box = 8.0
        fiber_length = 2.0
        diameter = 0.5
        cull_sq = (fiber_length + diameter) ** 2

        _check_candidates_against_existing_numba(
            candidate_centers,
            candidate_directions,
            centers,
            directions,
            len(centers),
            box,
            box,
            box,
            fiber_length,
            diameter,
            cull_sq,
        )
        _occupancy_counts_numba(centers, 2, 2, 2, box, box, box)
        _nearest_periodic_sq_numba(
            candidate_centers,
            centers,
            box,
            box,
            box,
        )
        _build_pairs_numba(centers, box, box, box, fiber_length + diameter)
        pairs = _build_pairs_center_cull_numba(
            centers,
            box,
            box,
            box,
            cull_sq,
        )
        center_updates = np.zeros_like(centers)
        rot_updates = np.zeros_like(directions)
        force_args = (
            box,
            box,
            box,
            fiber_length,
            diameter,
            cull_sq,
            0.1,
            0.05,
            0.01,
            0.25,
        )
        _apply_pair_forces_numba(
            pairs,
            centers,
            directions,
            center_updates,
            rot_updates,
            *force_args,
        )
        offsets, neighbors = _pairs_to_csr_numba(pairs, len(centers))
        max_overlap = np.zeros(len(centers), dtype=np.float64)
        _apply_forces_per_fiber_numba(
            centers,
            directions,
            offsets,
            neighbors,
            center_updates,
            rot_updates,
            max_overlap,
            *force_args,
        )
        _apply_updates_numba(
            centers.copy(),
            directions.copy(),
            center_updates,
            rot_updates,
            box,
            box,
            box,
        )
    finally:
        numba_set_num_threads(previous_threads)
    return float(time.perf_counter() - t0)


class SAMLiteGenerator:
    """Sequential Addition + Migration generator for straight fibers."""

    def __init__(
        self,
        A2: Array,
        fiber_length: float,
        fiber_diameter: float,
        vf_target: float,
        cell_size: Tuple[float, float, float],
        batch_vf: float = 0.01,
        tau_translate: float = 0.20,
        tau_rotate: float = 0.03,
        w_orient: float = 0.05,
        tol_A: float = 1e-3,
        tol_overlap: float = 1e-6,
        max_iter_relax: int = 2000,
        min_gap: float = 0.0,
        fast_mode: bool = False,
        bias_free_volume: bool = True,
        use_local_relax: bool = True,
        strict_insertion: bool = False,
        soft_insert: bool = False,
        soft_insert_start: float = 0.72,
        seed: int = 123,
        num_threads: int = 1,
        geometry_backend: str = "numba",
        cupy_min_candidate_pairs: int = 50_000,
        center_distribution: str = "uniform",
        cluster_fraction: float = 0.0,
        cluster_count: int = 4,
        cluster_sigma_rel: float = 0.12,
    ) -> None:
        self.A2 = self._sanitize_A2(np.asarray(A2, dtype=float))
        a2_key = tuple(float(round(value, 12)) for value in self.A2.ravel())
        self._A2_transform = _angular_gaussian_transform(a2_key)
        self.fiber_length = float(fiber_length)
        self.fiber_diameter = float(fiber_diameter)
        self.vf_target = float(vf_target)
        self.cell_size = np.asarray(cell_size, dtype=float)
        self.batch_vf = float(batch_vf)
        self.tau_translate = float(tau_translate)
        self.tau_rotate = float(tau_rotate)
        self.w_orient = float(w_orient)
        self.tol_A = float(tol_A)
        self.tol_overlap = float(tol_overlap)
        self.max_iter_relax = int(max_iter_relax)
        self.min_gap = float(min_gap)
        self.fast_mode = bool(fast_mode)
        self.bias_free_volume = bool(bias_free_volume)
        self.use_local_relax = bool(use_local_relax)
        self.strict_insertion = bool(strict_insertion)
        self.soft_insert = bool(soft_insert)
        self.soft_insert_start = float(soft_insert_start)
        self.num_threads = max(1, int(num_threads))
        backend = str(geometry_backend).strip().lower()
        if backend not in {"numba", "cupy", "auto"}:
            raise ValueError(
                "geometry_backend must be 'numba', 'cupy', or 'auto'."
            )
        self.geometry_backend_requested = backend
        self.geometry_backend_effective = "numba"
        self.cupy_min_candidate_pairs = max(
            1,
            int(cupy_min_candidate_pairs),
        )
        self._cupy_geometry_disabled = False
        self._cupy_candidate_calls = 0
        self._cupy_candidate_pairs = 0
        self._cupy_candidate_s = 0.0
        self._cupy_candidate_fallbacks = 0
        self._cupy_force_calls = 0
        self._cupy_force_pairs = 0
        self._cupy_force_s = 0.0
        self._cupy_force_fallbacks = 0
        self._full_d_eff = self.fiber_diameter + self.min_gap
        self.collision_diameter_scale = 1.0
        self.d_eff = self._full_d_eff
        self.rng = np.random.default_rng(seed)
        center_distribution = str(center_distribution).strip().lower()
        if center_distribution not in {"uniform", "gaussian_mixture"}:
            raise ValueError(
                "center_distribution must be 'uniform' or 'gaussian_mixture'."
            )
        if not 0.0 <= float(cluster_fraction) <= 1.0:
            raise ValueError("cluster_fraction must lie in [0, 1].")
        if int(cluster_count) < 1:
            raise ValueError("cluster_count must be positive.")
        if float(cluster_sigma_rel) <= 0.0:
            raise ValueError("cluster_sigma_rel must be positive.")
        self.center_distribution = center_distribution
        self.cluster_fraction = float(cluster_fraction)
        self.cluster_count = int(cluster_count)
        self.cluster_sigma_rel = float(cluster_sigma_rel)
        self._cluster_centers = (
            self.rng.random((self.cluster_count, 3)) * self.cell_size
            if self.center_distribution == "gaussian_mixture"
            else np.empty((0, 3), dtype=float)
        )
        self.last_run_stats: Dict[str, float] = {}
        self._compaction_removed_fibers = 0
        self._last_added_indices = np.empty(0, dtype=np.int32)
        self._free_volume_cache: Dict[Tuple[int, int, int], Array] = {}

        self.V_cell = float(np.prod(self.cell_size))
        self.V_fiber = math.pi * self.fiber_diameter ** 2 * self.fiber_length / 4.0

        est = max(64, int(2.0 * self.vf_target * self.V_cell / self.V_fiber))
        self._centers = np.empty((est, 3), dtype=float)
        self._directions = np.empty((est, 3), dtype=float)
        self._n_fibers = 0

        self._half_cell = 0.5 * self.cell_size
        self._rot_scale = self.rotational_scale(self.fiber_length, self.fiber_diameter)
        self._cull_padding = max(0.25, 0.05 * self.fiber_diameter)
        self._cull_radius = self.fiber_length + self.d_eff + self._cull_padding
        if NUMBA_AVAILABLE:
            try:
                numba_set_num_threads(self.num_threads)
            except Exception:
                pass

    def _ensure_capacity(self, n_extra: int) -> None:
        needed = self._n_fibers + n_extra
        if needed > self._centers.shape[0]:
            new_cap = max(needed, int(self._centers.shape[0] * 1.5))
            new_c = np.empty((new_cap, 3), dtype=float)
            new_d = np.empty((new_cap, 3), dtype=float)
            new_c[:self._n_fibers] = self._centers[:self._n_fibers]
            new_d[:self._n_fibers] = self._directions[:self._n_fibers]
            self._centers = new_c
            self._directions = new_d

    @property
    def centers(self) -> Array:
        return self._centers[:self._n_fibers]

    @property
    def directions(self) -> Array:
        return self._directions[:self._n_fibers]

    def run(
        self,
        verbose: bool = True,
        stop_callback: Optional[Callable[["SAMLiteGenerator"], bool]] = None,
    ) -> Tuple[Array, Array, Array]:
        iteration = 0
        total_attempts = 0
        total_relax_iters = 0
        stalled_batches = 0
        stopped_by_callback = False
        last_info = {"iters": 0, "max_overlap": 0.0, "A_err": 0.0}
        total_add_batch_s = 0.0
        total_sample_centers_s = 0.0
        total_sample_dirs_s = 0.0
        total_candidate_checks_s = 0.0
        total_relax_s = 0.0
        total_relax_local_s = 0.0
        total_pair_build_s = 0.0
        total_pair_forces_s = 0.0
        total_orientation_corr_s = 0.0
        total_compute_a4_s = 0.0
        total_apply_updates_s = 0.0
        cleanup_passes_used = 0
        stopped_by_stall = False
        total_soft_inserted_batches = 0.0
        cupy_calls_before = self._cupy_candidate_calls
        cupy_pairs_before = self._cupy_candidate_pairs
        cupy_s_before = self._cupy_candidate_s
        cupy_fallbacks_before = self._cupy_candidate_fallbacks
        cupy_force_calls_before = self._cupy_force_calls
        cupy_force_pairs_before = self._cupy_force_pairs
        cupy_force_s_before = self._cupy_force_s
        cupy_force_fallbacks_before = self._cupy_force_fallbacks

        if verbose:
            n_est = self.target_fiber_count()
            n_threads = numba_get_num_threads() if NUMBA_AVAILABLE else 1
            print(
                "SAMLite | "
                f"Vf_obj={100.0 * self.vf_target:.2f}% | "
                f"batch={100.0 * self.batch_vf:.2f}% | "
                f"fibras_est={n_est} | hilos={n_threads}"
            )

        def record_relax_stats(info: Dict[str, float], *, local: bool = False) -> None:
            nonlocal total_relax_iters, total_relax_s, total_relax_local_s
            nonlocal total_pair_build_s, total_pair_forces_s
            nonlocal total_orientation_corr_s, total_compute_a4_s, total_apply_updates_s

            total_relax_iters += int(info["iters"])
            if local:
                total_relax_local_s += info.get("wall_s", 0.0)
            else:
                total_relax_s += info.get("wall_s", 0.0)
            total_pair_build_s += info.get("pair_build_s", 0.0)
            total_pair_forces_s += info.get("pair_forces_s", 0.0)
            total_orientation_corr_s += info.get("orientation_corr_s", 0.0)
            total_compute_a4_s += info.get("compute_a4_s", 0.0)
            total_apply_updates_s += info.get("apply_updates_s", 0.0)

        while self._n_fibers < self.target_fiber_count():
            iteration += 1

            vf_before = self.current_vf()
            effective_batch_vf = self._choose_effective_batch_vf(vf_before)

            added, attempts, batch_stats = self.add_batch(effective_batch_vf)
            total_attempts += attempts
            total_add_batch_s += batch_stats["wall_s"]
            total_sample_centers_s += batch_stats["sample_centers_s"]
            total_sample_dirs_s += batch_stats["sample_dirs_s"]
            total_candidate_checks_s += batch_stats["candidate_checks_s"]
            total_soft_inserted_batches += batch_stats.get("soft_inserted", 0.0)
            vf_after = self.current_vf()
            progress = vf_after / max(self.vf_target, 1e-12)

            vf_added = vf_after - vf_before

            if verbose:
                print(f"|- Batch {iteration:02d} | Posicionamiento: {added} fibras ({attempts} intentos). "
                      f"Vf +{vf_added*100:.4f}% => {vf_after*100:.4f}%")

            relax_iters, target_overlap, apply_orientation = self._relax_schedule(progress)
            use_local_relax = (
                self.use_local_relax
                and
                self.fast_mode
                and added > 0
                and self._n_fibers >= 256
                and iteration % 4 != 0
                and progress < 0.97
            )
            if use_local_relax:
                local_info = self.relax_local(
                    self._last_added_indices,
                    verbose=verbose,
                    max_iter=relax_iters,
                    target_overlap=target_overlap,
                )
                last_info = local_info
                record_relax_stats(local_info, local=True)
                if local_info["max_overlap"] > max(3.0 * target_overlap, 0.10 * self.d_eff):
                    last_info = self.relax(
                        verbose=verbose,
                        max_iter=min(self.max_iter_relax, max(16, relax_iters)),
                        target_overlap=target_overlap,
                        apply_orientation=apply_orientation,
                        parallel_per_fiber=(
                            self.num_threads > 1 and self._n_fibers >= 512
                        ),
                    )
                    record_relax_stats(last_info)
            else:
                last_info = self.relax(
                    verbose=verbose,
                    max_iter=relax_iters,
                    target_overlap=target_overlap,
                    apply_orientation=apply_orientation,
                    parallel_per_fiber=(
                        self.num_threads > 1 and self._n_fibers >= 512
                    ),
                )
                record_relax_stats(last_info)

            if stop_callback is not None and self._n_fibers > 0:
                if stop_callback(self):
                    stopped_by_callback = True
                    if verbose:
                        print("|  +- Detencion temprana: objetivo voxelizado alcanzado.")
                    break

            if added == 0:
                stalled_batches += 1
                if verbose:
                    print("|  +- Insercion estancada, ejecutando limpieza estricta.")
                last_info = self.relax(
                    verbose=verbose,
                    max_iter=max(self.max_iter_relax, 80),
                    target_overlap=self.tol_overlap,
                    apply_orientation=True,
                    parallel_per_fiber=(
                        self.num_threads > 1 and self._n_fibers >= 512
                    ),
                )
                record_relax_stats(last_info)
                if stalled_batches >= 3:
                    stopped_by_stall = True
                    if verbose:
                        print("+- Detencion por seguridad: no fue posible insertar nuevas fibras.")
                    break
            else:
                stalled_batches = 0

            if self._n_fibers > self.target_fiber_count():
                if verbose:
                    print("+- Detencion por seguridad: Se ha excedido el objetivo de Vf.")
                break

        cleanup_passes = 3 if self.fast_mode else 2
        cleanup_overlap = self.measure_max_overlap()
        cleanup_A_err = self.orientation_error()
        for _ in range(cleanup_passes):
            if (
                cleanup_overlap <= self.tol_overlap
                and cleanup_A_err <= self.tol_A
            ):
                break
            cleanup_passes_used += 1
            last_info = self.relax(
                verbose=verbose,
                max_iter=max(self.max_iter_relax, 80),
                target_overlap=self.tol_overlap,
                apply_orientation=True,
                parallel_per_fiber=(
                    self.num_threads > 1 and self._n_fibers >= 512
                ),
            )
            record_relax_stats(last_info)
            cleanup_overlap = self.measure_max_overlap()
            cleanup_A_err = self.orientation_error()

        orientation_opt_info = {
            "iterations": 0.0,
            "accepted_steps": 0.0,
            "line_search_evaluations": 0.0,
            "contact_projections": 0.0,
            "initial_error": float(cleanup_A_err),
            "final_error": float(cleanup_A_err),
            "initial_overlap": float(cleanup_overlap),
            "final_overlap": float(cleanup_overlap),
            "converged": float(cleanup_A_err <= self.tol_A),
            "wall_s": 0.0,
        }
        if self._n_fibers >= 2 and cleanup_overlap <= self.tol_overlap:
            orientation_opt_info = self.optimize_A2_riemannian(
                max_iter=48 if self.fast_mode else 96,
                tolerance=max(1e-4, 0.25 * self.tol_A),
                overlap_limit=self.tol_overlap,
            )
            cleanup_overlap = self.measure_max_overlap()
            cleanup_A_err = self.orientation_error()

        target_n_fibers = self.target_fiber_count()
        discrete_target_vf = target_n_fibers * self.V_fiber / self.V_cell
        measured_final_overlap = cleanup_overlap
        self.last_run_stats = {
            "batches": float(iteration),
            "attempts": float(total_attempts),
            "relax_iters": float(total_relax_iters),
            "final_overlap": float(measured_final_overlap),
            "final_A_err": float(cleanup_A_err),
            "final_A2_error_rel": float(cleanup_A_err),
            "final_vf": float(self.current_vf()),
            "target_vf": float(self.vf_target),
            "target_n_fibers": float(target_n_fibers),
            "discrete_target_vf": float(discrete_target_vf),
            "vf_quantization_error": float(discrete_target_vf - self.vf_target),
            "target_reached": float(self._n_fibers >= target_n_fibers),
            "stopped_by_callback": float(stopped_by_callback),
            "stopped_by_stall": float(stopped_by_stall),
            "stalled_batches": float(stalled_batches),
            "n_fibers": float(self._n_fibers),
            "numba_available": float(NUMBA_AVAILABLE),
            "numba_threads": float(numba_get_num_threads() if NUMBA_AVAILABLE else 1),
            "add_batch_s": float(total_add_batch_s),
            "sample_centers_s": float(total_sample_centers_s),
            "sample_dirs_s": float(total_sample_dirs_s),
            "candidate_checks_s": float(total_candidate_checks_s),
            "relax_s": float(total_relax_s),
            "relax_local_s": float(total_relax_local_s),
            "pair_build_s": float(total_pair_build_s),
            "pair_forces_s": float(total_pair_forces_s),
            "orientation_corr_s": float(total_orientation_corr_s),
            "compute_a4_s": float(total_compute_a4_s),
            "apply_updates_s": float(total_apply_updates_s),
            "cleanup_passes": float(cleanup_passes_used),
            "soft_inserted_batches": float(total_soft_inserted_batches),
            "collision_diameter_scale": float(self.collision_diameter_scale),
            "use_local_relax": float(self.use_local_relax),
            "orientation_optimizer_iterations": float(
                orientation_opt_info["iterations"]
            ),
            "orientation_optimizer_accepted_steps": float(
                orientation_opt_info["accepted_steps"]
            ),
            "orientation_optimizer_line_search_evaluations": float(
                orientation_opt_info["line_search_evaluations"]
            ),
            "orientation_optimizer_contact_projections": float(
                orientation_opt_info["contact_projections"]
            ),
            "orientation_optimizer_initial_error": float(
                orientation_opt_info["initial_error"]
            ),
            "orientation_optimizer_final_error": float(
                orientation_opt_info["final_error"]
            ),
            "orientation_optimizer_s": float(orientation_opt_info["wall_s"]),
            "cupy_candidate_calls": float(
                self._cupy_candidate_calls - cupy_calls_before
            ),
            "cupy_candidate_pairs": float(
                self._cupy_candidate_pairs - cupy_pairs_before
            ),
            "cupy_candidate_s": float(
                self._cupy_candidate_s - cupy_s_before
            ),
            "cupy_candidate_fallbacks": float(
                self._cupy_candidate_fallbacks - cupy_fallbacks_before
            ),
            "cupy_force_calls": float(
                self._cupy_force_calls - cupy_force_calls_before
            ),
            "cupy_force_pairs": float(
                self._cupy_force_pairs - cupy_force_pairs_before
            ),
            "cupy_force_s": float(
                self._cupy_force_s - cupy_force_s_before
            ),
            "cupy_force_fallbacks": float(
                self._cupy_force_fallbacks - cupy_force_fallbacks_before
            ),
        }

        if verbose:
            print(
                "SAMLite terminado | "
                f"fibras={self._n_fibers} | "
                f"Vf={100.0 * self.current_vf():.4f}% | "
                f"solap={measured_final_overlap:.4e} | "
                f"intentos={total_attempts} | relax_iters={total_relax_iters}"
            )

        return self.export_arrays()

    def run_collective_sam(
        self,
        *,
        batch_vf: Optional[float] = None,
        max_iter_per_batch: int = 600,
        target_overlap: Optional[float] = None,
        max_restarts: int = 3,
        restart_patience: int = 250,
        min_batch_fibers: int = 1,
        stop_callback: Optional[Callable[["SAMLiteGenerator"], bool]] = None,
        verbose: bool = False,
    ) -> Tuple[Array, Array, Array]:
        """Add soft batches and migrate each one before activating the next."""
        t0 = time.perf_counter()
        self.set_collision_diameter_scale(1.0)
        overlap_limit = (
            self.tol_overlap
            if target_overlap is None
            else max(0.0, float(target_overlap))
        )
        target_count = self.target_fiber_count()
        requested_batch_vf = (
            self.batch_vf if batch_vf is None else float(batch_vf)
        )
        nominal_batch_count = max(
            1,
            int(round(requested_batch_vf * self.V_cell / self.V_fiber)),
        )
        current_batch_count = nominal_batch_count
        minimum_batch_count = max(1, int(min_batch_fibers))

        total_iters = 0
        total_restarts = 0
        total_force_evaluations = 0
        total_pair_build_s = 0.0
        total_pair_forces_s = 0.0
        total_apply_updates_s = 0.0
        total_sample_centers_s = 0.0
        total_sample_dirs_s = 0.0
        successful_batches = 0
        failed_batches = 0
        attempted_fibers = 0
        stopped_by_callback = False
        stopped_by_stall = False
        consecutive_single_failures = 0
        smallest_attempted_batch = nominal_batch_count
        last_info: Dict[str, float] = {
            "final_overlap": 0.0,
            "final_A2_error_rel": float(self.orientation_error()),
            "converged": 1.0,
        }

        cupy_force_calls_before = self._cupy_force_calls
        cupy_force_pairs_before = self._cupy_force_pairs
        cupy_force_s_before = self._cupy_force_s
        cupy_force_fallbacks_before = self._cupy_force_fallbacks

        max_batch_attempts = max(
            32,
            4 * int(math.ceil(target_count / nominal_batch_count)) + 24,
        )
        batch_attempt = 0
        while self._n_fibers < target_count and batch_attempt < max_batch_attempts:
            batch_attempt += 1
            remaining = target_count - self._n_fibers
            count_progress = self._n_fibers / max(target_count, 1)
            if count_progress >= 0.98:
                progress_batch_cap = 2
            elif count_progress >= 0.95:
                progress_batch_cap = 5
            elif count_progress >= 0.90:
                progress_batch_cap = 10
            elif count_progress >= 0.85:
                progress_batch_cap = 20
            else:
                progress_batch_cap = nominal_batch_count
            n_add = min(
                remaining,
                progress_batch_cap,
                max(minimum_batch_count, current_batch_count),
            )
            smallest_attempted_batch = min(smallest_attempted_batch, n_add)

            accepted_n = self._n_fibers
            accepted_centers = self.centers.copy()
            accepted_directions = self.directions.copy()

            sample_t0 = time.perf_counter()
            candidate_centers = self._sample_candidate_centers(
                n_add,
                self.current_vf() / max(self.vf_target, 1e-12),
                self.d_eff,
            )
            total_sample_centers_s += time.perf_counter() - sample_t0
            sample_t0 = time.perf_counter()
            candidate_directions = self._sample_directions_batch(
                n_add,
                target_A2=self._remaining_orientation_target(n_add),
            )
            total_sample_dirs_s += time.perf_counter() - sample_t0
            self._ensure_capacity(n_add)
            insert_stop = accepted_n + n_add
            self._centers[accepted_n:insert_stop] = candidate_centers
            self._directions[accepted_n:insert_stop] = candidate_directions
            self._n_fibers = insert_stop
            attempted_fibers += n_add

            gradient_budget = min(
                120,
                max(20, int(max_iter_per_batch) // 3),
            )
            gradient_info = self.relax(
                max_iter=gradient_budget,
                target_overlap=overlap_limit,
                apply_orientation=True,
                pair_rebuild_interval=2,
                effective_tau_override=0.501,
                max_center_step_factor=0.50,
                parallel_per_fiber=(
                    self.num_threads > 1 and self._n_fibers >= 512
                ),
                verbose=False,
            )
            gradient_iters = int(gradient_info.get("iters", 0.0))
            total_iters += gradient_iters
            total_force_evaluations += gradient_iters
            total_pair_build_s += float(
                gradient_info.get("pair_build_s", 0.0)
            )
            total_pair_forces_s += float(
                gradient_info.get("pair_forces_s", 0.0)
            )
            total_apply_updates_s += float(
                gradient_info.get("apply_updates_s", 0.0)
            )
            gradient_converged = bool(
                float(gradient_info.get("max_overlap", math.inf))
                <= overlap_limit
                and float(gradient_info.get("A_err", math.inf))
                <= self.tol_A
            )

            if gradient_converged:
                last_info = {
                    "converged": 1.0,
                    "final_overlap": float(
                        gradient_info.get("max_overlap", 0.0)
                    ),
                    "final_A2_error_rel": float(
                        gradient_info.get("A_err", 0.0)
                    ),
                    "iters": float(gradient_iters),
                    "restarts": 0.0,
                }
            else:
                fire_budget = max(
                    50,
                    int(max_iter_per_batch) - gradient_iters,
                )
                last_info = self.relax_collective_fire(
                    max_iter=fire_budget,
                    target_overlap=overlap_limit,
                    collision_diameter_scale=1.0,
                    apply_orientation=True,
                    pair_rebuild_interval=2,
                    max_restarts=max_restarts,
                    restart_patience=min(
                        max(25, int(restart_patience)),
                        max(25, fire_budget // 2),
                    ),
                    verbose=False,
                )
                total_iters += int(last_info.get("iters", 0.0))
                total_restarts += int(last_info.get("restarts", 0.0))
                total_force_evaluations += int(
                    last_info.get("force_evaluations", 0.0)
                )
                total_pair_build_s += float(
                    last_info.get("pair_build_s", 0.0)
                )
                total_pair_forces_s += float(
                    last_info.get("pair_forces_s", 0.0)
                )
                total_apply_updates_s += float(
                    last_info.get("apply_updates_s", 0.0)
                )

            if verbose:
                print(
                    "[SAM-FIRE] "
                    f"lote={batch_attempt} | +{n_add} | "
                    f"fibras={self._n_fibers}/{target_count} | "
                    f"overlap={float(last_info.get('final_overlap', math.inf)):.6f} | "
                    f"A2={float(last_info.get('final_A2_error_rel', math.inf)):.6f} | "
                    f"iters={int(last_info.get('iters', 0.0))} | "
                    f"ok={bool(last_info.get('converged', 0.0))}",
                    flush=True,
                )

            if bool(last_info.get("converged", 0.0)):
                successful_batches += 1
                consecutive_single_failures = 0
                current_batch_count = min(
                    nominal_batch_count,
                    max(n_add + 1, int(math.ceil(1.25 * n_add))),
                )
                if stop_callback is not None and stop_callback(self):
                    stopped_by_callback = True
                    break
                continue

            failed_batches += 1
            self._n_fibers = accepted_n
            self._centers[:accepted_n] = accepted_centers
            self._directions[:accepted_n] = accepted_directions
            if n_add > minimum_batch_count:
                current_batch_count = max(
                    minimum_batch_count,
                    n_add // 2,
                )
                continue

            consecutive_single_failures += 1
            if consecutive_single_failures >= 4:
                stopped_by_stall = True
                break

        if self._n_fibers < target_count and batch_attempt >= max_batch_attempts:
            stopped_by_stall = True

        final_overlap = float(self.measure_max_overlap())
        final_a2_error = float(self.orientation_error())
        orientation_opt_info = {
            "iterations": 0.0,
            "accepted_steps": 0.0,
            "line_search_evaluations": 0.0,
            "contact_projections": 0.0,
            "initial_error": final_a2_error,
            "final_error": final_a2_error,
            "wall_s": 0.0,
        }
        if (
            self._n_fibers >= 2
            and final_overlap <= overlap_limit
            and final_a2_error > max(1e-4, 0.25 * self.tol_A)
        ):
            orientation_opt_info = self.optimize_A2_riemannian(
                max_iter=64,
                tolerance=max(1e-4, 0.25 * self.tol_A),
                overlap_limit=overlap_limit,
            )
            final_overlap = float(self.measure_max_overlap())
            final_a2_error = float(self.orientation_error())

        discrete_target_vf = target_count * self.V_fiber / self.V_cell
        self.last_run_stats = {
            "batches": float(successful_batches),
            "attempts": float(attempted_fibers),
            "relax_iters": float(total_iters),
            "final_overlap": final_overlap,
            "final_A_err": final_a2_error,
            "final_A2_error_rel": final_a2_error,
            "final_vf": float(self.current_vf()),
            "target_vf": float(self.vf_target),
            "target_n_fibers": float(target_count),
            "discrete_target_vf": float(discrete_target_vf),
            "vf_quantization_error": float(discrete_target_vf - self.vf_target),
            "target_reached": float(self._n_fibers >= target_count),
            "stopped_by_callback": float(stopped_by_callback),
            "stopped_by_stall": float(stopped_by_stall),
            "stalled_batches": float(failed_batches),
            "n_fibers": float(self._n_fibers),
            "numba_available": float(NUMBA_AVAILABLE),
            "numba_threads": float(
                numba_get_num_threads() if NUMBA_AVAILABLE else 1
            ),
            "add_batch_s": float(
                total_sample_centers_s + total_sample_dirs_s
            ),
            "sample_centers_s": float(total_sample_centers_s),
            "sample_dirs_s": float(total_sample_dirs_s),
            "candidate_checks_s": 0.0,
            "relax_s": float(time.perf_counter() - t0),
            "relax_local_s": 0.0,
            "pair_build_s": float(total_pair_build_s),
            "pair_forces_s": float(total_pair_forces_s),
            "orientation_corr_s": 0.0,
            "compute_a4_s": 0.0,
            "apply_updates_s": float(total_apply_updates_s),
            "cleanup_passes": 0.0,
            "soft_inserted_batches": float(successful_batches),
            "collision_diameter_scale": 1.0,
            "use_local_relax": 0.0,
            "orientation_optimizer_iterations": float(
                orientation_opt_info["iterations"]
            ),
            "orientation_optimizer_accepted_steps": float(
                orientation_opt_info["accepted_steps"]
            ),
            "orientation_optimizer_line_search_evaluations": float(
                orientation_opt_info["line_search_evaluations"]
            ),
            "orientation_optimizer_contact_projections": float(
                orientation_opt_info["contact_projections"]
            ),
            "orientation_optimizer_s": float(
                orientation_opt_info["wall_s"]
            ),
            "cupy_candidate_calls": 0.0,
            "cupy_candidate_pairs": 0.0,
            "cupy_candidate_s": 0.0,
            "cupy_candidate_fallbacks": 0.0,
            "cupy_force_calls": float(
                self._cupy_force_calls - cupy_force_calls_before
            ),
            "cupy_force_pairs": float(
                self._cupy_force_pairs - cupy_force_pairs_before
            ),
            "cupy_force_s": float(
                self._cupy_force_s - cupy_force_s_before
            ),
            "cupy_force_fallbacks": float(
                self._cupy_force_fallbacks - cupy_force_fallbacks_before
            ),
            "collective_fire_iters": float(total_iters),
            "collective_fire_restarts": float(total_restarts),
            "collective_fire_force_evaluations": float(
                total_force_evaluations
            ),
            "collective_fire_pair_build_s": float(total_pair_build_s),
            "collective_fire_pair_forces_s": float(total_pair_forces_s),
            "collective_fire_apply_updates_s": float(total_apply_updates_s),
            "collective_fire_s": float(time.perf_counter() - t0),
            "collective_fire_successful_batches": float(successful_batches),
            "collective_fire_failed_batches": float(failed_batches),
            "collective_fire_smallest_batch": float(
                smallest_attempted_batch
            ),
        }

        if verbose:
            print(
                "[SAM-FIRE] "
                f"fibras={self._n_fibers}/{target_count} | "
                f"lotes={successful_batches} | fallidos={failed_batches} | "
                f"overlap={final_overlap:.6f} | A2={final_a2_error:.6f}",
                flush=True,
            )
        return self.export_arrays()

    def export_arrays(self) -> Tuple[Array, Array, Array]:
        centers = self.centers.copy()
        directions = self.directions.copy()
        lengths = np.full(self._n_fibers, self.fiber_length, dtype=float)
        return centers, directions, lengths

    def current_vf(self) -> float:
        return self._n_fibers * self.V_fiber / self.V_cell

    def target_fiber_count(self, vf_target: Optional[float] = None) -> int:
        """Return the integer fiber count with minimum continuous Vf error."""
        target = self.vf_target if vf_target is None else float(vf_target)
        if target <= 0.0:
            return 0
        continuous_count = target * self.V_cell / max(self.V_fiber, 1e-16)
        return max(1, int(math.floor(continuous_count + 0.5)))

    def compute_A2_vec(self) -> Array:
        if self._n_fibers == 0:
            return np.zeros((3, 3), dtype=float)
        return np.einsum(
            "ni,nj->ij",
            self.directions,
            self.directions,
        ) / self._n_fibers

    def orientation_error(self) -> float:
        return self.relative_tensor_error(self.compute_A2_vec(), self.A2)

    def _orientation_addition_scores(self, directions: Array) -> Array:
        """Score candidates by the A2 residual after appending one direction."""
        if len(directions) == 0:
            return np.empty(0, dtype=float)
        current_sum = np.einsum(
            "ni,nj->ij",
            self.directions,
            self.directions,
        )
        candidate_outer = np.einsum("ni,nj->nij", directions, directions)
        target_sum = (self._n_fibers + 1.0) * self.A2
        residual = current_sum[None, :, :] + candidate_outer - target_sum
        return np.einsum("nij,nij->n", residual, residual)

    def _remaining_orientation_target(self, count: int) -> Array:
        """Return the realizable A2 that best compensates the current prefix."""
        count = max(1, int(count))
        current_sum = np.einsum(
            "ni,nj->ij",
            self.directions,
            self.directions,
        )
        desired = (
            (self._n_fibers + count) * self.A2 - current_sum
        ) / count
        desired = 0.5 * (desired + desired.T)
        vals, vecs = np.linalg.eigh(desired)
        vals = np.clip(vals, 1e-6, None)
        vals /= np.sum(vals)
        return vecs @ np.diag(vals) @ vecs.T

    def optimize_A2_riemannian(
        self,
        *,
        max_iter: int = 64,
        tolerance: Optional[float] = None,
        overlap_limit: Optional[float] = None,
        max_line_search: int = 12,
    ) -> Dict[str, float]:
        """Refine A2 with spectral Riemannian descent on (S2)^N.

        The Armijo search makes the A2 objective monotone. Candidate rotations
        are accepted only while the requested overlap constraint remains
        satisfied.
        """
        t0 = time.perf_counter()
        n = self._n_fibers
        target_error = self.tol_A if tolerance is None else max(
            0.0,
            float(tolerance),
        )
        allowed_overlap = self.tol_overlap if overlap_limit is None else float(
            overlap_limit
        )
        initial_error = self.orientation_error()
        initial_overlap = self.measure_max_overlap()
        result = {
            "iterations": 0.0,
            "accepted_steps": 0.0,
            "line_search_evaluations": 0.0,
            "contact_projections": 0.0,
            "initial_error": float(initial_error),
            "final_error": float(initial_error),
            "initial_overlap": float(initial_overlap),
            "final_overlap": float(initial_overlap),
            "converged": float(initial_error <= target_error),
            "wall_s": 0.0,
        }
        if (
            n < 2
            or initial_error <= target_error
            or initial_overlap > allowed_overlap + 1e-12
        ):
            result["wall_s"] = float(time.perf_counter() - t0)
            return result

        target_norm = np.linalg.norm(self.A2.ravel()) + 1e-16

        def objective_gradient(directions: Array) -> Tuple[float, Array, Array]:
            moment = np.einsum("ni,nj->ij", directions, directions) / n
            residual = moment - self.A2
            objective = 0.5 * float(np.sum(residual * residual))
            raw = np.einsum("ij,nj->ni", residual, directions)
            tangent = raw - directions * np.sum(
                directions * raw,
                axis=1,
                keepdims=True,
            )
            gradient = (2.0 / n) * tangent
            return objective, gradient, moment

        directions = self.directions.copy()
        objective, gradient, moment = objective_gradient(directions)
        error = math.sqrt(max(0.0, 2.0 * objective)) / target_norm
        spectral_step = max(1.0, 0.5 * n)
        accepted_steps = 0
        evaluations = 0
        current_overlap = initial_overlap
        iterations = 0
        contact_projections = 0

        for iteration in range(1, max(1, int(max_iter)) + 1):
            iterations = iteration
            if error <= target_error:
                break

            grad_norm_sq = float(np.sum(gradient * gradient))
            saddle_escape = grad_norm_sq <= 1e-24
            if saddle_escape:
                delta = self.A2 - moment
                values, vectors = np.linalg.eigh(delta)
                target_axis = vectors[:, int(np.argmax(values))]
                source_axis = vectors[:, int(np.argmin(values))]
                scores = (
                    np.abs(directions @ source_axis)
                    - np.abs(directions @ target_axis)
                )
                index = int(np.argmax(scores))
                tangent = target_axis - directions[index] * float(
                    np.dot(directions[index], target_axis)
                )
                tangent_norm = float(np.linalg.norm(tangent))
                if tangent_norm <= 1e-14:
                    break
                search = np.zeros_like(directions)
                search[index] = tangent / tangent_norm
                directional_derivative = 0.0
                trial_step = 0.05
            else:
                search = -gradient
                directional_derivative = -grad_norm_sq
                trial_step = spectral_step

            old_directions = directions.copy()
            old_gradient = gradient
            old_objective = objective
            accepted = False

            for _ in range(max(1, int(max_line_search))):
                evaluations += 1
                candidate = old_directions + trial_step * search
                candidate /= np.maximum(
                    np.linalg.norm(candidate, axis=1)[:, None],
                    1e-16,
                )
                candidate_objective, candidate_gradient, candidate_moment = (
                    objective_gradient(candidate)
                )
                armijo_rhs = (
                    old_objective + 1e-4 * trial_step * directional_derivative
                )
                objective_ok = (
                    candidate_objective < old_objective
                    if saddle_escape
                    else candidate_objective <= armijo_rhs
                )
                if objective_ok:
                    self._directions[:n] = candidate
                    candidate_overlap = self.measure_max_overlap()
                    if candidate_overlap <= allowed_overlap + 1e-12:
                        accepted = True
                        current_overlap = candidate_overlap
                        directions = candidate
                        objective = candidate_objective
                        gradient = candidate_gradient
                        moment = candidate_moment
                        break
                    self._directions[:n] = old_directions
                trial_step *= 0.5

            if not accepted:
                self._directions[:n] = old_directions
                can_project_contacts = (
                    contact_projections < 2
                    and allowed_overlap > 0.0
                    and current_overlap > 0.25 * allowed_overlap
                )
                if can_project_contacts:
                    self.relax(
                        verbose=False,
                        max_iter=48,
                        target_overlap=0.5 * allowed_overlap,
                        apply_orientation=False,
                        pair_rebuild_interval=2,
                        tau_rotate_override=1e-6,
                    )
                    self._directions[:n] = old_directions
                    projected_overlap = self.measure_max_overlap()
                    if projected_overlap < current_overlap - 1e-10:
                        contact_projections += 1
                        current_overlap = projected_overlap
                        directions = old_directions
                        objective, gradient, moment = objective_gradient(
                            directions
                        )
                        error = (
                            math.sqrt(max(0.0, 2.0 * objective))
                            / target_norm
                        )
                        spectral_step = max(1.0, 0.25 * n)
                        continue
                break

            accepted_steps += 1
            error = math.sqrt(max(0.0, 2.0 * objective)) / target_norm
            displacement = directions - old_directions
            transported_old_gradient = old_gradient - directions * np.sum(
                directions * old_gradient,
                axis=1,
                keepdims=True,
            )
            gradient_change = gradient - transported_old_gradient
            curvature = float(np.sum(displacement * gradient_change))
            displacement_sq = float(np.sum(displacement * displacement))
            if curvature > 1e-20:
                spectral_step = float(
                    np.clip(
                        displacement_sq / curvature,
                        1e-3,
                        50.0 * n,
                    )
                )
            else:
                spectral_step = min(2.0 * trial_step, 50.0 * n)

        final_error = self.orientation_error()
        final_overlap = current_overlap
        result.update(
            {
                "iterations": float(iterations),
                "accepted_steps": float(accepted_steps),
                "line_search_evaluations": float(evaluations),
                "contact_projections": float(contact_projections),
                "final_error": float(final_error),
                "final_overlap": float(final_overlap),
                "converged": float(final_error <= target_error),
                "wall_s": float(time.perf_counter() - t0),
            }
        )
        return result

    def set_collision_diameter_scale(self, scale: float) -> None:
        """Change the collision diameter without changing physical fiber volume."""
        scale = float(scale)
        if not (0.0 < scale <= 1.0):
            raise ValueError("collision diameter scale must be in (0, 1].")
        self.collision_diameter_scale = scale
        self.d_eff = self._full_d_eff * scale
        self._cull_radius = self.fiber_length + self.d_eff + self._cull_padding
        self._free_volume_cache.clear()

    def retain_first_fibers(self, count: int) -> None:
        """Discard fibers after ``count`` while preserving deterministic order."""
        count = int(count)
        if count < 0 or count > self._n_fibers:
            raise ValueError("fiber count is outside the current geometry.")
        self._n_fibers = count
        self._last_added_indices = self._last_added_indices[
            self._last_added_indices < count
        ]

    def retain_best_oriented_fibers(self, count: int) -> int:
        """Greedily retain the subset whose second moment best matches A2."""
        count = int(count)
        if count < 0 or count > self._n_fibers:
            raise ValueError("fiber count is outside the current geometry.")
        removed = 0
        while self._n_fibers > count:
            n = self._n_fibers
            if n == 1:
                self._n_fibers = 0
                removed += 1
                break
            directions = self.directions
            moment_sum = np.einsum("ni,nj->ij", directions, directions)
            outer = np.einsum("ni,nj->nij", directions, directions)
            candidate_A2 = (moment_sum[None, :, :] - outer) / (n - 1.0)
            residual = candidate_A2 - self.A2[None, :, :]
            scores = np.einsum("nij,nij->n", residual, residual)
            self.remove_fibers(np.array([int(np.argmin(scores))]))
            removed += 1
        self._last_added_indices = np.empty(0, dtype=np.int32)
        return removed

    def remove_fibers(self, indices: Array) -> int:
        """Remove selected fiber indices while preserving the remaining order."""
        if self._n_fibers == 0:
            return 0
        indices = np.unique(np.asarray(indices, dtype=np.int64))
        indices = indices[(indices >= 0) & (indices < self._n_fibers)]
        if len(indices) == 0:
            return 0
        keep = np.ones(self._n_fibers, dtype=bool)
        keep[indices] = False
        new_n = int(np.count_nonzero(keep))
        self._centers[:new_n] = self.centers[keep]
        self._directions[:new_n] = self.directions[keep]
        self._n_fibers = new_n
        self._last_added_indices = np.empty(0, dtype=np.int32)
        return int(len(indices))

    def drop_worst_overlapping_fibers(
        self,
        *,
        target_overlap: float,
        max_remove: int,
    ) -> int:
        """Drop a bounded half-set of the fibers with the worst remaining contacts."""
        n = self._n_fibers
        max_remove = max(0, int(max_remove))
        if n < 2 or max_remove == 0:
            return 0
        dummy_centers = np.zeros((n, 3), dtype=float)
        dummy_rotations = np.zeros((n, 3), dtype=float)
        overlap_per_fiber = np.zeros(n, dtype=float)
        pairs = self.build_neighbor_pairs_vec()
        neighbor_offsets, neighbor_indices = _pairs_to_csr_numba(pairs, n)
        _apply_forces_per_fiber_numba(
            self.centers,
            self.directions,
            neighbor_offsets,
            neighbor_indices,
            dummy_centers,
            dummy_rotations,
            overlap_per_fiber,
            self.cell_size[0],
            self.cell_size[1],
            self.cell_size[2],
            self.fiber_length,
            self.d_eff,
            self._cull_radius ** 2,
            1e-6,
            float(target_overlap),
            0.0,
            1.0 / max(self._rot_scale, 1e-12),
        )
        unresolved = np.flatnonzero(overlap_per_fiber > float(target_overlap))
        if len(unresolved) == 0:
            return 0
        remove_count = min(max_remove, max(1, int(math.ceil(len(unresolved) / 2.0))))
        order = unresolved[np.argsort(overlap_per_fiber[unresolved])[::-1]]
        return self.remove_fibers(order[:remove_count])

    def measure_max_overlap(self) -> float:
        """Measure overlap for the current collision diameter without moving fibers."""
        if self._n_fibers < 2:
            return 0.0
        cupy_result = self._cupy_all_pair_contact_forces(
            effective_tau=1e-6,
            target_overlap=self.tol_overlap,
            effective_tau_rotate=0.0,
        )
        if cupy_result is not None:
            return float(np.max(cupy_result[2]))

        pairs = self.build_neighbor_pairs_vec()
        if len(pairs) == 0:
            return 0.0
        center_updates = np.zeros((self._n_fibers, 3), dtype=float)
        rot_updates = np.zeros((self._n_fibers, 3), dtype=float)
        return float(
            self._apply_pair_forces_vec(
                pairs,
                center_updates,
                rot_updates,
                effective_tau=0.0,
            )
        )

    def measure_max_overlap_prefix(self, count: int) -> float:
        """Measure the retained prefix used by the voxelized geometry."""
        count = max(0, min(int(count), self._n_fibers))
        previous = self._n_fibers
        try:
            self._n_fibers = count
            return self.measure_max_overlap()
        finally:
            self._n_fibers = previous

    def shake_overlapping_fibers(
        self,
        *,
        target_overlap: float,
        translate_fraction: float = 0.04,
        rotate_radians: float = 0.01,
    ) -> int:
        """Apply a deterministic random kick only to currently overlapping fibers."""
        n = self._n_fibers
        if n < 2:
            return 0
        dummy_centers = np.zeros((n, 3), dtype=float)
        dummy_rotations = np.zeros((n, 3), dtype=float)
        overlap_per_fiber = np.zeros(n, dtype=float)
        pairs = self.build_neighbor_pairs_vec()
        neighbor_offsets, neighbor_indices = _pairs_to_csr_numba(pairs, n)
        _apply_forces_per_fiber_numba(
            self.centers,
            self.directions,
            neighbor_offsets,
            neighbor_indices,
            dummy_centers,
            dummy_rotations,
            overlap_per_fiber,
            self.cell_size[0],
            self.cell_size[1],
            self.cell_size[2],
            self.fiber_length,
            self.d_eff,
            self._cull_radius ** 2,
            1e-6,
            float(target_overlap),
            0.0,
            1.0 / max(self._rot_scale, 1e-12),
        )
        indices = np.flatnonzero(overlap_per_fiber > float(target_overlap))
        if len(indices) == 0:
            return 0

        center_updates = np.zeros((n, 3), dtype=float)
        rot_updates = np.zeros((n, 3), dtype=float)
        translation = self.rng.standard_normal((len(indices), 3))
        translation_norm = np.linalg.norm(translation, axis=1)
        translation /= np.maximum(translation_norm[:, None], 1e-12)
        center_updates[indices] = (
            float(translate_fraction) * self.d_eff * translation
        )

        tangent = self.rng.standard_normal((len(indices), 3))
        local_dirs = self.directions[indices]
        tangent -= local_dirs * np.sum(tangent * local_dirs, axis=1)[:, None]
        tangent_norm = np.linalg.norm(tangent, axis=1)
        tangent /= np.maximum(tangent_norm[:, None], 1e-12)
        rot_updates[indices] = float(rotate_radians) * tangent
        _apply_updates_numba(
            self.centers,
            self.directions,
            center_updates,
            rot_updates,
            self.cell_size[0],
            self.cell_size[1],
            self.cell_size[2],
        )
        return int(len(indices))

    def reinsert_overlapping_fibers(
        self,
        *,
        target_overlap: float,
        max_attempts_per_fiber: int = 750,
        hard_diameter_factor: float = 0.50,
        max_fibers: Optional[int] = None,
    ) -> Dict[str, int]:
        """Relocate unresolved fibers one at a time at the current diameter."""
        n = self._n_fibers
        if n < 2:
            return {"candidates": 0, "reinserted": 0, "failed": 0}

        dummy_centers = np.zeros((n, 3), dtype=float)
        dummy_rotations = np.zeros((n, 3), dtype=float)
        overlap_per_fiber = np.zeros(n, dtype=float)
        pairs = self.build_neighbor_pairs_vec()
        neighbor_offsets, neighbor_indices = _pairs_to_csr_numba(pairs, n)
        _apply_forces_per_fiber_numba(
            self.centers,
            self.directions,
            neighbor_offsets,
            neighbor_indices,
            dummy_centers,
            dummy_rotations,
            overlap_per_fiber,
            self.cell_size[0],
            self.cell_size[1],
            self.cell_size[2],
            self.fiber_length,
            self.d_eff,
            self._cull_radius ** 2,
            1e-6,
            float(target_overlap),
            0.0,
            1.0 / max(self._rot_scale, 1e-12),
        )
        unresolved = np.flatnonzero(overlap_per_fiber > float(target_overlap))
        if len(unresolved) == 0:
            return {"candidates": 0, "reinserted": 0, "failed": 0}
        unresolved = unresolved[
            np.argsort(overlap_per_fiber[unresolved])[::-1]
        ]
        if max_fibers is not None:
            unresolved = unresolved[:max(0, int(max_fibers))]

        candidates_checked = 0
        reinserted = 0
        failed = 0
        hard_d = max(
            0.0,
            min(
                self.d_eff - float(target_overlap),
                float(hard_diameter_factor) * self.d_eff,
            ),
        )
        progress = self.current_vf() / max(self.vf_target, 1e-12)
        chunk = 64

        for idx_value in unresolved:
            idx = int(idx_value)
            placed = False
            attempts = 0
            while attempts < max_attempts_per_fiber and not placed:
                sample_count = min(chunk, max_attempts_per_fiber - attempts)
                centers_cand = self._sample_candidate_centers(
                    sample_count,
                    progress,
                    hard_d,
                )
                dirs_cand = self._sample_directions_batch(sample_count)
                for k in range(sample_count):
                    attempts += 1
                    candidates_checked += 1
                    cc = centers_cand[k]
                    dd = dirs_cand[k]
                    overlaps = _check_candidate_serial_numba(
                        cc[0],
                        cc[1],
                        cc[2],
                        dd[0],
                        dd[1],
                        dd[2],
                        self.centers,
                        self.directions,
                        n,
                        self.cell_size[0],
                        self.cell_size[1],
                        self.cell_size[2],
                        self.fiber_length,
                        hard_d,
                        self._cull_radius ** 2,
                        idx,
                    )
                    if not overlaps:
                        self._centers[idx] = cc
                        self._directions[idx] = dd
                        reinserted += 1
                        placed = True
                        break
            if not placed:
                failed += 1

        return {
            "candidates": int(candidates_checked),
            "reinserted": int(reinserted),
            "failed": int(failed),
        }

    def relax_collective_fire(
        self,
        *,
        max_iter: int = 2000,
        target_overlap: Optional[float] = None,
        collision_diameter_scale: Optional[float] = None,
        apply_orientation: bool = True,
        pair_rebuild_interval: int = 1,
        static_neighbor_min_density: float = 0.45,
        static_neighbor_max_mib: float = 64.0,
        dt_initial: float = 0.10,
        dt_max: float = 0.80,
        alpha_initial: float = 0.10,
        max_center_step_factor: float = 0.12,
        max_rotation_step: float = 0.035,
        max_restarts: int = 3,
        restart_patience: int = 250,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Collectively remove contacts with a periodic rigid-fiber FIRE solver."""
        t0 = time.perf_counter()
        n = self._n_fibers
        overlap_limit = (
            self.tol_overlap
            if target_overlap is None
            else max(0.0, float(target_overlap))
        )
        if n < 2:
            return {
                "iters": 0.0,
                "converged": 1.0,
                "final_overlap": 0.0,
                "final_A2_error_rel": float(self.orientation_error()),
                "restarts": 0.0,
                "pair_build_s": 0.0,
                "pair_forces_s": 0.0,
                "apply_updates_s": 0.0,
                "wall_s": float(time.perf_counter() - t0),
            }

        self.set_collision_diameter_scale(
            1.0
            if collision_diameter_scale is None
            else float(collision_diameter_scale)
        )
        rebuild_interval = max(1, int(pair_rebuild_interval))
        max_iterations = max(1, int(max_iter))
        dt = max(1e-5, float(dt_initial))
        dt_ceiling = max(dt, float(dt_max))
        alpha_start = min(0.99, max(1e-5, float(alpha_initial)))
        alpha = alpha_start
        n_positive = 0
        restarts = 0
        plateau_iteration = 0

        center_velocity = np.zeros((n, 3), dtype=float)
        rotation_velocity = np.zeros((n, 3), dtype=float)
        neighbors_initialized = False
        neighbor_offsets = np.zeros(n + 1, dtype=np.int32)
        neighbor_indices = np.empty(0, dtype=np.int32)
        neighbor_strategy = "center_cull"
        neighbor_density = 0.0
        neighbor_builds = 0
        all_pair_count = n * (n - 1) // 2
        complete_csr_mib = (
            n * (n - 1) * np.dtype(np.int32).itemsize / 1024.0**2
        )
        allow_static_neighbors = (
            all_pair_count > 0
            and complete_csr_mib <= max(0.0, float(static_neighbor_max_mib))
        )
        pair_build_s = 0.0
        pair_forces_s = 0.0
        apply_updates_s = 0.0
        force_evaluations = 0

        rotation_gain = max(
            self.tau_rotate,
            2.0
            * self._rot_scale
            / max(self.fiber_length * self.fiber_length, 1e-12),
        )
        orientation_activation_overlap = max(
            5.0 * overlap_limit,
            0.20 * self._full_d_eff,
        )

        def evaluate_forces(
            iteration: int,
        ) -> Tuple[Array, Array, float, float]:
            nonlocal neighbors_initialized, neighbor_offsets, neighbor_indices
            nonlocal neighbor_strategy, neighbor_density, neighbor_builds
            nonlocal pair_build_s, pair_forces_s, force_evaluations

            if (
                not neighbors_initialized
                or (
                    neighbor_strategy == "center_cull"
                    and iteration % rebuild_interval == 0
                )
            ):
                pair_t0 = time.perf_counter()
                pairs = self.build_neighbor_pairs_vec()
                neighbor_density = len(pairs) / max(all_pair_count, 1)
                if (
                    not neighbors_initialized
                    and allow_static_neighbors
                    and neighbor_density >= float(static_neighbor_min_density)
                ):
                    neighbor_offsets, neighbor_indices = (
                        _complete_neighbor_csr_numba(n)
                    )
                    neighbor_strategy = "static_all_pairs"
                else:
                    neighbor_offsets, neighbor_indices = _pairs_to_csr_numba(
                        pairs,
                        n,
                    )
                neighbors_initialized = True
                neighbor_builds += 1
                pair_build_s += time.perf_counter() - pair_t0

            center_force = np.zeros((n, 3), dtype=float)
            rotation_force = np.zeros((n, 3), dtype=float)
            overlap_per_fiber = np.zeros(n, dtype=float)
            force_t0 = time.perf_counter()
            cupy_result = self._cupy_contact_forces(
                neighbor_offsets,
                neighbor_indices,
                effective_tau=0.30,
                target_overlap=overlap_limit,
                effective_tau_rotate=rotation_gain,
            )
            if cupy_result is not None:
                (
                    center_force[:],
                    rotation_force[:],
                    overlap_per_fiber[:],
                ) = cupy_result
            else:
                _apply_forces_per_fiber_numba(
                    self.centers,
                    self.directions,
                    neighbor_offsets,
                    neighbor_indices,
                    center_force,
                    rotation_force,
                    overlap_per_fiber,
                    self.cell_size[0],
                    self.cell_size[1],
                    self.cell_size[2],
                    self.fiber_length,
                    self.d_eff,
                    self._cull_radius ** 2,
                    0.30,
                    overlap_limit,
                    rotation_gain,
                    1.0 / max(self._rot_scale, 1e-12),
                )
            pair_forces_s += time.perf_counter() - force_t0
            force_evaluations += 1

            max_overlap = float(np.max(overlap_per_fiber))
            a2_error = float(self.orientation_error())
            if apply_orientation:
                orientation_scale = max(
                    0.0,
                    1.0
                    - max_overlap
                    / max(orientation_activation_overlap, 1e-12),
                )
                if orientation_scale > 0.0:
                    self._apply_orientation_correction_vec(
                        self.A2 - self.compute_A2_vec(),
                        rotation_force,
                        scale=orientation_scale,
                    )
            return center_force, rotation_force, max_overlap, a2_error

        def score(overlap: float, a2_error: float) -> float:
            overlap_residual = max(0.0, overlap - overlap_limit)
            orientation_residual = (
                max(0.0, a2_error - self.tol_A)
                if apply_orientation
                else 0.0
            )
            return (
                overlap_residual / max(self._full_d_eff, 1e-12)
                + orientation_residual
            )

        (
            center_force,
            rotation_force,
            max_overlap,
            a2_error,
        ) = evaluate_forces(0)
        best_score = score(max_overlap, a2_error)
        best_overlap = max_overlap
        best_a2_error = a2_error
        best_centers = self.centers.copy()
        best_directions = self.directions.copy()
        plateau_score = float(best_score)
        converged = (
            max_overlap <= overlap_limit
            and (not apply_orientation or a2_error <= self.tol_A)
        )
        iterations = 0

        for iteration in range(1, max_iterations + 1):
            iterations = iteration
            if converged:
                break

            center_velocity += 0.5 * dt * center_force
            rotation_velocity += 0.5 * dt * rotation_force

            center_step = dt * center_velocity
            center_norm = np.linalg.norm(center_step, axis=1)
            max_center_step = (
                max(1e-6, float(max_center_step_factor)) * self.d_eff
            )
            center_mask = center_norm > max_center_step
            if np.any(center_mask):
                center_step[center_mask] *= (
                    max_center_step / center_norm[center_mask]
                )[:, None]
                center_velocity[center_mask] = (
                    center_step[center_mask] / dt
                )

            rotation_step = dt * rotation_velocity
            rotation_norm = np.linalg.norm(rotation_step, axis=1)
            rotation_mask = rotation_norm > float(max_rotation_step)
            if np.any(rotation_mask):
                rotation_step[rotation_mask] *= (
                    float(max_rotation_step) / rotation_norm[rotation_mask]
                )[:, None]
                rotation_velocity[rotation_mask] = (
                    rotation_step[rotation_mask] / dt
                )

            update_t0 = time.perf_counter()
            _apply_updates_numba(
                self.centers,
                self.directions,
                center_step,
                rotation_step,
                self.cell_size[0],
                self.cell_size[1],
                self.cell_size[2],
            )
            apply_updates_s += time.perf_counter() - update_t0

            (
                new_center_force,
                new_rotation_force,
                max_overlap,
                a2_error,
            ) = evaluate_forces(iteration)
            center_velocity += 0.5 * dt * new_center_force
            rotation_velocity += 0.5 * dt * new_rotation_force
            rotation_velocity -= self.directions * np.sum(
                self.directions * rotation_velocity,
                axis=1,
                keepdims=True,
            )

            power = float(
                np.sum(center_velocity * new_center_force)
                + np.sum(rotation_velocity * new_rotation_force)
            )
            if power > 0.0:
                center_speed = float(np.linalg.norm(center_velocity))
                center_force_norm = float(np.linalg.norm(new_center_force))
                if center_speed > 0.0 and center_force_norm > 0.0:
                    center_velocity = (
                        (1.0 - alpha) * center_velocity
                        + alpha
                        * new_center_force
                        * (center_speed / center_force_norm)
                    )
                rotation_speed = float(np.linalg.norm(rotation_velocity))
                rotation_force_norm = float(np.linalg.norm(new_rotation_force))
                if rotation_speed > 0.0 and rotation_force_norm > 0.0:
                    rotation_velocity = (
                        (1.0 - alpha) * rotation_velocity
                        + alpha
                        * new_rotation_force
                        * (rotation_speed / rotation_force_norm)
                    )
                n_positive += 1
                if n_positive > 5:
                    dt = min(dt * 1.1, dt_ceiling)
                    alpha *= 0.99
            else:
                dt = max(1e-5, 0.5 * dt)
                alpha = alpha_start
                n_positive = 0
                center_velocity.fill(0.0)
                rotation_velocity.fill(0.0)

            center_force = new_center_force
            rotation_force = new_rotation_force
            current_score = score(max_overlap, a2_error)
            if current_score < best_score - 1e-10:
                best_score = current_score
                best_overlap = max_overlap
                best_a2_error = a2_error
                best_centers = self.centers.copy()
                best_directions = self.directions.copy()

            converged = (
                max_overlap <= overlap_limit
                and (not apply_orientation or a2_error <= self.tol_A)
            )
            if converged:
                best_overlap = max_overlap
                best_a2_error = a2_error
                best_centers = self.centers.copy()
                best_directions = self.directions.copy()
                break

            patience = max(1, int(restart_patience))
            if iteration - plateau_iteration >= patience:
                score_reduction = plateau_score - best_score
                required_reduction = max(
                    1e-5,
                    0.03 * max(abs(plateau_score), overlap_limit),
                )
                if (
                    score_reduction < required_reduction
                    and restarts < max(0, int(max_restarts))
                ):
                    self._centers[:n] = best_centers
                    self._directions[:n] = best_directions
                    self.shake_overlapping_fibers(
                        target_overlap=overlap_limit,
                        translate_fraction=0.08 / (1.0 + restarts),
                        rotate_radians=0.015 / (1.0 + restarts),
                    )
                    center_velocity.fill(0.0)
                    rotation_velocity.fill(0.0)
                    dt = max(
                        1e-5,
                        float(dt_initial) / (1.0 + restarts),
                    )
                    alpha = alpha_start
                    n_positive = 0
                    restarts += 1
                    if neighbor_strategy == "center_cull":
                        neighbors_initialized = False
                    (
                        center_force,
                        rotation_force,
                        max_overlap,
                        a2_error,
                    ) = evaluate_forces(iteration)
                plateau_iteration = iteration
                plateau_score = float(best_score)

            if verbose and (iteration == 1 or iteration % 100 == 0):
                print(
                    "[FIRE] "
                    f"iter={iteration} | overlap={max_overlap:.6f} | "
                    f"A2={a2_error:.6f} | dt={dt:.4f} | "
                    f"restarts={restarts}",
                    flush=True,
                )

        if not converged:
            self._centers[:n] = best_centers
            self._directions[:n] = best_directions
            max_overlap = float(best_overlap)
            a2_error = float(best_a2_error)

        # The cached sparse graph accelerates migration; acceptance still uses
        # an exact all-pair contact check at the returned configuration.
        max_overlap = float(self.measure_max_overlap())
        a2_error = float(self.orientation_error())
        converged = bool(
            max_overlap <= overlap_limit
            and (not apply_orientation or a2_error <= self.tol_A)
        )

        return {
            "iters": float(iterations),
            "converged": float(converged),
            "final_overlap": float(max_overlap),
            "final_A2_error_rel": float(a2_error),
            "best_score": float(best_score),
            "restarts": float(restarts),
            "force_evaluations": float(force_evaluations),
            "neighbor_builds": float(neighbor_builds),
            "neighbor_density": float(neighbor_density),
            "static_neighbor_graph": float(
                neighbor_strategy == "static_all_pairs"
            ),
            "pair_build_s": float(pair_build_s),
            "pair_forces_s": float(pair_forces_s),
            "apply_updates_s": float(apply_updates_s),
            "final_dt": float(dt),
            "wall_s": float(time.perf_counter() - t0),
        }

    def compact_collision_diameter_fire_rescue(
        self,
        *,
        max_iter: int = 2500,
        target_overlap: Optional[float] = None,
        max_restarts: int = 3,
        restart_patience: int = 250,
        pair_rebuild_interval: int = 4,
        max_fiber_removals: int = 0,
        removal_batch: int = 8,
        max_rescue_passes: int = 3,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Restore the physical diameter, then remove only blocking contacts."""
        t0 = time.perf_counter()
        overlap_limit = (
            self.tol_overlap
            if target_overlap is None
            else max(0.0, float(target_overlap))
        )
        initial_scale = float(self.collision_diameter_scale)
        total_iters = 0
        total_restarts = 0
        total_force_evaluations = 0
        total_neighbor_builds = 0
        total_pair_build_s = 0.0
        total_pair_forces_s = 0.0
        total_apply_updates_s = 0.0
        total_relax_s = 0.0
        static_neighbor_calls = 0
        fire_calls = 0
        removed_fibers = 0
        best_score = math.inf
        final_info: Dict[str, float] = {}

        def run_fire(iteration_budget: int) -> Dict[str, float]:
            nonlocal total_iters, total_restarts, total_force_evaluations
            nonlocal total_neighbor_builds, total_pair_build_s
            nonlocal total_pair_forces_s, total_apply_updates_s
            nonlocal total_relax_s, static_neighbor_calls, fire_calls
            nonlocal best_score
            info = self.relax_collective_fire(
                max_iter=max(1, int(iteration_budget)),
                target_overlap=overlap_limit,
                collision_diameter_scale=1.0,
                apply_orientation=True,
                pair_rebuild_interval=max(1, int(pair_rebuild_interval)),
                max_restarts=max_restarts,
                restart_patience=restart_patience,
                verbose=verbose,
            )
            fire_calls += 1
            total_iters += int(info.get("iters", 0.0))
            total_restarts += int(info.get("restarts", 0.0))
            total_force_evaluations += int(
                info.get("force_evaluations", 0.0)
            )
            total_neighbor_builds += int(info.get("neighbor_builds", 0.0))
            total_pair_build_s += float(info.get("pair_build_s", 0.0))
            total_pair_forces_s += float(info.get("pair_forces_s", 0.0))
            total_apply_updates_s += float(info.get("apply_updates_s", 0.0))
            total_relax_s += float(info.get("wall_s", 0.0))
            static_neighbor_calls += int(info.get("static_neighbor_graph", 0.0))
            best_score = min(best_score, float(info.get("best_score", math.inf)))
            return info

        final_info = run_fire(max_iter)
        removal_limit = max(0, int(max_fiber_removals))
        passes = max(0, int(max_rescue_passes))
        batch_limit = max(1, int(removal_batch))

        for _ in range(passes):
            if bool(final_info.get("converged", 0.0)):
                break
            remaining_removals = max(
                0,
                removal_limit - self._compaction_removed_fibers,
            )
            if remaining_removals == 0:
                break
            removed_now = self.drop_worst_overlapping_fibers(
                target_overlap=overlap_limit,
                max_remove=min(batch_limit, remaining_removals),
            )
            if removed_now == 0:
                break
            removed_fibers += removed_now
            self._compaction_removed_fibers += removed_now
            final_info = run_fire(max_iter)

        final_overlap = float(self.measure_max_overlap())
        final_a2_error = float(self.orientation_error())
        converged = bool(
            final_overlap <= overlap_limit
            and final_a2_error <= self.tol_A
        )
        return {
            "iters": float(total_iters),
            "relax_iters": float(total_iters),
            "converged": float(converged),
            "final_overlap": final_overlap,
            "final_A2_error_rel": final_a2_error,
            "best_score": float(best_score),
            "restarts": float(total_restarts),
            "force_evaluations": float(total_force_evaluations),
            "neighbor_builds": float(total_neighbor_builds),
            "static_neighbor_calls": float(static_neighbor_calls),
            "pair_build_s": float(total_pair_build_s),
            "pair_forces_s": float(total_pair_forces_s),
            "apply_updates_s": float(total_apply_updates_s),
            "initial_scale": initial_scale,
            "final_scale": 1.0,
            "stages": float(fire_calls),
            "successful_stages": float(converged),
            "subdivisions": 0.0,
            "stage_failures": float(fire_calls - int(converged)),
            "initial_relax_passes": 0.0,
            "shake_attempts": float(total_restarts),
            "shaken_fibers": 0.0,
            "reinsert_candidates": 0.0,
            "reinserted_fibers": 0.0,
            "reinsert_failures": 0.0,
            "removed_fibers": float(removed_fibers),
            "relax_s": float(total_relax_s),
            "wall_s": float(time.perf_counter() - t0),
            "collective_fire": 1.0,
        }

    def compact_collision_diameter_fire(
        self,
        *,
        n_stages: int = 6,
        max_total_iter: int = 2500,
        max_iter_per_stage: int = 600,
        target_overlap: Optional[float] = None,
        max_restarts: int = 3,
        restart_patience: int = 250,
        min_scale_step: float = 0.01,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Grow the contact diameter with collective FIRE at every stage."""
        t0 = time.perf_counter()
        overlap_limit = (
            self.tol_overlap
            if target_overlap is None
            else max(0.0, float(target_overlap))
        )
        initial_scale = float(self.collision_diameter_scale)
        stage_count = max(1, int(n_stages))
        total_budget = max(1, int(max_total_iter))
        stage_budget = max(1, int(max_iter_per_stage))

        if initial_scale >= 1.0 - 1e-12:
            info = self.relax_collective_fire(
                max_iter=total_budget,
                target_overlap=overlap_limit,
                collision_diameter_scale=1.0,
                apply_orientation=True,
                max_restarts=max_restarts,
                restart_patience=restart_patience,
                verbose=verbose,
            )
            return {
                **info,
                "initial_scale": initial_scale,
                "final_scale": 1.0,
                "stages": 1.0,
                "successful_stages": float(info["converged"]),
                "subdivisions": 0.0,
                "stage_failures": float(not bool(info["converged"])),
            }

        pending = np.linspace(
            initial_scale,
            1.0,
            stage_count + 1,
            dtype=float,
        )[1:].tolist()
        accepted_scale = initial_scale
        accepted_centers = self.centers.copy()
        accepted_directions = self.directions.copy()

        attempted_stages = 0
        successful_stages = 0
        subdivisions = 0
        stage_failures = 0
        total_iters = 0
        total_restarts = 0
        total_force_evaluations = 0
        total_pair_build_s = 0.0
        total_pair_forces_s = 0.0
        total_apply_updates_s = 0.0
        best_score = math.inf
        final_overlap = float(self.measure_max_overlap())
        final_a2_error = float(self.orientation_error())
        max_attempts = max(2 * stage_count, stage_count + 12)

        while (
            pending
            and total_iters < total_budget
            and attempted_stages < max_attempts
        ):
            requested_scale = float(pending.pop(0))
            remaining_budget = total_budget - total_iters
            current_budget = min(stage_budget, remaining_budget)
            attempted_stages += 1

            self._centers[:self._n_fibers] = accepted_centers
            self._directions[:self._n_fibers] = accepted_directions
            self.set_collision_diameter_scale(requested_scale)

            info = self.relax_collective_fire(
                max_iter=current_budget,
                target_overlap=overlap_limit,
                collision_diameter_scale=requested_scale,
                apply_orientation=True,
                max_restarts=max_restarts,
                restart_patience=min(
                    max(25, int(restart_patience)),
                    max(25, current_budget // 2),
                ),
                verbose=verbose,
            )
            total_iters += int(info.get("iters", 0.0))
            total_restarts += int(info.get("restarts", 0.0))
            total_force_evaluations += int(
                info.get("force_evaluations", 0.0)
            )
            total_pair_build_s += float(info.get("pair_build_s", 0.0))
            total_pair_forces_s += float(info.get("pair_forces_s", 0.0))
            total_apply_updates_s += float(
                info.get("apply_updates_s", 0.0)
            )
            best_score = min(best_score, float(info.get("best_score", math.inf)))
            final_overlap = float(info.get("final_overlap", math.inf))
            final_a2_error = float(
                info.get("final_A2_error_rel", math.inf)
            )

            if bool(info.get("converged", 0.0)):
                accepted_scale = requested_scale
                accepted_centers = self.centers.copy()
                accepted_directions = self.directions.copy()
                successful_stages += 1
                continue

            stage_failures += 1
            scale_step = requested_scale - accepted_scale
            can_subdivide = (
                scale_step > max(1e-4, float(min_scale_step))
                and total_iters < total_budget
                and attempted_stages < max_attempts
            )
            if can_subdivide:
                midpoint = accepted_scale + 0.5 * scale_step
                pending.insert(0, requested_scale)
                pending.insert(0, midpoint)
                subdivisions += 1
                continue
            break

        converged = bool(
            accepted_scale >= 1.0 - 1e-12
            and final_overlap <= overlap_limit
            and final_a2_error <= self.tol_A
        )
        if converged:
            self._centers[:self._n_fibers] = accepted_centers
            self._directions[:self._n_fibers] = accepted_directions
        else:
            self._centers[:self._n_fibers] = accepted_centers
            self._directions[:self._n_fibers] = accepted_directions
            self.set_collision_diameter_scale(1.0)
            final_overlap = float(self.measure_max_overlap())
            final_a2_error = float(self.orientation_error())

        return {
            "iters": float(total_iters),
            "converged": float(converged),
            "final_overlap": float(final_overlap),
            "final_A2_error_rel": float(final_a2_error),
            "best_score": float(best_score),
            "restarts": float(total_restarts),
            "force_evaluations": float(total_force_evaluations),
            "pair_build_s": float(total_pair_build_s),
            "pair_forces_s": float(total_pair_forces_s),
            "apply_updates_s": float(total_apply_updates_s),
            "initial_scale": float(initial_scale),
            "final_scale": float(1.0 if converged else accepted_scale),
            "stages": float(attempted_stages),
            "successful_stages": float(successful_stages),
            "subdivisions": float(subdivisions),
            "stage_failures": float(stage_failures),
            "wall_s": float(time.perf_counter() - t0),
        }

    def compact_collision_diameter(
        self,
        *,
        n_stages: int = 5,
        max_iter_per_stage: Optional[int] = None,
        target_overlap: Optional[float] = None,
        max_passes_per_stage: int = 2,
        min_scale_step: Optional[float] = None,
        rotation_gain: Optional[float] = None,
        final_rotation_gain: float = 2.0,
        max_fiber_removals: int = 0,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Grow the collision diameter to its physical value through relaxed stages."""
        t0 = time.perf_counter()
        target_overlap = self.tol_overlap if target_overlap is None else float(target_overlap)
        max_iter = (
            self.max_iter_relax
            if max_iter_per_stage is None
            else max(1, int(max_iter_per_stage))
        )
        max_passes = max(1, int(max_passes_per_stage))
        initial_scale = float(self.collision_diameter_scale)
        if rotation_gain is None:
            rotation_gain = 12.0
        scale_step_floor = (
            max(0.001, 0.5 * target_overlap / max(self._full_d_eff, 1e-12))
            if min_scale_step is None
            else max(0.001, float(min_scale_step))
        )
        max_diameter_increment = max(2.0 * target_overlap, 0.02 * self._full_d_eff)
        diameter_limited_stages = int(
            math.ceil(
                (1.0 - initial_scale)
                * self._full_d_eff
                / max(max_diameter_increment, 1e-12)
            )
        )
        effective_n_stages = max(1, int(n_stages), diameter_limited_stages)
        rotation_mobility_scale = self._rot_scale / max(
            self.fiber_length * self.fiber_length,
            1e-12,
        )

        def compaction_tau_rotate(scale: float) -> float:
            scale_progress = (float(scale) - initial_scale) / max(
                1.0 - initial_scale,
                1e-12,
            )
            scale_progress = min(1.0, max(0.0, scale_progress))
            gain = float(final_rotation_gain) + (
                float(rotation_gain) - float(final_rotation_gain)
            ) * (1.0 - scale_progress) ** 2
            return max(self.tau_rotate, gain * rotation_mobility_scale)

        initial_tau_rotate = compaction_tau_rotate(initial_scale)
        active_fraction = 1.0
        total_iters = 0
        total_relax_s = 0.0
        total_pair_build_s = 0.0
        total_pair_forces_s = 0.0
        attempted_stages = 0
        successful_stages = 0
        subdivisions = 0
        initial_relax_passes = 0
        shake_attempts = 0
        shaken_fibers = 0
        reinsert_candidates = 0
        reinserted_fibers = 0
        reinsert_failures = 0
        removed_fibers = 0

        if initial_scale >= 1.0 - 1e-12:
            self.set_collision_diameter_scale(1.0)
            final_overlap = self.measure_max_overlap()
            return {
                "initial_scale": initial_scale,
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
                "final_overlap": final_overlap,
                "converged": float(final_overlap <= target_overlap),
                "tau_rotate": float(initial_tau_rotate),
                "wall_s": float(time.perf_counter() - t0),
            }

        initial_overlap = self.measure_max_overlap()
        if initial_overlap > target_overlap:
            for _ in range(max_passes):
                initial_relax_passes += 1
                info = self.relax(
                    verbose=verbose,
                    max_iter=max_iter,
                    target_overlap=target_overlap,
                    apply_orientation=False,
                    pair_rebuild_interval=1,
                    effective_tau_override=0.08,
                    max_center_step_factor=0.10,
                    tau_rotate_override=initial_tau_rotate,
                    parallel_per_fiber=(
                        self.num_threads > 1 and self._n_fibers >= 512
                    ),
                    active_fraction=active_fraction,
                )
                total_iters += int(info["iters"])
                total_relax_s += float(info.get("wall_s", 0.0))
                total_pair_build_s += float(info.get("pair_build_s", 0.0))
                total_pair_forces_s += float(info.get("pair_forces_s", 0.0))
                initial_overlap = self.measure_max_overlap()
                if initial_overlap <= target_overlap:
                    break

        base_targets = np.linspace(
            initial_scale,
            1.0,
            effective_n_stages + 1,
            dtype=float,
        )[1:].tolist()
        pending = base_targets if initial_overlap <= target_overlap else []
        accepted_scale = initial_scale
        converged = initial_overlap <= target_overlap
        max_attempted_stages = max(12, 3 * len(base_targets))

        while pending and attempted_stages < max_attempted_stages:
            stage_scale = float(pending.pop(0))
            attempted_stages += 1
            self.set_collision_diameter_scale(stage_scale)
            stage_tau_rotate = compaction_tau_rotate(stage_scale)
            stage_overlap = math.inf

            for _ in range(max_passes):
                info = self.relax(
                    verbose=verbose,
                    max_iter=max_iter,
                    target_overlap=target_overlap,
                    apply_orientation=False,
                    pair_rebuild_interval=1,
                    effective_tau_override=0.08,
                    max_center_step_factor=0.10,
                    tau_rotate_override=stage_tau_rotate,
                    parallel_per_fiber=(
                        self.num_threads > 1 and self._n_fibers >= 512
                    ),
                    active_fraction=active_fraction,
                )
                total_iters += int(info["iters"])
                total_relax_s += float(info.get("wall_s", 0.0))
                total_pair_build_s += float(info.get("pair_build_s", 0.0))
                total_pair_forces_s += float(info.get("pair_forces_s", 0.0))
                stage_overlap = self.measure_max_overlap()
                if stage_overlap <= target_overlap:
                    break

            if stage_overlap <= target_overlap:
                accepted_scale = stage_scale
                successful_stages += 1
                continue

            scale_step = stage_scale - accepted_scale
            if scale_step > scale_step_floor and attempted_stages < max_attempted_stages:
                self.set_collision_diameter_scale(accepted_scale)
                midpoint = accepted_scale + 0.5 * scale_step
                pending.insert(0, stage_scale)
                pending.insert(0, midpoint)
                subdivisions += 1
                continue

            reinsertion = self.reinsert_overlapping_fibers(
                target_overlap=target_overlap,
            )
            reinsert_candidates += int(reinsertion["candidates"])
            reinserted_fibers += int(reinsertion["reinserted"])
            reinsert_failures += int(reinsertion["failed"])
            if reinsertion["reinserted"] > 0:
                info = self.relax(
                    verbose=verbose,
                    max_iter=max_iter,
                    target_overlap=target_overlap,
                    apply_orientation=False,
                    pair_rebuild_interval=1,
                    effective_tau_override=0.05,
                    max_center_step_factor=0.08,
                    tau_rotate_override=0.5 * stage_tau_rotate,
                    parallel_per_fiber=(
                        self.num_threads > 1 and self._n_fibers >= 512
                    ),
                    active_fraction=active_fraction,
                )
                total_iters += int(info["iters"])
                total_relax_s += float(info.get("wall_s", 0.0))
                total_pair_build_s += float(info.get("pair_build_s", 0.0))
                total_pair_forces_s += float(info.get("pair_forces_s", 0.0))
                stage_overlap = self.measure_max_overlap()
                if stage_overlap <= target_overlap:
                    accepted_scale = stage_scale
                    successful_stages += 1
                    continue

            remaining_removals = max(
                0,
                int(max_fiber_removals) - self._compaction_removed_fibers,
            )
            if remaining_removals > 0:
                removed_now = self.drop_worst_overlapping_fibers(
                    target_overlap=target_overlap,
                    max_remove=remaining_removals,
                )
                removed_fibers += removed_now
                self._compaction_removed_fibers += removed_now
                if removed_now > 0:
                    info = self.relax(
                        verbose=verbose,
                        max_iter=max_iter,
                        target_overlap=target_overlap,
                        apply_orientation=False,
                        pair_rebuild_interval=1,
                        effective_tau_override=0.05,
                        max_center_step_factor=0.08,
                        tau_rotate_override=0.5 * stage_tau_rotate,
                        parallel_per_fiber=(
                            self.num_threads > 1 and self._n_fibers >= 512
                        ),
                        active_fraction=active_fraction,
                    )
                    total_iters += int(info["iters"])
                    total_relax_s += float(info.get("wall_s", 0.0))
                    total_pair_build_s += float(info.get("pair_build_s", 0.0))
                    total_pair_forces_s += float(info.get("pair_forces_s", 0.0))
                    stage_overlap = self.measure_max_overlap()
                    if stage_overlap <= target_overlap:
                        accepted_scale = stage_scale
                        successful_stages += 1
                        continue

            best_overlap = stage_overlap
            for shake_index in range(3):
                centers_before = self.centers.copy()
                directions_before = self.directions.copy()
                shake_attempts += 1
                shaken_fibers += self.shake_overlapping_fibers(
                    target_overlap=target_overlap,
                    translate_fraction=0.04 / (1.0 + shake_index),
                    rotate_radians=0.012 / (1.0 + shake_index),
                )
                info = self.relax(
                    verbose=verbose,
                    max_iter=max_iter,
                    target_overlap=target_overlap,
                    apply_orientation=False,
                    pair_rebuild_interval=1,
                    effective_tau_override=0.05,
                    max_center_step_factor=0.08,
                    tau_rotate_override=0.5 * stage_tau_rotate,
                    parallel_per_fiber=(
                        self.num_threads > 1 and self._n_fibers >= 512
                    ),
                    active_fraction=active_fraction,
                )
                total_iters += int(info["iters"])
                total_relax_s += float(info.get("wall_s", 0.0))
                total_pair_build_s += float(info.get("pair_build_s", 0.0))
                total_pair_forces_s += float(info.get("pair_forces_s", 0.0))
                shaken_overlap = self.measure_max_overlap()
                if shaken_overlap < best_overlap:
                    best_overlap = shaken_overlap
                    stage_overlap = shaken_overlap
                    if stage_overlap <= target_overlap:
                        break
                else:
                    self._centers[:self._n_fibers] = centers_before
                    self._directions[:self._n_fibers] = directions_before

            if stage_overlap <= target_overlap:
                accepted_scale = stage_scale
                successful_stages += 1
                continue

            converged = False
            break

        if pending:
            converged = False

        self.set_collision_diameter_scale(1.0)
        final_tau_rotate = compaction_tau_rotate(1.0)
        final_overlap = self.measure_max_overlap()
        if final_overlap > target_overlap:
            for _ in range(max_passes):
                info = self.relax(
                    verbose=verbose,
                    max_iter=max_iter,
                    target_overlap=target_overlap,
                    apply_orientation=False,
                    pair_rebuild_interval=1,
                    effective_tau_override=0.08,
                    max_center_step_factor=0.10,
                    tau_rotate_override=final_tau_rotate,
                    parallel_per_fiber=(
                        self.num_threads > 1 and self._n_fibers >= 512
                    ),
                    active_fraction=active_fraction,
                )
                total_iters += int(info["iters"])
                total_relax_s += float(info.get("wall_s", 0.0))
                total_pair_build_s += float(info.get("pair_build_s", 0.0))
                total_pair_forces_s += float(info.get("pair_forces_s", 0.0))
                final_overlap = self.measure_max_overlap()
                if final_overlap <= target_overlap:
                    break

        converged = converged and final_overlap <= target_overlap
        return {
            "initial_scale": initial_scale,
            "final_scale": float(self.collision_diameter_scale),
            "stages": float(attempted_stages),
            "successful_stages": float(successful_stages),
            "subdivisions": float(subdivisions),
            "initial_relax_passes": float(initial_relax_passes),
            "shake_attempts": float(shake_attempts),
            "shaken_fibers": float(shaken_fibers),
            "reinsert_candidates": float(reinsert_candidates),
            "reinserted_fibers": float(reinserted_fibers),
            "reinsert_failures": float(reinsert_failures),
            "removed_fibers": float(removed_fibers),
            "relax_iters": float(total_iters),
            "relax_s": float(total_relax_s),
            "pair_build_s": float(total_pair_build_s),
            "pair_forces_s": float(total_pair_forces_s),
            "final_overlap": float(final_overlap),
            "converged": float(converged),
            "tau_rotate": float(initial_tau_rotate),
            "final_tau_rotate": float(final_tau_rotate),
            "wall_s": float(time.perf_counter() - t0),
        }

    def _choose_effective_batch_vf(self, current_vf: float) -> float:
        remaining = self.vf_target - current_vf
        if not self.fast_mode:
            return min(self.batch_vf, remaining * 0.5 + 1e-6)

        progress = current_vf / max(self.vf_target, 1e-12)
        if progress < 0.20:
            multiplier = 3.0
        elif progress < 0.45:
            multiplier = 2.0
        elif progress < 0.75:
            multiplier = 1.4
        elif progress < 0.90:
            multiplier = 1.0
        else:
            multiplier = 0.75

        return min(self.batch_vf * multiplier, remaining * 0.9 + 1e-6)

    def _get_free_volume_grid(self, cell_w: float) -> Tuple[int, int, int, Array]:
        Lx, Ly, Lz = self.cell_size
        nx = max(2, int(math.floor(Lx / cell_w)))
        ny = max(2, int(math.floor(Ly / cell_w)))
        nz = max(2, int(math.floor(Lz / cell_w)))
        key = (nx, ny, nz)
        centers = self._free_volume_cache.get(key)
        if centers is None:
            xs = (np.arange(nx, dtype=float) + 0.5) * (Lx / nx)
            ys = (np.arange(ny, dtype=float) + 0.5) * (Ly / ny)
            zs = (np.arange(nz, dtype=float) + 0.5) * (Lz / nz)
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
            centers = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
            self._free_volume_cache[key] = centers
        return nx, ny, nz, centers

    def _sample_available_volume_centers(self, chunk: int, progress: float, hard_d: float) -> Array:
        cell_w = max(1.5 * self.d_eff, min(0.35 * self.fiber_length, 0.12 * float(np.min(self.cell_size))))
        nx, ny, nz, grid_centers = self._get_free_volume_grid(cell_w)
        m = len(grid_centers)
        if m == 0 or self._n_fibers == 0:
            return self.rng.random((chunk, 3)) * self.cell_size

        existing = self.centers
        nearest_sq = _nearest_periodic_sq_numba(
            grid_centers,
            existing,
            self.cell_size[0],
            self.cell_size[1],
            self.cell_size[2],
        )

        clearance = np.sqrt(nearest_sq)
        threshold = max(0.35 * self._cull_radius, 0.5 * self.fiber_length + hard_d)
        scores = np.maximum(clearance - threshold, 0.0)
        if float(np.max(scores)) <= 1e-12:
            scores = clearance.copy()

        keep_ratio = 0.25 if progress > 0.75 else 0.40
        cutoff = np.quantile(scores, max(0.0, 1.0 - keep_ratio))
        mask = scores >= cutoff
        selected = np.nonzero(mask)[0]
        if len(selected) == 0:
            selected = np.arange(m)
        weights = scores[selected] + 1e-6
        chosen = self.rng.choice(selected, size=chunk, p=weights / np.sum(weights))

        rx = self.rng.random(chunk) - 0.5
        ry = self.rng.random(chunk) - 0.5
        rz = self.rng.random(chunk) - 0.5
        dx = self.cell_size[0] / nx
        dy = self.cell_size[1] / ny
        dz = self.cell_size[2] / nz

        centers = grid_centers[chosen].copy()
        centers[:, 0] += rx * dx
        centers[:, 1] += ry * dy
        centers[:, 2] += rz * dz
        centers[:] = np.mod(centers, self.cell_size)
        return centers

    def _sample_candidate_centers(self, chunk: int, progress: float, hard_d: float) -> Array:
        centers = self.rng.random((chunk, 3)) * self.cell_size
        if self.center_distribution == "gaussian_mixture" and self.cluster_fraction > 0.0:
            effective_cluster_fraction = self.cluster_fraction * max(
                0.10,
                1.0 - float(progress) ** 2,
            )
            n_clustered = min(
                chunk,
                max(0, int(round(float(chunk) * effective_cluster_fraction))),
            )
            if n_clustered:
                components = self.rng.integers(0, self.cluster_count, size=n_clustered)
                noise = self.rng.normal(size=(n_clustered, 3))
                noise *= self.cluster_sigma_rel * self.cell_size
                centers[:n_clustered] = np.mod(
                    self._cluster_centers[components] + noise,
                    self.cell_size,
                )
            return centers
        if (not self.bias_free_volume) or self._n_fibers < 64:
            return centers

        Lx, Ly, Lz = self.cell_size
        cell_w = min(self._cull_radius, float(np.min(self.cell_size)))
        nx = max(1, int(math.floor(Lx / cell_w)))
        ny = max(1, int(math.floor(Ly / cell_w)))
        nz = max(1, int(math.floor(Lz / cell_w)))
        total_cells = nx * ny * nz
        if total_cells <= 1:
            return centers

        existing = self.centers
        occ = _occupancy_counts_numba(existing, nx, ny, nz, Lx, Ly, Lz)

        exponent = 1.2 + 1.6 * progress
        weights = 1.0 / np.power(1.0 + occ, exponent)
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            return centers

        biased_fraction = 0.35 + 0.55 * progress
        n_biased = min(chunk, max(0, int(round(chunk * biased_fraction))))
        if n_biased == 0:
            return centers

        chosen = self.rng.choice(total_cells, size=n_biased, p=weights / total_weight)
        rx = self.rng.random(n_biased)
        ry = self.rng.random(n_biased)
        rz = self.rng.random(n_biased)

        cell_dx = Lx / nx
        cell_dy = Ly / ny
        cell_dz = Lz / nz

        cx = chosen // (ny * nz)
        rem = chosen % (ny * nz)
        cy = rem // nz
        cz = rem % nz

        centers[:n_biased, 0] = (cx + rx) * cell_dx
        centers[:n_biased, 1] = (cy + ry) * cell_dy
        centers[:n_biased, 2] = (cz + rz) * cell_dz

        if self.fast_mode and progress > 0.40:
            av_fraction = min(0.85, 0.25 + 0.75 * progress)
            n_av = min(chunk, max(1, int(round(chunk * av_fraction))))
            centers[:n_av] = self._sample_available_volume_centers(n_av, progress, hard_d)
        return centers

    def _sample_directions_batch(
        self,
        chunk: int,
        target_A2: Optional[Array] = None,
    ) -> Array:
        if target_A2 is None:
            transform = self._A2_transform
        else:
            sanitized = self._sanitize_A2(np.asarray(target_A2, dtype=float))
            key = tuple(float(round(value, 6)) for value in sanitized.ravel())
            transform = _angular_gaussian_transform(key)
        gaussian = self.rng.standard_normal((chunk, 3)) @ transform.T
        norms = np.linalg.norm(gaussian, axis=1)
        bad = norms < 1e-12
        if np.any(bad):
            gaussian[bad] = np.array([1.0, 0.0, 0.0], dtype=float)
            norms[bad] = 1.0
        return gaussian / norms[:, None]

    @classmethod
    def sample_directions_for_A2(
        cls,
        A2: Array,
        rng: np.random.Generator,
        count: int,
    ) -> Array:
        sanitized = cls._sanitize_A2(np.asarray(A2, dtype=float))
        key = tuple(float(round(value, 12)) for value in sanitized.ravel())
        transform = _angular_gaussian_transform(key)
        gaussian = rng.standard_normal((int(count), 3)) @ transform.T
        norms = np.linalg.norm(gaussian, axis=1)
        bad = norms < 1e-12
        if np.any(bad):
            gaussian[bad] = np.array([1.0, 0.0, 0.0], dtype=float)
            norms[bad] = 1.0
        return gaussian / norms[:, None]

    def _relax_schedule(self, progress: float) -> Tuple[int, float, bool]:
        if not self.fast_mode:
            return self.max_iter_relax, self.tol_overlap, True

        if progress < 0.25:
            return min(self.max_iter_relax, 10), 3.0 * self.tol_overlap, False
        if progress < 0.50:
            return min(self.max_iter_relax, 16), 2.0 * self.tol_overlap, False
        if progress < 0.75:
            return min(self.max_iter_relax, 24), 1.5 * self.tol_overlap, False
        if progress < 0.90:
            return min(self.max_iter_relax, 36), 1.2 * self.tol_overlap, True
        if progress < 0.98:
            return min(self.max_iter_relax, 72), self.tol_overlap, True
        return min(self.max_iter_relax, 96), self.tol_overlap, True

    def _collect_local_indices(self, seed_indices: Array, radius_scale: float = 1.8) -> Array:
        if len(seed_indices) == 0 or self._n_fibers == 0:
            return np.empty(0, dtype=np.int32)
        if len(seed_indices) >= self._n_fibers:
            return np.arange(self._n_fibers, dtype=np.int32)

        radius_sq = (radius_scale * self._cull_radius) ** 2
        all_centers = self.centers
        mask = np.zeros(self._n_fibers, dtype=bool)
        for idx in seed_indices:
            delta = all_centers - all_centers[idx]
            delta -= np.round(delta / self.cell_size) * self.cell_size
            dist_sq = np.einsum("ij,ij->i", delta, delta)
            mask |= dist_sq <= radius_sq
        return np.flatnonzero(mask).astype(np.int32)

    def relax_local(
        self,
        seed_indices: Array,
        verbose: bool = False,
        max_iter: Optional[int] = None,
        target_overlap: Optional[float] = None,
    ) -> Dict[str, float]:
        t_relax0 = time.perf_counter()
        local_idx = self._collect_local_indices(seed_indices)
        n_local = len(local_idx)
        if n_local == 0:
            return {
                "iters": 0,
                "max_overlap": 0.0,
                "A_err": 0.0,
                "local_n": 0.0,
                "wall_s": 0.0,
                "pair_build_s": 0.0,
                "pair_forces_s": 0.0,
                "apply_updates_s": 0.0,
            }
        if n_local > 0.85 * self._n_fibers:
            info = self.relax(
                verbose=verbose,
                max_iter=max_iter,
                target_overlap=target_overlap,
                apply_orientation=False,
            )
            info["local_n"] = float(self._n_fibers)
            return info

        max_iter_relax = self.max_iter_relax if max_iter is None else int(max_iter)
        target_overlap = self.tol_overlap if target_overlap is None else float(target_overlap)
        Lx, Ly, Lz = self.cell_size
        local_centers = self._centers[local_idx].copy()
        local_dirs = self._directions[local_idx].copy()
        pair_cache = None
        effective_tau = max(self.tau_translate, 0.30)
        stagnation_count = 0
        overlap_history: List[float] = []
        t_pair_build_s = 0.0
        t_pair_forces_s = 0.0
        t_apply_updates_s = 0.0

        for it in range(1, max_iter_relax + 1):
            center_updates = np.zeros((n_local, 3), dtype=float)
            rot_updates = np.zeros((n_local, 3), dtype=float)

            local_pair_rebuild_interval = 12 if self.fast_mode else 4
            if pair_cache is None or it % local_pair_rebuild_interval == 1:
                t0 = time.perf_counter()
                pair_cache = _build_pairs_center_cull_numba(
                    local_centers,
                    Lx,
                    Ly,
                    Lz,
                    self._cull_radius ** 2,
                )
                t_pair_build_s += time.perf_counter() - t0

            if len(pair_cache) > 0:
                t0 = time.perf_counter()
                max_overlap = _apply_pair_forces_numba(
                    pair_cache, local_centers, local_dirs,
                    center_updates, rot_updates,
                    Lx, Ly, Lz,
                    self.fiber_length, self.d_eff, self._cull_radius ** 2,
                    effective_tau, self.tol_overlap, self.tau_rotate,
                    1.0 / max(self._rot_scale, 1e-12),
                )
                t_pair_forces_s += time.perf_counter() - t0
            else:
                max_overlap = 0.0

            if max_overlap < target_overlap:
                self._centers[local_idx] = local_centers
                self._directions[local_idx] = local_dirs
                return {
                    "iters": it,
                    "max_overlap": max_overlap,
                    "A_err": 0.0,
                    "local_n": float(n_local),
                    "wall_s": float(time.perf_counter() - t_relax0),
                    "pair_build_s": float(t_pair_build_s),
                    "pair_forces_s": float(t_pair_forces_s),
                    "apply_updates_s": float(t_apply_updates_s),
                }

            disp_norms = np.linalg.norm(center_updates, axis=1)
            max_center_step = 0.45 * self.d_eff
            mask = disp_norms > max_center_step
            if np.any(mask):
                center_updates[mask] *= (max_center_step / disp_norms[mask])[:, None]

            rot_norms = np.linalg.norm(rot_updates, axis=1)
            max_rot_step = 0.08
            mask_rot = rot_norms > max_rot_step
            if np.any(mask_rot):
                rot_updates[mask_rot] *= (max_rot_step / rot_norms[mask_rot])[:, None]

            t0 = time.perf_counter()
            _apply_updates_numba(
                local_centers,
                local_dirs,
                center_updates,
                rot_updates,
                Lx, Ly, Lz,
            )
            t_apply_updates_s += time.perf_counter() - t0

            prev_overlap = overlap_history[-1] if overlap_history else math.inf
            overlap_history.append(max_overlap)
            if max_overlap < prev_overlap - 1e-8 * self.d_eff:
                stagnation_count = 0
            else:
                stagnation_count += 1
                if stagnation_count >= 10:
                    effective_tau *= 0.8
                    stagnation_count = 0

            if effective_tau < 0.01:
                self._centers[local_idx] = local_centers
                self._directions[local_idx] = local_dirs
                return {
                    "iters": it,
                    "max_overlap": max_overlap,
                    "A_err": 0.0,
                    "local_n": float(n_local),
                    "wall_s": float(time.perf_counter() - t_relax0),
                    "pair_build_s": float(t_pair_build_s),
                    "pair_forces_s": float(t_pair_forces_s),
                    "apply_updates_s": float(t_apply_updates_s),
                }

        self._centers[local_idx] = local_centers
        self._directions[local_idx] = local_dirs
        return {
            "iters": max_iter_relax,
            "max_overlap": max_overlap,
            "A_err": 0.0,
            "local_n": float(n_local),
            "wall_s": float(time.perf_counter() - t_relax0),
            "pair_build_s": float(t_pair_build_s),
            "pair_forces_s": float(t_pair_forces_s),
            "apply_updates_s": float(t_apply_updates_s),
        }

    def _candidate_overlap_flags(
        self,
        candidate_centers: Array,
        candidate_directions: Array,
        *,
        n_existing: int,
        hard_d: float,
        cull_r_sq: float,
    ) -> Array:
        pair_count = len(candidate_centers) * int(n_existing)
        use_cupy = (
            self.geometry_backend_requested in {"cupy", "auto"}
            and not self._cupy_geometry_disabled
            and pair_count >= self.cupy_min_candidate_pairs
        )
        if use_cupy:
            t0 = time.perf_counter()
            try:
                overlaps = _check_candidates_against_existing_cupy(
                    candidate_centers,
                    candidate_directions,
                    self._centers,
                    self._directions,
                    n_existing,
                    self.cell_size[0],
                    self.cell_size[1],
                    self.cell_size[2],
                    self.fiber_length,
                    hard_d,
                    cull_r_sq,
                )
                self._cupy_candidate_calls += 1
                self._cupy_candidate_pairs += pair_count
                self._cupy_candidate_s += time.perf_counter() - t0
                self.geometry_backend_effective = "cupy_hybrid"
                return overlaps
            except Exception:
                self._cupy_candidate_fallbacks += 1
                self._cupy_candidate_s += time.perf_counter() - t0
                self._cupy_geometry_disabled = True
                self.geometry_backend_effective = "numba_fallback"

        return _check_candidates_against_existing_numba(
            candidate_centers,
            candidate_directions,
            self._centers[:n_existing],
            self._directions[:n_existing],
            n_existing,
            self.cell_size[0],
            self.cell_size[1],
            self.cell_size[2],
            self.fiber_length,
            hard_d,
            cull_r_sq,
        )

    def _cupy_contact_forces(
        self,
        neighbor_offsets: Array,
        neighbor_indices: Array,
        *,
        effective_tau: float,
        target_overlap: float,
        effective_tau_rotate: float,
    ) -> Optional[Tuple[Array, Array, Array]]:
        use_cupy = (
            self.geometry_backend_requested in {"cupy", "auto"}
            and not self._cupy_geometry_disabled
            and len(neighbor_indices) >= self.cupy_min_candidate_pairs
        )
        if not use_cupy:
            return None
        t0 = time.perf_counter()
        try:
            result = _apply_forces_per_fiber_cupy(
                self.centers,
                self.directions,
                neighbor_offsets,
                neighbor_indices,
                self.cell_size[0],
                self.cell_size[1],
                self.cell_size[2],
                self.fiber_length,
                self.d_eff,
                self._cull_radius ** 2,
                effective_tau,
                target_overlap,
                effective_tau_rotate,
                1.0 / max(self._rot_scale, 1e-12),
            )
            self._cupy_force_calls += 1
            self._cupy_force_pairs += len(neighbor_indices)
            self._cupy_force_s += time.perf_counter() - t0
            self.geometry_backend_effective = "cupy_hybrid"
            return result
        except Exception:
            self._cupy_force_fallbacks += 1
            self._cupy_force_s += time.perf_counter() - t0
            self._cupy_geometry_disabled = True
            self.geometry_backend_effective = "numba_fallback"
            return None

    def _cupy_all_pair_contact_forces(
        self,
        *,
        effective_tau: float,
        target_overlap: float,
        effective_tau_rotate: float,
    ) -> Optional[Tuple[Array, Array, Array]]:
        directed_pairs = self._n_fibers * max(0, self._n_fibers - 1)
        use_cupy = (
            self.geometry_backend_requested in {"cupy", "auto"}
            and not self._cupy_geometry_disabled
            and directed_pairs >= self.cupy_min_candidate_pairs
        )
        if not use_cupy:
            return None

        t0 = time.perf_counter()
        try:
            dummy = np.zeros(1, dtype=np.int32)
            result = _apply_forces_per_fiber_cupy(
                self.centers,
                self.directions,
                dummy,
                dummy,
                self.cell_size[0],
                self.cell_size[1],
                self.cell_size[2],
                self.fiber_length,
                self.d_eff,
                self._cull_radius ** 2,
                effective_tau,
                target_overlap,
                effective_tau_rotate,
                1.0 / max(self._rot_scale, 1e-12),
                all_pairs=True,
            )
            self._cupy_force_calls += 1
            self._cupy_force_pairs += directed_pairs
            self._cupy_force_s += time.perf_counter() - t0
            self.geometry_backend_effective = "cupy_hybrid"
            return result
        except Exception:
            self._cupy_force_fallbacks += 1
            self._cupy_force_s += time.perf_counter() - t0
            self._cupy_geometry_disabled = True
            self.geometry_backend_effective = "numba_fallback"
            return None

    def add_batch(self, effective_batch_vf: Optional[float] = None) -> Tuple[int, int, Dict[str, float]]:
        t_batch0 = time.perf_counter()
        bvf = effective_batch_vf if effective_batch_vf is not None else self.batch_vf
        requested = max(1, round(bvf * self.V_cell / self.V_fiber))
        remaining = max(0, self.target_fiber_count() - self._n_fibers)
        n_to_add = min(requested, remaining)
        if n_to_add == 0:
            self._last_added_indices = np.empty(0, dtype=np.int32)
            return 0, 0, {
                "wall_s": float(time.perf_counter() - t_batch0),
                "sample_centers_s": 0.0,
                "sample_dirs_s": 0.0,
                "candidate_checks_s": 0.0,
                "soft_inserted": 0.0,
            }
        max_attempts = max(500, 1000 * n_to_add)
        self._ensure_capacity(n_to_add)
        start_idx = self._n_fibers

        progress = self.current_vf() / max(self.vf_target, 1e-12)
        if self.strict_insertion:
            severe_factor = 1.0
        else:
            severe_factor = 0.85 - 0.35 * progress
            if self.fast_mode:
                severe_factor -= 0.08 * min(1.0, progress)
            severe_factor = max(0.30, severe_factor)
        hard_d = severe_factor * self.d_eff
        cull_r_sq = self._cull_radius ** 2
        use_soft_insert = (
            self.soft_insert
            and self.fast_mode
            and progress >= self.soft_insert_start
            and self._n_fibers > 0
        )

        Lx, Ly, Lz = self.cell_size
        L = self.fiber_length

        attempts = 0
        added = 0

        chunk = max(64, n_to_add * 4)
        t_sample_centers_s = 0.0
        t_sample_dirs_s = 0.0
        t_candidate_checks_s = 0.0
        soft_inserted = 0.0
        soft_attempt_limit = max(5_000, int(80 * n_to_add))

        def append_soft(n_extra: int) -> None:
            nonlocal added, attempts, t_sample_centers_s, t_sample_dirs_s, soft_inserted
            if n_extra <= 0:
                return
            t0 = time.perf_counter()
            centers_soft = self._sample_candidate_centers(n_extra, progress, hard_d)
            t_sample_centers_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            dirs_soft = self._sample_directions_batch(
                n_extra,
                target_A2=self._remaining_orientation_target(n_extra),
            )
            t_sample_dirs_s += time.perf_counter() - t0
            insert_start = self._n_fibers
            insert_stop = insert_start + n_extra
            self._centers[insert_start:insert_stop] = centers_soft
            self._directions[insert_start:insert_stop] = dirs_soft
            self._n_fibers = insert_stop
            added += n_extra
            attempts += n_extra
            soft_inserted = 1.0

        if use_soft_insert:
            append_soft(n_to_add)
            self._last_added_indices = np.arange(start_idx, start_idx + added, dtype=np.int32)
            return added, attempts, {
                "wall_s": float(time.perf_counter() - t_batch0),
                "sample_centers_s": float(t_sample_centers_s),
                "sample_dirs_s": float(t_sample_dirs_s),
                "candidate_checks_s": float(t_candidate_checks_s),
                "soft_inserted": soft_inserted,
            }

        while added < n_to_add and attempts < max_attempts:
            t0 = time.perf_counter()
            centers_cand = self._sample_candidate_centers(chunk, progress, hard_d)
            t_sample_centers_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            remaining_to_add = max(1, n_to_add - added)
            dirs_cand = self._sample_directions_batch(
                chunk,
                target_A2=self._remaining_orientation_target(
                    remaining_to_add
                ),
            )
            t_sample_dirs_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            n_existing = self._n_fibers
            if n_existing == 0:
                overlap_existing = np.zeros(chunk, dtype=np.uint8)
            else:
                overlap_existing = self._candidate_overlap_flags(
                    centers_cand,
                    dirs_cand,
                    n_existing=n_existing,
                    hard_d=hard_d,
                    cull_r_sq=cull_r_sq,
                )

            accepted_centers = np.empty((chunk, 3), dtype=float)
            accepted_dirs = np.empty((chunk, 3), dtype=float)
            n_accepted_local = 0
            available = overlap_existing == 0
            attempts += int(np.count_nonzero(~available))
            while np.any(available):
                available_indices = np.flatnonzero(available)
                local_scores = self._orientation_addition_scores(
                    dirs_cand[available_indices]
                )
                best_local = int(np.argmin(local_scores))
                k = int(available_indices[best_local])
                available[k] = False
                attempts += 1
                if attempts > max_attempts:
                    break

                cc = centers_cand[k]
                dd = dirs_cand[k]
                overlap = False
                if n_accepted_local > 0:
                    overlap = _check_candidate_serial_numba(
                        cc[0], cc[1], cc[2],
                        dd[0], dd[1], dd[2],
                        accepted_centers,
                        accepted_dirs,
                        n_accepted_local,
                        Lx, Ly, Lz, L, hard_d, cull_r_sq, -1
                    )

                if not overlap:
                    idx = self._n_fibers
                    self._centers[idx] = cc
                    self._directions[idx] = dd
                    accepted_centers[n_accepted_local] = cc
                    accepted_dirs[n_accepted_local] = dd
                    n_accepted_local += 1
                    self._n_fibers += 1
                    added += 1
                    if added >= n_to_add:
                        break
            t_candidate_checks_s += time.perf_counter() - t0
            fallback_progress = min(self.soft_insert_start, 0.30)
            if (
                self.soft_insert
                and self.fast_mode
                and self.soft_insert_start < 1.0
                and progress >= fallback_progress
                and added < n_to_add
                and attempts >= soft_attempt_limit
            ):
                append_soft(n_to_add - added)
                break

        self._last_added_indices = np.arange(start_idx, start_idx + added, dtype=np.int32)
        batch_stats = {
            "wall_s": float(time.perf_counter() - t_batch0),
            "sample_centers_s": float(t_sample_centers_s),
            "sample_dirs_s": float(t_sample_dirs_s),
            "candidate_checks_s": float(t_candidate_checks_s),
            "soft_inserted": soft_inserted,
        }
        return added, attempts, batch_stats

    def relax(
        self,
        verbose: bool = False,
        max_iter: Optional[int] = None,
        target_overlap: Optional[float] = None,
        apply_orientation: bool = True,
        pair_rebuild_interval: Optional[int] = None,
        effective_tau_override: Optional[float] = None,
        max_center_step_factor: Optional[float] = None,
        tau_rotate_override: Optional[float] = None,
        parallel_per_fiber: bool = False,
        active_fraction: float = 1.0,
    ) -> Dict[str, float]:
        t_relax0 = time.perf_counter()
        n = self._n_fibers
        if n == 0:
            return {
                "iters": 0,
                "max_overlap": 0.0,
                "A_err": 0.0,
                "wall_s": 0.0,
                "pair_build_s": 0.0,
                "pair_forces_s": 0.0,
                "orientation_corr_s": 0.0,
                "compute_a4_s": 0.0,
                "apply_updates_s": 0.0,
            }

        pair_cache = None
        max_iter_relax = self.max_iter_relax if max_iter is None else int(max_iter)
        target_overlap = self.tol_overlap if target_overlap is None else float(target_overlap)
        rebuild_interval = (
            6 if self.fast_mode else 5
        ) if pair_rebuild_interval is None else max(1, int(pair_rebuild_interval))

        overlap_history = []

        effective_tau = (
            max(self.tau_translate, 0.30) if self.fast_mode else self.tau_translate
        )
        if effective_tau_override is not None:
            effective_tau = max(1e-6, float(effective_tau_override))
        effective_tau_rotate = (
            self.tau_rotate
            if tau_rotate_override is None
            else max(1e-6, float(tau_rotate_override))
        )
        stagnation_count = 0
        A_err = self.orientation_error()
        A_delta = self.A2 - self.compute_A2_vec()
        t_pair_build_s = 0.0
        t_pair_forces_s = 0.0
        t_orientation_corr_s = 0.0
        t_compute_a4_s = 0.0
        t_apply_updates_s = 0.0
        max_overlap = 0.0
        cupy_force_possible = (
            self.geometry_backend_requested in {"cupy", "auto"}
            and not self._cupy_geometry_disabled
        )

        def _relax_result(iters: int, overlap: float, a_err: float) -> Dict[str, float]:
            return {
                "iters": int(iters),
                "max_overlap": float(overlap),
                "A_err": float(a_err),
                "wall_s": float(time.perf_counter() - t_relax0),
                "pair_build_s": float(t_pair_build_s),
                "pair_forces_s": float(t_pair_forces_s),
                "orientation_corr_s": float(t_orientation_corr_s),
                "compute_a4_s": float(t_compute_a4_s),
                "apply_updates_s": float(t_apply_updates_s),
            }

        for it in range(1, max_iter_relax + 1):
            a2_interval = 5 if self.fast_mode else 3
            if apply_orientation and (it == 1 or it % a2_interval == 0):
                t0 = time.perf_counter()
                A2_cur = self.compute_A2_vec()
                A_err = self.relative_tensor_error(A2_cur, self.A2)
                A_delta = self.A2 - A2_cur
                t_compute_a4_s += time.perf_counter() - t0

            center_updates = np.zeros((n, 3), dtype=float)
            rot_updates = np.zeros((n, 3), dtype=float)

            cupy_force_result = None
            if cupy_force_possible:
                t0 = time.perf_counter()
                cupy_force_result = self._cupy_all_pair_contact_forces(
                    effective_tau=effective_tau,
                    target_overlap=target_overlap,
                    effective_tau_rotate=effective_tau_rotate,
                )
                t_pair_forces_s += time.perf_counter() - t0
                if self._cupy_geometry_disabled:
                    cupy_force_possible = False

            if cupy_force_result is None:
                if pair_cache is None or it % rebuild_interval == 1:
                    t0 = time.perf_counter()
                    pair_cache = self.build_neighbor_pairs_vec()
                    if parallel_per_fiber:
                        neighbor_offsets, neighbor_indices = _pairs_to_csr_numba(
                            pair_cache,
                            n,
                        )
                    t_pair_build_s += time.perf_counter() - t0

            if cupy_force_result is not None:
                center_updates[:], rot_updates[:], max_overlap_per_fiber = (
                    cupy_force_result
                )
                max_overlap = float(np.max(max_overlap_per_fiber))
            elif parallel_per_fiber:
                t0 = time.perf_counter()
                max_overlap_per_fiber = np.zeros(n, dtype=float)
                _apply_forces_per_fiber_numba(
                    self.centers,
                    self.directions,
                    neighbor_offsets,
                    neighbor_indices,
                    center_updates,
                    rot_updates,
                    max_overlap_per_fiber,
                    self.cell_size[0],
                    self.cell_size[1],
                    self.cell_size[2],
                    self.fiber_length,
                    self.d_eff,
                    self._cull_radius ** 2,
                    effective_tau,
                    target_overlap,
                    effective_tau_rotate,
                    1.0 / max(self._rot_scale, 1e-12),
                )
                max_overlap = float(np.max(max_overlap_per_fiber))
                t_pair_forces_s += time.perf_counter() - t0
            elif len(pair_cache) > 0:
                t0 = time.perf_counter()
                max_overlap = self._apply_pair_forces_vec(
                    pair_cache, center_updates, rot_updates,
                    effective_tau=effective_tau,
                    tau_rotate_override=effective_tau_rotate,
                    force_tolerance=target_overlap,
                )
                t_pair_forces_s += time.perf_counter() - t0
            else:
                max_overlap = 0.0

            orientation_ok = (not apply_orientation) or A_err <= self.tol_A
            if max_overlap <= target_overlap and orientation_ok:
                if verbose:
                    print(
                        f"\r|  +- Relax {it:4d}/{max_iter_relax} "
                        f"| Solap={max_overlap:.4e} | A_err={A_err:.4e} (Convergido)       "
                    )
                return _relax_result(it, max_overlap, A_err)

            if active_fraction < 1.0:
                active = self.rng.random(n) < max(0.01, float(active_fraction))
                if not np.any(active):
                    active[int(self.rng.integers(0, n))] = True
                center_updates[~active] = 0.0
                rot_updates[~active] = 0.0

            orientation_overlap_scale = max(
                10.0 * max(target_overlap, self.tol_overlap, 1e-8),
                0.10 * self.d_eff,
            )
            orient_scale = max(
                0.0,
                1.0 - max_overlap / orientation_overlap_scale,
            )
            if apply_orientation and orient_scale > 0.01:
                t0 = time.perf_counter()
                self._apply_orientation_correction_vec(A_delta, rot_updates,
                                                       scale=orient_scale)
                t_orientation_corr_s += time.perf_counter() - t0

            if n > 0:
                disp_norms = np.linalg.norm(center_updates, axis=1)
                step_factor = (
                    0.45 if self.fast_mode else 0.30
                ) if max_center_step_factor is None else float(max_center_step_factor)
                max_center_step = max(1e-6, step_factor) * self.d_eff
                mask = disp_norms > max_center_step
                if np.any(mask):
                    center_updates[mask] *= (max_center_step / disp_norms[mask])[:, None]

            rot_norms = np.linalg.norm(rot_updates, axis=1)
            max_rot_step = 0.08 if self.fast_mode else 0.05
            mask_rot = rot_norms > max_rot_step
            if np.any(mask_rot):
                rot_updates[mask_rot] *= (max_rot_step / rot_norms[mask_rot])[:, None]

            c = self.centers
            d = self.directions
            t0 = time.perf_counter()
            _apply_updates_numba(
                c,
                d,
                center_updates,
                rot_updates,
                self.cell_size[0],
                self.cell_size[1],
                self.cell_size[2],
            )
            t_apply_updates_s += time.perf_counter() - t0

            prev_overlap = overlap_history[-1] if overlap_history else math.inf
            overlap_history.append(max_overlap)

            if max_overlap <= target_overlap:
                stagnation_count = 0
            elif max_overlap < prev_overlap - 1e-8 * self.d_eff:
                stagnation_count = 0
            else:
                stagnation_count += 1
                if stagnation_count >= 15:
                    effective_tau *= 0.8
                    effective_tau_rotate *= 0.8
                    stagnation_count = 0

            if verbose:
                if it % 20 == 0 or it == 1:
                    print(
                        f"\r|  +- Relax {it:4d}/{max_iter_relax} "
                        f"| Solap={max_overlap:.4e} "
                        f"| tau={effective_tau:.3f} | A_err={A_err:.4e}      ",
                        end="",
                        flush=True,
                    )

            if (
                max_overlap > target_overlap
                and effective_tau < (0.01 if self.fast_mode else 0.005)
            ):
                if verbose:
                    print(
                        f"\r|  +- Relax {it:4d}/{max_iter_relax} "
                        f"| Solap={max_overlap:.4e} | A_err={A_err:.4e} (tau agotado)    "
                    )
                return _relax_result(it, max_overlap, A_err)

        if verbose:
            print(
                f"\r|  +- Relax {max_iter_relax:4d}/{max_iter_relax} "
                f"| Solap={max_overlap:.4e} | A_err={A_err:.4e} (Limite Iters) "
            )
        return _relax_result(
            max_iter_relax,
            self.measure_max_overlap(),
            self.orientation_error() if apply_orientation else A_err,
        )

    def _apply_pair_forces_vec(
        self,
        pairs: Array,
        center_updates: Array,
        rot_updates: Array,
        effective_tau: float,
        tau_rotate_override: Optional[float] = None,
        force_tolerance: Optional[float] = None,
    ) -> float:
        Lx, Ly, Lz = self.cell_size
        eps_inv = 1.0 / max(self._rot_scale, 1e-12)
        pair_cull_sq = self._cull_radius ** 2
        return _apply_pair_forces_numba(
            pairs, self.centers, self.directions,
            center_updates, rot_updates,
            Lx, Ly, Lz, self.fiber_length, self.d_eff, pair_cull_sq,
            effective_tau,
            self.tol_overlap if force_tolerance is None else float(force_tolerance),
            self.tau_rotate if tau_rotate_override is None else float(tau_rotate_override),
            eps_inv,
        )

    def _apply_orientation_correction_vec(self, A_delta: Array, rot_updates: Array, scale: float = 1.0) -> None:
        dirs = self.directions
        raw = np.einsum("ij,nj->ni", A_delta, dirs)
        pp_dot = np.sum(dirs * raw, axis=1, keepdims=True)
        projected = raw - dirs * pp_dot
        rot_updates += scale * self.w_orient * self.fiber_length * projected

    def build_neighbor_pairs_vec(self) -> Array:
        Lx, Ly, Lz = self.cell_size
        return _build_pairs_center_cull_numba(
            self.centers,
            Lx,
            Ly,
            Lz,
            self._cull_radius ** 2,
        )

    def compute_A4_vec(self) -> Array:
        if self._n_fibers == 0:
            return np.zeros((3, 3, 3, 3), dtype=float)
        dirs = self.directions
        outer2 = np.einsum('ni,nj->nij', dirs, dirs)
        A4 = np.einsum('nij,nkl->ijkl', outer2, outer2) / self._n_fibers
        return A4

    @staticmethod
    def relative_tensor_error(A: Array, B: Array) -> float:
        num = np.linalg.norm((A - B).ravel())
        den = np.linalg.norm(B.ravel()) + 1e-16
        return float(num / den)

    @staticmethod
    def closure_orthotropic_A4(A2: Array) -> Array:
        vals, vecs = np.linalg.eigh(A2)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        a1, a2, a3 = vals
        A4p = np.zeros((3, 3, 3, 3), dtype=float)

        A4p[0, 0, 0, 0] = a1 * a1
        A4p[1, 1, 1, 1] = a2 * a2
        A4p[2, 2, 2, 2] = a3 * a3

        pairs = [
            ((0, 0, 1, 1), a1 * a2),
            ((0, 0, 2, 2), a1 * a3),
            ((1, 1, 2, 2), a2 * a3),
        ]
        for (i, j, k, l), val in pairs:
            for idx in set([
                (i, j, k, l), (i, j, l, k), (j, i, k, l), (j, i, l, k),
                (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)
            ]):
                A4p[idx] = val

        Q = vecs
        A4 = np.einsum('ia,jb,kc,ld,abcd->ijkl', Q, Q, Q, Q, A4p)
        return A4

    @staticmethod
    def rotational_scale(length: float, diameter: float) -> float:
        a = length / diameter
        return diameter**2 * (2.0/3.0 + a * (a**3/12.0 + a**2/6.0 + 3.0*a/16.0 + 1.0/15.0))

    @staticmethod
    def _sanitize_A2(A2: Array) -> Array:
        A2 = 0.5 * (A2 + A2.T)
        vals, vecs = np.linalg.eigh(A2)
        vals = np.clip(vals, 1e-8, None)
        vals /= np.sum(vals)
        A2 = vecs @ np.diag(vals) @ vecs.T
        A2 = 0.5 * (A2 + A2.T)
        return A2
