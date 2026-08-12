"""Two-kernel Schur estimator with voxel-independent online complexity."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import json
from pathlib import Path
import shutil
import sysconfig
import tempfile
import time
from typing import Callable, Sequence

import numpy as np
from scipy import fft as scipy_fft
from scipy.linalg import eigvalsh


MANDEL_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
MANDEL_FACTORS = np.array((1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)))
KERNEL_NAMES = ("transverse", "longitudinal")
_CUPY_FFT_RUNTIME_READY = False


def _cupy_fft_runtime() -> object:
    """Import CuPy after exposing CUDA wheels installed inside a virtualenv.

    Some virtualenvs contain ``nvidia-cufft-cu12`` but their library directory
    is not present in the process loader path.  Loading its dependencies with
    ``RTLD_GLOBAL`` makes the fix local to this Python process and avoids a
    machine-wide linker or driver change.
    """
    global _CUPY_FFT_RUNTIME_READY
    if not _CUPY_FFT_RUNTIME_READY:
        # Use the active environment's purelib only.  Mixing user-site CUDA
        # wheels with the virtualenv wheel can silently cross CUDA ABIs.
        root = Path(sysconfig.get_path("purelib")) / "nvidia"
        if root.is_dir():
            # CuPy is installed as a CUDA 12 wheel here.  The separately
            # installed CUDA 13 package must not be mixed into that ABI.
            for directory in sorted(root.glob("*/lib")):
                if directory.parent.name == "cu13":
                    continue
                for library in sorted(directory.glob("lib*.so*")):
                    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        _CUPY_FFT_RUNTIME_READY = True

    import cupy as cp
    # Resolve cuFFT now, so ``auto`` can fall back before a campaign begins.
    from cupy.cuda import cufft  # noqa: F401
    return cp


def _normalize_fft_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    if value not in {"scipy", "cupy", "auto"}:
        raise ValueError("fft_backend must be 'scipy', 'cupy', or 'auto'.")
    if value != "auto":
        return value
    try:
        _cupy_fft_runtime()
    except Exception:
        return "scipy"
    return "cupy"


def _half_spectrum_is_exact(shape: tuple[int, int, int]) -> bool:
    # With the signed-Nyquist convention used by the Galerkin projector, an
    # even leading axis breaks Hermitian symmetry of the directional kernel.
    return all(int(size) % 2 == 1 for size in shape[:-1])


def _frequency_unit_vectors(
    shape: tuple[int, int, int],
    *,
    half_spectrum: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies = [np.fft.fftfreq(size) for size in shape]
    if half_spectrum:
        frequencies[-1] = np.fft.rfftfreq(shape[-1])
    grids = np.meshgrid(*frequencies, indexing="ij")
    frequencies = np.asarray(grids, dtype=np.float64)
    norms = np.sqrt(np.sum(frequencies * frequencies, axis=0))
    nonzero = norms > 0.0
    directions = np.zeros_like(frequencies)
    directions[:, nonzero] = frequencies[:, nonzero] / norms[nonzero]
    return directions, nonzero


def _mandel_loads_to_tensor(fields: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    flat = np.asarray(fields)
    if flat.ndim != 3 or flat.shape[1] != 6 or flat.shape[2] != int(np.prod(shape)):
        raise ValueError("fields must have shape (n_fields, 6, nvox).")
    tensor = np.zeros((flat.shape[0], 3, 3) + shape, dtype=flat.dtype)
    for component, (ii, jj) in enumerate(MANDEL_PAIRS):
        values = (flat[:, component] / MANDEL_FACTORS[component]).reshape(
            (flat.shape[0],) + shape
        )
        tensor[:, ii, jj] = values
        tensor[:, jj, ii] = values
    return tensor


def traction_kernel_features(
    stress_fields: np.ndarray,
    shape: tuple[int, int, int],
    *,
    directions: np.ndarray | None = None,
    nonzero: np.ndarray | None = None,
    half_spectrum: bool = True,
    fft_workers: int = 1,
    fft_backend: str = "scipy",
) -> tuple[np.ndarray, np.ndarray]:
    """Return transverse/longitudinal Fourier traction features.

    For a compatible projected residual, its traction is identical to that of
    the unprojected stress. Parseval then gives the reference-inverse energy as
    ``<t_perp,t_perp>/mu0 + <t_parallel,t_parallel>/(lambda0+2 mu0)``.
    """
    half_spectrum = bool(half_spectrum and _half_spectrum_is_exact(shape))
    if directions is None or nonzero is None:
        directions, nonzero = _frequency_unit_vectors(
            shape, half_spectrum=half_spectrum
        )
    flat = np.asarray(stress_fields)
    if flat.ndim != 3 or flat.shape[1:] != (6, int(np.prod(shape))):
        raise ValueError("stress_fields must have shape (n_fields, 6, nvox).")
    mandel = flat.reshape((flat.shape[0], 6) + shape)
    backend = _normalize_fft_backend(fft_backend)
    if backend == "cupy":
        try:
            cp = _cupy_fft_runtime()
        except Exception as error:
            raise RuntimeError(
                "fft_backend='cupy' requires a working CuPy/CUDA installation."
            ) from error

        gpu_mandel = cp.asarray(mandel)
        gpu_directions = cp.asarray(directions)
        gpu_nonzero = cp.asarray(nonzero)
        transform = cp.fft.rfftn if half_spectrum else cp.fft.fftn
        fourier = transform(gpu_mandel, axes=(2, 3, 4))
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        traction = cp.empty(
            (flat.shape[0], 3) + tuple(directions.shape[1:]), dtype=cp.complex128
        )
        traction[:, 0] = (
            fourier[:, 0] * gpu_directions[0]
            + inv_sqrt2 * fourier[:, 5] * gpu_directions[1]
            + inv_sqrt2 * fourier[:, 4] * gpu_directions[2]
        )
        traction[:, 1] = (
            inv_sqrt2 * fourier[:, 5] * gpu_directions[0]
            + fourier[:, 1] * gpu_directions[1]
            + inv_sqrt2 * fourier[:, 3] * gpu_directions[2]
        )
        traction[:, 2] = (
            inv_sqrt2 * fourier[:, 4] * gpu_directions[0]
            + inv_sqrt2 * fourier[:, 3] * gpu_directions[1]
            + fourier[:, 2] * gpu_directions[2]
        )
        longitudinal = cp.einsum("bixyz,ixyz->bxyz", traction, gpu_directions)
        transverse = traction - gpu_directions[None, ...] * longitudinal[:, None, ...]
        mask = gpu_nonzero[None, None, ...]
        transverse *= mask
        longitudinal *= gpu_nonzero[None, ...]
        if half_spectrum:
            weights = cp.full(gpu_directions.shape[-1], np.sqrt(2.0), dtype=cp.float64)
            weights[0] = 1.0
            if shape[-1] % 2 == 0:
                weights[-1] = 1.0
            transverse *= weights[None, None, None, None, :]
            longitudinal *= weights[None, None, None, :]
        return (
            cp.asnumpy(transverse.reshape(transverse.shape[0], -1)),
            cp.asnumpy(longitudinal.reshape(longitudinal.shape[0], -1)),
        )

    transform = scipy_fft.rfftn if half_spectrum else scipy_fft.fftn
    fourier = transform(
        mandel,
        axes=(2, 3, 4),
        workers=max(1, int(fft_workers)),
    )
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    traction = np.empty(
        (flat.shape[0], 3) + tuple(directions.shape[1:]),
        dtype=np.complex128,
    )
    traction[:, 0] = (
        fourier[:, 0] * directions[0]
        + inv_sqrt2 * fourier[:, 5] * directions[1]
        + inv_sqrt2 * fourier[:, 4] * directions[2]
    )
    traction[:, 1] = (
        inv_sqrt2 * fourier[:, 5] * directions[0]
        + fourier[:, 1] * directions[1]
        + inv_sqrt2 * fourier[:, 3] * directions[2]
    )
    traction[:, 2] = (
        inv_sqrt2 * fourier[:, 4] * directions[0]
        + inv_sqrt2 * fourier[:, 3] * directions[1]
        + fourier[:, 2] * directions[2]
    )
    longitudinal = np.einsum(
        "bixyz,ixyz->bxyz", traction, directions, optimize=True
    )
    transverse = traction - directions[None, ...] * longitudinal[:, None, ...]
    transverse[:, :, ~nonzero] = 0.0
    longitudinal[:, ~nonzero] = 0.0
    if half_spectrum:
        weights = np.full(directions.shape[-1], np.sqrt(2.0), dtype=np.float64)
        weights[0] = 1.0
        if shape[-1] % 2 == 0:
            weights[-1] = 1.0
        transverse *= weights[None, None, None, None, :]
        longitudinal *= weights[None, None, None, :]
    return (
        transverse.reshape(transverse.shape[0], -1),
        longitudinal.reshape(longitudinal.shape[0], -1),
    )


def _blocked_gram(
    left: np.ndarray,
    right: np.ndarray,
    *,
    feature_block: int,
) -> np.ndarray:
    if left.shape[1] != right.shape[1]:
        raise ValueError("Feature dimensions do not match.")
    gram = np.zeros((left.shape[0], right.shape[0]), dtype=np.complex128)
    compensation = np.zeros_like(gram)
    for start in range(0, left.shape[1], int(feature_block)):
        stop = min(start + int(feature_block), left.shape[1])
        partial = left[:, start:stop] @ np.conjugate(right[:, start:stop]).T
        corrected = partial - compensation
        updated = gram + corrected
        compensation = (updated - gram) - corrected
        gram = updated
    return gram


def compile_two_kernel_contractions(
    *,
    output_path: Path,
    shape: tuple[int, int, int],
    basis_fields: np.ndarray,
    coefficient_names: Sequence[str],
    affine_stress_batch: Callable[[int, np.ndarray], np.ndarray],
    atom_batch_size: int = 6,
    feature_block: int = 131_072,
    half_spectrum: bool = True,
    fft_workers: int = 1,
    fft_backend: str = "scipy",
) -> dict[str, float | int | str]:
    """Compile BB/BK/KK contractions and save ``online_estimator.npz``.

    ``affine_stress_batch(q, strains)`` applies affine stiffness atom ``q`` to
    strain fields shaped ``(batch, 6, nvox)``. Temporary Fourier features are
    isolated below the output directory and removed after successful assembly.
    """
    started = time.perf_counter()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(value) for value in shape)
    nvox = int(np.prod(shape))
    half_spectrum = bool(half_spectrum and _half_spectrum_is_exact(shape))
    basis = np.asarray(basis_fields)
    if basis.ndim != 3 or basis.shape[1:] != (6, nvox):
        raise ValueError("basis_fields must have shape (rank, 6, nvox).")
    rank = int(basis.shape[0])
    q_count = int(len(coefficient_names))
    atom_count = 6 + rank
    directions, nonzero = _frequency_unit_vectors(
        shape, half_spectrum=half_spectrum
    )
    frequency_count = int(np.prod(directions.shape[1:]))
    work_dir = Path(tempfile.mkdtemp(prefix=".online_estimator_", dir=output_path.parent))

    feature_paths: list[tuple[Path, Path]] = []
    try:
        for q in range(q_count):
            transverse_path = work_dir / f"q{q:02d}_transverse.npy"
            longitudinal_path = work_dir / f"q{q:02d}_longitudinal.npy"
            transverse_map = np.lib.format.open_memmap(
                transverse_path,
                mode="w+",
                dtype=np.complex128,
                shape=(atom_count, 3 * frequency_count),
            )
            longitudinal_map = np.lib.format.open_memmap(
                longitudinal_path,
                mode="w+",
                dtype=np.complex128,
                shape=(atom_count, frequency_count),
            )
            for start in range(0, atom_count, max(1, int(atom_batch_size))):
                stop = min(start + max(1, int(atom_batch_size)), atom_count)
                strains = np.zeros((stop - start, 6, nvox), dtype=np.float64)
                for local, atom_id in enumerate(range(start, stop)):
                    if atom_id < 6:
                        strains[local, atom_id] = 1.0
                    else:
                        strains[local] = basis[atom_id - 6]
                stress = np.asarray(affine_stress_batch(q, strains), dtype=np.float64)
                if stress.shape != strains.shape:
                    raise ValueError(
                        f"affine_stress_batch returned {stress.shape}, expected {strains.shape}."
                    )
                transverse, longitudinal = traction_kernel_features(
                    stress,
                    shape,
                    directions=directions,
                    nonzero=nonzero,
                    half_spectrum=half_spectrum,
                    fft_workers=fft_workers,
                    fft_backend=fft_backend,
                )
                transverse_map[start:stop] = transverse
                longitudinal_map[start:stop] = longitudinal
            transverse_map.flush()
            longitudinal_map.flush()
            del transverse_map, longitudinal_map
            feature_paths.append((transverse_path, longitudinal_path))

        bb = np.zeros((2, q_count, q_count, 6, 6), dtype=np.float64)
        bk = np.zeros((2, q_count, q_count, 6, rank), dtype=np.float64)
        kk = np.zeros((2, q_count, q_count, rank, rank), dtype=np.float64)
        normalization = float(nvox) ** 2
        imaginary_norm_squared = 0.0
        gram_norm_squared = 0.0
        for p in range(q_count):
            left_features = [np.load(path, mmap_mode="r") for path in feature_paths[p]]
            for q in range(p, q_count):
                right_features = [np.load(path, mmap_mode="r") for path in feature_paths[q]]
                for kernel_id, (left, right) in enumerate(zip(left_features, right_features)):
                    gram = _blocked_gram(left, right, feature_block=int(feature_block))
                    imaginary_norm_squared += float(np.linalg.norm(gram.imag)) ** 2
                    gram_norm_squared += float(np.linalg.norm(gram)) ** 2
                    gram_real = gram.real / normalization
                    bb[kernel_id, p, q] = gram_real[:6, :6]
                    bk[kernel_id, p, q] = gram_real[:6, 6:]
                    kk[kernel_id, p, q] = gram_real[6:, 6:]
                    if p != q:
                        transpose = gram_real.T
                        bb[kernel_id, q, p] = transpose[:6, :6]
                        bk[kernel_id, q, p] = transpose[:6, 6:]
                        kk[kernel_id, q, p] = transpose[6:, 6:]

        np.savez_compressed(
            output_path,
            BB=bb,
            BK=bk,
            KK=kk,
            coefficient_names=np.asarray(tuple(coefficient_names)),
            kernel_names=np.asarray(KERNEL_NAMES),
            basis_rank=np.asarray(rank, dtype=np.int64),
            nvox=np.asarray(nvox, dtype=np.int64),
            grid_shape=np.asarray(shape, dtype=np.int64),
            format_version=np.asarray(1, dtype=np.int64),
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "basis_rank": rank,
        "coefficient_count": q_count,
        "nvox": nvox,
        "half_spectrum": bool(half_spectrum),
        "fft_workers": int(fft_workers),
        "fft_backend": _normalize_fft_backend(fft_backend),
        "global_imaginary_ratio": float(
            np.sqrt(imaginary_norm_squared)
            / max(np.sqrt(gram_norm_squared), np.finfo(float).tiny)
        ),
        "assembly_wall_s": float(time.perf_counter() - started),
        "output_path": str(output_path),
    }


class IncrementalTwoKernelCompiler:
    """Append-only compiler for nested reduced spaces.

    Fourier traction features are stored once per affine atom and basis vector.
    Appending a tangential block computes only its new ``BK`` columns and the
    new rows/columns of ``KK``. The resulting contractions are algebraically
    identical to a fresh call to :func:`compile_two_kernel_contractions`.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        work_dir: Path,
        shape: tuple[int, int, int],
        coefficient_names: Sequence[str],
        affine_stress_batch: Callable[[int, np.ndarray], np.ndarray],
        max_rank: int,
        atom_batch_size: int = 6,
        feature_block: int = 131_072,
        sketch_size: int | None = None,
        sketch_seed: int = 20260818,
        half_spectrum: bool = True,
        fft_workers: int = 1,
        fft_backend: str = "scipy",
        resume: bool = False,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.state_path = self.work_dir / "state.json"
        self.shape = tuple(int(value) for value in shape)
        self.nvox = int(np.prod(self.shape))
        self.coefficient_names = tuple(str(value) for value in coefficient_names)
        self.q_count = len(self.coefficient_names)
        self.affine_stress_batch = affine_stress_batch
        self.max_rank = int(max_rank)
        self.atom_batch_size = max(1, int(atom_batch_size))
        self.feature_block = max(1, int(feature_block))
        self.sketch_size = None if sketch_size is None else int(sketch_size)
        self.sketch_seed = int(sketch_seed)
        self.half_spectrum = bool(half_spectrum)
        self.fft_workers = max(1, int(fft_workers))
        self.fft_backend = _normalize_fft_backend(fft_backend)
        if self.max_rank < 1:
            raise ValueError("max_rank must be positive.")
        if self.q_count < 1:
            raise ValueError("coefficient_names must not be empty.")
        if self.sketch_size is not None and self.sketch_size < 1:
            raise ValueError("sketch_size must be positive when provided.")
        if resume and self.state_path.is_file():
            stored = json.loads(self.state_path.read_text(encoding="utf-8"))
            # States written before the real-FFT optimization contain the full
            # complex spectrum and remain resumable without numerical changes.
            if "half_spectrum" not in stored:
                self.half_spectrum = False
        self.half_spectrum = bool(
            self.half_spectrum and _half_spectrum_is_exact(self.shape)
        )
        self.directions, self.nonzero = _frequency_unit_vectors(
            self.shape, half_spectrum=self.half_spectrum
        )
        frequency_count = int(np.prod(self.directions.shape[1:]))
        self.full_feature_sizes = (3 * frequency_count, frequency_count)
        self.feature_indices: list[np.ndarray | None] = []
        self.feature_scales: list[float] = []
        for kernel, full_size in enumerate(self.full_feature_sizes):
            if self.sketch_size is None or self.sketch_size >= full_size:
                self.feature_indices.append(None)
                self.feature_scales.append(1.0)
            else:
                rng = np.random.default_rng(self.sketch_seed + kernel)
                indices = np.sort(
                    rng.choice(full_size, size=self.sketch_size, replace=False)
                )
                self.feature_indices.append(indices)
                self.feature_scales.append(
                    float(np.sqrt(full_size / float(self.sketch_size)))
                )

        if resume:
            self._resume()
        else:
            self._initialize()
        self._open_memmaps()

    @property
    def rank(self) -> int:
        return int(self._rank)

    def _feature_path(self, q: int, kernel: int) -> Path:
        return self.work_dir / f"q{q:02d}_{KERNEL_NAMES[kernel]}.npy"

    def _contraction_path(self, name: str) -> Path:
        return self.work_dir / f"{name}.npy"

    def _write_state(self) -> None:
        payload = {
            "format_version": self.FORMAT_VERSION,
            "shape": list(self.shape),
            "coefficient_names": list(self.coefficient_names),
            "max_rank": self.max_rank,
            "rank": self.rank,
            "atom_batch_size": self.atom_batch_size,
            "feature_block": self.feature_block,
            "sketch_size": self.sketch_size,
            "sketch_seed": self.sketch_seed,
            "half_spectrum": self.half_spectrum,
            "fft_backend": self.fft_backend,
        }
        self.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _initialize(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=False)
        atom_capacity = 6 + self.max_rank
        feature_sizes = tuple(
            full_size if indices is None else len(indices)
            for full_size, indices in zip(
                self.full_feature_sizes, self.feature_indices, strict=True
            )
        )
        for q in range(self.q_count):
            for kernel, feature_size in enumerate(feature_sizes):
                np.lib.format.open_memmap(
                    self._feature_path(q, kernel),
                    mode="w+",
                    dtype=np.complex128,
                    shape=(atom_capacity, feature_size),
                ).flush()
        np.lib.format.open_memmap(
            self._contraction_path("BB"),
            mode="w+",
            dtype=np.float64,
            shape=(2, self.q_count, self.q_count, 6, 6),
        ).flush()
        np.lib.format.open_memmap(
            self._contraction_path("BK"),
            mode="w+",
            dtype=np.float64,
            shape=(2, self.q_count, self.q_count, 6, self.max_rank),
        ).flush()
        np.lib.format.open_memmap(
            self._contraction_path("KK"),
            mode="w+",
            dtype=np.float64,
            shape=(2, self.q_count, self.q_count, self.max_rank, self.max_rank),
        ).flush()
        self._rank = 0
        self._write_state()

    def _resume(self) -> None:
        if not self.state_path.is_file():
            raise FileNotFoundError(f"Missing incremental compiler state: {self.state_path}")
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        expected = {
            "format_version": self.FORMAT_VERSION,
            "shape": list(self.shape),
            "coefficient_names": list(self.coefficient_names),
            "max_rank": self.max_rank,
        }
        if "half_spectrum" in state:
            expected["half_spectrum"] = self.half_spectrum
        if self.sketch_size is not None:
            expected["sketch_size"] = self.sketch_size
            expected["sketch_seed"] = self.sketch_seed
        elif state.get("sketch_size") is not None:
            raise ValueError("Cannot resume a sketched compiler as an exact compiler.")
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"Incremental compiler state mismatch for {key}: "
                    f"{state.get(key)!r} != {value!r}."
                )
        self._rank = int(state["rank"])
        if not 0 <= self._rank <= self.max_rank:
            raise ValueError(f"Invalid stored rank {self._rank}.")

    def _open_memmaps(self) -> None:
        self.features = [
            [np.load(self._feature_path(q, kernel), mmap_mode="r+") for kernel in range(2)]
            for q in range(self.q_count)
        ]
        self.BB = np.load(self._contraction_path("BB"), mmap_mode="r+")
        self.BK = np.load(self._contraction_path("BK"), mmap_mode="r+")
        self.KK = np.load(self._contraction_path("KK"), mmap_mode="r+")
        if self.rank == 0:
            self._compile_macroscopic_atoms()

    def _features_for_strains(self, q: int, strains: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        stress = np.asarray(self.affine_stress_batch(q, strains), dtype=np.float64)
        if stress.shape != strains.shape:
            raise ValueError(
                f"affine_stress_batch returned {stress.shape}, expected {strains.shape}."
            )
        features = traction_kernel_features(
            stress,
            self.shape,
            directions=self.directions,
            nonzero=self.nonzero,
            half_spectrum=self.half_spectrum,
            fft_workers=self.fft_workers,
            fft_backend=self.fft_backend,
        )
        sampled: list[np.ndarray] = []
        for kernel, values in enumerate(features):
            indices = self.feature_indices[kernel]
            if indices is None:
                sampled.append(values)
            else:
                sampled.append(
                    values[:, indices] * self.feature_scales[kernel]
                )
        return sampled[0], sampled[1]

    def _compile_macroscopic_atoms(self) -> None:
        strains = np.zeros((6, 6, self.nvox), dtype=np.float64)
        strains[np.arange(6), np.arange(6)] = 1.0
        for q in range(self.q_count):
            transverse, longitudinal = self._features_for_strains(q, strains)
            self.features[q][0][:6] = transverse
            self.features[q][1][:6] = longitudinal
            self.features[q][0].flush()
            self.features[q][1].flush()

        normalization = float(self.nvox) ** 2
        for p in range(self.q_count):
            for q in range(p, self.q_count):
                for kernel in range(2):
                    gram = _blocked_gram(
                        self.features[p][kernel][:6],
                        self.features[q][kernel][:6],
                        feature_block=self.feature_block,
                    ).real / normalization
                    self.BB[kernel, p, q] = gram
                    self.BB[kernel, q, p] = gram.T
        self.BB.flush()
        self._write_state()

    def append(self, basis_fields: np.ndarray) -> dict[str, float | int]:
        started = time.perf_counter()
        new_basis = np.asarray(basis_fields, dtype=np.float64)
        if new_basis.ndim != 3 or new_basis.shape[1:] != (6, self.nvox):
            raise ValueError("basis_fields must have shape (new_rank, 6, nvox).")
        count = int(new_basis.shape[0])
        if count == 0:
            return {"old_rank": self.rank, "new_rank": self.rank, "append_wall_s": 0.0}
        old_rank = self.rank
        new_rank = old_rank + count
        if new_rank > self.max_rank:
            raise ValueError(
                f"Appending {count} fields would exceed max_rank={self.max_rank}."
            )

        feature_start = 6 + old_rank
        feature_stop = 6 + new_rank
        feature_started = time.perf_counter()
        for q in range(self.q_count):
            for start in range(0, count, self.atom_batch_size):
                stop = min(start + self.atom_batch_size, count)
                transverse, longitudinal = self._features_for_strains(q, new_basis[start:stop])
                self.features[q][0][feature_start + start : feature_start + stop] = transverse
                self.features[q][1][feature_start + start : feature_start + stop] = longitudinal
            self.features[q][0].flush()
            self.features[q][1].flush()
        feature_wall_s = float(time.perf_counter() - feature_started)

        normalization = float(self.nvox) ** 2
        new_slice = slice(feature_start, feature_stop)
        basis_new_slice = slice(old_rank, new_rank)
        atom_count = 6 + new_rank
        # Fuse all affine (p,q) products into one rectangular Gram per
        # frequency block.  This preserves compensated accumulation while
        # reading each memmapped feature row once instead of once per pair.
        fused_block = min(int(self.feature_block), 32_768)
        contraction_started = time.perf_counter()
        for kernel in range(2):
            feature_count = int(self.features[0][kernel].shape[1])
            gram = np.zeros(
                (self.q_count * atom_count, self.q_count * count),
                dtype=np.complex128,
            )
            compensation = np.zeros_like(gram)
            for block_start in range(0, feature_count, fused_block):
                block_stop = min(block_start + fused_block, feature_count)
                left = np.concatenate(
                    [
                        np.asarray(
                            self.features[q][kernel][
                                :atom_count, block_start:block_stop
                            ]
                        )
                        for q in range(self.q_count)
                    ],
                    axis=0,
                )
                right = np.concatenate(
                    [
                        np.asarray(
                            self.features[q][kernel][
                                new_slice, block_start:block_stop
                            ]
                        )
                        for q in range(self.q_count)
                    ],
                    axis=0,
                )
                partial = left @ np.conjugate(right).T
                corrected = partial - compensation
                updated = gram + corrected
                compensation = (updated - gram) - corrected
                gram = updated
            columns = (
                gram.real.reshape(
                    self.q_count,
                    atom_count,
                    self.q_count,
                    count,
                )
                / normalization
            )
            self.BK[kernel, :, :, :, basis_new_slice] = np.transpose(
                columns[:, :6], (0, 2, 1, 3)
            )
            kk_columns = columns[:, 6:]
            self.KK[kernel, :, :, :new_rank, basis_new_slice] = np.transpose(
                kk_columns, (0, 2, 1, 3)
            )
            for p in range(self.q_count):
                for q in range(self.q_count):
                    self.KK[
                        kernel, p, q, basis_new_slice, :new_rank
                    ] = kk_columns[q, :, p, :].T
        contraction_wall_s = float(time.perf_counter() - contraction_started)

        self.BK.flush()
        self.KK.flush()
        self._rank = new_rank
        self._write_state()
        return {
            "old_rank": old_rank,
            "appended_rank": count,
            "new_rank": new_rank,
            "feature_wall_s": feature_wall_s,
            "contraction_wall_s": contraction_wall_s,
            "fused_feature_block": int(fused_block),
            "append_wall_s": float(time.perf_counter() - started),
        }

    def estimator(self) -> "TwoKernelEstimator":
        rank = self.rank
        return TwoKernelEstimator(
            np.asarray(self.BB, dtype=np.float64).copy(),
            np.asarray(self.BK[..., :rank], dtype=np.float64).copy(),
            np.asarray(self.KK[..., :rank, :rank], dtype=np.float64).copy(),
            self.coefficient_names,
        )

    def save(self, output_path: Path) -> None:
        estimator = self.estimator()
        np.savez_compressed(
            Path(output_path),
            BB=estimator.BB,
            BK=estimator.BK,
            KK=estimator.KK,
            coefficient_names=np.asarray(self.coefficient_names),
            kernel_names=np.asarray(KERNEL_NAMES),
            basis_rank=np.asarray(self.rank, dtype=np.int64),
            nvox=np.asarray(self.nvox, dtype=np.int64),
            grid_shape=np.asarray(self.shape, dtype=np.int64),
            format_version=np.asarray(2, dtype=np.int64),
        )

    def close(self) -> None:
        for maps in getattr(self, "features", []):
            for array in maps:
                array.flush()
        for name in ("BB", "BK", "KK"):
            array = getattr(self, name, None)
            if array is not None:
                array.flush()

    def cleanup(self) -> None:
        self.close()
        self.features = []
        self.BB = self.BK = self.KK = None
        shutil.rmtree(self.work_dir, ignore_errors=True)


@dataclass(frozen=True)
class TwoKernelEstimator:
    BB: np.ndarray
    BK: np.ndarray
    KK: np.ndarray
    coefficient_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        bb = np.asarray(self.BB, dtype=np.float64)
        bk = np.asarray(self.BK, dtype=np.float64)
        kk = np.asarray(self.KK, dtype=np.float64)
        if bb.ndim != 5 or bb.shape[0] != 2 or bb.shape[-2:] != (6, 6):
            raise ValueError("BB must have shape (2, Q, Q, 6, 6).")
        q_count = bb.shape[1]
        if bb.shape[2] != q_count or bk.shape[:3] != (2, q_count, q_count):
            raise ValueError("Inconsistent BB/BK coefficient dimensions.")
        rank = bk.shape[-1]
        if bk.shape[-2] != 6 or kk.shape != (2, q_count, q_count, rank, rank):
            raise ValueError("Inconsistent BK/KK reduced dimensions.")
        object.__setattr__(self, "BB", bb)
        object.__setattr__(self, "BK", bk)
        object.__setattr__(self, "KK", kk)

    @classmethod
    def load(cls, path: Path) -> "TwoKernelEstimator":
        with np.load(path) as payload:
            names = tuple(str(value) for value in payload.get("coefficient_names", ()))
            return cls(payload["BB"], payload["BK"], payload["KK"], names)

    @property
    def rank(self) -> int:
        return int(self.BK.shape[-1])

    def kernel_energy_matrices(
        self,
        coefficients: np.ndarray,
        amplitudes: np.ndarray,
    ) -> np.ndarray:
        coefficients = np.asarray(coefficients, dtype=np.float64)
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
        q_count = self.BB.shape[1]
        if coefficients.shape != (q_count,):
            raise ValueError(f"coefficients must have shape ({q_count},).")
        if amplitudes.shape != (self.rank, 6):
            raise ValueError(f"amplitudes must have shape ({self.rank}, 6).")
        weights = np.multiply.outer(coefficients, coefficients)
        bb = np.einsum("pq,tpqlm->tlm", weights, self.BB, optimize=True)
        bk = np.einsum("pq,tpqlj->tlj", weights, self.BK, optimize=True)
        kk = np.einsum("pq,tpqij->tij", weights, self.KK, optimize=True)
        cross = np.einsum("tli,im->tlm", bk, amplitudes, optimize=True)
        reduced = np.einsum(
            "il,tij,jm->tlm", amplitudes, kk, amplitudes, optimize=True
        )
        energy = bb + cross + np.swapaxes(cross, -1, -2) + reduced
        return 0.5 * (energy + np.swapaxes(energy, -1, -2))

    def kernel_energy_matrices_batch(
        self,
        coefficients: np.ndarray,
        amplitudes: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the two kernel energies for many materials at once."""
        coefficients = np.asarray(coefficients, dtype=np.float64)
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
        if coefficients.ndim != 2 or coefficients.shape[1] != self.BB.shape[1]:
            raise ValueError(
                "coefficients must have shape (n_candidates, n_coefficients)."
            )
        if amplitudes.ndim != 3 or amplitudes.shape[1:] != (self.rank, 6):
            raise ValueError(
                "amplitudes must have shape (n_candidates, rank, 6)."
            )
        weights = np.einsum("np,nq->npq", coefficients, coefficients, optimize=True)
        bb = np.einsum("npq,tpqlm->ntlm", weights, self.BB, optimize=True)
        bk = np.einsum("npq,tpqlj->ntlj", weights, self.BK, optimize=True)
        kk = np.einsum("npq,tpqij->ntij", weights, self.KK, optimize=True)
        cross = np.einsum("ntli,nim->ntlm", bk, amplitudes, optimize=True)
        reduced = np.einsum(
            "nrl,ntrs,nsm->ntlm", amplitudes, kk, amplitudes, optimize=True
        )
        energy = bb + cross + np.swapaxes(cross, -1, -2) + reduced
        return 0.5 * (energy + np.swapaxes(energy, -1, -2))

    def kernel_cross_matrices_batch(
        self,
        coefficients_left: np.ndarray,
        amplitudes_left: np.ndarray,
        coefficients_right: np.ndarray,
        amplitudes_right: np.ndarray,
    ) -> np.ndarray:
        """Evaluate cross-material residual Gram matrices in both kernels.

        Entry ``[i, j, t]`` is ``R_i.T @ P_t @ R_j`` before division by the
        isotropic reference eigenvalues.  The contraction is entirely
        offline-online: no voxel field or FFT is needed at evaluation time.
        The matrix is intentionally not symmetrized because its transpose is
        the reverse-material product used by Schur projection scores.
        """
        coeff_left = np.asarray(coefficients_left, dtype=np.float64)
        coeff_right = np.asarray(coefficients_right, dtype=np.float64)
        amp_left = np.asarray(amplitudes_left, dtype=np.float64)
        amp_right = np.asarray(amplitudes_right, dtype=np.float64)
        q_count = self.BB.shape[1]
        rank = self.rank
        if coeff_left.ndim != 2 or coeff_left.shape[1] != q_count:
            raise ValueError(
                "coefficients_left must have shape (n_left, n_coefficients)."
            )
        if coeff_right.ndim != 2 or coeff_right.shape[1] != q_count:
            raise ValueError(
                "coefficients_right must have shape (n_right, n_coefficients)."
            )
        if amp_left.shape != (len(coeff_left), rank, 6):
            raise ValueError("amplitudes_left has an incompatible shape.")
        if amp_right.shape != (len(coeff_right), rank, 6):
            raise ValueError("amplitudes_right has an incompatible shape.")

        weights = np.einsum(
            "ip,jq->ijpq", coeff_left, coeff_right, optimize=True
        )
        bb = np.einsum(
            "ijpq,tpqlm->ijtlm", weights, self.BB, optimize=True
        )
        right_cross = np.einsum(
            "ijpq,tpqlv,jvm->ijtlm",
            weights,
            self.BK,
            amp_right,
            optimize=True,
        )
        left_cross = np.einsum(
            "ijpq,tqpmv,ivl->ijtlm",
            weights,
            self.BK,
            amp_left,
            optimize=True,
        )
        reduced = np.einsum(
            "ijpq,tpquv,iul,jvm->ijtlm",
            weights,
            self.KK,
            amp_left,
            amp_right,
            optimize=True,
        )
        return bb + right_cross + left_cross + reduced

    def cross_energy_matrices_batch(
        self,
        coefficients_left: np.ndarray,
        amplitudes_left: np.ndarray,
        coefficients_right: np.ndarray,
        amplitudes_right: np.ndarray,
        *,
        lambda0: float,
        mu0: float,
    ) -> np.ndarray:
        """Apply one fixed isotropic inverse to cross-material residuals."""
        if mu0 <= 0.0 or lambda0 + 2.0 * mu0 <= 0.0:
            raise ValueError("The isotropic reference must be SPD.")
        kernels = self.kernel_cross_matrices_batch(
            coefficients_left,
            amplitudes_left,
            coefficients_right,
            amplitudes_right,
        )
        return kernels[:, :, 0] / float(mu0) + kernels[:, :, 1] / (
            float(lambda0) + 2.0 * float(mu0)
        )

    def reference_atom_gram(self, *, lambda0: float, mu0: float) -> np.ndarray:
        """Assemble the residual-atom Gram matrix for one fixed reference.

        Atom ordering is ``(affine coefficient, macro/basis atom)``.  This
        representation exposes a factorization shared by every material and
        avoids forming all cross-material pairs explicitly.
        """
        if mu0 <= 0.0 or lambda0 + 2.0 * mu0 <= 0.0:
            raise ValueError("The isotropic reference must be SPD.")
        q_count = self.BB.shape[1]
        atom_count = 6 + self.rank
        kernels = np.zeros(
            (2, q_count, atom_count, q_count, atom_count), dtype=np.float64
        )
        for kernel in range(2):
            kernels[kernel, :, :6, :, :6] = np.transpose(
                self.BB[kernel], (0, 2, 1, 3)
            )
            upper = np.transpose(self.BK[kernel], (0, 2, 1, 3))
            kernels[kernel, :, :6, :, 6:] = upper
            kernels[kernel, :, 6:, :, :6] = np.transpose(
                upper, (2, 3, 0, 1)
            )
            kernels[kernel, :, 6:, :, 6:] = np.transpose(
                self.KK[kernel], (0, 2, 1, 3)
            )
        gram = kernels[0] / float(mu0)
        gram += kernels[1] / (float(lambda0) + 2.0 * float(mu0))
        dimension = q_count * atom_count
        gram = gram.reshape(dimension, dimension)
        return 0.5 * (gram + gram.T)

    def residual_atom_coefficients_batch(
        self,
        coefficients: np.ndarray,
        amplitudes: np.ndarray,
    ) -> np.ndarray:
        """Map material residuals to the shared affine atom coordinates."""
        coeffs = np.asarray(coefficients, dtype=np.float64)
        amps = np.asarray(amplitudes, dtype=np.float64)
        q_count = self.BB.shape[1]
        if coeffs.ndim != 2 or coeffs.shape[1] != q_count:
            raise ValueError(
                "coefficients must have shape (n_materials, n_coefficients)."
            )
        if amps.shape != (len(coeffs), self.rank, 6):
            raise ValueError("amplitudes has an incompatible shape.")
        atoms = np.zeros(
            (len(coeffs), q_count, 6 + self.rank, 6), dtype=np.float64
        )
        identity = np.eye(6, dtype=np.float64)
        atoms[:, :, :6, :] = coeffs[:, :, None, None] * identity[None, None]
        atoms[:, :, 6:, :] = coeffs[:, :, None, None] * amps[:, None]
        return atoms.reshape(len(coeffs), q_count * (6 + self.rank), 6)

    def reference_gram_factor(
        self,
        *,
        lambda0: float,
        mu0: float,
        relative_eigenvalue_tolerance: float = 1.0e-12,
        max_rank: int | None = None,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        """Return ``F`` such that the fixed-reference Gram is ``F.T @ F``.

        Setting ``max_rank`` enables a controlled spectral compression.  With
        ``max_rank=None``, only roundoff-level nonpositive eigenvalues are
        removed and the cross residual energy is retained to the requested
        relative tolerance.
        """
        tolerance = float(relative_eigenvalue_tolerance)
        if tolerance < 0.0:
            raise ValueError("relative_eigenvalue_tolerance must be non-negative.")
        gram = self.reference_atom_gram(lambda0=lambda0, mu0=mu0)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        largest = float(max(eigenvalues[-1], 0.0))
        threshold = tolerance * largest
        keep = np.flatnonzero(eigenvalues > threshold)
        if max_rank is not None:
            requested = int(max_rank)
            if requested < 1:
                raise ValueError("max_rank must be positive when provided.")
            keep = keep[-requested:]
        selected = eigenvalues[keep]
        factor = np.sqrt(selected)[:, None] * eigenvectors[:, keep].T
        positive_sum = float(np.sum(np.clip(eigenvalues, 0.0, None)))
        retained_sum = float(np.sum(selected))
        metadata: dict[str, float | int] = {
            "atom_dimension": int(len(eigenvalues)),
            "embedding_rank": int(len(keep)),
            "largest_eigenvalue": largest,
            "smallest_retained_eigenvalue": float(selected[0]) if len(selected) else 0.0,
            "most_negative_eigenvalue": float(eigenvalues[0]),
            "retained_trace_fraction": retained_sum / max(positive_sum, np.finfo(float).tiny),
            "relative_eigenvalue_tolerance": tolerance,
        }
        return factor, metadata

    def residual_embeddings_batch(
        self,
        coefficients: np.ndarray,
        amplitudes: np.ndarray,
        factor: np.ndarray,
    ) -> np.ndarray:
        """Embed many six-output residuals in a shared Gram feature space."""
        atom_coefficients = self.residual_atom_coefficients_batch(
            coefficients, amplitudes
        )
        factor = np.asarray(factor, dtype=np.float64)
        if factor.ndim != 2 or factor.shape[1] != atom_coefficients.shape[1]:
            raise ValueError("factor has an incompatible atom dimension.")
        return np.einsum(
            "kd,ndl->nkl", factor, atom_coefficients, optimize=True
        )

    @staticmethod
    def global_influence_scores_from_embeddings(
        probe_embeddings: np.ndarray,
        candidate_embeddings: np.ndarray,
        probe_weights: np.ndarray,
        *,
        pseudoinverse_rcond: float = 1.0e-12,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Score Schur gains from precomputed residual embeddings.

        This exposes the covariance condensation independently of how probes
        are weighted.  In particular, a caller can concentrate the same exact
        algebra on the predicted tail without constructing material pairs.
        """
        probes = np.asarray(probe_embeddings, dtype=np.float64)
        candidates = np.asarray(candidate_embeddings, dtype=np.float64)
        weights = np.asarray(probe_weights, dtype=np.float64)
        if probes.ndim != 3 or candidates.ndim != 3 or probes.shape[1:] != candidates.shape[1:]:
            raise ValueError("Probe and candidate embeddings must have shape (n, k, 6).")
        if weights.shape != (len(probes),):
            raise ValueError("probe_weights must contain one value per probe.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("probe_weights must be finite and non-negative.")
        if not np.any(weights > 0.0):
            raise ValueError("At least one probe weight must be positive.")
        rcond = float(pseudoinverse_rcond)
        if rcond < 0.0:
            raise ValueError("pseudoinverse_rcond must be non-negative.")

        covariance = np.einsum(
            "n,nkl,nml->km", weights, probes, probes, optimize=True
        )
        self_gram = np.einsum("nkl,nkr->nlr", candidates, candidates, optimize=True)
        covariance_action = np.einsum(
            "km,nmr->nkr", covariance, candidates, optimize=True
        )
        compressed = np.einsum(
            "nkl,nkr->nlr", candidates, covariance_action, optimize=True
        )
        inverse = np.linalg.pinv(self_gram, rcond=rcond)
        scores = np.einsum("nij,nji->n", inverse, compressed, optimize=True)
        return np.asarray(scores, dtype=np.float64), {
            "probe_count": int(len(probes)),
            "candidate_count": int(len(candidates)),
            "pair_count_avoided": int(len(probes) * len(candidates)),
        }

    def global_influence_scores_batch(
        self,
        probe_coefficients: np.ndarray,
        probe_amplitudes: np.ndarray,
        candidate_coefficients: np.ndarray,
        candidate_amplitudes: np.ndarray,
        probe_weights: np.ndarray,
        *,
        lambda0: float,
        mu0: float,
        relative_eigenvalue_tolerance: float = 1.0e-12,
        max_embedding_rank: int | None = 192,
        pseudoinverse_rcond: float = 1.0e-12,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        """Score global residual reduction without material-pair enumeration.

        For residual embeddings ``E_i`` and candidate ``E_j``, the exact
        cross energy is ``H_ij = E_i.T @ E_j``.  Condensing all probes into
        ``Sigma = sum_i w_i E_i E_i.T`` changes the explicit pair sum into

        ``trace((E_j.T E_j)^dagger E_j.T Sigma E_j)``.

        Candidate work is therefore linear in their count.  A fixed spectral
        cap also prevents the embedding dimension from growing indefinitely
        with the Ritz basis.
        """
        factor, metadata = self.reference_gram_factor(
            lambda0=lambda0,
            mu0=mu0,
            relative_eigenvalue_tolerance=relative_eigenvalue_tolerance,
            max_rank=max_embedding_rank,
        )
        probes = self.residual_embeddings_batch(
            probe_coefficients, probe_amplitudes, factor
        )
        candidates = self.residual_embeddings_batch(
            candidate_coefficients, candidate_amplitudes, factor
        )
        scores, score_metadata = self.global_influence_scores_from_embeddings(
            probes,
            candidates,
            probe_weights,
            pseudoinverse_rcond=pseudoinverse_rcond,
        )
        metadata = {
            **metadata,
            **score_metadata,
        }
        return np.asarray(scores, dtype=np.float64), metadata

    def energy_matrix(
        self,
        coefficients: np.ndarray,
        amplitudes: np.ndarray,
        *,
        lambda0: float,
        mu0: float,
    ) -> np.ndarray:
        if mu0 <= 0.0 or lambda0 + 2.0 * mu0 <= 0.0:
            raise ValueError("The isotropic reference must be SPD.")
        kernels = self.kernel_energy_matrices(coefficients, amplitudes)
        energy = kernels[0] / float(mu0)
        energy += kernels[1] / float(lambda0 + 2.0 * mu0)
        return 0.5 * (energy + energy.T)


def hierarchical_matrix_upper_bound(
    coarse_output: np.ndarray,
    enriched_output: np.ndarray,
    enriched_tail_upper_bound: np.ndarray,
    *,
    loewner_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Lift an enriched-space tail bound to a coarse Ritz-space bound.

    For nested Ritz spaces, ``coarse_output - enriched_output`` is positive
    semidefinite and known exactly from reduced solves. If ``tail_upper``
    bounds the enriched-space error, their sum bounds the coarse-space error.
    """
    matrices = tuple(
        0.5 * (np.asarray(value, dtype=np.float64) + np.asarray(value, dtype=np.float64).T)
        for value in (coarse_output, enriched_output, enriched_tail_upper_bound)
    )
    if any(value.shape != (6, 6) for value in matrices):
        raise ValueError("Hierarchical bound inputs must have shape (6, 6).")
    if any(not np.all(np.isfinite(value)) for value in matrices):
        raise ValueError("Hierarchical bound inputs must be finite.")
    coarse, enriched, tail_upper = matrices
    drop = coarse - enriched
    tolerance = float(loewner_tolerance)
    if tolerance < 0.0:
        raise ValueError("loewner_tolerance must be non-negative.")
    if float(np.min(np.linalg.eigvalsh(drop))) < -tolerance:
        raise ValueError("The supplied outputs do not form a nested Ritz hierarchy.")
    if float(np.min(np.linalg.eigvalsh(tail_upper))) < -tolerance:
        raise ValueError("The enriched-space tail bound is not positive semidefinite.")
    upper = drop + tail_upper
    return 0.5 * (upper + upper.T)


def isotropic_stiffness_mandel(lambda0: float, mu0: float) -> np.ndarray:
    if mu0 <= 0.0 or lambda0 + 2.0 * mu0 <= 0.0:
        raise ValueError("The isotropic reference must be SPD.")
    matrix = np.zeros((6, 6), dtype=np.float64)
    matrix[:3, :3] = float(lambda0)
    matrix[np.arange(3), np.arange(3)] += 2.0 * float(mu0)
    matrix[3:, 3:] = 2.0 * float(mu0) * np.eye(3)
    return matrix


def isotropic_reference_family(count: int = 129) -> np.ndarray:
    if count < 2:
        raise ValueError("count must be at least 2.")
    indices = np.arange(count, dtype=np.float64)
    nodes = np.cos(np.pi * indices / float(count - 1))
    poisson = np.sort(0.02 + 0.47 * nodes)
    mu = np.ones_like(poisson)
    lam = 2.0 * mu * poisson / (1.0 - 2.0 * poisson)
    return np.column_stack((lam, mu, poisson))


def generalized_coercivity(
    phase_stiffnesses: Sequence[np.ndarray],
    *,
    lambda0: float,
    mu0: float,
) -> float:
    reference = isotropic_stiffness_mandel(lambda0, mu0)
    values = [float(np.min(eigvalsh(np.asarray(C, dtype=np.float64), reference))) for C in phase_stiffnesses]
    beta = min(values)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"Non-positive generalized coercivity beta={beta}.")
    return beta


@dataclass(frozen=True)
class ReferenceSelection:
    index: int
    lambda0: float
    mu0: float
    poisson0: float
    beta: float
    beta_safe: float
    energy_matrix: np.ndarray
    upper_bound_matrix: np.ndarray
    objective: float


@dataclass(frozen=True)
class ReferenceEnvelopeSelection:
    weights: np.ndarray
    active_indices: np.ndarray
    upper_bound_matrix: np.ndarray
    objective: float
    best_single_index: int
    best_single_objective: float
    iterations: int
    relative_duality_gap: float


def isotropic_reference_upper_bounds(
    kernel_energy_matrices: np.ndarray,
    phase_stiffnesses: Sequence[np.ndarray],
    *,
    references: np.ndarray | None = None,
    beta_margin: float = 1.0e-10,
) -> dict[str, np.ndarray]:
    kernels = np.asarray(kernel_energy_matrices, dtype=np.float64)
    if kernels.shape != (2, 6, 6):
        raise ValueError("kernel_energy_matrices must have shape (2, 6, 6).")
    if not 0.0 <= beta_margin < 1.0:
        raise ValueError("beta_margin must lie in [0, 1).")
    family = isotropic_reference_family(129) if references is None else np.asarray(references)
    if family.ndim != 2 or family.shape[1] != 3 or family.shape[0] == 0:
        raise RuntimeError("The isotropic reference family is empty.")

    # Isotropy makes C0^{-1/2} analytic on volumetric/deviatoric subspaces,
    # so all 129 generalized spectra can be evaluated in one batched eigensolve.
    volumetric_vector = np.zeros(6, dtype=np.float64)
    volumetric_vector[:3] = 1.0 / np.sqrt(3.0)
    p_vol = np.outer(volumetric_vector, volumetric_vector)
    p_dev = np.eye(6) - p_vol
    lam = family[:, 0].astype(np.float64)
    mu = family[:, 1].astype(np.float64)
    vol_eigenvalue = 3.0 * lam + 2.0 * mu
    dev_eigenvalue = 2.0 * mu
    if np.any(vol_eigenvalue <= 0.0) or np.any(dev_eigenvalue <= 0.0):
        raise ValueError("Every reference in the family must be SPD.")
    inverse_roots = (
        p_vol[None, ...] / np.sqrt(vol_eigenvalue)[:, None, None]
        + p_dev[None, ...] / np.sqrt(dev_eigenvalue)[:, None, None]
    )
    beta_values = np.full(family.shape[0], np.inf, dtype=np.float64)
    for phase in phase_stiffnesses:
        stiffness = np.asarray(phase, dtype=np.float64)
        transformed = np.einsum(
            "nij,jk,nkl->nil", inverse_roots, stiffness, inverse_roots, optimize=True
        )
        beta_values = np.minimum(beta_values, np.linalg.eigvalsh(transformed)[:, 0])
    if np.any(~np.isfinite(beta_values)) or np.any(beta_values <= 0.0):
        raise ValueError("The generalized coercivity family is not strictly positive.")
    beta_safe_values = beta_values * (1.0 - float(beta_margin))
    energies = kernels[0][None, ...] / mu[:, None, None]
    energies += kernels[1][None, ...] / (lam + 2.0 * mu)[:, None, None]
    energies = 0.5 * (energies + np.swapaxes(energies, -1, -2))
    upper_bounds = energies / beta_safe_values[:, None, None]
    objectives = np.linalg.norm(upper_bounds, axis=(1, 2))
    return {
        "family": family,
        "lambda0": lam,
        "mu0": mu,
        "beta": beta_values,
        "beta_safe": beta_safe_values,
        "energy_matrices": energies,
        "upper_bound_matrices": upper_bounds,
        "objectives": objectives,
    }


def optimize_isotropic_reference(
    kernel_energy_matrices: np.ndarray,
    phase_stiffnesses: Sequence[np.ndarray],
    *,
    references: np.ndarray | None = None,
    beta_margin: float = 1.0e-10,
) -> ReferenceSelection:
    family_data = isotropic_reference_upper_bounds(
        kernel_energy_matrices,
        phase_stiffnesses,
        references=references,
        beta_margin=beta_margin,
    )
    family = family_data["family"]
    lam = family_data["lambda0"]
    mu = family_data["mu0"]
    beta_values = family_data["beta"]
    beta_safe_values = family_data["beta_safe"]
    energies = family_data["energy_matrices"]
    upper_bounds = family_data["upper_bound_matrices"]
    objectives = family_data["objectives"]
    index = int(np.argmin(objectives))
    return ReferenceSelection(
        index=index,
        lambda0=float(lam[index]),
        mu0=float(mu[index]),
        poisson0=float(family[index, 2]),
        beta=float(beta_values[index]),
        beta_safe=float(beta_safe_values[index]),
        energy_matrix=energies[index],
        upper_bound_matrix=upper_bounds[index],
        objective=float(objectives[index]),
    )


def optimize_isotropic_reference_batch(
    kernel_energy_matrices: np.ndarray,
    phase_stiffnesses: Sequence[np.ndarray],
    *,
    references: np.ndarray | None = None,
    beta_margin: float = 1.0e-10,
) -> dict[str, np.ndarray]:
    """Optimize the isotropic reference independently for many candidates.

    This is the batched equivalent of :func:`optimize_isotropic_reference`.
    It keeps the same 129-reference family and the same generalized
    coercivity calculation, while moving the candidate loop into NumPy's
    batched dense linear algebra.
    """
    kernels = np.asarray(kernel_energy_matrices, dtype=np.float64)
    if kernels.ndim != 4 or kernels.shape[1:] != (2, 6, 6):
        raise ValueError(
            "kernel_energy_matrices must have shape (n_candidates, 2, 6, 6)."
        )
    if not 0.0 <= beta_margin < 1.0:
        raise ValueError("beta_margin must lie in [0, 1).")

    family = isotropic_reference_family(129) if references is None else np.asarray(references)
    if family.ndim != 2 or family.shape[1] != 3 or family.shape[0] == 0:
        raise RuntimeError("The isotropic reference family is empty.")
    phase_arrays = [np.asarray(value, dtype=np.float64) for value in phase_stiffnesses]
    if not phase_arrays:
        raise ValueError("At least one phase stiffness is required.")
    candidate_count = int(kernels.shape[0])
    if any(value.shape != (candidate_count, 6, 6) for value in phase_arrays):
        raise ValueError(
            "Every phase stiffness must have shape (n_candidates, 6, 6)."
        )

    volumetric_vector = np.zeros(6, dtype=np.float64)
    volumetric_vector[:3] = 1.0 / np.sqrt(3.0)
    p_vol = np.outer(volumetric_vector, volumetric_vector)
    p_dev = np.eye(6) - p_vol
    lam = family[:, 0].astype(np.float64)
    mu = family[:, 1].astype(np.float64)
    vol_eigenvalue = 3.0 * lam + 2.0 * mu
    dev_eigenvalue = 2.0 * mu
    if np.any(vol_eigenvalue <= 0.0) or np.any(dev_eigenvalue <= 0.0):
        raise ValueError("Every reference in the family must be SPD.")
    inverse_roots = (
        p_vol[None, ...] / np.sqrt(vol_eigenvalue)[:, None, None]
        + p_dev[None, ...] / np.sqrt(dev_eigenvalue)[:, None, None]
    )

    beta_values = np.full((candidate_count, family.shape[0]), np.inf, dtype=np.float64)
    for stiffness in phase_arrays:
        transformed = np.einsum(
            "fij,njk,fkl->nfil",
            inverse_roots,
            stiffness,
            inverse_roots,
            optimize=True,
        )
        beta_values = np.minimum(beta_values, np.linalg.eigvalsh(transformed)[..., 0])
    if np.any(~np.isfinite(beta_values)) or np.any(beta_values <= 0.0):
        raise ValueError("The generalized coercivity family is not strictly positive.")

    beta_safe_values = beta_values * (1.0 - float(beta_margin))
    denominator = lam + 2.0 * mu
    energies = (
        kernels[:, 0, None, :, :] / mu[None, :, None, None]
        + kernels[:, 1, None, :, :] / denominator[None, :, None, None]
    )
    energies = 0.5 * (energies + np.swapaxes(energies, -1, -2))
    upper_bounds = energies / beta_safe_values[:, :, None, None]
    objectives = np.linalg.norm(upper_bounds, axis=(-2, -1))
    index = np.argmin(objectives, axis=1)
    rows = np.arange(candidate_count)
    return {
        "family": family,
        "index": index.astype(np.int64),
        "lambda0": lam[index],
        "mu0": mu[index],
        "poisson0": family[index, 2],
        "beta": beta_values[rows, index],
        "beta_safe": beta_safe_values[rows, index],
        "energy_matrices": energies[rows, index],
        "upper_bound_matrices": upper_bounds[rows, index],
        "objectives": objectives[rows, index],
    }


def optimize_convex_reference_envelope(
    upper_bound_matrices: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-12,
    max_iterations: int = 10_000,
) -> ReferenceEnvelopeSelection:
    """Minimize the Frobenius norm over a convex family of valid bounds.

    If every input matrix satisfies ``E <= U_j`` in Loewner order, every
    convex combination also bounds ``E``. Frank-Wolfe with exact line search
    keeps that invariant at every iteration and needs only small dense algebra.
    """
    bounds = np.asarray(upper_bound_matrices, dtype=np.float64)
    if bounds.ndim != 3 or bounds.shape[1:] != (6, 6) or len(bounds) == 0:
        raise ValueError("upper_bound_matrices must have shape (n, 6, 6).")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("upper_bound_matrices must be finite.")
    if relative_tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("The optimization tolerance and iteration limit must be positive.")
    bounds = 0.5 * (bounds + np.swapaxes(bounds, -1, -2))
    minimum_eigenvalues = np.linalg.eigvalsh(bounds)[:, 0]
    if np.min(minimum_eigenvalues) < -1.0e-12:
        raise ValueError("Every upper-bound matrix must be positive semidefinite.")

    objectives = np.linalg.norm(bounds, axis=(1, 2))
    best = int(np.argmin(objectives))
    weights = np.zeros(len(bounds), dtype=np.float64)
    weights[best] = 1.0
    current = bounds[best].copy()
    relative_gap = np.inf
    iteration = 0
    for iteration in range(1, int(max_iterations) + 1):
        inner = np.einsum("nij,ij->n", bounds, current, optimize=True)
        vertex = int(np.argmin(inner))
        current_norm_sq = float(np.sum(current * current))
        duality_gap = 2.0 * max(current_norm_sq - float(inner[vertex]), 0.0)
        relative_gap = duality_gap / max(2.0 * current_norm_sq, np.finfo(float).tiny)
        if relative_gap <= float(relative_tolerance):
            break
        direction = bounds[vertex] - current
        denominator = float(np.sum(direction * direction))
        if denominator <= np.finfo(float).tiny:
            break
        step = float(np.clip(-np.sum(current * direction) / denominator, 0.0, 1.0))
        weights *= 1.0 - step
        weights[vertex] += step
        current += step * direction

    weights = np.maximum(weights, 0.0)
    weights /= float(np.sum(weights))
    current = np.einsum("n,nij->ij", weights, bounds, optimize=True)
    current = 0.5 * (current + current.T)
    return ReferenceEnvelopeSelection(
        weights=weights,
        active_indices=np.flatnonzero(weights > 1.0e-12),
        upper_bound_matrix=current,
        objective=float(np.linalg.norm(current)),
        best_single_index=best,
        best_single_objective=float(objectives[best]),
        iterations=iteration,
        relative_duality_gap=float(relative_gap),
    )
