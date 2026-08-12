#!/usr/bin/env python3
"""Compile and validate the voxel-independent two-kernel Schur estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constitutive_transfer.schur_estimator import (
    TwoKernelEstimator,
    compile_two_kernel_contractions,
)
import schur_energy_indicators as qoi
import rom_reduced_operator as reduced


RUN_DEFAULT = (
    PROJECT_ROOT
    / "results"
    / "fixed_geometry_ffthompy"
    / "fixed_geometry_ar15_vf20_sobol8_center_fields"
)
ROM_DEFAULT = RUN_DEFAULT / "rom_tangential_r48_center_m1_m2_m3_m5_m7_v13_v3_basis"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _affine_stress_factory(
    matrix_idx: np.ndarray,
    fiber_groups: list[tuple[np.ndarray, np.ndarray]],
):
    matrix_bases = reduced._isotropic_bases()

    def apply(q: int, strains: np.ndarray) -> np.ndarray:
        stress = np.zeros_like(strains, dtype=np.float64)
        if q < 2:
            if len(matrix_idx):
                stress[:, :, matrix_idx] = np.einsum(
                    "ab,lbn->lan",
                    matrix_bases[q],
                    strains[:, :, matrix_idx],
                    optimize=True,
                )
        else:
            local_q = q - 2
            for indices, bases in fiber_groups:
                stress[:, :, indices] = np.einsum(
                    "ab,lbn->lan",
                    bases[local_q],
                    strains[:, :, indices],
                    optimize=True,
                )
        return stress

    return apply


def _direct_energy(
    *,
    coefficients: np.ndarray,
    amplitudes: np.ndarray,
    basis: np.ndarray,
    shape: tuple[int, int, int],
    matrix_idx: np.ndarray,
    fiber_groups: list[tuple[np.ndarray, np.ndarray]],
    lambda0: float,
    mu0: float,
) -> np.ndarray:
    nvox = int(np.prod(shape))
    nvec, nonzero = qoi._frequency_unit_vectors(shape)
    strains = qoi._candidate_strain_loads(amplitudes, basis)
    stress = qoi._apply_material_to_strain_batch(
        strains,
        coeffs=coefficients,
        matrix_idx=matrix_idx,
        fiber_groups=fiber_groups,
    )
    residual = qoi._project_compatible_fourier(
        stress,
        shape=shape,
        nvec=nvec,
        nonzero=nonzero,
    )
    correction = qoi._reference_inverse_fourier(
        residual,
        nvec=nvec,
        nonzero=nonzero,
        lam0=lambda0,
        mu0=mu0,
    )
    return qoi._energy_matrix_from_fourier(
        qoi._tensor_loads_to_mandel_flat(residual),
        qoi._tensor_loads_to_mandel_flat(correction),
        nvox=nvox,
    )


def _slice_estimator(estimator: TwoKernelEstimator, rank: int) -> TwoKernelEstimator:
    return TwoKernelEstimator(
        estimator.BB,
        estimator.BK[..., :rank],
        estimator.KK[..., :rank, :rank],
        estimator.coefficient_names,
    )


def _validate(
    *,
    estimator: TwoKernelEstimator,
    basis: np.ndarray,
    operators: dict[str, np.ndarray],
    shape: tuple[int, int, int],
    matrix_idx: np.ndarray,
    fiber_groups: list[tuple[np.ndarray, np.ndarray]],
    points: int,
    seed: int,
) -> list[dict[str, float | int]]:
    candidates = qoi._build_candidates(max(1, int(points)), int(seed))
    full_rank = int(basis.shape[0])
    ranks = sorted({min(full_rank, value) for value in (6, 24, full_rank) if value > 0})
    records: list[dict[str, float | int]] = []
    for rank in ranks:
        rank_estimator = _slice_estimator(estimator, rank)
        for _, row in candidates.iterrows():
            coefficients = reduced._material_coefficients(row.to_dict())
            _, amplitudes, _ = reduced._rom_ceff(
                coefficients,
                operators["Kq"][:, :rank, :rank],
                operators["Bq"][:, :rank],
                operators["Dq"],
            )
            lambda0, mu0 = qoi._reference_lame(coefficients)
            dense = rank_estimator.energy_matrix(
                coefficients,
                amplitudes,
                lambda0=lambda0,
                mu0=mu0,
            )
            direct_energy = _direct_energy(
                coefficients=coefficients,
                amplitudes=amplitudes,
                basis=basis[:rank],
                shape=shape,
                matrix_idx=matrix_idx,
                fiber_groups=fiber_groups,
                lambda0=lambda0,
                mu0=mu0,
            )
            relative_error = float(
                np.linalg.norm(dense - direct_energy)
                / max(np.linalg.norm(direct_energy), np.finfo(float).tiny)
            )
            absolute_error = float(np.linalg.norm(dense - direct_energy))
            records.append(
                {
                    "rank": int(rank),
                    "material_id": int(row["material_id"]),
                    "relative_error": relative_error,
                    "absolute_error": absolute_error,
                    "direct_energy_norm": float(np.linalg.norm(direct_energy)),
                }
            )
    return records


def _benchmark(
    estimator: TwoKernelEstimator,
    operators: dict[str, np.ndarray],
    *,
    points: int,
    seed: int,
) -> dict[str, float | int]:
    candidates = qoi._build_candidates(int(points), int(seed))
    times: list[float] = []
    started = time.perf_counter()
    for _, row in candidates.iterrows():
        coefficients = reduced._material_coefficients(row.to_dict())
        _, amplitudes, _ = reduced._rom_ceff(
            coefficients, operators["Kq"], operators["Bq"], operators["Dq"]
        )
        lambda0, mu0 = qoi._reference_lame(coefficients)
        t0 = time.perf_counter()
        estimator.energy_matrix(
            coefficients, amplitudes, lambda0=lambda0, mu0=mu0
        )
        times.append(time.perf_counter() - t0)
    wall = time.perf_counter() - started
    return {
        "points": int(points),
        "median_estimator_s": float(np.median(times)),
        "p95_estimator_s": float(np.quantile(times, 0.95)),
        "sweep_wall_s_including_rom": float(wall),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", default=RUN_DEFAULT)
    parser.add_argument("--rom-dir", type=Path, default=ROM_DEFAULT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--atom-batch-size", type=int, default=6)
    parser.add_argument("--feature-block", type=int, default=131_072)
    parser.add_argument("--fft-workers", type=int, default=1)
    parser.add_argument(
        "--fft-backend",
        choices=("scipy", "cupy", "auto"),
        default="scipy",
        help="Use CuPy only for offline feature transforms; stored contractions stay float64.",
    )
    parser.add_argument("--validation-points", type=int, default=3)
    parser.add_argument("--validation-seed", type=int, default=20261101)
    parser.add_argument("--benchmark-points", type=int, default=4096)
    parser.add_argument(
        "--validation-rtol",
        type=float,
        default=1.0e-8,
        help="Diagnostic dense/direct relative tolerance; independent of the 1e-4 ROM floor.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Validate and benchmark an existing online_estimator.npz without recompiling it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rom_dir = args.rom_dir.resolve()
    output = rom_dir / "online_estimator.npz"
    if output.exists() and not args.overwrite and not args.reuse_existing:
        raise FileExistsError(f"Ya existe {output}; usa --overwrite para regenerarlo.")

    phase = np.load(run_dir / "_fixed_geometry" / "phase.npy").astype(np.uint8)
    ori = np.load(run_dir / "_fixed_geometry" / "ori.npy").astype(np.float64)
    shape = tuple(int(value) for value in phase.shape)
    basis = qoi._load_basis_fields(rom_dir).astype(np.float64).reshape(-1, 6, phase.size)
    with np.load(rom_dir / "reduced_operators.npz") as payload:
        operators = {key: np.asarray(payload[key], dtype=np.float64) for key in ("Kq", "Bq", "Dq")}
        names = tuple(str(value) for value in payload["coefficient_names"])
    matrix_idx, fiber_groups = qoi._geometry_groups(phase, ori)

    if args.reuse_existing:
        if not output.is_file():
            raise FileNotFoundError(f"No existe {output} para --reuse-existing.")
        compile_meta = {
            "output_path": str(output),
            "reused_existing": True,
        }
    else:
        compile_meta = compile_two_kernel_contractions(
            output_path=output,
            shape=shape,
            basis_fields=basis,
            coefficient_names=names,
            affine_stress_batch=_affine_stress_factory(matrix_idx, fiber_groups),
            atom_batch_size=int(args.atom_batch_size),
            feature_block=int(args.feature_block),
            fft_workers=int(args.fft_workers),
            fft_backend=str(args.fft_backend),
        )
    estimator = TwoKernelEstimator.load(output)
    validation = _validate(
        estimator=estimator,
        basis=basis,
        operators=operators,
        shape=shape,
        matrix_idx=matrix_idx,
        fiber_groups=fiber_groups,
        points=int(args.validation_points),
        seed=int(args.validation_seed),
    )
    max_relative_error = max(record["relative_error"] for record in validation)
    max_absolute_error = max(record["absolute_error"] for record in validation)
    if max_relative_error >= float(args.validation_rtol):
        raise RuntimeError(
            "Estimador denso/Fourier no coincide: "
            f"error relativo maximo={max_relative_error:.3e}, "
            f"umbral={float(args.validation_rtol):.3e}."
        )
    benchmark = _benchmark(
        estimator,
        operators,
        points=int(args.benchmark_points),
        seed=int(args.validation_seed) + 1,
    )
    manifest = {
        "compile": compile_meta,
        "validation": validation,
        "validation_max_relative_error": float(max_relative_error),
        "validation_max_absolute_error": float(max_absolute_error),
        "validation_threshold": float(args.validation_rtol),
        "scientific_rom_floor": 1.0e-4,
        "benchmark": benchmark,
        "online_independent_of_nvox": True,
    }
    _write_json(rom_dir / "online_estimator_manifest.json", manifest)
    print(
        "[SCHUR] listo | "
        f"r={estimator.rank} | validation={max_relative_error:.3e} | "
        f"median={benchmark['median_estimator_s'] * 1e3:.3f} ms | "
        f"sweep={benchmark['sweep_wall_s_including_rom']:.3f} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
