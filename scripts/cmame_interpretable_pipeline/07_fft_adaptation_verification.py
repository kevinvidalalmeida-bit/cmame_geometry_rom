#!/usr/bin/env python3
"""Generate reproducible verification results for the FFTHomPy adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from env_bootstrap import ensure_configured_venv


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_DEFAULT = HERE / "campaign_config.json"
ensure_configured_venv(CONFIG_DEFAULT)

import numpy as np
import pandas as pd

for path in (ROOT / "FFT", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pipeline.fft_solver import (
    TI_stiffness_voigt,
    rotate_C_mandel,
    rotation_matrix_from_vector,
    solve_homogenization,
    voigt_to_mandel,
)
import rom_reduced_operator as reduced


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    command.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "cmame_method" / "fft_adaptation_verification",
    )
    command.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return command


def isotropic_mandel(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    matrix = np.zeros((6, 6), dtype=np.float64)
    matrix[:3, :3] = lam
    matrix[np.arange(3), np.arange(3)] += 2.0 * mu
    matrix[3:, 3:] = 2.0 * mu * np.eye(3)
    return matrix


def laminate_normal_x1(C0: np.ndarray, C1: np.ndarray, vf1: float) -> np.ndarray:
    traction = np.array((0, 5, 4))
    in_plane = np.array((1, 2, 3))
    phases = ((1.0 - vf1, C0), (vf1, C1))
    average_inverse = np.zeros((3, 3))
    coupling = np.zeros((3, 3))
    coupling_t = np.zeros((3, 3))
    schur = np.zeros((3, 3))
    for weight, C in phases:
        Caa = C[np.ix_(traction, traction)]
        Cab = C[np.ix_(traction, in_plane)]
        Cba = C[np.ix_(in_plane, traction)]
        Cbb = C[np.ix_(in_plane, in_plane)]
        inverse = np.linalg.inv(Caa)
        average_inverse += weight * inverse
        coupling += weight * inverse @ Cab
        coupling_t += weight * Cba @ inverse
        schur += weight * (Cbb - Cba @ inverse @ Cab)
    effective_aa = np.linalg.inv(average_inverse)
    effective_ab = effective_aa @ coupling
    effective_ba = coupling_t @ effective_aa
    effective_bb = schur + coupling_t @ effective_aa @ coupling
    effective = np.zeros((6, 6))
    effective[np.ix_(traction, traction)] = effective_aa
    effective[np.ix_(traction, in_plane)] = effective_ab
    effective[np.ix_(in_plane, traction)] = effective_ba
    effective[np.ix_(in_plane, in_plane)] = effective_bb
    return 0.5 * (effective + effective.T)


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_values = np.asarray(actual)
    expected_values = np.asarray(expected)
    return float(
        np.linalg.norm((actual_values - expected_values).reshape(-1))
        / max(np.linalg.norm(expected_values.reshape(-1)), np.finfo(float).eps)
    )


def solve_case(
    *,
    output_dir: Path,
    name: str,
    phase: np.ndarray,
    ori: np.ndarray,
    material: dict[str, float],
    backend: str,
    return_fields: bool = False,
) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
    parameters: dict[str, Any] = {
        "input_dir": str(output_dir / name),
        "seed": 20260902,
        "phase_array": phase,
        "ori_array": ori,
        **material,
        "fft_backend": backend,
        "solver_profile": "truth",
        "solver_maxiter": 1000,
        "store_solution_fields": bool(return_fields),
        "solution_field_return_in_memory": bool(return_fields),
        "solution_field_dtype": "float64",
        "solution_field_load_ids": list(range(6)),
        "gpu_timing_sync": backend == "cupy",
    }
    ceff = solve_homogenization(parameters)
    fields = parameters.get("_solution_fields_result")
    return np.asarray(ceff, dtype=np.float64), fields


def main() -> int:
    args = parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    shape = (8, 8, 8)
    phase_matrix = np.zeros(shape, dtype=np.uint8)
    ori_zero = np.zeros((*shape, 3), dtype=np.float64)
    isotropic = {
        "Em": 7.25,
        "nu_m": 0.29,
        "Ef_L": 31.0,
        "Ef_T": 8.0,
        "nu_LT": 0.22,
        "nu_TT": 0.35,
        "G_LT": 4.1,
    }
    actual, _ = solve_case(
        output_dir=output_dir,
        name="homogeneous_isotropic_cpu",
        phase=phase_matrix,
        ori=ori_zero,
        material=isotropic,
        backend="scipy",
    )
    error = relative_error(actual, isotropic_mandel(isotropic["Em"], isotropic["nu_m"]))
    rows.append(
        {
            "verification": "homogeneous_isotropic",
            "backend": "scipy",
            "relative_error": error,
            "threshold": 1.0e-12,
            "passed": error < 1.0e-12,
        }
    )

    phase_fiber = np.ones(shape, dtype=np.uint8)
    axis = np.array((2.0, -1.0, 3.0), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    ori_rotated = np.broadcast_to(axis, (*shape, 3)).copy()
    rotated_material = {
        "Em": 3.0,
        "nu_m": 0.34,
        "Ef_L": 48.0,
        "Ef_T": 13.0,
        "nu_LT": 0.21,
        "nu_TT": 0.37,
        "G_LT": 5.2,
    }
    actual, _ = solve_case(
        output_dir=output_dir,
        name="homogeneous_rotated_ti_cpu",
        phase=phase_fiber,
        ori=ori_rotated,
        material=rotated_material,
        backend="scipy",
    )
    local = voigt_to_mandel(
        TI_stiffness_voigt(
            rotated_material["Ef_L"],
            rotated_material["Ef_T"],
            rotated_material["nu_LT"],
            rotated_material["nu_TT"],
            rotated_material["G_LT"],
        )
    )
    expected = rotate_C_mandel(local, rotation_matrix_from_vector(axis))
    error = relative_error(actual, expected)
    rows.append(
        {
            "verification": "homogeneous_rotated_TI_Mandel",
            "backend": "scipy",
            "relative_error": error,
            "threshold": 1.0e-12,
            "passed": error < 1.0e-12,
        }
    )

    n = 12
    laminate_phase = np.zeros((n, n, n), dtype=np.uint8)
    laminate_phase[: n // 2] = 1
    laminate_ori = np.zeros((n, n, n, 3), dtype=np.float64)
    laminate_ori[..., 0] = 1.0
    E0, nu0, E1, nu1 = 3.0, 0.31, 11.0, 0.23
    laminate_material = {
        "Em": E0,
        "nu_m": nu0,
        "Ef_L": E1,
        "Ef_T": E1,
        "nu_LT": nu1,
        "nu_TT": nu1,
        "G_LT": E1 / (2.0 * (1.0 + nu1)),
    }
    laminate_cpu, cpu_fields = solve_case(
        output_dir=output_dir,
        name="periodic_laminate_cpu",
        phase=laminate_phase,
        ori=laminate_ori,
        material=laminate_material,
        backend="scipy",
        return_fields=True,
    )
    laminate_exact = laminate_normal_x1(
        isotropic_mandel(E0, nu0), isotropic_mandel(E1, nu1), 0.5
    )
    error = relative_error(laminate_cpu, laminate_exact)
    rows.append(
        {
            "verification": "periodic_laminate_analytic",
            "backend": "scipy",
            "relative_error": error,
            "threshold": 2.0e-9,
            "passed": error < 2.0e-9,
        }
    )
    rows.append(
        {
            "verification": "snapshot_transport_six_float64_fields",
            "backend": "scipy",
            "relative_error": 0.0,
            "threshold": 0.0,
            "passed": bool(
                cpu_fields is not None
                and len(cpu_fields) == 6
                and all(np.asarray(field).dtype == np.float64 for field in cpu_fields)
            ),
        }
    )

    reconstruction = reduced.affine_constitutive_reconstruction_error(
        laminate_phase, laminate_ori, laminate_material
    )
    affine_error = float(reconstruction["relative_frobenius_error"])
    rows.append(
        {
            "verification": "voxelwise_affine_constitutive_reconstruction",
            "backend": "numpy",
            "relative_error": affine_error,
            "threshold": 1.0e-12,
            "passed": bool(
                affine_error < 1.0e-12
                and float(reconstruction["maximum_voxel_group_relative_error"])
                < 1.0e-12
            ),
        }
    )

    gpu_error: str | None = None
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        laminate_gpu, gpu_fields = solve_case(
            output_dir=output_dir,
            name="periodic_laminate_gpu",
            phase=laminate_phase,
            ori=laminate_ori,
            material=laminate_material,
            backend="cupy",
            return_fields=True,
        )
        tensor_error = relative_error(laminate_gpu, laminate_cpu)
        field_error = max(
            relative_error(np.asarray(gpu), np.asarray(cpu))
            for gpu, cpu in zip(gpu_fields or (), cpu_fields or (), strict=True)
        )
        rows.extend(
            [
                {
                    "verification": "GPU_CPU_effective_tensor_parity",
                    "backend": "cupy_vs_scipy",
                    "relative_error": tensor_error,
                    "threshold": 1.0e-9,
                    "passed": tensor_error < 1.0e-9,
                },
                {
                    "verification": "GPU_CPU_snapshot_field_parity",
                    "backend": "cupy_vs_scipy",
                    "relative_error": field_error,
                    "threshold": 1.0e-8,
                    "passed": field_error < 1.0e-8,
                },
            ]
        )
    except Exception as exc:  # recorded explicitly; --require-gpu can make it fatal
        gpu_error = f"{type(exc).__name__}: {exc}"

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "fft_adaptation_verification.csv", index=False)
    summary = {
        "solver_profile": "truth",
        "solver_real_dtype": "float64",
        "solver_rtol": 1.0e-10,
        "all_executed_checks_passed": bool(results["passed"].all()),
        "gpu_error": gpu_error,
        "require_gpu": bool(args.require_gpu),
        "result_count": int(len(results)),
        "results_csv": str(output_dir / "fft_adaptation_verification.csv"),
    }
    (output_dir / "fft_adaptation_verification.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not summary["all_executed_checks_passed"]:
        raise RuntimeError("At least one FFTHomPy adaptation verification failed.")
    if args.require_gpu and gpu_error is not None:
        raise RuntimeError(f"GPU verification was required but failed: {gpu_error}")
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
