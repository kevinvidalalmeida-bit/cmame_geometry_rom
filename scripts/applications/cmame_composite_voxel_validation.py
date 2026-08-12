#!/usr/bin/env python3
"""Validate composite-voxel coarse graining against a fine FFT reference.

The experiment answers a narrow question before changing the paper workflow:
can a 64^3 composite voxel solve approach the existing 128^3 truth better than
a binary 64^3 solve, without increasing the global grid resolution?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
FFT_ROOT = ROOT / "FFT"
for path in (SCRIPTS, FFT_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import cmame_campaign_common as common
from pipeline.fft_solver import (
    ElasticTensor,
    Enu_to_KG,
    TI_stiffness_voigt,
    _pack_cfield_sym21,
    mandel_to_tensor4,
    rotate_C_mandel,
    rotation_matrix_from_vector,
    solve_homogenization,
    tensor4_to_mandel,
    voigt_to_mandel,
)


FINE_GRID_DEFAULT = ROOT / "results" / "cmame_method" / "voxel_scaling" / "N128"
OUT_DEFAULT = ROOT / "results" / "cmame_method" / "composite_voxel_validation"
MATERIAL_CENTER = {
    "Em": 2.75,
    "nu_m": 0.385,
    "Ef_L": 233.5,
    "Ef_T": 14.5,
    "G_LT": 19.0,
    "nu_LT": 0.23,
    "nu_TT": 0.375,
}
MANDEL_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
MANDEL_FACTORS = np.array((1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _relative_error(C: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(C - reference) / max(np.linalg.norm(reference), np.finfo(float).eps))


def _phase_blocks(phase: np.ndarray, factor: int) -> np.ndarray:
    n = int(phase.shape[0])
    if phase.shape != (n, n, n) or n % int(factor) != 0:
        raise ValueError("Fine phase must be cubic and divisible by the coarsening factor.")
    coarse = n // int(factor)
    return phase.reshape(coarse, factor, coarse, factor, coarse, factor).transpose(
        0, 2, 4, 1, 3, 5
    )


def _ori_blocks(ori: np.ndarray, factor: int) -> np.ndarray:
    n = int(ori.shape[0])
    coarse = n // int(factor)
    return ori.reshape(coarse, factor, coarse, factor, coarse, factor, 3).transpose(
        0, 2, 4, 1, 3, 5, 6
    )


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-14:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / norm
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis = -axis
    return axis


def _principal_axis(axes: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes, dtype=np.float64).reshape(-1, 3)
    if axes.size == 0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1.0e-14]
    if len(axes) == 0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    axes = axes / np.linalg.norm(axes, axis=1)[:, None]
    second_moment = np.einsum("ni,nj->ij", axes, axes, optimize=True) / float(len(axes))
    values, vectors = np.linalg.eigh(0.5 * (second_moment + second_moment.T))
    return _canonical_axis(vectors[:, int(np.argmax(values))])


def _interface_normal(mask: np.ndarray) -> np.ndarray:
    fiber = np.argwhere(mask)
    matrix = np.argwhere(~mask)
    if len(fiber) == 0 or len(matrix) == 0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    normal = fiber.mean(axis=0) - matrix.mean(axis=0)
    return _canonical_axis(normal)


def _mandel_basis_tensor(component: int) -> np.ndarray:
    tensor = np.zeros((3, 3), dtype=np.float64)
    ii, jj = MANDEL_PAIRS[int(component)]
    value = 1.0 / MANDEL_FACTORS[int(component)]
    tensor[ii, jj] = value
    tensor[jj, ii] = value
    return tensor


def _stress_tensor_to_mandel(stress: np.ndarray) -> np.ndarray:
    return np.array(
        [
            MANDEL_FACTORS[index] * stress[ii, jj]
            for index, (ii, jj) in enumerate(MANDEL_PAIRS)
        ],
        dtype=np.float64,
    )


def _sym_jump_vector(axis: int, normal: np.ndarray) -> np.ndarray:
    direction = np.zeros(3, dtype=np.float64)
    direction[int(axis)] = 1.0
    return 0.5 * (np.outer(direction, normal) + np.outer(normal, direction))


def laminate_composite_stiffness(
    Cm: np.ndarray,
    Cf: np.ndarray,
    fiber_fraction: float,
    normal: np.ndarray,
) -> np.ndarray:
    """Rank-one laminate stiffness for one mixed voxel in Mandel notation."""
    ff = float(np.clip(fiber_fraction, 0.0, 1.0))
    if ff <= 1.0e-12:
        return 0.5 * (Cm + Cm.T)
    if ff >= 1.0 - 1.0e-12:
        return 0.5 * (Cf + Cf.T)
    fm = 1.0 - ff
    normal = _canonical_axis(normal)
    Cm4 = mandel_to_tensor4(0.5 * (Cm + Cm.T))
    Cf4 = mandel_to_tensor4(0.5 * (Cf + Cf.T))
    dC4 = Cf4 - Cm4
    chat4 = fm * Cf4 + ff * Cm4
    cavg4 = fm * Cm4 + ff * Cf4

    jump_basis = np.stack([_sym_jump_vector(axis, normal) for axis in range(3)], axis=0)
    traction_operator = np.einsum(
        "j,ijmn,kmn->ik",
        normal,
        chat4,
        jump_basis,
        optimize=True,
    )
    out = np.zeros((6, 6), dtype=np.float64)
    for component in range(6):
        strain = _mandel_basis_tensor(component)
        traction_mismatch = np.einsum(
            "j,ijmn,mn->i",
            normal,
            dC4,
            strain,
            optimize=True,
        )
        jump = np.linalg.solve(traction_operator, -traction_mismatch)
        jump_tensor = np.einsum("k,kmn->mn", jump, jump_basis, optimize=True)
        stress = np.einsum("ijmn,mn->ij", cavg4, strain, optimize=True)
        stress += fm * ff * np.einsum("ijmn,mn->ij", dC4, jump_tensor, optimize=True)
        out[:, component] = _stress_tensor_to_mandel(stress)
    return 0.5 * (out + out.T)


def _center_stiffnesses(material: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    bulk, shear = Enu_to_KG(float(material["Em"]), float(material["nu_m"]))
    Cm = ElasticTensor(bulk=bulk, mu=shear).mandel.astype(np.float64)
    Cf = voigt_to_mandel(
        TI_stiffness_voigt(
            float(material["Ef_L"]),
            float(material["Ef_T"]),
            float(material["nu_LT"]),
            float(material["nu_TT"]),
            float(material["G_LT"]),
        )
    ).astype(np.float64)
    return Cm, Cf


def coarsen_binary_from_fine(
    phase_fine: np.ndarray,
    ori_fine: np.ndarray,
    *,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_blocks = _phase_blocks(np.asarray(phase_fine, dtype=np.uint8), factor)
    ori_blocks = _ori_blocks(np.asarray(ori_fine, dtype=np.float64), factor)
    coarse_shape = phase_blocks.shape[:3]
    fractions = phase_blocks.mean(axis=(3, 4, 5))
    phase = (fractions >= 0.5).astype(np.uint8)
    ori = np.zeros(coarse_shape + (3,), dtype=np.float32)
    for index in np.ndindex(coarse_shape):
        mask = phase_blocks[index] != 0
        if np.any(mask):
            ori[index] = _principal_axis(ori_blocks[index][mask]).astype(np.float32)
        else:
            ori[index] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return phase, ori, fractions


def build_composite_cfield(
    phase_fine: np.ndarray,
    ori_fine: np.ndarray,
    *,
    factor: int,
    material: dict[str, float],
    rule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    phase_binary, ori_coarse, fractions = coarsen_binary_from_fine(
        phase_fine,
        ori_fine,
        factor=factor,
    )
    phase_blocks = _phase_blocks(np.asarray(phase_fine, dtype=np.uint8), factor)
    ori_blocks = _ori_blocks(np.asarray(ori_fine, dtype=np.float64), factor)
    Cm, Cf_local = _center_stiffnesses(material)
    coarse_shape = phase_binary.shape
    cfield = np.empty((6, 6) + coarse_shape, dtype=np.float64)
    mixed_count = 0
    fallback_count = 0
    for index in np.ndindex(coarse_shape):
        phi = float(fractions[index])
        if phi <= 1.0e-12:
            C = Cm
        else:
            mask = phase_blocks[index] != 0
            axis = _principal_axis(ori_blocks[index][mask])
            Cf_rot = rotate_C_mandel(Cf_local, rotation_matrix_from_vector(axis))
            if phi >= 1.0 - 1.0e-12:
                C = Cf_rot
            elif rule == "voigt":
                mixed_count += 1
                C = (1.0 - phi) * Cm + phi * Cf_rot
            elif rule == "laminate":
                mixed_count += 1
                try:
                    C = laminate_composite_stiffness(
                        Cm,
                        Cf_rot,
                        fiber_fraction=phi,
                        normal=_interface_normal(mask),
                    )
                except np.linalg.LinAlgError:
                    fallback_count += 1
                    C = (1.0 - phi) * Cm + phi * Cf_rot
            else:
                raise ValueError(f"Unknown composite rule: {rule}")
        cfield[:, :, index[0], index[1], index[2]] = C
    packed = _pack_cfield_sym21(cfield)
    metadata = {
        "rule": str(rule),
        "coarsening_factor": int(factor),
        "fine_grid": int(phase_fine.shape[0]),
        "coarse_grid": int(phase_binary.shape[0]),
        "mixed_voxels": int(mixed_count),
        "mixed_voxel_fraction": float(mixed_count / np.prod(coarse_shape)),
        "laminate_fallback_to_voigt": int(fallback_count),
        "fiber_fraction_mean": float(fractions.mean()),
        "fiber_fraction_binary_mean": float(phase_binary.mean()),
        "cfield_storage": "sym21",
    }
    return phase_binary, ori_coarse, packed, metadata


def _solver_parameters(
    *,
    phase: np.ndarray,
    ori: np.ndarray,
    output_dir: Path,
    profile: str,
    cfield: np.ndarray | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        **MATERIAL_CENTER,
        "input_dir": str(output_dir),
        "seed": 20260824,
        "phase_array": phase,
        "ori_array": ori,
        "fft_backend": "cupy",
        "solver_profile": str(profile),
        "solver_maxiter": 2000,
        "require_convergence": True,
        "solver_fft_form": "r",
        "cfield_storage": "sym21",
        "cfield_indexed": False,
        "projection_storage": "full",
        "projection_backend": "cupy",
        "postprocess_assembly": "scalar",
        "load_batch_size": 1,
        "solver_verbose": False,
        "solver_timing_path": str(output_dir / "solver_timing.json"),
        "free_gpu_memory_after_solve": True,
    }
    if cfield is not None:
        params["Cfield_array"] = cfield
        params["precomputed_cfield_unique_entries"] = -1
    return params


def _solve_case(
    *,
    name: str,
    phase: np.ndarray,
    ori: np.ndarray,
    cfield: np.ndarray | None,
    out_dir: Path,
    profile: str,
    overwrite: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    ceff_path = case_dir / "Ceff.npy"
    if ceff_path.is_file() and not overwrite:
        ceff = np.load(ceff_path)
    else:
        started = time.perf_counter()
        ceff = solve_homogenization(
            _solver_parameters(
                phase=phase,
                ori=ori,
                output_dir=case_dir,
                profile=profile,
                cfield=cfield,
            )
        )
        np.save(ceff_path, ceff)
        _write_json(case_dir / "case_wall_time.json", {"solve_wall_s": time.perf_counter() - started})
    timing_path = case_dir / "solver_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else {}
    return ceff, timing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-grid-dir", type=Path, default=FINE_GRID_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--coarse-grid", type=int, default=64)
    parser.add_argument("--methods", nargs="*", choices=("binary", "voigt", "laminate"), default=["binary", "voigt", "laminate"])
    parser.add_argument("--profile", choices=("truth", "snapshot", "timing"), default="truth")
    parser.add_argument("--venv-path", type=Path, default=common.DEFAULT_VENV_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-runtime-reexec", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fine_dir = args.fine_grid_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_fine = np.load(fine_dir / "phase.npy").astype(np.uint8)
    ori_fine = np.load(fine_dir / "ori.npy").astype(np.float64)
    fine_grid = int(phase_fine.shape[0])
    coarse_grid = int(args.coarse_grid)
    if fine_grid % coarse_grid != 0:
        raise ValueError(f"fine grid {fine_grid} must be divisible by coarse grid {coarse_grid}.")
    factor = fine_grid // coarse_grid
    reference_path = fine_dir / "Ceff_truth.npy"
    if not reference_path.is_file():
        reference_path = fine_dir / "Ceff.npy"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing fine-grid reference Ceff in {fine_dir}.")
    reference = np.load(reference_path)

    phase_binary, ori_binary, fractions = coarsen_binary_from_fine(
        phase_fine,
        ori_fine,
        factor=factor,
    )
    np.save(out_dir / "phase_binary.npy", phase_binary)
    np.save(out_dir / "ori_binary.npy", ori_binary)
    np.save(out_dir / "fiber_fraction.npy", fractions.astype(np.float32))

    composite_payloads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for method in args.methods:
        if method == "binary":
            continue
        method_dir = out_dir / method
        cfield_path = method_dir / "Cfield_sym21.npy"
        composite_manifest_path = method_dir / "composite_voxel_manifest.json"
        if cfield_path.is_file() and composite_manifest_path.is_file() and not args.overwrite:
            composite_payloads[method] = (
                phase_binary,
                ori_binary,
                np.load(cfield_path),
                json.loads(composite_manifest_path.read_text(encoding="utf-8")),
            )
            continue
        started = time.perf_counter()
        payload = build_composite_cfield(
            phase_fine,
            ori_fine,
            factor=factor,
            material=MATERIAL_CENTER,
            rule=method,
        )
        payload[3]["cfield_build_wall_s"] = float(time.perf_counter() - started)
        composite_payloads[method] = payload
        method_dir.mkdir(parents=True, exist_ok=True)
        np.save(cfield_path, payload[2])
        _write_json(composite_manifest_path, payload[3])

    if args.build_only:
        _write_json(
            out_dir / "campaign_manifest.json",
            {
                "status": "build_only",
                "fine_grid_dir": str(fine_dir),
                "fine_grid": fine_grid,
                "coarse_grid": coarse_grid,
                "coarsening_factor": factor,
                "methods": list(args.methods),
                "reference_path": str(reference_path),
            },
        )
        print(f"[COMPOSITE-VOXEL] build only | out={out_dir}", flush=True)
        return 0

    if not args.no_runtime_reexec:
        common.prepare_runtime(
            args.venv_path,
            Path(__file__),
            marker_name="CMAME_COMPOSITE_VOXEL_CUDA_READY",
        )

    rows: list[dict[str, Any]] = []
    if "binary" in args.methods:
        ceff, timing = _solve_case(
            name="binary",
            phase=phase_binary,
            ori=ori_binary,
            cfield=None,
            out_dir=out_dir,
            profile=str(args.profile),
            overwrite=bool(args.overwrite),
        )
        rows.append(
            {
                "method": "binary",
                "coarse_grid": coarse_grid,
                "relative_error_vs_fine": _relative_error(ceff, reference),
                "ceff_min_eig": float(np.linalg.eigvalsh(0.5 * (ceff + ceff.T)).min()),
                "solver_total_wall_s": float(timing.get("solver_total_wall_s", np.nan)),
                "cfield_precomputed": bool(timing.get("cfield_precomputed", False)),
            }
        )

    for method, (phase, ori, cfield, metadata) in composite_payloads.items():
        ceff, timing = _solve_case(
            name=method,
            phase=phase,
            ori=ori,
            cfield=cfield,
            out_dir=out_dir,
            profile=str(args.profile),
            overwrite=bool(args.overwrite),
        )
        rows.append(
            {
                "method": method,
                "coarse_grid": coarse_grid,
                "relative_error_vs_fine": _relative_error(ceff, reference),
                "ceff_min_eig": float(np.linalg.eigvalsh(0.5 * (ceff + ceff.T)).min()),
                "solver_total_wall_s": float(timing.get("solver_total_wall_s", np.nan)),
                "cfield_precomputed": bool(timing.get("cfield_precomputed", False)),
                "mixed_voxels": int(metadata["mixed_voxels"]),
                "mixed_voxel_fraction": float(metadata["mixed_voxel_fraction"]),
                "cfield_build_wall_s": float(metadata["cfield_build_wall_s"]),
                "laminate_fallback_to_voigt": int(metadata["laminate_fallback_to_voigt"]),
            }
        )

    table = pd.DataFrame(rows).sort_values("relative_error_vs_fine")
    table.to_csv(out_dir / "composite_voxel_validation.csv", index=False)
    best = table.iloc[0].to_dict()
    binary = table.loc[table["method"] == "binary"]
    binary_error = float(binary.iloc[0]["relative_error_vs_fine"]) if len(binary) else np.nan
    _write_json(
        out_dir / "campaign_manifest.json",
        {
            "status": "complete",
            "fine_grid_dir": str(fine_dir),
            "reference_path": str(reference_path),
            "fine_grid": fine_grid,
            "coarse_grid": coarse_grid,
            "coarsening_factor": factor,
            "profile": str(args.profile),
            "material": MATERIAL_CENTER,
            "methods": list(args.methods),
            "best_method": str(best["method"]),
            "best_relative_error_vs_fine": float(best["relative_error_vs_fine"]),
            "binary_relative_error_vs_fine": binary_error,
            "best_improvement_factor_over_binary": float(binary_error / best["relative_error_vs_fine"])
            if np.isfinite(binary_error) and float(best["relative_error_vs_fine"]) > 0.0
            else None,
        },
    )
    print(
        "[COMPOSITE-VOXEL] complete | "
        f"best={best['method']} error={best['relative_error_vs_fine']:.3e} | out={out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
