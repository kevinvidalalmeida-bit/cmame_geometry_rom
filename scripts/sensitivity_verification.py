#!/usr/bin/env python3
"""Check affine and physical-parameter sensitivities of a fixed-geometry ROM."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
RUN_DEFAULT = (
    PROJECT_ROOT
    / "results"
    / "fixed_geometry_ffthompy"
    / "fixed_geometry_ar15_vf20_sobol8_center_fields"
)
ROM_DEFAULT = (
    RUN_DEFAULT
    / "rom_tangential_r66_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v30"
)
DEFAULT_VENV_PATH = (
    Path.home() / "Documentos/ANDRES/COMPUTATIONAL_WORKSPACEV4/.venv"
)

PHYSICAL_NAMES = ["Em", "nu_m", "Ef_L", "Ef_T", "G_LT", "nu_LT", "nu_TT"]
FOM_FD_SCHEMES = ("central", "edge-auto")

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import fft_homogenization_solver as sweep
import rom_reduced_operator as reduced


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
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_out_dir(run_dir: Path, out_name: str, *, overwrite: bool) -> Path:
    out_dir = run_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=overwrite)
    return out_dir


def _relative_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(float).eps))


def _full_ceff_from_row(row: pd.Series | dict[str, Any]) -> np.ndarray | None:
    matrix = np.zeros((6, 6), dtype=float)
    for ii in range(6):
        for jj in range(6):
            column = f"Ceff_{ii + 1}{jj + 1}"
            if column not in row:
                return None
            matrix[ii, jj] = float(row[column])
    return 0.5 * (matrix + matrix.T)


def _rom_ceff_from_coeffs(
    coeffs: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> np.ndarray:
    C, _, _ = reduced._rom_ceff(coeffs, Kq, Bq, Dq)
    return C


def _rom_affine_sensitivities(
    coeffs: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    K = np.tensordot(coeffs, Kq, axes=(0, 0))
    B = np.tensordot(coeffs, Bq, axes=(0, 0))
    D = np.tensordot(coeffs, Dq, axes=(0, 0))
    X = np.linalg.solve(K, B)
    C = D - B.T @ X
    C = 0.5 * (C + C.T)

    gradients = np.empty((len(coeffs), 6, 6), dtype=float)
    for q in range(len(coeffs)):
        dC = (
            Dq[q]
            - Bq[q].T @ X
            - X.T @ Bq[q]
            + X.T @ Kq[q] @ X
        )
        gradients[q] = 0.5 * (dC + dC.T)
    return C, gradients


def _affine_fd_sensitivities(
    coeffs: np.ndarray,
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    *,
    rel_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    gradients = np.empty((len(coeffs), 6, 6), dtype=float)
    steps = np.empty(len(coeffs), dtype=float)
    for q, value in enumerate(coeffs):
        step = float(rel_step) * max(abs(float(value)), 1.0)
        cp = coeffs.copy()
        cm = coeffs.copy()
        cp[q] += step
        cm[q] -= step
        gradients[q] = (
            _rom_ceff_from_coeffs(cp, Kq, Bq, Dq)
            - _rom_ceff_from_coeffs(cm, Kq, Bq, Dq)
        ) / (2.0 * step)
        steps[q] = step
    return gradients, steps


def _sampled_from_row(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    return {name: float(row[name]) for name in PHYSICAL_NAMES}


def _material_row_from_sampled(
    sampled: dict[str, float],
    *,
    material_id: int,
    label: str,
) -> dict[str, Any]:
    material = sweep._material_derived(sampled)
    sweep._validate_material(material)
    return {
        "material_id": int(material_id),
        "material_label": str(label),
        **material,
    }


def _coefficient_jacobian_physical(
    sampled: dict[str, float],
    *,
    rel_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    base_coeffs = reduced._material_coefficients(
        _material_row_from_sampled(sampled, material_id=0, label="base")
    )
    jac = np.empty((len(base_coeffs), len(PHYSICAL_NAMES)), dtype=float)
    steps = np.empty(len(PHYSICAL_NAMES), dtype=float)
    for jj, name in enumerate(PHYSICAL_NAMES):
        low, high = sweep.MATERIAL_BOUNDS[name]
        value = float(sampled[name])
        step = float(rel_step) * max(abs(value), 1.0)
        max_centered = 0.45 * min(value - low, high - value)
        if max_centered > 0.0:
            step = min(step, max_centered)
        if step <= 0.0:
            raise ValueError(f"No se puede perturbar {name} dentro del dominio.")
        plus = dict(sampled)
        minus = dict(sampled)
        plus[name] = value + step
        minus[name] = value - step
        coeff_plus = reduced._material_coefficients(
            _material_row_from_sampled(plus, material_id=1, label=f"{name}_plus")
        )
        coeff_minus = reduced._material_coefficients(
            _material_row_from_sampled(minus, material_id=2, label=f"{name}_minus")
        )
        jac[:, jj] = (coeff_plus - coeff_minus) / (2.0 * step)
        steps[jj] = step
    return jac, steps


def _physical_fd_sensitivities_rom(
    sampled: dict[str, float],
    Kq: np.ndarray,
    Bq: np.ndarray,
    Dq: np.ndarray,
    *,
    steps: np.ndarray,
) -> np.ndarray:
    gradients = np.empty((len(PHYSICAL_NAMES), 6, 6), dtype=float)
    for jj, name in enumerate(PHYSICAL_NAMES):
        value = float(sampled[name])
        step = float(steps[jj])
        plus = dict(sampled)
        minus = dict(sampled)
        plus[name] = value + step
        minus[name] = value - step
        coeff_plus = reduced._material_coefficients(
            _material_row_from_sampled(plus, material_id=1, label=f"{name}_plus")
        )
        coeff_minus = reduced._material_coefficients(
            _material_row_from_sampled(minus, material_id=2, label=f"{name}_minus")
        )
        gradients[jj] = (
            _rom_ceff_from_coeffs(coeff_plus, Kq, Bq, Dq)
            - _rom_ceff_from_coeffs(coeff_minus, Kq, Bq, Dq)
        ) / (2.0 * step)
    return gradients


def _central_steps_for_physical(
    sampled: dict[str, float],
    *,
    rel_step: float,
) -> np.ndarray:
    return _coefficient_jacobian_physical(sampled, rel_step=rel_step)[1]


def _fom_fd_stencils(
    sampled: dict[str, float],
    *,
    rel_step: float,
    scheme: str,
) -> list[dict[str, Any]]:
    if scheme not in FOM_FD_SCHEMES:
        raise ValueError(f"fom_fd_scheme desconocido: {scheme}")
    central_steps = _central_steps_for_physical(sampled, rel_step=rel_step)
    stencils: list[dict[str, Any]] = []
    for jj, name in enumerate(PHYSICAL_NAMES):
        low, high = sweep.MATERIAL_BOUNDS[name]
        value = float(sampled[name])
        requested_step = float(rel_step) * max(abs(value), 1.0)
        central_step = float(central_steps[jj])
        if scheme == "central":
            stencils.append(
                {
                    "name": name,
                    "step": central_step,
                    "scheme": "central",
                    "direction": "centered",
                    "points": [(1.0, value + central_step), (-1.0, value - central_step)],
                    "denominator": 2.0 * central_step,
                    "requested_step": requested_step,
                    "central_step": central_step,
                    "boundary_limited": central_step < 0.5 * requested_step,
                }
            )
            continue

        lower_room = value - float(low)
        upper_room = float(high) - value
        can_central = (
            value - requested_step >= float(low)
            and value + requested_step <= float(high)
        )
        if can_central:
            stencils.append(
                {
                    "name": name,
                    "step": requested_step,
                    "scheme": "central",
                    "direction": "centered",
                    "points": [(1.0, value + requested_step), (-1.0, value - requested_step)],
                    "denominator": 2.0 * requested_step,
                    "requested_step": requested_step,
                    "central_step": central_step,
                    "boundary_limited": False,
                }
            )
        elif lower_room >= upper_room:
            step = min(requested_step, 0.45 * lower_room)
            if step <= 0.0:
                raise ValueError(f"No se puede perturbar {name} hacia abajo.")
            stencils.append(
                {
                    "name": name,
                    "step": step,
                    "scheme": "one_sided_second_order",
                    "direction": "backward",
                    "points": [(3.0, value), (-4.0, value - step), (1.0, value - 2.0 * step)],
                    "denominator": 2.0 * step,
                    "requested_step": requested_step,
                    "central_step": central_step,
                    "boundary_limited": True,
                }
            )
        else:
            step = min(requested_step, 0.45 * upper_room)
            if step <= 0.0:
                raise ValueError(f"No se puede perturbar {name} hacia arriba.")
            stencils.append(
                {
                    "name": name,
                    "step": step,
                    "scheme": "one_sided_second_order",
                    "direction": "forward",
                    "points": [(-3.0, value), (4.0, value + step), (-1.0, value + 2.0 * step)],
                    "denominator": 2.0 * step,
                    "requested_step": requested_step,
                    "central_step": central_step,
                    "boundary_limited": True,
                }
            )
    return stencils


def _prepare_runtime(venv_path: Path, *, no_reexec: bool, marker_name: str) -> None:
    if no_reexec:
        return
    venv_python = venv_path.expanduser().resolve() / "bin/python"
    if not venv_python.is_file():
        raise FileNotFoundError(f"No existe el Python del entorno: {venv_python}")

    env = os.environ.copy()
    cuda_dirs = sorted(
        str(path)
        for path in (venv_path / "lib").glob("python*/site-packages/nvidia/*/lib")
        if path.is_dir()
    )
    if cuda_dirs:
        current = env.get("LD_LIBRARY_PATH", "")
        current_dirs = [value for value in current.split(":") if value]
        env["LD_LIBRARY_PATH"] = ":".join(
            [*cuda_dirs, *[value for value in current_dirs if value not in cuda_dirs]]
        )

    marker = ":".join(cuda_dirs)
    if env.get(marker_name) != marker:
        env[marker_name] = marker
        os.execve(
            str(venv_python),
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            env,
        )
    os.environ.update(env)

    try:
        ctypes.CDLL("libcublas.so.12")
        ctypes.CDLL("libcudart.so.12")
    except OSError as exc:
        raise RuntimeError(
            "No se pudieron cargar las librerias CUDA del entorno virtual. "
            f"Detalle: {exc}"
        ) from exc


def _load_design_row(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        payload = _load_json(manifest_path)
        design = payload.get("geometry", {}).get("design")
        if isinstance(design, dict):
            return design
    geometry_manifest = _load_json(run_dir / "_fixed_geometry" / "geometry_manifest.json")
    design = geometry_manifest.get("design")
    if not isinstance(design, dict):
        raise RuntimeError("No se pudo recuperar design_row de la geometria fija.")
    return design


def _configure_runtime(args: argparse.Namespace) -> dict[str, Any]:
    runtime_args = argparse.Namespace(
        geometry_backend=args.geometry_backend,
        generator_cores=args.generator_cores,
        compute_rve_metrics=False,
        solver_tol=args.solver_tol,
    )
    runtime = sweep._configure_fft_runtime(runtime_args)
    sobol_gpu = runtime["sobol_gpu"]
    sobol_gpu.check_cupy_gpu()
    sobol_gpu.warmup_gpu_once()
    return runtime


def _solve_full_order(
    *,
    run_dir: Path,
    out_dir: Path,
    material_row: dict[str, Any],
    design_row: dict[str, Any],
    runtime: dict[str, Any],
    seed: int,
    tag: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    sobol_gpu = runtime["sobol_gpu"]
    config = runtime["config"]
    geometry_dir = run_dir / "_fixed_geometry"
    phase = np.load(geometry_dir / "phase.npy").astype(np.uint8)
    ori = np.load(geometry_dir / "ori.npy").astype(np.float32)
    material_dir = out_dir / f"fom_{tag}"
    material_dir.mkdir(parents=True, exist_ok=True)
    _write_json(material_dir / "material.json", material_row)
    params = sweep._material_solver_params(
        sobol_gpu=sobol_gpu,
        config=config,
        material_row=material_row,
        design_row=design_row,
        material_dir=material_dir,
        seed=int(seed),
        save_solution_fields=False,
    )
    params["phase_array"] = phase
    params["ori_array"] = ori
    params["preloaded_geometry"] = True
    params["free_gpu_memory_after_solve"] = True

    print(
        "[SENS-FOM] FFTHomPy/CuPy | "
        f"{tag} | Em={material_row['Em']:.6g} | Ef_L={material_row['Ef_L']:.6g}",
        flush=True,
    )
    t0 = time.perf_counter()
    ceff = np.asarray(sobol_gpu.solve_homogenization(params), dtype=float)
    wall_s = float(time.perf_counter() - t0)
    np.save(material_dir / "Ceff.npy", ceff)
    timing: dict[str, Any] = {}
    timing_path = material_dir / "solver_timing.json"
    if timing_path.is_file():
        timing = _load_json(timing_path)
    return 0.5 * (ceff + ceff.T), {
        "tag": tag,
        "material_dir": str(material_dir),
        "solve_wall_s": wall_s,
        "solver_total_wall_s": float(timing.get("solver_total_wall_s", np.nan)),
        "problem_calculate_s": float(timing.get("problem_calculate_s", np.nan)),
    }


def _full_order_fd_physical(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    out_dir: Path,
    sampled: dict[str, float],
    base_material_id: int,
    stencils: list[dict[str, Any]],
    base_ceff: np.ndarray | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    _prepare_runtime(
        args.venv_path,
        no_reexec=bool(args.no_runtime_reexec),
        marker_name="FIXED_GEOMETRY_SENS_CUDA_READY",
    )
    runtime = _configure_runtime(args)
    design_row = _load_design_row(run_dir)
    gradients = np.empty((len(PHYSICAL_NAMES), 6, 6), dtype=float)
    records: list[dict[str, Any]] = []
    base_ceff_cache = (
        np.asarray(base_ceff, dtype=float)
        if base_ceff is not None and np.isfinite(np.asarray(base_ceff, dtype=float)).all()
        else None
    )
    for jj, stencil in enumerate(stencils):
        name = str(stencil["name"])
        numerator = np.zeros((6, 6), dtype=float)
        for point_index, (weight, parameter_value) in enumerate(stencil["points"]):
            if math.isclose(float(parameter_value), float(sampled[name]), rel_tol=0.0, abs_tol=0.0):
                if base_ceff_cache is None:
                    row = _material_row_from_sampled(
                        sampled,
                        material_id=100000 + 10 * jj + point_index,
                        label=f"sensitivity_{base_material_id}_{name}_base",
                    )
                    ceff, timing = _solve_full_order(
                        run_dir=run_dir,
                        out_dir=out_dir,
                        material_row=row,
                        design_row=design_row,
                        runtime=runtime,
                        seed=int(args.geometry_seed),
                        tag=f"m{base_material_id:04d}_{name}_base",
                    )
                    base_ceff_cache = ceff
                    side = "base_solved"
                else:
                    ceff = base_ceff_cache
                    timing = {
                        "tag": f"m{base_material_id:04d}_{name}_base",
                        "material_dir": "",
                        "solve_wall_s": 0.0,
                        "solver_total_wall_s": 0.0,
                        "problem_calculate_s": 0.0,
                    }
                    side = "base_reused"
            else:
                perturbed = dict(sampled)
                perturbed[name] = float(parameter_value)
                side = f"p{point_index}"
                if float(parameter_value) > float(sampled[name]):
                    side = f"plus{point_index}"
                elif float(parameter_value) < float(sampled[name]):
                    side = f"minus{point_index}"
                row = _material_row_from_sampled(
                    perturbed,
                    material_id=100000 + 10 * jj + point_index,
                    label=f"sensitivity_{base_material_id}_{name}_{side}",
                )
                ceff, timing = _solve_full_order(
                    run_dir=run_dir,
                    out_dir=out_dir,
                    material_row=row,
                    design_row=design_row,
                    runtime=runtime,
                    seed=int(args.geometry_seed),
                    tag=f"m{base_material_id:04d}_{name}_{side}",
                )
            numerator += float(weight) * ceff
            records.append(
                {
                    "physical_parameter": name,
                    "side": side,
                    "weight": float(weight),
                    "parameter_value": float(parameter_value),
                    "step": float(stencil["step"]),
                    "requested_step": float(stencil["requested_step"]),
                    "central_step": float(stencil["central_step"]),
                    "fd_scheme": str(stencil["scheme"]),
                    "fd_direction": str(stencil["direction"]),
                    "boundary_limited": bool(stencil["boundary_limited"]),
                    **timing,
                }
            )
        gradients[jj] = numerator / float(stencil["denominator"])
    return gradients, pd.DataFrame(records)


def _tensor_component_records(
    *,
    prefix: str,
    names: list[str],
    tensors: np.ndarray,
    material_id: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        row: dict[str, Any] = {
            "material_id": int(material_id),
            f"{prefix}_name": name,
            f"{prefix}_index": int(idx),
            "tensor_frobenius_norm": float(np.linalg.norm(tensors[idx])),
        }
        for ii in range(6):
            for jj in range(6):
                row[f"dCeff_{ii + 1}{jj + 1}"] = float(tensors[idx, ii, jj])
        rows.append(row)
    return rows


def _error_rows(
    *,
    material_id: int,
    names: list[str],
    analytic: np.ndarray,
    reference: np.ndarray,
    kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        diff = analytic[idx] - reference[idx]
        rows.append(
            {
                "material_id": int(material_id),
                "sensitivity_kind": kind,
                "name": name,
                "index": int(idx),
                "analytic_norm": float(np.linalg.norm(analytic[idx])),
                "reference_norm": float(np.linalg.norm(reference[idx])),
                "absolute_frobenius_error": float(np.linalg.norm(diff)),
                "relative_frobenius_error": _relative_frobenius(analytic[idx], reference[idx]),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcula sensibilidades analiticas de un ROM tangencial fijo y "
            "opcionalmente las compara con diferencias finitas FFTHomPy/CuPy."
        )
    )
    parser.add_argument("run_dir", type=Path, nargs="?", default=RUN_DEFAULT)
    parser.add_argument("--rom-dir", type=Path, default=ROM_DEFAULT)
    parser.add_argument("--material-csv", type=Path, default=None)
    parser.add_argument("--material-ids", type=int, nargs="*", default=[0])
    parser.add_argument("--out-name", default="sensitivity_r66_center")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--affine-rel-step", type=float, default=1e-6)
    parser.add_argument("--physical-rel-step", type=float, default=1e-6)
    parser.add_argument("--fom-rel-step", type=float, default=5e-3)
    parser.add_argument(
        "--fom-fd-scheme",
        choices=FOM_FD_SCHEMES,
        default="central",
        help=(
            "central reproduce el chequeo historico; edge-auto usa central si "
            "cabe el paso solicitado y diferencias unilaterales de segundo "
            "orden hacia dentro cerca de un borde del dominio."
        ),
    )
    parser.add_argument("--run-full-order-fd", action="store_true")
    parser.add_argument("--geometry-backend", choices=("numba", "cupy", "auto"), default="numba")
    parser.add_argument("--generator-cores", type=int, default=2)
    parser.add_argument("--solver-tol", type=float, default=1e-3)
    parser.add_argument("--geometry-seed", type=int, default=20260811)
    parser.add_argument("--venv-path", type=Path, default=DEFAULT_VENV_PATH)
    parser.add_argument("--no-runtime-reexec", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rom_dir = args.rom_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No existe run_dir: {run_dir}")
    if not rom_dir.is_dir():
        raise FileNotFoundError(f"No existe rom_dir: {rom_dir}")
    material_csv = (
        args.material_csv
        if args.material_csv is not None
        else run_dir / "fixed_geometry_ffthompy_results.csv"
    )
    if not material_csv.is_file():
        raise FileNotFoundError(f"No existe material CSV: {material_csv}")

    out_dir = _make_out_dir(run_dir, str(args.out_name), overwrite=bool(args.overwrite))
    material_df = pd.read_csv(material_csv)
    operators = np.load(rom_dir / "reduced_operators.npz")
    Kq = np.asarray(operators["Kq"], dtype=float)
    Bq = np.asarray(operators["Bq"], dtype=float)
    Dq = np.asarray(operators["Dq"], dtype=float)

    affine_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    rom_affine_error_rows: list[dict[str, Any]] = []
    rom_physical_error_rows: list[dict[str, Any]] = []
    fom_error_rows: list[dict[str, Any]] = []
    fom_timing_frames: list[pd.DataFrame] = []
    tensors_payload: dict[str, np.ndarray] = {}

    for material_id in [int(value) for value in args.material_ids]:
        matches = material_df.loc[material_df["material_id"].astype(int) == material_id]
        if matches.empty:
            raise KeyError(f"material_id={material_id} no existe en {material_csv}")
        row = matches.iloc[0]
        sampled = _sampled_from_row(row)
        coeffs = reduced._material_coefficients(row.to_dict())
        C_rom, affine_sens = _rom_affine_sensitivities(coeffs, Kq, Bq, Dq)
        C_rom_check = _rom_ceff_from_coeffs(coeffs, Kq, Bq, Dq)
        affine_fd, affine_steps = _affine_fd_sensitivities(
            coeffs,
            Kq,
            Bq,
            Dq,
            rel_step=float(args.affine_rel_step),
        )
        jac_physical, physical_steps_rom = _coefficient_jacobian_physical(
            sampled,
            rel_step=float(args.physical_rel_step),
        )
        physical_sens = np.einsum("qab,qj->jab", affine_sens, jac_physical, optimize=True)
        physical_fd_rom = _physical_fd_sensitivities_rom(
            sampled,
            Kq,
            Bq,
            Dq,
            steps=physical_steps_rom,
        )

        affine_rows.extend(
            _tensor_component_records(
                prefix="affine",
                names=reduced.COEFF_NAMES,
                tensors=affine_sens,
                material_id=material_id,
            )
        )
        physical_rows.extend(
            _tensor_component_records(
                prefix="physical",
                names=PHYSICAL_NAMES,
                tensors=physical_sens,
                material_id=material_id,
            )
        )
        rom_affine_error_rows.extend(
            _error_rows(
                material_id=material_id,
                names=reduced.COEFF_NAMES,
                analytic=affine_sens,
                reference=affine_fd,
                kind="rom_affine_central_fd",
            )
        )
        rom_physical_error_rows.extend(
            _error_rows(
                material_id=material_id,
                names=PHYSICAL_NAMES,
                analytic=physical_sens,
                reference=physical_fd_rom,
                kind="rom_physical_central_fd",
            )
        )
        for idx, step in enumerate(affine_steps):
            rom_affine_error_rows[-len(affine_steps) + idx]["fd_step"] = float(step)
        for idx, step in enumerate(physical_steps_rom):
            rom_physical_error_rows[-len(physical_steps_rom) + idx]["fd_step"] = float(step)

        fom_gradients = None
        fom_steps = None
        fom_stencils = None
        full_ceff = _full_ceff_from_row(row)
        if args.run_full_order_fd:
            fom_stencils = _fom_fd_stencils(
                sampled,
                rel_step=float(args.fom_rel_step),
                scheme=str(args.fom_fd_scheme),
            )
            fom_steps = np.array([float(stencil["step"]) for stencil in fom_stencils])
            fom_gradients, timing_df = _full_order_fd_physical(
                args=args,
                run_dir=run_dir,
                out_dir=out_dir,
                sampled=sampled,
                base_material_id=material_id,
                stencils=fom_stencils,
                base_ceff=full_ceff,
            )
            timing_df.insert(0, "material_id", int(material_id))
            fom_timing_frames.append(timing_df)
            fom_error_rows.extend(
                _error_rows(
                    material_id=material_id,
                    names=PHYSICAL_NAMES,
                    analytic=physical_sens,
                    reference=fom_gradients,
                    kind="fom_physical_central_fd",
                )
            )
            for idx, step in enumerate(fom_steps):
                fom_error_rows[-len(fom_steps) + idx]["fd_step"] = float(step)
                if fom_stencils is not None:
                    stencil = fom_stencils[idx]
                    fom_error_rows[-len(fom_steps) + idx]["requested_step"] = float(stencil["requested_step"])
                    fom_error_rows[-len(fom_steps) + idx]["central_step"] = float(stencil["central_step"])
                    fom_error_rows[-len(fom_steps) + idx]["fd_scheme"] = str(stencil["scheme"])
                    fom_error_rows[-len(fom_steps) + idx]["fd_direction"] = str(stencil["direction"])
                    fom_error_rows[-len(fom_steps) + idx]["boundary_limited"] = bool(stencil["boundary_limited"])

        tensors_payload[f"material_{material_id:04d}_C_rom"] = C_rom
        tensors_payload[f"material_{material_id:04d}_affine_sens"] = affine_sens
        tensors_payload[f"material_{material_id:04d}_physical_sens"] = physical_sens
        tensors_payload[f"material_{material_id:04d}_rom_affine_fd"] = affine_fd
        tensors_payload[f"material_{material_id:04d}_rom_physical_fd"] = physical_fd_rom
        tensors_payload[f"material_{material_id:04d}_coefficients"] = coeffs
        tensors_payload[f"material_{material_id:04d}_coefficient_jacobian"] = jac_physical
        tensors_payload[f"material_{material_id:04d}_C_rom_formula_diff_norm"] = np.array(
            [np.linalg.norm(C_rom - C_rom_check)],
            dtype=float,
        )
        if full_ceff is not None:
            tensors_payload[f"material_{material_id:04d}_C_full_order"] = full_ceff
            tensors_payload[f"material_{material_id:04d}_C_rom_rel_error"] = np.array(
                [_relative_frobenius(C_rom, full_ceff)],
                dtype=float,
            )
        if fom_gradients is not None:
            tensors_payload[f"material_{material_id:04d}_fom_physical_fd"] = fom_gradients
            tensors_payload[f"material_{material_id:04d}_fom_physical_steps"] = fom_steps

        print(
            "[SENS] "
            f"material={material_id} | "
            f"max_affine_self={max(row['relative_frobenius_error'] for row in rom_affine_error_rows if row['material_id'] == material_id):.3e} | "
            f"max_physical_self={max(row['relative_frobenius_error'] for row in rom_physical_error_rows if row['material_id'] == material_id):.3e}",
            flush=True,
        )

    affine_df = pd.DataFrame(affine_rows)
    physical_df = pd.DataFrame(physical_rows)
    rom_affine_errors = pd.DataFrame(rom_affine_error_rows)
    rom_physical_errors = pd.DataFrame(rom_physical_error_rows)
    fom_errors = pd.DataFrame(fom_error_rows)

    affine_df.to_csv(out_dir / "rom_affine_sensitivity_tensors.csv", index=False)
    physical_df.to_csv(out_dir / "rom_physical_sensitivity_tensors.csv", index=False)
    rom_affine_errors.to_csv(out_dir / "rom_affine_selfcheck_errors.csv", index=False)
    rom_physical_errors.to_csv(out_dir / "rom_physical_selfcheck_errors.csv", index=False)
    if len(fom_errors):
        fom_errors.to_csv(out_dir / "fom_physical_fd_errors.csv", index=False)
    if fom_timing_frames:
        pd.concat(fom_timing_frames, ignore_index=True).to_csv(
            out_dir / "fom_fd_solver_timing.csv",
            index=False,
        )
    np.savez_compressed(out_dir / "sensitivity_tensors.npz", **tensors_payload)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "rom_dir": str(rom_dir),
        "material_csv": str(material_csv.resolve()),
        "out_dir": str(out_dir),
        "material_ids": [int(value) for value in args.material_ids],
        "coefficient_names": reduced.COEFF_NAMES,
        "physical_names": PHYSICAL_NAMES,
        "affine_rel_step": float(args.affine_rel_step),
        "physical_rel_step": float(args.physical_rel_step),
        "run_full_order_fd": bool(args.run_full_order_fd),
        "rom_affine_selfcheck": {
            "mean_relative_error": float(rom_affine_errors["relative_frobenius_error"].mean()),
            "max_relative_error": float(rom_affine_errors["relative_frobenius_error"].max()),
        },
        "rom_physical_selfcheck": {
            "mean_relative_error": float(rom_physical_errors["relative_frobenius_error"].mean()),
            "max_relative_error": float(rom_physical_errors["relative_frobenius_error"].max()),
        },
    }
    if len(fom_errors):
        summary["fom_physical_fd"] = {
            "fom_rel_step": float(args.fom_rel_step),
            "fom_fd_scheme": str(args.fom_fd_scheme),
            "mean_relative_error": float(fom_errors["relative_frobenius_error"].mean()),
            "max_relative_error": float(fom_errors["relative_frobenius_error"].max()),
            "median_relative_error": float(fom_errors["relative_frobenius_error"].median()),
            "boundary_limited_parameters": (
                sorted(fom_errors.loc[fom_errors.get("boundary_limited", False).astype(bool), "name"].unique().tolist())
                if "boundary_limited" in fom_errors
                else []
            ),
        }
    _write_json(out_dir / "sensitivity_summary.json", summary)

    text = f"""# Fixed-Geometry Sensitivity Check

- Source run: `{run_dir}`
- ROM: `{rom_dir}`
- Material IDs: `{summary['material_ids']}`
- Affine coefficients: `{reduced.COEFF_NAMES}`
- Physical parameters: `{PHYSICAL_NAMES}`

## ROM Self-Checks

| Check | Mean relative error | Max relative error |
|---|---:|---:|
| Affine analytic vs ROM central FD | {summary['rom_affine_selfcheck']['mean_relative_error']:.6e} | {summary['rom_affine_selfcheck']['max_relative_error']:.6e} |
| Physical chain-rule vs ROM central FD | {summary['rom_physical_selfcheck']['mean_relative_error']:.6e} | {summary['rom_physical_selfcheck']['max_relative_error']:.6e} |
"""
    if "fom_physical_fd" in summary:
        text += f"""
## FFTHomPy/CuPy Physical Finite Differences

| Check | Mean relative error | Median relative error | Max relative error |
|---|---:|---:|---:|
| ROM chain-rule vs full-order {summary['fom_physical_fd']['fom_fd_scheme']} FD | {summary['fom_physical_fd']['mean_relative_error']:.6e} | {summary['fom_physical_fd']['median_relative_error']:.6e} | {summary['fom_physical_fd']['max_relative_error']:.6e} |
"""
        if summary["fom_physical_fd"]["boundary_limited_parameters"]:
            text += (
                "\nBoundary-limited parameters: "
                f"`{summary['fom_physical_fd']['boundary_limited_parameters']}`.\n"
            )
    text += """
Detailed tensors and per-parameter errors are stored in the CSV/NPZ files in
this directory. The full-order finite-difference check perturbs physical
material parameters, then compares against the ROM derivative obtained by
chain rule from the seven affine stiffness coefficients.
"""
    (out_dir / "sensitivity_summary.md").write_text(text, encoding="utf-8")

    print(
        "[SENS] listo | "
        f"out={out_dir} | "
        f"affine_self_max={summary['rom_affine_selfcheck']['max_relative_error']:.3e} | "
        f"physical_self_max={summary['rom_physical_selfcheck']['max_relative_error']:.3e}",
        flush=True,
    )
    if "fom_physical_fd" in summary:
        print(
            "[SENS] FOM FD | "
            f"mean={summary['fom_physical_fd']['mean_relative_error']:.3e} | "
            f"max={summary['fom_physical_fd']['max_relative_error']:.3e}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
