"""Solver FFT para RVEs de fibras cortas TI con FFTHomPy/CuPy."""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
FFTHOMPY_WRAPPER_DIR = WORKSPACE_DIR / "ffthompy_core" / "ffthompy"
if str(FFTHOMPY_WRAPPER_DIR) not in sys.path:
    sys.path.append(str(FFTHOMPY_WRAPPER_DIR))

from ffthompy.general.base import get_base_dir
from ffthompy.mechanics.matcoef import ElasticTensor
from ffthompy.tensors.fft import (
    set_cupy_plan_mode,
    set_fft_backend,
    set_fft_workers,
    get_fft_backend_status,
)
from ffthompy.tensors.operators import set_cupy_fused_matvec, set_cupy_unscaled_fft_pair
from ffthompy.problem import Problem

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore
    _CUPY_AVAILABLE = False

def Enu_to_KG(E: float, nu: float) -> Tuple[float, float]:
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    return K, G

TOL     = 1e-5
MAXITER = 2000

SOLVER_PROFILES: Dict[str, Dict[str, Any]] = {
    "truth": {"solver_real_dtype": "float64", "solver_rtol": 1.0e-10, "solver_atol": 0.0},
    "snapshot": {"solver_real_dtype": "float64", "solver_rtol": 1.0e-8, "solver_atol": 0.0},
    "reference": {"solver_real_dtype": "float64", "solver_rtol": 1.0e-6, "solver_atol": 0.0},
    "reference32": {"solver_real_dtype": "float32", "solver_rtol": 1.0e-6, "solver_atol": 0.0},
    "timing": {"solver_real_dtype": "float32", "solver_rtol": 1.0e-5, "solver_atol": 0.0},
    # Fast feasibility mode aligned with the declared 1e-4 ROM floor.
    # It is not used for high-precision truth validation.
    "rom_floor": {"solver_real_dtype": "float32", "solver_rtol": 1.0e-4, "solver_atol": 0.0},
}

base_dir = get_base_dir()

_CFIELD_ASSIGN_CHUNK_VOXELS = 2_000_000
_SYM21_PAIRS = (
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 2), (2, 3), (2, 4), (2, 5),
    (3, 3), (3, 4), (3, 5),
    (4, 4), (4, 5),
    (5, 5),
)

AFFINE_SENSITIVITY_NAMES = (
    "matrix_lambda",
    "matrix_mu",
    "fiber_C_TT",
    "fiber_C_TT_cross",
    "fiber_C_LT",
    "fiber_C_LL",
    "fiber_G_LT",
)


def _normalize_cupy_plan_mode(mode: str) -> str:
    mode = str(mode).lower().strip()
    if mode == "measure":
        print("  [SOLVER][WARN] cupy_plan_mode='measure' no existe en este ffthompy_core; usando 'manual'.", flush=True)
        return "manual"
    if mode not in {"auto", "manual", "none"}:
        print(f"  [SOLVER][WARN] cupy_plan_mode={mode!r} invalido; usando 'auto'.", flush=True)
        return "auto"
    return mode


def _free_cupy_pools_if_requested(force: bool = False) -> None:
    if cp is None:
        return
    if force:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def _cupy_memory_snapshot(label: str, enabled: bool) -> Dict[str, Any]:
    if not enabled or not _CUPY_AVAILABLE:
        return {}
    try:
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        pinned_pool = cp.get_default_pinned_memory_pool()
        return {
            "label": str(label),
            "device_free_mib": float(free_b) / (1024 ** 2),
            "device_total_mib": float(total_b) / (1024 ** 2),
            "device_used_mib": float(total_b - free_b) / (1024 ** 2),
            "pool_used_mib": float(pool.used_bytes()) / (1024 ** 2),
            "pool_total_mib": float(pool.total_bytes()) / (1024 ** 2),
            "pinned_pool_free_blocks": int(pinned_pool.n_free_blocks()),
        }
    except Exception as exc:
        return {"label": str(label), "error": str(exc)}


def _sync_cupy_for_timing(enabled: bool) -> float:
    if not enabled or not _CUPY_AVAILABLE:
        return 0.0
    t0 = time.perf_counter()
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        return 0.0
    return time.perf_counter() - t0


def _pack_cfield_sym21(Cfield: Any) -> Any:
    if Cfield.shape[0] == 21:
        return Cfield
    if Cfield.shape[0] != 6 or Cfield.shape[1] != 6:
        raise ValueError(f"Cfield no tiene forma compatible para sym21: {Cfield.shape}")
    if _CUPY_AVAILABLE and hasattr(Cfield, "get"):
        return cp.stack([Cfield[ii, jj] for ii, jj in _SYM21_PAIRS], axis=0)
    return np.stack([Cfield[ii, jj] for ii, jj in _SYM21_PAIRS], axis=0)


def _to_numpy_array(value: Any) -> np.ndarray:
    if cp is not None and hasattr(value, "get"):
        return cp.asnumpy(value)
    return np.asarray(value)


def _save_stress_slices_if_requested(
    *,
    prob: Any,
    material_conf: Dict[str, Any],
    Ngrid: np.ndarray,
    solver_fft_form: Any,
    p: Dict[str, Any],
) -> Dict[str, Any]:
    out_path_raw = p.get("stress_slice_out_path")
    if not out_path_raw:
        return {}

    from ffthompy.materials import Material

    out_path = Path(str(out_path_raw))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    load_ids = [int(value) for value in p.get("stress_slice_load_ids", [0])]
    component_ids = [int(value) for value in p.get("stress_slice_components", [0])]
    axis = int(p.get("stress_slice_axis", 2))
    if axis not in {0, 1, 2}:
        raise ValueError("stress_slice_axis debe ser 0, 1 o 2.")
    slice_index = int(p.get("stress_slice_index", int(Ngrid[axis]) // 2))
    slice_index = max(0, min(slice_index, int(Ngrid[axis]) - 1))

    solutions = prob.output.get("sol_primal")
    if solutions is None:
        raise RuntimeError("No hay sol_primal; activa store_solution_fields para guardar cortes de esfuerzo.")

    mat = Material(material_conf)
    A = mat.get_A_GaNi(Ngrid, "primal", fft_form=solver_fft_form)
    component_names = ["sigma11", "sigma22", "sigma33", "sigma23_mandel", "sigma13_mandel", "sigma12_mandel"]

    save_projection = bool(p.get("stress_slice_save_projection", False))

    arrays: Dict[str, np.ndarray] = {}
    for load_id in load_ids:
        solution = solutions[load_id]
        if not hasattr(solution, "val"):
            raise RuntimeError(f"La solucion para load_id={load_id} no esta disponible.")
        stress = A(solution)
        stress_val = stress.val
        for component_id in component_ids:
            if component_id < 0 or component_id >= 6:
                raise ValueError("stress_slice_components debe contener indices entre 0 y 5.")
            component = stress_val[component_id]
            if axis == 0:
                field_slice = component[slice_index, :, :]
            elif axis == 1:
                field_slice = component[:, slice_index, :]
            else:
                field_slice = component[:, :, slice_index]
            key = f"{component_names[component_id]}_load{load_id}"
            arrays[key] = _to_numpy_array(field_slice).astype(np.float32)
            if save_projection:
                component_np = _to_numpy_array(component)
                proj = np.nanmax(component_np, axis=axis)
                arrays[f"{key}_proj"] = proj.astype(np.float32)
        del stress

    metadata = {
        "load_ids": load_ids,
        "component_ids": component_ids,
        "axis": axis,
        "slice_index": slice_index,
        "grid_shape": [int(v) for v in Ngrid],
        "load_description": "load0..load5 = E11,E22,E33,E23,E13,E12 unitarios en notacion Mandel",
        "units": "GPa para deformacion macroscopica unitaria",
        "has_projection": save_projection,
    }
    arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(out_path, **arrays)
    return {"stress_slice_path": str(out_path), **metadata}


def _save_solution_fields_if_requested(
    *,
    prob: Any,
    Ngrid: np.ndarray,
    p: Dict[str, Any],
) -> Dict[str, Any]:
    out_path_raw = p.get("solution_field_out_path")
    return_in_memory = bool(p.get("solution_field_return_in_memory", False))
    field_consumer = p.get("solution_field_consumer")
    consume_in_memory = callable(field_consumer)
    if not out_path_raw and not return_in_memory and not consume_in_memory:
        return {}

    out_path = Path(str(out_path_raw)) if out_path_raw else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    load_ids = [int(value) for value in p.get("solution_field_load_ids", list(range(6)))]
    solutions = prob.output.get("sol_primal")
    if solutions is None:
        raise RuntimeError(
            "No hay sol_primal; activa store_solution_fields para guardar campos tangenciales."
        )

    arrays: Dict[str, np.ndarray] = {}
    save_total = bool(p.get("solution_field_save_total", False))
    requested_field_dtype = p.get("solution_field_dtype", p.get("solver_real_dtype"))
    if requested_field_dtype is None and load_ids:
        requested_field_dtype = _to_numpy_array(solutions[load_ids[0]].val).dtype
    field_dtype = np.dtype(requested_field_dtype or "float32")
    if field_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("solution_field_dtype debe ser float32 o float64.")
    saved_load_ids = []
    field_shape: list[int] = []
    for load_id in load_ids:
        if load_id < 0 or load_id >= len(solutions):
            raise ValueError("solution_field_load_ids debe contener indices validos.")
        solution = solutions[load_id]
        if not hasattr(solution, "val"):
            raise RuntimeError(f"La solucion para load_id={load_id} no esta disponible.")
        total = _to_numpy_array(solution.val).astype(field_dtype, copy=False)
        fluctuation = total.copy()
        fluctuation[load_id] -= field_dtype.type(1.0)
        field_shape = list(fluctuation.shape)
        if consume_in_memory:
            field_consumer(load_id, fluctuation)
        else:
            arrays[f"fluctuation_load{load_id}"] = fluctuation
        if save_total and not consume_in_memory:
            arrays[f"total_load{load_id}"] = total
        saved_load_ids.append(int(load_id))

    metadata = {
        "load_ids": saved_load_ids,
        "grid_shape": [int(v) for v in Ngrid],
        "field_shape": field_shape,
        "field": "zero-mean compatible fluctuation strain in Mandel/Kelvin order used by FFTHomPy",
        "macro_subtracted": True,
        "saved_total_fields": bool(save_total and not consume_in_memory),
        "dtype": str(field_dtype),
        "load_description": "load0..load5 = E11,E22,E33,E23,E13,E12 unitarios en notacion Mandel",
    }
    if consume_in_memory:
        p["_solution_fields_consumed"] = tuple(saved_load_ids)
        return {
            "solution_field_path": "",
            "solution_field_format": "memory_consumer",
            **metadata,
        }
    if return_in_memory:
        p["_solution_fields_result"] = tuple(
            arrays[f"fluctuation_load{load_id}"] for load_id in saved_load_ids
        )
        return {
            "solution_field_path": "",
            "solution_field_format": "memory",
            **metadata,
        }

    if out_path is None:
        raise RuntimeError("Falta solution_field_out_path para guardar los campos.")
    field_format = str(p.get("solution_field_format", "npy_dir")).strip().lower()
    if field_format == "npz_compressed":
        arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=True))
        np.savez_compressed(out_path, **arrays)
        return {"solution_field_path": str(out_path), **metadata}
    if field_format == "npz":
        arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=True))
        np.savez(out_path, **arrays)
        return {"solution_field_path": str(out_path), **metadata}
    if field_format != "npy_dir":
        raise ValueError("solution_field_format debe ser npy_dir, npz o npz_compressed.")

    field_dir = out_path.with_suffix("")
    field_dir.mkdir(parents=True, exist_ok=True)
    for load_id in saved_load_ids:
        np.save(field_dir / f"fluctuation_load{load_id}.npy", arrays[f"fluctuation_load{load_id}"])
        if save_total:
            np.save(field_dir / f"total_load{load_id}.npy", arrays[f"total_load{load_id}"])
    (field_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "solution_field_path": str(out_path),
        "solution_field_dir": str(field_dir),
        **metadata,
    }


def _save_solution_sensitivities_if_requested(
    *,
    prob: Any,
    Ngrid: np.ndarray,
    p: Dict[str, Any],
) -> Dict[str, Any]:
    out_path_raw = p.get("solution_sensitivity_out_path")
    return_in_memory = bool(p.get("solution_sensitivity_return_in_memory", False))
    field_consumer = p.get("solution_sensitivity_consumer")
    consume_in_memory = callable(field_consumer)
    if not out_path_raw and not return_in_memory and not consume_in_memory:
        return {}

    payload = prob.output.get("affine_sensitivity_primal")
    if not isinstance(payload, dict) or "fields" not in payload:
        raise RuntimeError("No hay sensibilidades afines; activa solution_sensitivity_*.")

    fields = np.asarray(payload["fields"])
    coefficient_names = [str(value) for value in payload.get("coefficient_names", AFFINE_SENSITIVITY_NAMES)]
    load_ids = [int(value) for value in payload.get("load_ids", list(range(fields.shape[1])))]
    if fields.ndim != 6 or fields.shape[2] != 6:
        raise RuntimeError(f"Sensibilidades afines con forma incompatible: {fields.shape}.")
    if fields.shape[0] != len(coefficient_names) or fields.shape[1] != len(load_ids):
        raise RuntimeError("Metadatos de sensibilidades afines inconsistentes.")

    requested_field_dtype = p.get(
        "solution_sensitivity_dtype",
        p.get("solution_field_dtype", p.get("solver_real_dtype")),
    )
    field_dtype = np.dtype(requested_field_dtype or fields.dtype)
    if field_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("solution_sensitivity_dtype debe ser float32 o float64.")

    out_path = Path(str(out_path_raw)) if out_path_raw else None
    arrays: Dict[str, np.ndarray] = {}
    consumed: list[tuple[int, int]] = []
    for q, name in enumerate(coefficient_names):
        for local_load, load_id in enumerate(load_ids):
            field = fields[q, local_load].astype(field_dtype, copy=False)
            if consume_in_memory:
                field_consumer(q, name, load_id, field)
                consumed.append((int(q), int(load_id)))
            else:
                arrays[f"{name}_load{load_id}"] = field

    metadata = {
        "coefficient_names": coefficient_names,
        "load_ids": load_ids,
        "grid_shape": [int(v) for v in Ngrid],
        "field_shape": list(fields.shape[2:]),
        "field": "zero-mean exact discrete sensitivity strain d(fluctuation)/dgamma_q",
        "dtype": str(field_dtype),
        "load_description": "load0..load5 = E11,E22,E33,E23,E13,E12 unitarios en notacion Mandel",
        "sensitivity_summary": prob.output.get("affine_sensitivity_summary_primal", {}),
    }
    if consume_in_memory:
        p["_solution_sensitivities_consumed"] = tuple(consumed)
        return {
            "solution_sensitivity_path": "",
            "solution_sensitivity_format": "memory_consumer",
            **metadata,
        }
    if return_in_memory:
        p["_solution_sensitivities_result"] = fields.astype(field_dtype, copy=False)
        return {
            "solution_sensitivity_path": "",
            "solution_sensitivity_format": "memory",
            **metadata,
        }

    if out_path is None:
        raise RuntimeError("Falta solution_sensitivity_out_path para guardar sensibilidades.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=True))
    np.savez_compressed(out_path, **arrays)
    return {
        "solution_sensitivity_path": str(out_path),
        "solution_sensitivity_format": "npz_compressed",
        **metadata,
    }


def _save_stress_volumes_if_requested(
    *,
    prob: Any,
    material_conf: Dict[str, Any],
    Ngrid: np.ndarray,
    solver_fft_form: Any,
    p: Dict[str, Any],
) -> Dict[str, Any]:
    out_path_raw = p.get("stress_volume_out_path")
    if not out_path_raw:
        return {}

    from ffthompy.materials import Material

    out_path = Path(str(out_path_raw))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    load_ids = [int(value) for value in p.get("stress_volume_load_ids", [0])]
    component_ids = [int(value) for value in p.get("stress_volume_components", [0])]
    pair_items = p.get("stress_volume_pairs", None)
    if pair_items is not None:
        pairs = [(int(load_id), int(component_id)) for load_id, component_id in pair_items]
        load_ids = sorted({load_id for load_id, _ in pairs})
    else:
        pairs = [(load_id, component_id) for load_id in load_ids for component_id in component_ids]

    solutions = prob.output.get("sol_primal")
    if solutions is None:
        raise RuntimeError("No hay sol_primal; activa store_solution_fields para guardar volumenes de esfuerzo.")

    mat = Material(material_conf)
    A = mat.get_A_GaNi(Ngrid, "primal", fft_form=solver_fft_form)
    component_names = ["sigma11", "sigma22", "sigma33", "sigma23_mandel", "sigma13_mandel", "sigma12_mandel"]

    arrays: Dict[str, np.ndarray] = {}
    by_load: Dict[int, list[int]] = {}
    for load_id, component_id in pairs:
        by_load.setdefault(int(load_id), []).append(int(component_id))

    for load_id in load_ids:
        solution = solutions[load_id]
        if not hasattr(solution, "val"):
            raise RuntimeError(f"La solucion para load_id={load_id} no esta disponible.")
        stress = A(solution)
        stress_val = stress.val
        for component_id in by_load.get(load_id, []):
            if component_id < 0 or component_id >= 6:
                raise ValueError("stress_volume_components debe contener indices entre 0 y 5.")
            key = f"{component_names[component_id]}_load{load_id}"
            arrays[key] = _to_numpy_array(stress_val[component_id]).astype(np.float32)
        del stress

    metadata = {
        "load_ids": load_ids,
        "component_ids": component_ids,
        "pairs": [[int(load_id), int(component_id)] for load_id, component_id in pairs],
        "grid_shape": [int(v) for v in Ngrid],
        "load_description": "load0..load5 = E11,E22,E33,E23,E13,E12 unitarios en notacion Mandel",
        "units": "GPa para deformacion macroscopica unitaria",
    }
    arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(out_path, **arrays)
    return {"stress_volume_path": str(out_path), **metadata}


def _is_sym21_storage(storage: str) -> bool:
    return str(storage).strip().lower() in {"sym21", "symmetric21", "packed21"}


def _matrix_sym21_values(matrix: Any, xp: Any, dtype: Any) -> Any:
    return xp.asarray([matrix[ii, jj] for ii, jj in _SYM21_PAIRS], dtype=dtype)

def TI_stiffness_voigt(E_L: float, E_T: float, nu_LT: float, nu_TT: float, G_LT: float) -> np.ndarray:
    G_TT = E_T / (2.0 * (1.0 + nu_TT))
    S = np.zeros((6, 6), dtype=float)
    S[0, 0] = 1.0 / E_L;  S[1, 1] = 1.0 / E_T;  S[2, 2] = 1.0 / E_T
    S[0, 1] = S[1, 0] = -nu_LT / E_L
    S[0, 2] = S[2, 0] = -nu_LT / E_L
    S[1, 2] = S[2, 1] = -nu_TT / E_T
    S[3, 3] = 1.0 / G_TT;  S[4, 4] = 1.0 / G_LT;  S[5, 5] = 1.0 / G_LT
    return np.linalg.inv(S)


def voigt_to_mandel(Cv: np.ndarray) -> np.ndarray:
    T = np.diag([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])
    return T @ Cv @ T


def mandel_to_tensor4(Cm: np.ndarray) -> np.ndarray:
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    fac = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)], dtype=float)
    C4 = np.zeros((3, 3, 3, 3), dtype=float)
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            val = Cm[I, J] / (fac[I] * fac[J])
            C4[i, j, k, l] = val;  C4[j, i, k, l] = val
            C4[i, j, l, k] = val;  C4[j, i, l, k] = val
    return C4


def tensor4_to_mandel(C4: np.ndarray) -> np.ndarray:
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    fac = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)], dtype=float)
    Cm = np.zeros((6, 6), dtype=float)
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            Cm[I, J] = fac[I] * fac[J] * C4[i, j, k, l]
    return Cm


def rotation_matrix_from_vector(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    e1 = a / n
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, e1)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e2 = np.cross(e1, tmp)
    e2n = np.linalg.norm(e2)
    if e2n < 1e-12:
        tmp = np.array([0.0, 0.0, 1.0])
        e2 = np.cross(e1, tmp)
        e2n = np.linalg.norm(e2)
        if e2n < 1e-12:
            return np.eye(3)
    e2 /= e2n
    e3 = np.cross(e1, e2)
    return np.column_stack((e1, e2, e3))


def rotate_C_mandel(Cm_local: np.ndarray, R: np.ndarray) -> np.ndarray:
    C4 = mandel_to_tensor4(Cm_local)
    C4g = np.einsum('ip,jq,kr,ls,pqrs->ijkl', R, R, R, R, C4)
    return tensor4_to_mandel(C4g)


def _group_quantized_orientations(
    fiber_orientations: np.ndarray,
    quantization: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Agrupa orientaciones cuantizadas y devuelve indice inverso y medias."""
    orientations = np.asarray(fiber_orientations, dtype=np.float64)
    quantized = np.round(orientations * quantization).astype(np.int64)

    minimum = quantized.min(axis=0)
    shifted = quantized - minimum
    widths = shifted.max(axis=0) + 1
    keys = (shifted[:, 0] * widths[1] + shifted[:, 1]) * widths[2] + shifted[:, 2]

    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    sums = np.column_stack(
        [
            np.bincount(inverse, weights=orientations[:, axis], minlength=len(counts))
            for axis in range(3)
        ]
    )
    means = sums / counts[:, None]
    norms = np.linalg.norm(means, axis=1)
    valid = norms >= 1e-12
    means[valid] /= norms[valid, None]
    means[~valid] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return inverse, means


def _affine_stiffness_bases() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, ...]]:
    """Return the seven Mandel stiffness bases used by the affine ROM."""
    lam = np.zeros((6, 6), dtype=np.float64)
    mu = np.zeros((6, 6), dtype=np.float64)
    lam[:3, :3] = 1.0
    mu[0, 0] = mu[1, 1] = mu[2, 2] = 2.0
    mu[3, 3] = mu[4, 4] = mu[5, 5] = 2.0

    c_tt = np.zeros((6, 6), dtype=np.float64)
    c_tt_cross = np.zeros((6, 6), dtype=np.float64)
    c_lt = np.zeros((6, 6), dtype=np.float64)
    c_ll = np.zeros((6, 6), dtype=np.float64)
    g_lt = np.zeros((6, 6), dtype=np.float64)

    c_tt[1, 1] = c_tt[2, 2] = 1.0
    c_tt[3, 3] = 1.0

    c_tt_cross[1, 2] = c_tt_cross[2, 1] = 1.0
    c_tt_cross[3, 3] = -1.0

    c_lt[0, 1] = c_lt[1, 0] = 1.0
    c_lt[0, 2] = c_lt[2, 0] = 1.0

    c_ll[0, 0] = 1.0

    g_lt[4, 4] = 2.0
    g_lt[5, 5] = 2.0

    return (lam, mu), (c_tt, c_tt_cross, c_lt, c_ll, g_lt)


def _build_affine_sensitivity_cfields_cpu(
    phase: np.ndarray,
    ori: np.ndarray,
    dtype: np.dtype,
    *,
    quantization: float = 1.0e4,
    assign_chunk_voxels: int = _CFIELD_ASSIGN_CHUNK_VOXELS,
) -> np.ndarray:
    """Build full sym21 fields for dA/dgamma_q on CPU."""
    DTYPE = np.dtype(dtype).type
    phase_arr = np.asarray(phase, dtype=np.uint8)
    ori_arr = np.asarray(ori, dtype=np.float64)
    Nx, Ny, Nz = phase_arr.shape
    cfields = np.zeros((len(AFFINE_SENSITIVITY_NAMES), 21, Nx, Ny, Nz), dtype=DTYPE)
    matrix_bases, fiber_bases = _affine_stiffness_bases()

    matrix_mask = phase_arr == 0
    if np.any(matrix_mask):
        for q, basis in enumerate(matrix_bases):
            cfields[q, :, matrix_mask] = _matrix_sym21_values(basis, np, DTYPE)[:, None]

    fiber_mask = phase_arr != 0
    if not np.any(fiber_mask):
        return cfields

    fiber_idx = np.argwhere(fiber_mask).astype(np.int64, copy=False)
    inv_idx, means = _group_quantized_orientations(
        ori_arr[fiber_mask],
        quantization,
    )
    group_bases = np.empty((len(means), len(fiber_bases), 21), dtype=DTYPE)
    for group_id, axis in enumerate(means):
        rotation = rotation_matrix_from_vector(axis)
        for local_q, basis in enumerate(fiber_bases):
            group_bases[group_id, local_q] = _matrix_sym21_values(
                rotate_C_mandel(basis, rotation),
                np,
                DTYPE,
            )

    chunk_size = max(1, int(assign_chunk_voxels))
    n_fiber = int(fiber_idx.shape[0])
    for start in range(0, n_fiber, chunk_size):
        end = min(start + chunk_size, n_fiber)
        group_fi = fiber_idx[start:end]
        ix = group_fi[:, 0]
        iy = group_fi[:, 1]
        iz = group_fi[:, 2]
        cfields[2:, :, ix, iy, iz] = np.transpose(
            group_bases[inv_idx[start:end]],
            (1, 2, 0),
        )
    return cfields


def _build_affine_sensitivity_cfields_gpu(
    phase: np.ndarray,
    ori: np.ndarray,
    dtype: np.dtype,
    *,
    quantization: float = 1.0e4,
    assign_chunk_voxels: int = _CFIELD_ASSIGN_CHUNK_VOXELS,
) -> Any:
    """Build full sym21 fields for dA/dgamma_q on GPU."""
    if cp is None:
        raise RuntimeError("CuPy no esta disponible para construir sensibilidades GPU.")
    gpu_dtype = cp.float64 if np.dtype(dtype) == np.dtype(np.float64) else cp.float32
    phase_arr = np.asarray(phase, dtype=np.uint8)
    ori_arr = np.asarray(ori, dtype=np.float64)
    Nx, Ny, Nz = phase_arr.shape
    cfields = cp.zeros((len(AFFINE_SENSITIVITY_NAMES), 21, Nx, Ny, Nz), dtype=gpu_dtype)
    matrix_bases, fiber_bases = _affine_stiffness_bases()

    matrix_idx = np.argwhere(phase_arr == 0).astype(np.int32, copy=False)
    if matrix_idx.size:
        matrix_idx_gpu = cp.asarray(matrix_idx, dtype=cp.int32)
        ix = matrix_idx_gpu[:, 0]
        iy = matrix_idx_gpu[:, 1]
        iz = matrix_idx_gpu[:, 2]
        for q, basis in enumerate(matrix_bases):
            values = cp.asarray(_matrix_sym21_values(basis, np, np.dtype(dtype).type), dtype=gpu_dtype)
            cfields[q, :, ix, iy, iz] = values[None, :]

    fiber_mask = phase_arr != 0
    if not np.any(fiber_mask):
        return cfields

    fiber_idx = np.argwhere(fiber_mask).astype(np.int32, copy=False)
    inv_idx, means = _group_quantized_orientations(
        ori_arr[fiber_mask],
        quantization,
    )
    group_bases_cpu = np.empty((len(means), len(fiber_bases), 21), dtype=np.dtype(dtype))
    for group_id, axis in enumerate(means):
        rotation = rotation_matrix_from_vector(axis)
        for local_q, basis in enumerate(fiber_bases):
            group_bases_cpu[group_id, local_q] = _matrix_sym21_values(
                rotate_C_mandel(basis, rotation),
                np,
                np.dtype(dtype).type,
            )
    group_bases_gpu = cp.asarray(group_bases_cpu, dtype=gpu_dtype)
    inv_idx_gpu = cp.asarray(inv_idx, dtype=cp.int32)
    fiber_idx_gpu = cp.asarray(fiber_idx, dtype=cp.int32)

    chunk_size = max(1, int(assign_chunk_voxels))
    n_fiber = int(fiber_idx.shape[0])
    for start in range(0, n_fiber, chunk_size):
        end = min(start + chunk_size, n_fiber)
        group_fi = fiber_idx_gpu[start:end]
        ix = group_fi[:, 0]
        iy = group_fi[:, 1]
        iz = group_fi[:, 2]
        cfields[2:, :, ix, iy, iz] = cp.transpose(
            group_bases_gpu[inv_idx_gpu[start:end]],
            (1, 2, 0),
        )
    return cfields


def _build_cfield_gpu(
    phase: np.ndarray,
    ori: np.ndarray,
    Cm: np.ndarray,
    Cf_local: np.ndarray,
    QUANT: float = 1e4,
    storage: str = "full",
    rotation_batch_size: int = 0,
    assign_chunk_voxels: int = _CFIELD_ASSIGN_CHUNK_VOXELS,
    indexed: bool = False,
) -> Tuple[Any, int, float]:
    t0 = time.perf_counter()
    Nx, Ny, Nz = phase.shape
    packed_sym21 = _is_sym21_storage(storage)
    gpu_dtype = cp.float64 if np.asarray(Cm).dtype == np.float64 else cp.float32
    if indexed and not packed_sym21:
        raise ValueError("Cfield indexado requiere storage='sym21'.")

    fiber_mask_cpu = (phase == 1)
    fiber_idx_cpu = np.argwhere(fiber_mask_cpu).astype(np.int32, copy=False)

    Cm_gpu = cp.asarray(Cm, dtype=gpu_dtype)
    if indexed:
        Cm_sym = _matrix_sym21_values(Cm, cp, gpu_dtype)
        Cfield = None
    elif packed_sym21:
        Cm_sym = _matrix_sym21_values(Cm, cp, gpu_dtype)
        Cfield = cp.empty((21, Nx, Ny, Nz), dtype=gpu_dtype)
        Cfield[:] = Cm_sym[:, None, None, None]
    else:
        Cfield = cp.empty((6, 6, Nx, Ny, Nz), dtype=gpu_dtype)
        Cfield[:] = Cm_gpu[:, :, None, None, None]

    if fiber_idx_cpu.size == 0:
        if indexed:
            index_map = cp.zeros((Nx, Ny, Nz), dtype=cp.int32)
            table = cp.ascontiguousarray(Cm_sym[None, :])
            return (index_map, table), 0, time.perf_counter() - t0
        return Cfield, 0, time.perf_counter() - t0

    fiber_ori_cpu = ori[fiber_mask_cpu].astype(np.float64, copy=False)
    inv_idx, means = _group_quantized_orientations(fiber_ori_cpu, QUANT)
    n_unique = len(means)

    R_batch = np.empty((n_unique, 3, 3), dtype=np.float64)
    for g_id in range(n_unique):
        R_batch[g_id] = rotation_matrix_from_vector(means[g_id])

    Cf_local_f64 = Cf_local.astype(np.float64)
    C4_local = mandel_to_tensor4(Cf_local_f64)

    pairs = [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]
    fac = np.array(
        [1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)],
        dtype=np.dtype(Cm.dtype),
    )
    fac_gpu = cp.asarray(fac)
    fac_outer = fac_gpu[:, None] * fac_gpu[None, :]

    C4_gpu = cp.asarray(C4_local, dtype=gpu_dtype)
    rot_chunk = int(rotation_batch_size)
    if rot_chunk <= 0:
        rot_chunk = int(n_unique)
    rot_chunk = max(1, min(rot_chunk, int(n_unique)))

    if packed_sym21:
        Cm_rot = cp.empty((n_unique, 21), dtype=gpu_dtype)
    else:
        Cm_rot = cp.empty((n_unique, 6, 6), dtype=gpu_dtype)

    for rot_start in range(0, n_unique, rot_chunk):
        rot_end = min(rot_start + rot_chunk, n_unique)
        R_gpu = cp.asarray(R_batch[rot_start:rot_end], dtype=gpu_dtype)
        T1 = cp.einsum('gip,pqrs->giqrs', R_gpu, C4_gpu)
        T2 = cp.einsum('gjq,giqrs->gijrs', R_gpu, T1)
        T3 = cp.einsum('gkr,gijrs->gijks', R_gpu, T2)
        C4_rot = cp.einsum('gls,gijks->gijkl', R_gpu, T3)

        if packed_sym21:
            for K, (I, J) in enumerate(_SYM21_PAIRS):
                i, j = pairs[I]
                k, l = pairs[J]
                Cm_rot[rot_start:rot_end, K] = (
                    fac_outer[I, J] * C4_rot[:, i, j, k, l]
                )
        else:
            for I, (i, j) in enumerate(pairs):
                for J, (k, l) in enumerate(pairs):
                    Cm_rot[rot_start:rot_end, I, J] = (
                        fac_outer[I, J] * C4_rot[:, i, j, k, l]
                    )
        del R_gpu, T1, T2, T3, C4_rot

    inv_idx_gpu = cp.asarray(inv_idx, dtype=cp.int32)
    fiber_idx_gpu = cp.asarray(fiber_idx_cpu, dtype=cp.int32)
    if indexed:
        index_map = cp.zeros((Nx, Ny, Nz), dtype=cp.int32)
        table = cp.ascontiguousarray(
            cp.concatenate((Cm_sym[None, :], Cm_rot), axis=0)
        )

    n_fiber = int(fiber_idx_cpu.shape[0])
    chunk_size = max(1, int(assign_chunk_voxels))
    for start in range(0, n_fiber, chunk_size):
        end = min(start + chunk_size, n_fiber)
        group_fi = fiber_idx_gpu[start:end]
        ix = group_fi[:, 0]
        iy = group_fi[:, 1]
        iz = group_fi[:, 2]
        if indexed:
            index_map[ix, iy, iz] = inv_idx_gpu[start:end] + 1
        elif packed_sym21:
            Cfield[:, ix, iy, iz] = cp.transpose(
                Cm_rot[inv_idx_gpu[start:end]],
                (1, 0),
            )
        else:
            Cfield[:, :, ix, iy, iz] = cp.transpose(
                Cm_rot[inv_idx_gpu[start:end]],
                (1, 2, 0),
            )

    cfield_build_s = time.perf_counter() - t0
    if indexed:
        return (index_map, table), n_unique, cfield_build_s
    return Cfield, n_unique, cfield_build_s


def _build_cfield_cpu(
    phase: np.ndarray,
    ori: np.ndarray,
    Cm: np.ndarray,
    Cf_local: np.ndarray,
    DTYPE: type,
    QUANT: float = 1e4,
    storage: str = "full",
) -> Tuple[np.ndarray, int, float]:
    t0 = time.perf_counter()
    Nx, Ny, Nz = phase.shape
    packed_sym21 = _is_sym21_storage(storage)

    if packed_sym21:
        Cm_sym = _matrix_sym21_values(Cm, np, DTYPE)
        Cfield = np.empty((21, Nx, Ny, Nz), dtype=DTYPE)
        Cfield[:] = Cm_sym[:, None, None, None]
    else:
        Cfield = np.empty((6, 6, Nx, Ny, Nz), dtype=DTYPE)
        Cfield[:] = Cm[:, :, None, None, None]

    fiber_mask = (phase == 1)
    fiber_idx  = np.argwhere(fiber_mask)

    if fiber_idx.size == 0:
        return Cfield, 0, time.perf_counter() - t0

    fiber_ori = ori[fiber_mask]

    fiber_ori_f64 = fiber_ori.astype(np.float64, copy=False)
    inv_idx, means = _group_quantized_orientations(fiber_ori_f64, QUANT)
    n_unique = len(means)

    for g_id in range(n_unique):
        group_mask = (inv_idx == g_id)
        R = rotation_matrix_from_vector(means[g_id])
        Cf_rot = rotate_C_mandel(Cf_local.astype(np.float64), R).astype(DTYPE)

        group_fiber_idx = fiber_idx[group_mask]
        ix = group_fiber_idx[:, 0]
        iy = group_fiber_idx[:, 1]
        iz = group_fiber_idx[:, 2]
        if packed_sym21:
            Cf_rot_sym = _matrix_sym21_values(Cf_rot, np, DTYPE)
            Cfield[:, ix, iy, iz] = Cf_rot_sym[:, None]
        else:
            Cfield[:, :, ix, iy, iz] = Cf_rot[:, :, None]

    return Cfield, n_unique, time.perf_counter() - t0


def solve_homogenization(p: Dict[str, Any]) -> np.ndarray:
    """Ejecuta la homogeneizacion FFT y devuelve Ceff."""
    solver_total_t0 = time.perf_counter()
    solver_verbose = bool(p.get("solver_verbose", False))

    profile_name = str(p.get("solver_profile", "")).strip().lower()
    if profile_name and profile_name not in SOLVER_PROFILES:
        valid_profiles = ", ".join(sorted(SOLVER_PROFILES))
        raise ValueError(
            f"solver_profile={profile_name!r} invalido; usa {valid_profiles}."
        )
    profile = SOLVER_PROFILES.get(profile_name, {})

    fft_backend = p.get('fft_backend', 'scipy')
    solver_fft_form_raw = p.get('solver_fft_form', 0)
    solver_fft_form = 0 if str(solver_fft_form_raw).strip() in {'0', 'standard'} else solver_fft_form_raw
    if solver_fft_form not in {0, 'c', 'r'}:
        print(
            f"  [SOLVER][WARN] solver_fft_form={solver_fft_form_raw!r} invalido; usando 0.",
            flush=True,
        )
        solver_fft_form = 0
    use_gpu = (fft_backend == 'cupy') and _CUPY_AVAILABLE
    gpu_memory_profile = bool(p.get("gpu_memory_profile", False)) and use_gpu
    gpu_timing_sync = bool(p.get("gpu_timing_sync", False)) and use_gpu
    gpu_timing_sync_s = 0.0
    cupy_memory_snapshots = []

    def record_gpu_memory(label: str) -> None:
        snapshot = _cupy_memory_snapshot(label, gpu_memory_profile)
        if snapshot:
            cupy_memory_snapshots.append(snapshot)

    record_gpu_memory("start")
    solver_real_dtype = str(p.get(
        'solver_real_dtype',
        profile.get('solver_real_dtype', 'float32' if use_gpu else 'float64'),
    ))
    real_dtype = np.dtype(solver_real_dtype)
    if real_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("solver_real_dtype debe ser 'float32' o 'float64'.")
    DTYPE = real_dtype.type
    use_cfield_material_fast_path = bool(p.get('use_cfield_material_fast_path', True))
    cfield_origin = str(p.get('cfield_origin', 'zero')).strip().lower()
    if cfield_origin not in {'centered', 'c', 'zero', '0', 'native', 'grid'}:
        print(
            f"  [SOLVER][WARN] cfield_origin={cfield_origin!r} invalido; usando 'zero'.",
            flush=True,
        )
        cfield_origin = 'zero'
    cfield_storage = str(p.get('cfield_storage', 'full')).strip().lower()
    if cfield_storage not in {'full', 'sym21', 'symmetric21', 'packed21'}:
        print(
            f"  [SOLVER][WARN] cfield_storage={cfield_storage!r} invalido; usando 'full'.",
            flush=True,
        )
        cfield_storage = 'full'
    if cfield_storage in {'symmetric21', 'packed21'}:
        cfield_storage = 'sym21'
    cfield_rotation_batch_size = max(0, int(p.get('cfield_rotation_batch_size', 0)))
    cfield_assign_chunk_voxels = max(
        1,
        int(p.get('cfield_assign_chunk_voxels', _CFIELD_ASSIGN_CHUNK_VOXELS)),
    )
    cfield_indexed = bool(p.get('cfield_indexed', False))
    if cfield_indexed and (
        not use_gpu
        or cfield_storage != 'sym21'
        or bool(p.get('force_disk_cfield', False))
    ):
        print(
            "  [SOLVER][WARN] cfield_indexed requiere GPU, build optimizado, "
            "storage sym21 y Cfield en memoria; usando campo denso.",
            flush=True,
        )
        cfield_indexed = False
    cupy_fused_matvec = bool(p.get('cupy_fused_matvec', True))
    cupy_unscaled_fft_pair = bool(p.get('cupy_unscaled_fft_pair', True))
    cupy_lazy_scalars = bool(p.get('cupy_lazy_scalars', True))
    cupy_fused_cg_updates = bool(p.get('cupy_fused_cg_updates', True))
    cupy_fused_xr_rr = bool(p.get('cupy_fused_xr_rr', False))
    cupy_fused_dot = bool(p.get('cupy_fused_dot', False))
    cupy_residual_check_every = max(1, int(p.get('cupy_residual_check_every', 1)))
    fast_macro_add = bool(p.get('fast_macro_add', True))
    check_macro_mean = bool(p.get('check_macro_mean', False))
    solution_field_requested = bool(
        p.get("solution_field_out_path")
        or p.get("solution_field_return_in_memory", False)
        or callable(p.get("solution_field_consumer"))
    )
    solution_sensitivity_requested = bool(
        p.get("solution_sensitivity_out_path")
        or p.get("solution_sensitivity_return_in_memory", False)
        or callable(p.get("solution_sensitivity_consumer"))
    )
    stress_slice_requested = bool(p.get("stress_slice_out_path"))
    stress_volume_requested = bool(p.get("stress_volume_out_path"))
    store_solution_fields = bool(
        p.get('store_solution_fields', False)
        or solution_field_requested
        or stress_slice_requested
        or stress_volume_requested
    )
    load_batch_size = max(1, int(p.get('load_batch_size', 1)))
    postprocess_batch_size = max(1, int(p.get('postprocess_batch_size', 6)))
    postprocess_assembly = str(p.get('postprocess_assembly', 'gemm')).strip().lower()
    if postprocess_assembly not in {'scalar', 'einsum', 'gemm'}:
        print(
            f"  [SOLVER][WARN] postprocess_assembly={postprocess_assembly!r} invalido; usando 'gemm'.",
            flush=True,
        )
        postprocess_assembly = 'gemm'
    projection_storage = str(p.get('projection_storage', 'full')).strip().lower()
    if projection_storage not in {'full', 'sym21', 'symmetric21', 'packed21', 'direct', 'formula', 'elastic_direct'}:
        print(
            f"  [SOLVER][WARN] projection_storage={projection_storage!r} invalido; usando 'full'.",
            flush=True,
        )
        projection_storage = 'full'
    if projection_storage in {'symmetric21', 'packed21'}:
        projection_storage = 'sym21'
    projection_backend = str(p.get('projection_backend', 'auto')).strip().lower()
    if projection_backend not in {'auto', 'numpy', 'cpu', 'cupy', 'gpu'}:
        print(
            f"  [SOLVER][WARN] projection_backend={projection_backend!r} invalido; usando 'auto'.",
            flush=True,
        )
        projection_backend = 'auto'
    if projection_backend == 'cpu':
        projection_backend = 'numpy'
    elif projection_backend == 'gpu':
        projection_backend = 'cupy'

    PHASE_PATH = os.path.join(p['input_dir'], "phase.npy")
    ORI_PATH   = os.path.join(p['input_dir'], "ori.npy")

    preloaded = ("phase_array" in p) and ("ori_array" in p)
    if solver_verbose:
        print(
            f"  [SOLVER] RVE | backend={fft_backend} | "
            f"GPU={'SI' if use_gpu else 'NO'} | preloaded={'SI' if preloaded else 'NO'}",
            flush=True,
        )
    io_t0 = time.perf_counter()
    if preloaded:
        phase = np.asarray(p["phase_array"], dtype=np.uint8)
        ori   = np.asarray(p["ori_array"], dtype=DTYPE)
    else:
        phase = np.load(PHASE_PATH).astype(np.uint8)
        ori   = np.load(ORI_PATH).astype(DTYPE)
    input_loading_s = time.perf_counter() - io_t0
    record_gpu_memory("after_input")

    Nx, Ny, Nz = phase.shape
    dim   = 3
    Y     = np.ones(dim, dtype=DTYPE)
    Ngrid = np.array([Nx, Ny, Nz], dtype=np.int32)

    Km_, Gm_ = Enu_to_KG(p['Em'], p['nu_m'])
    mat_m = ElasticTensor(bulk=Km_, mu=Gm_)
    Cm    = mat_m.mandel.astype(DTYPE)

    Cf_voigt = TI_stiffness_voigt(p['Ef_L'], p['Ef_T'], p['nu_LT'], p['nu_TT'], p['G_LT'])
    Cf_local = voigt_to_mandel(Cf_voigt).astype(DTYPE)

    n_unique = 0
    cfield_build_s = 0.0
    Cfield_index = None
    Cfield_table = None
    cfield_precomputed = ("Cfield_array" in p) or ("Cfield_input_path" in p)
    cfield_precomputed_source = ""

    if cfield_precomputed:
        cfield_t0 = time.perf_counter()
        if "Cfield_array" in p:
            Cfield_raw = p["Cfield_array"]
            cfield_precomputed_source = "Cfield_array"
        else:
            cfield_input_path = Path(str(p["Cfield_input_path"]))
            Cfield_raw = np.load(cfield_input_path)
            cfield_precomputed_source = str(cfield_input_path)
        Cfield_shape = tuple(int(value) for value in np.shape(Cfield_raw))
        if Cfield_shape[:1] == (21,) and Cfield_shape[1:] == (Nx, Ny, Nz):
            cfield_storage = "sym21"
        elif Cfield_shape[:2] == (6, 6) and Cfield_shape[2:] == (Nx, Ny, Nz):
            if cfield_storage == "sym21":
                pass
            else:
                cfield_storage = "full"
        else:
            raise ValueError(
                "Cfield precomputado incompatible: "
                f"shape={Cfield_shape}, grid={(Nx, Ny, Nz)}."
            )
        if cfield_indexed:
            print(
                "  [SOLVER][WARN] Cfield precomputado desactiva cfield_indexed.",
                flush=True,
            )
            cfield_indexed = False
        if use_gpu:
            Cfield = cp.asarray(Cfield_raw, dtype=real_dtype)
        else:
            Cfield = np.asarray(Cfield_raw, dtype=real_dtype)
        n_unique = int(p.get("precomputed_cfield_unique_entries", -1))
        cfield_build_s = time.perf_counter() - cfield_t0
        if solver_verbose:
            print(
                f"  [SOLVER] Cfield precomputado | storage={cfield_storage} | "
                f"shape={Cfield_shape} | load={cfield_build_s:.3f}s",
                flush=True,
            )
    elif use_gpu:
        if solver_verbose:
            print(
                f"  [SOLVER] Cfield GPU ({Nx}x{Ny}x{Nz}) | storage={cfield_storage}",
                flush=True,
            )
        Cfield_payload, n_unique, cfield_build_s = _build_cfield_gpu(
            phase,
            ori,
            Cm,
            Cf_local,
            storage=cfield_storage,
            rotation_batch_size=cfield_rotation_batch_size,
            assign_chunk_voxels=cfield_assign_chunk_voxels,
            indexed=cfield_indexed,
        )
        if cfield_indexed:
            Cfield_index, Cfield_table = Cfield_payload
            Cfield = None
        else:
            Cfield = Cfield_payload
        sync_s = _sync_cupy_for_timing(gpu_timing_sync)
        cfield_build_s += sync_s
        gpu_timing_sync_s += sync_s
        if solver_verbose:
            print(
                f"  [SOLVER] Cfield GPU listo | orientaciones={n_unique} | "
                f"build={cfield_build_s:.3f}s",
                flush=True,
            )
    else:
        fiber_mask = (phase == 1)
        n_fiber_voxels = int(fiber_mask.sum())
        if solver_verbose:
            print(
                f"  [SOLVER] Cfield CPU ({Nx}x{Ny}x{Nz}) | "
                f"fibras={n_fiber_voxels} | storage={cfield_storage}",
                flush=True,
            )
        Cfield, n_unique, cfield_build_s = _build_cfield_cpu(
            phase, ori, Cm, Cf_local, DTYPE, storage=cfield_storage
        )

    if cfield_indexed:
        cfield_pack_s = 0.0
        if solver_verbose:
            print(
                "  [SOLVER] Cfield indexado | "
                f"index={tuple(Cfield_index.shape)} | table={tuple(Cfield_table.shape)}",
                flush=True,
            )
    elif cfield_storage == 'sym21':
        if Cfield.shape[0] == 21:
            cfield_pack_s = 0.0
            if solver_verbose:
                print(
                    f"  [SOLVER] Cfield sym21 directo | shape={tuple(Cfield.shape)}",
                    flush=True,
                )
        else:
            pack_t0 = time.perf_counter()
            Cfield = _pack_cfield_sym21(Cfield)
            sync_s = _sync_cupy_for_timing(gpu_timing_sync)
            cfield_pack_s = time.perf_counter() - pack_t0
            gpu_timing_sync_s += sync_s
            if solver_verbose:
                print(
                    f"  [SOLVER] Cfield empaquetado sym21 | shape={tuple(Cfield.shape)} | "
                    f"pack={cfield_pack_s:.3f}s",
                    flush=True,
                )
    else:
        cfield_pack_s = 0.0
    record_gpu_memory("after_cfield")

    affine_sensitivity_cfields = None
    affine_sensitivity_build_s = 0.0
    if solution_sensitivity_requested:
        sens_t0 = time.perf_counter()
        if use_gpu:
            affine_sensitivity_cfields = _build_affine_sensitivity_cfields_gpu(
                phase,
                ori,
                real_dtype,
                assign_chunk_voxels=cfield_assign_chunk_voxels,
            )
            sync_s = _sync_cupy_for_timing(gpu_timing_sync)
            gpu_timing_sync_s += sync_s
        else:
            affine_sensitivity_cfields = _build_affine_sensitivity_cfields_cpu(
                phase,
                ori,
                real_dtype,
                assign_chunk_voxels=cfield_assign_chunk_voxels,
            )
        affine_sensitivity_build_s = time.perf_counter() - sens_t0
        record_gpu_memory("after_affine_sensitivity_cfields")

    force_disk = p.get('force_disk_cfield', False)
    CFIELD_PATH = None

    if force_disk:
        if use_gpu:
            Cfield_cpu = cp.asnumpy(Cfield)
        else:
            Cfield_cpu = Cfield
        CFIELD_PATH = os.path.join(p['input_dir'], f"Cfield_{p['seed']}.npy")
        np.save(CFIELD_PATH, Cfield_cpu)
        del Cfield, Cfield_cpu
        gc.collect()
        material_conf = {
            'Cfield_path': CFIELD_PATH,
            'Y': Y,
            'order': None,
            'P': Ngrid,
            'use_cfield_fast_path': use_cfield_material_fast_path,
            'cfield_origin': cfield_origin,
            'cfield_storage': cfield_storage,
        }
    elif cfield_indexed:
        material_conf = {
            'Cfield_index': Cfield_index,
            'Cfield_table': Cfield_table,
            'Y': Y,
            'order': None,
            'P': Ngrid,
            'use_cfield_fast_path': use_cfield_material_fast_path,
            'cfield_origin': cfield_origin,
            'cfield_storage': 'sym21_indexed',
        }
    else:
        material_conf = {
            'Cfield': Cfield,
            'Y': Y,
            'order': None,
            'P': Ngrid,
            'use_cfield_fast_path': use_cfield_material_fast_path,
            'cfield_origin': cfield_origin,
            'cfield_storage': cfield_storage,
        }

    solver_tol_is_alias = 'solver_rtol' not in p and 'solver_tol' in p
    solver_rtol = float(p.get(
        'solver_rtol',
        p.get('solver_tol', profile.get('solver_rtol', TOL)),
    ))
    solver_atol = float(p.get('solver_atol', profile.get('solver_atol', 0.0)))
    if not np.isfinite(solver_rtol) or solver_rtol < 0.0:
        raise ValueError("solver_rtol debe ser finito y no negativo.")
    if not np.isfinite(solver_atol) or solver_atol < 0.0:
        raise ValueError("solver_atol debe ser finito y no negativo.")
    if solver_rtol == 0.0 and solver_atol == 0.0:
        raise ValueError("Se requiere solver_rtol > 0 o solver_atol > 0.")
    require_convergence = bool(p.get('require_convergence', True))

    if use_gpu:
        cupy_plan_mode = _normalize_cupy_plan_mode(p.get('cupy_plan_mode', 'manual'))
    else:
        cupy_plan_mode = _normalize_cupy_plan_mode(p.get('cupy_plan_mode', 'auto'))

    set_fft_backend(fft_backend)
    set_fft_workers(p.get('fft_workers', 1))
    set_cupy_plan_mode(cupy_plan_mode)
    set_cupy_fused_matvec(cupy_fused_matvec)
    set_cupy_unscaled_fft_pair(cupy_unscaled_fft_pair)

    problem_conf = {
        'name':        'shortfiber_TI_opt',
        'physics':     'elasticity',
        'material':    material_conf,
        'solve':       {
            'kind':          'GaNi',
            'N':             Ngrid,
            'primaldual':    ['primal'],
            'fft_form':      solver_fft_form,
            'fft_backend':    fft_backend,
            'parallel':      p.get('internal_load_parallel', False),
            'parallel_backend': p.get('internal_load_backend', 'auto'),
            'parallel_workers': p.get('internal_load_workers', 2),
            'real_dtype': solver_real_dtype,
            'fast_macro_add': bool(fast_macro_add),
            'check_macro_mean': bool(check_macro_mean),
            'store_solution_fields': bool(store_solution_fields),
            'load_batch_size': int(load_batch_size),
            'postprocess_batch_size': int(postprocess_batch_size),
            'postprocess_assembly': str(postprocess_assembly),
            'projection_storage': str(projection_storage),
            'projection_backend': str(projection_backend),
            'cache_projection': bool(p.get('cache_projection', False)),
            'active_load_ids': p.get('active_load_ids', None),
            'partial_load_output': bool(p.get('partial_load_output', False)),
            'affine_sensitivity_cfields': affine_sensitivity_cfields,
            'affine_sensitivity_names': list(AFFINE_SENSITIVITY_NAMES),
            'affine_sensitivity_batch_size': int(
                p.get('solution_sensitivity_batch_size', load_batch_size)
            ),
        },
        'postprocess': [{'kind': 'GaNi', 'fft_form': solver_fft_form}],
        'solver':      {
            'kind': 'CG',
            'tol': solver_rtol,
            'rtol': solver_rtol,
            'atol': solver_atol,
            'maxiter': int(p.get('solver_maxiter', 1000)),
            'callback': p.get('solver_callback', 'none'),
            'real_dtype': solver_real_dtype,
            'keep_solution_on_device': bool(p.get('keep_solution_on_device', False)),
            'cupy_lazy_scalars': bool(cupy_lazy_scalars),
            'cupy_fused_cg_updates': bool(cupy_fused_cg_updates),
            'cupy_fused_xr_rr': bool(cupy_fused_xr_rr),
            'cupy_fused_dot': bool(cupy_fused_dot),
            'cupy_residual_check_every': int(cupy_residual_check_every),
            'fast_macro_add': bool(fast_macro_add),
            'check_macro_mean': bool(check_macro_mean),
            'load_batch_size': int(load_batch_size),
        },
    }

    prob = Problem(problem_conf)

    if solver_verbose:
        print(
            f"  [SOLVER] CG | rtol={solver_rtol:.0e} | atol={solver_atol:.0e} | "
            f"grid=({Nx},{Ny},{Nz}) | "
            f"fft_form={solver_fft_form} | plan={cupy_plan_mode} | "
            f"cfield={cfield_origin}/{cfield_storage} | "
            f"proj={projection_storage}/{projection_backend}",
            flush=True,
        )

    calculate_t0 = time.perf_counter()
    try:
        prob.calculate()
    except Exception as e:
        if CFIELD_PATH and os.path.exists(CFIELD_PATH):
            try:
                os.remove(CFIELD_PATH)
            except OSError:
                pass
        raise RuntimeError(f"FFTHomPy fallo durante prob.calculate(): {e}") from e
    gpu_timing_sync_s += _sync_cupy_for_timing(gpu_timing_sync)
    problem_calculate_s = time.perf_counter() - calculate_t0
    record_gpu_memory("after_calculate")

    partial_load_output = bool(p.get("partial_load_output", False))
    if partial_load_output:
        problem_postprocessing_s = 0.0
    else:
        postprocess_t0 = time.perf_counter()
        try:
            prob.postprocessing()
        except Exception as e:
            raise RuntimeError(f"FFTHomPy fallo durante prob.postprocessing(): {e}") from e
        gpu_timing_sync_s += _sync_cupy_for_timing(gpu_timing_sync)
        problem_postprocessing_s = time.perf_counter() - postprocess_t0
        record_gpu_memory("after_postprocess")

    if CFIELD_PATH and os.path.exists(CFIELD_PATH):
        try:
            os.remove(CFIELD_PATH)
        except Exception:
            pass

    try:
        if partial_load_output:
            partial_columns = prob.output.get("partial_columns_primal", {})
            active_load_ids = [int(value) for value in p.get("active_load_ids", [])]
            if active_load_ids and partial_columns:
                Ceff = np.column_stack(
                    [np.asarray(partial_columns[int(iL)], dtype=float) for iL in active_load_ids]
                )
            else:
                raise RuntimeError("No se generaron todas las columnas parciales solicitadas.")
        else:
            res_key = list(prob.output['mat_primal'].keys())[0]
            Ceff = prob.output['mat_primal'][res_key]
    except Exception as e:
        raise RuntimeError(f"Ceff ausente o incompleto: {e}") from e

    if cp is not None and hasattr(Ceff, "get"):
        Ceff_to_save = cp.asnumpy(Ceff)
    else:
        Ceff_to_save = np.asarray(Ceff)

    Ceff_to_save = np.asarray(Ceff_to_save, dtype=np.float64)
    expected_columns = len(p.get("active_load_ids") or []) if partial_load_output else 6
    if Ceff_to_save.shape != (6, expected_columns):
        raise RuntimeError(
            f"Ceff tiene forma {Ceff_to_save.shape}; se esperaba (6, {expected_columns})."
        )
    if not np.all(np.isfinite(Ceff_to_save)):
        raise RuntimeError("Ceff contiene valores NaN o infinitos.")

    application_timing = prob.output.get("application_timing", {})
    load_summary = (
        application_timing.get("primaldual", {})
        .get("primal", {})
        .get("load_solver_summary", {})
    )
    sensitivity_summary = prob.output.get("affine_sensitivity_summary_primal", {})
    if require_convergence:
        if not load_summary:
            raise RuntimeError("FFTHomPy no entrego diagnosticos de convergencia por carga.")
        if int(load_summary.get("load_count", -1)) != expected_columns:
            raise RuntimeError(
                "Numero incompleto de cargas FFT: "
                f"{load_summary.get('load_count')} de {expected_columns}."
            )
        if not bool(load_summary.get("all_converged", False)):
            raise RuntimeError(
                "Una o mas cargas FFT no convergieron: "
                f"residual_rel_max={load_summary.get('final_norm_res_rel_max')}, "
                f"maxiter={load_summary.get('maxiter')}."
            )
        if solution_sensitivity_requested:
            if not sensitivity_summary:
                raise RuntimeError("FFTHomPy no entrego diagnosticos de sensibilidad afin.")
            sensitivity_residual = float(sensitivity_summary.get("final_norm_res_rel_max", np.nan))
            maximum_allowed = 1.05 * solver_rtol
            if (
                not bool(sensitivity_summary.get("all_converged", False))
                or not np.isfinite(sensitivity_residual)
                or sensitivity_residual > maximum_allowed
            ):
                raise RuntimeError(
                    "Una o mas sensibilidades afines no convergieron: "
                    f"residual_rel_max={sensitivity_summary.get('final_norm_res_rel_max')}, "
                    f"allowed={maximum_allowed:.3e}."
                )

    ceff_diagnostics: Dict[str, Any] = {
        "shape": [int(value) for value in Ceff_to_save.shape],
        "finite": True,
        "complete": True,
    }
    if not partial_load_output:
        ceff_scale = max(float(np.linalg.norm(Ceff_to_save, ord="fro")), np.finfo(float).tiny)
        symmetry_defect = float(
            np.linalg.norm(Ceff_to_save - Ceff_to_save.T, ord="fro") / ceff_scale
        )
        symmetry_tol = float(p.get(
            "ceff_symmetry_rtol",
            max(10.0 * solver_rtol, 100.0 * np.finfo(real_dtype).eps),
        ))
        Ceff_symmetric = 0.5 * (Ceff_to_save + Ceff_to_save.T)
        eigenvalues = np.linalg.eigvalsh(Ceff_symmetric)
        min_eigenvalue = float(eigenvalues[0])
        spd_atol = float(p.get("ceff_spd_atol", 0.0))
        ceff_diagnostics.update({
            "symmetry_defect_rel_fro": symmetry_defect,
            "symmetry_tolerance": symmetry_tol,
            "minimum_eigenvalue": min_eigenvalue,
            "maximum_eigenvalue": float(eigenvalues[-1]),
            "spd_atol": spd_atol,
        })
        if symmetry_defect > symmetry_tol:
            raise RuntimeError(
                f"Ceff no es suficientemente simetrico: {symmetry_defect:.3e} > {symmetry_tol:.3e}."
            )
        if min_eigenvalue <= spd_atol:
            raise RuntimeError(
                f"Ceff no es SPD: lambda_min={min_eigenvalue:.6e} <= {spd_atol:.6e}."
            )
        if bool(p.get("symmetrize_ceff", True)):
            Ceff_to_save = Ceff_symmetric

    solution_field_info: Dict[str, Any] = {}
    if solution_field_requested:
        solution_field_info = _save_solution_fields_if_requested(
            prob=prob,
            Ngrid=Ngrid,
            p=p,
        )
    solution_sensitivity_info: Dict[str, Any] = {}
    if solution_sensitivity_requested:
        solution_sensitivity_info = _save_solution_sensitivities_if_requested(
            prob=prob,
            Ngrid=Ngrid,
            p=p,
        )

    stress_slice_info: Dict[str, Any] = {}
    if stress_slice_requested:
        stress_slice_info = _save_stress_slices_if_requested(
            prob=prob,
            material_conf=material_conf,
            Ngrid=Ngrid,
            solver_fft_form=solver_fft_form,
            p=p,
        )
    stress_volume_info: Dict[str, Any] = {}
    if stress_volume_requested:
        stress_volume_info = _save_stress_volumes_if_requested(
            prob=prob,
            material_conf=material_conf,
            Ngrid=Ngrid,
            solver_fft_form=solver_fft_form,
            p=p,
        )

    record_gpu_memory("after_save")
    if solver_verbose:
        print(
            f"  [SOLVER] Ceff calculado | "
            f"CG={problem_calculate_s:.2f}s | post={problem_postprocessing_s:.2f}s | "
            f"total={time.perf_counter() - solver_total_t0:.2f}s",
            flush=True,
        )

    solver_timing_path = p.get("solver_timing_path")
    if solver_timing_path:
        fiber_idx_size = int((phase == 1).sum())
        timing_payload = {
            "grid_shape": [int(Nx), int(Ny), int(Nz)],
            "n_fiber_voxels": fiber_idx_size,
            "n_unique_orientations": int(n_unique),
            "input_loading_s": float(input_loading_s),
            "preloaded_geometry": bool(preloaded),
            "cfield_build_s": float(cfield_build_s),
            "cfield_pack_s": float(cfield_pack_s),
            "cfield_precomputed": bool(cfield_precomputed),
            "cfield_precomputed_source": str(cfield_precomputed_source),
            "cfield_origin": str(cfield_origin),
            "cfield_storage": str(cfield_storage),
            "affine_sensitivity_requested": bool(solution_sensitivity_requested),
            "affine_sensitivity_build_s": float(affine_sensitivity_build_s),
            "cfield_rotation_batch_size": int(cfield_rotation_batch_size),
            "cfield_assign_chunk_voxels": int(cfield_assign_chunk_voxels),
            "cfield_indexed": bool(cfield_indexed),
            "use_cfield_material_fast_path": bool(use_cfield_material_fast_path),
            "cupy_fused_matvec": bool(cupy_fused_matvec),
            "cupy_unscaled_fft_pair": bool(cupy_unscaled_fft_pair),
            "cupy_lazy_scalars": bool(cupy_lazy_scalars),
            "cupy_fused_cg_updates": bool(cupy_fused_cg_updates),
            "cupy_fused_xr_rr": bool(cupy_fused_xr_rr),
            "cupy_fused_dot": bool(cupy_fused_dot),
            "cupy_residual_check_every": int(cupy_residual_check_every),
            "fast_macro_add": bool(fast_macro_add),
            "check_macro_mean": bool(check_macro_mean),
            "store_solution_fields": bool(store_solution_fields),
            "load_batch_size": int(load_batch_size),
            "postprocess_batch_size": int(postprocess_batch_size),
            "postprocess_assembly": str(postprocess_assembly),
            "projection_storage": str(projection_storage),
            "projection_backend": str(projection_backend),
            "cache_projection": bool(p.get("cache_projection", False)),
            "problem_calculate_s": float(problem_calculate_s),
            "problem_postprocessing_s": float(problem_postprocessing_s),
            "solver_total_wall_s": float(time.perf_counter() - solver_total_t0),
            "force_disk_cfield": bool(force_disk),
            "solver_tol": float(solver_rtol),
            "solver_rtol": float(solver_rtol),
            "solver_atol": float(solver_atol),
            "solver_tol_is_legacy_alias": bool(solver_tol_is_alias),
            "solver_profile": profile_name or "custom",
            "require_convergence": bool(require_convergence),
            "solver_maxiter": int(p.get("solver_maxiter", 1000)),
            "solver_real_dtype": solver_real_dtype,
            "keep_solution_on_device": bool(p.get('keep_solution_on_device', False)),
            "fft_backend": get_fft_backend_status(),
            "solver_fft_form": str(solver_fft_form),
            "cupy_available": bool(_CUPY_AVAILABLE),
            "cupy_used_for_cfield": bool(use_gpu),
            "cupy_plan_mode": str(cupy_plan_mode),
            "gpu_memory_profile": bool(gpu_memory_profile),
            "cupy_memory_snapshots": cupy_memory_snapshots,
            "gpu_timing_sync": bool(gpu_timing_sync),
            "gpu_timing_sync_s": float(gpu_timing_sync_s),
            "fft_workers": int(p.get("fft_workers", 1)),
            "internal_load_parallel": bool(p.get("internal_load_parallel", False)),
            "internal_load_backend": p.get("internal_load_backend", "auto"),
            "internal_load_workers": int(p.get("internal_load_workers", 2)),
            "active_load_ids": [
                int(value) for value in (p.get("active_load_ids") or [])
            ],
            "partial_load_output": bool(partial_load_output),
            "solver_callback": str(p.get("solver_callback", "none")),
            "solution_field_info": solution_field_info,
            "solution_sensitivity_info": solution_sensitivity_info,
            "affine_sensitivity_summary": sensitivity_summary,
            "stress_slice_info": stress_slice_info,
            "stress_volume_info": stress_volume_info,
            "ffthompy_solver_timing": prob.output.get("solver_timing", {}),
            "ffthompy_application_timing": application_timing,
            "load_solver_summary": load_summary,
            "ceff_diagnostics": ceff_diagnostics,
        }
        with open(solver_timing_path, "w", encoding="utf-8") as fh:
            json.dump(timing_payload, fh, indent=2)

    if bool(p.get("free_gpu_memory_after_solve", False)):
        _free_cupy_pools_if_requested(True)

    return Ceff_to_save
