from __future__ import annotations

import numpy as np
import scipy.fft as scipy_fft

try:
    import pyfftw.interfaces.scipy_fft as pyfftw_fft
    import pyfftw.interfaces.cache as pyfftw_cache

    pyfftw_cache.enable()
    PYFFTW_AVAILABLE = True
except ImportError:
    pyfftw_fft = None
    pyfftw_cache = None
    PYFFTW_AVAILABLE = False

try:
    import cupy as cp
    import cupyx.scipy.fft as cupy_fft
    from cupyx.scipy.fft import get_fft_plan as cupy_get_fft_plan

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cupy_fft = None
    cupy_get_fft_plan = None
    CUPY_AVAILABLE = False


FFT_WORKERS = 1
FFT_BACKEND = "scipy"
CUPY_PLAN_MODE = "auto"  # auto | manual | none
_CUPY_PLAN_CACHE = {}


def set_fft_workers(n):
    global FFT_WORKERS
    FFT_WORKERS = max(1, int(n))


def set_fft_backend(name):
    global FFT_BACKEND
    if name not in {"scipy", "pyfftw", "auto", "cupy"}:
        raise ValueError("FFT backend debe ser 'scipy', 'pyfftw', 'cupy' o 'auto'.")
    FFT_BACKEND = name


def set_cupy_plan_mode(mode):
    global CUPY_PLAN_MODE
    if mode not in {"auto", "manual", "none"}:
        raise ValueError("CUPY_PLAN_MODE debe ser 'auto', 'manual' o 'none'.")
    CUPY_PLAN_MODE = mode
    _CUPY_PLAN_CACHE.clear()
    if CUPY_AVAILABLE:
        cp.fft.config.enable_nd_planning = mode != "none"


def get_fft_backend():
    if FFT_BACKEND == "auto":
        if PYFFTW_AVAILABLE:
            return "pyfftw"
        return "scipy"
    if FFT_BACKEND == "pyfftw" and not PYFFTW_AVAILABLE:
        raise RuntimeError("FFT_BACKEND='pyfftw' pero pyfftw no esta instalado.")
    if FFT_BACKEND == "cupy" and not CUPY_AVAILABLE:
        raise RuntimeError("FFT_BACKEND='cupy' pero cupy no esta instalado.")
    return FFT_BACKEND


def get_fft_backend_status():
    return {
        "requested": FFT_BACKEND,
        "resolved": get_fft_backend(),
        "pyfftw_available": PYFFTW_AVAILABLE,
        "cupy_available": CUPY_AVAILABLE,
        "cupy_plan_mode": CUPY_PLAN_MODE,
    }


def is_cupy_array(x) -> bool:
    return bool(CUPY_AVAILABLE and isinstance(x, cp.ndarray))


def is_array_type(x) -> bool:
    if isinstance(x, np.ndarray):
        return True
    return is_cupy_array(x)


def _has_cupy_payload(obj) -> bool:
    if is_cupy_array(obj):
        return True
    return is_cupy_array(getattr(obj, "val", None))


def cupy_synchronize(*objs) -> bool:
    if not CUPY_AVAILABLE:
        return False

    if objs:
        needs_sync = any(_has_cupy_payload(obj) for obj in objs)
    else:
        try:
            needs_sync = get_fft_backend() == "cupy"
        except Exception:
            needs_sync = False

    if not needs_sync:
        return False

    try:
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        cp.cuda.Stream.null.synchronize()
    return True


def get_array_module(*xs):
    for x in xs:
        if is_cupy_array(x):
            return cp
    return np


def _array_module_shift(x):
    return cp.fft if is_cupy_array(x) else np.fft


def to_backend_array(x, prefer_backend: str | None = None, copy: bool = False):
    if prefer_backend is None:
        prefer_backend = get_fft_backend()

    if prefer_backend == "cupy":
        if not CUPY_AVAILABLE:
            raise RuntimeError("Backend CuPy solicitado pero cupy no esta disponible.")
        if is_cupy_array(x):
            return cp.array(x, copy=copy) if copy else x
        return cp.array(x, copy=copy) if copy else cp.asarray(x)

    if is_cupy_array(x):
        arr = cp.asnumpy(x)
        return np.array(arr, copy=copy) if copy else arr
    return np.array(x, copy=copy) if copy else np.asarray(x)


def to_host_array(x, copy: bool = False):
    if is_cupy_array(x):
        arr = cp.asnumpy(x)
        return np.array(arr, copy=copy) if copy else arr
    return np.array(x, copy=copy) if copy else np.asarray(x)


def _normalize_fft_args(N):
    N_tuple = tuple(int(v) for v in np.asarray(N, dtype=np.int32).tolist())
    prodN = int(np.prod(N_tuple))
    axes = tuple(range(-len(N_tuple), 0))
    return N_tuple, prodN, axes


def _cupy_fft_plan_key(x, axes, value_type, shape=None):
    input_shape = tuple(int(v) for v in x.shape)
    transform_shape = None if shape is None else tuple(int(v) for v in shape)
    return (input_shape, str(x.dtype), tuple(axes), value_type, transform_shape)


def _maybe_cupy_plan(x, axes, value_type, shape=None):
    if not is_cupy_array(x) or CUPY_PLAN_MODE != "manual":
        return None

    key = _cupy_fft_plan_key(x, axes, value_type, shape=shape)
    plan = _CUPY_PLAN_CACHE.get(key)
    if plan is None:
        plan = cupy_get_fft_plan(
            x,
            shape=shape,
            axes=axes,
            value_type=value_type,
        )
        _CUPY_PLAN_CACHE[key] = plan
    return plan


def _fft_module_for_value(x):
    backend = get_fft_backend()
    if backend == "cupy" or is_cupy_array(x):
        if not CUPY_AVAILABLE:
            raise RuntimeError("Backend CuPy solicitado pero cupy no esta disponible.")
        return cupy_fft
    if backend == "pyfftw":
        return pyfftw_fft
    return scipy_fft


def _fftshift_like(x, arr, axes):
    shift_mod = _array_module_shift(x)
    return shift_mod.fftshift(arr, axes=axes)


def _ifftshift_like(x, arr, axes):
    shift_mod = _array_module_shift(x)
    return shift_mod.ifftshift(arr, axes=axes)


def _cupy_complex_dtype(x):
    if not is_cupy_array(x):
        return None
    return cp.complex64 if x.dtype in (cp.float32, cp.complex64) else cp.complex128


def cfftnc_cached(x, N_tuple, prodN, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    shifted = _ifftshift_like(x, x, axes)
    if is_cupy_array(shifted):
        x_fft = shifted.astype(_cupy_complex_dtype(shifted), copy=False)
        plan = _maybe_cupy_plan(x_fft, axes, "C2C")
        out = fft_mod.fftn(x_fft, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.fftn(x_fft, s=N_tuple, axes=axes)
    else:
        out = fft_mod.fftn(shifted, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return (1.0 / prodN) * _fftshift_like(x, out, axes)


def icfftnc_cached(Fx, N_tuple, prodN, axes):
    Fx = to_backend_array(Fx, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(Fx)
    shifted = _ifftshift_like(Fx, Fx, axes)
    if is_cupy_array(shifted):
        plan = _maybe_cupy_plan(shifted.astype(_cupy_complex_dtype(shifted), copy=False), axes, "C2C")
        out = fft_mod.ifftn(shifted, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.ifftn(shifted, s=N_tuple, axes=axes)
        return _fftshift_like(Fx, out, axes).real * prodN
    out = fft_mod.ifftn(shifted, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return _fftshift_like(Fx, out, axes).real * prodN


def fftnc_cached(x, N_tuple, prodN, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        x_fft = x.astype(_cupy_complex_dtype(x), copy=False)
        plan = _maybe_cupy_plan(x_fft, axes, "C2C")
        out = fft_mod.fftn(x_fft, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.fftn(x_fft, s=N_tuple, axes=axes)
    else:
        out = fft_mod.fftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return (1.0 / prodN) * _fftshift_like(x, out, axes)


def icfftn_cached(Fx, N_tuple, prodN, axes):
    Fx = to_backend_array(Fx, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(Fx)
    shifted = _ifftshift_like(Fx, Fx, axes)
    if is_cupy_array(shifted):
        plan = _maybe_cupy_plan(shifted.astype(_cupy_complex_dtype(shifted), copy=False), axes, "C2C")
        out = fft_mod.ifftn(shifted, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.ifftn(shifted, s=N_tuple, axes=axes)
    else:
        out = fft_mod.ifftn(shifted, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return out.real * prodN


def fftn_cached(x, N_tuple, prodN, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        x_fft = x.astype(_cupy_complex_dtype(x), copy=False)
        plan = _maybe_cupy_plan(x_fft, axes, "C2C")
        out = fft_mod.fftn(x_fft, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.fftn(x_fft, s=N_tuple, axes=axes)
    else:
        out = fft_mod.fftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return (1.0 / prodN) * out


def fftn_unscaled_cached(x, N_tuple, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        x_fft = x.astype(_cupy_complex_dtype(x), copy=False)
        plan = _maybe_cupy_plan(x_fft, axes, "C2C")
        return fft_mod.fftn(x_fft, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.fftn(x_fft, s=N_tuple, axes=axes)
    return fft_mod.fftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)


def ifftn_cached(x, N_tuple, prodN, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        plan = _maybe_cupy_plan(x.astype(_cupy_complex_dtype(x), copy=False), axes, "C2C")
        out = fft_mod.ifftn(x, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.ifftn(x, s=N_tuple, axes=axes)
    else:
        out = fft_mod.ifftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return out.real * prodN


def ifftn_unscaled_real_cached(x, N_tuple, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        plan = _maybe_cupy_plan(x.astype(_cupy_complex_dtype(x), copy=False), axes, "C2C")
        out = fft_mod.ifftn(x, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.ifftn(x, s=N_tuple, axes=axes)
    else:
        out = fft_mod.ifftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return out.real


def rfftn_cached(x, N_tuple, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        plan = _maybe_cupy_plan(x, axes, "R2C")
        return fft_mod.rfftn(x, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.rfftn(x, s=N_tuple, axes=axes)
    return fft_mod.rfftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)


def irfftn_cached(x, N_tuple, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        plan = _maybe_cupy_plan(x, axes, "C2R", shape=N_tuple)
        return fft_mod.irfftn(x, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.irfftn(x, s=N_tuple, axes=axes)
    return fft_mod.irfftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)


def cfftnc(x, N):
    N_tuple, prodN, axes = _normalize_fft_args(N)
    return cfftnc_cached(x, N_tuple, prodN, axes)


def icfftnc(Fx, N):
    N_tuple, prodN, axes = _normalize_fft_args(N)
    return icfftnc_cached(Fx, N_tuple, prodN, axes)


def fftnc(x, N):
    N_tuple, prodN, axes = _normalize_fft_args(N)
    return fftnc_cached(x, N_tuple, prodN, axes)


def icfftn(Fx, N):
    N_tuple, prodN, axes = _normalize_fft_args(N)
    return icfftn_cached(Fx, N_tuple, prodN, axes)


def fftn(x, N):
    N_tuple, prodN, axes = _normalize_fft_args(N)
    return fftn_cached(x, N_tuple, prodN, axes)


def ifftn(x, N):
    N_tuple, prodN, axes = _normalize_fft_args(N)
    return ifftn_cached(x, N_tuple, prodN, axes)


def rfftn(x, N):
    N_tuple, _, axes = _normalize_fft_args(N)
    return rfftn_cached(x, N_tuple, axes)


def irfftn(x, N):
    N_tuple, _, axes = _normalize_fft_args(N)
    return irfftn_cached(x, N_tuple, axes)
