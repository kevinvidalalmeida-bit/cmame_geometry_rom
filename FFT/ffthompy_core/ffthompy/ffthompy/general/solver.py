#!/usr/bin/python
import numpy as np
import time
from ffthompy.general.base import Timer
from ffthompy.matvecs import VecTri
from ffthompy.tensors import Tensor
from ffthompy.tensors.fft import (
    CUPY_AVAILABLE,
    cp,
    cupy_synchronize,
    get_array_module,
    is_cupy_array,
    to_backend_array,
    to_host_array,
)
import scipy.sparse.linalg as spslin


_CUPY_CG_UPDATE_XR_KERNEL = None
_CUPY_CG_UPDATE_P_KERNEL = None
_CUPY_CG_BATCH_UPDATE_KERNELS = {}
_CUPY_CG_UPDATE_XR_RR_KERNELS = {}
_CUPY_CG_DOT_KERNELS = {}
CG_PROFILE_LABELS = (
    "initialization",
    "Afun",
    "dot_pap_alpha",
    "update_xr",
    "dot_rr_beta",
    "update_p",
    "residual_check",
    "callback",
)


def _new_cg_profile():
    return {
        label: {"seconds": 0.0, "calls": 0}
        for label in CG_PROFILE_LABELS
    }


def _record_cg_profile(stats, label, started_at):
    cupy_synchronize()
    stats[label]["seconds"] += time.perf_counter() - started_at
    stats[label]["calls"] += 1


def _scalar_to_float(x):
    if CUPY_AVAILABLE and is_cupy_array(x):
        return float(x.item())
    if isinstance(x, np.generic):
        return float(x.item())
    return float(x)


def _sqrt_scalar_to_float(x):
    if CUPY_AVAILABLE and is_cupy_array(x):
        return float(cp.sqrt(x).item())
    return np.double(x) ** 0.5


def _tensor_scalar_product_raw(y, x):
    assert(isinstance(x, Tensor))
    assert(y.val.shape == x.val.shape)
    assert(y.fft_form == x.fft_form)

    xp = get_array_module(y.val, x.val)
    yval = y.val if xp is np else to_backend_array(y.val, prefer_backend='cupy')
    xval = x.val if xp is np else to_backend_array(x.val, prefer_backend='cupy')

    if y.Fourier:
        if x.fft_form in ['r']:
            if x.N[-1] % 2 == 1:
                scal = (
                    xp.sum(yval[..., 0] * xp.conj(xval[..., 0])).real +
                    2 * xp.sum(yval[..., 1:] * xp.conj(xval[..., 1:])).real
                ) / np.prod(y.N) ** 2
            else:
                scal = (
                    xp.sum(yval[..., 0] * xp.conj(xval[..., 0])).real +
                    xp.sum(yval[..., -1] * xp.conj(xval[..., -1])).real +
                    2 * xp.sum(yval[..., 1:-1] * xp.conj(xval[..., 1:-1])).real
                ) / np.prod(y.N) ** 2
        else:
            scal = xp.sum(yval[:] * xp.conj(xval[:])).real
    else:
        scal = xp.sum(yval[:] * xval[:]) / np.prod(y.N)
    return scal


def _use_cupy_lazy_scalars(B, par):
    if not bool(par.get('cupy_lazy_scalars', False)):
        return False
    if 'scal' in par:
        return False
    return isinstance(B, Tensor) and is_cupy_array(B.val)


def _residual_check_every(B, par):
    if not (isinstance(B, Tensor) and is_cupy_array(B.val)):
        return 1
    return max(1, int(par.get('cupy_residual_check_every', 1)))


def _solver_tolerances(par):
    """Return relative/absolute tolerances while preserving the legacy alias."""
    rtol = float(par.get('rtol', par.get('tol', 1e-6)))
    atol = float(par.get('atol', 0.0))
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("CG rtol must be finite and non-negative.")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("CG atol must be finite and non-negative.")
    if rtol == 0.0 and atol == 0.0:
        raise ValueError("CG requires rtol > 0 or atol > 0.")
    return rtol, atol


def _finalize_cg_info(res, rhs_norm, threshold, maxiter):
    norm_res = float(res.get('norm_res', np.inf))
    denom = max(float(rhs_norm), np.finfo(float).tiny)
    res['rhs_norm'] = float(rhs_norm)
    res['threshold'] = float(threshold)
    res['norm_res_rel'] = float(norm_res / denom)
    res['converged'] = bool(np.isfinite(norm_res) and norm_res <= float(threshold))
    res['hit_maxiter'] = bool(res.get('kit', 0) >= int(maxiter) and not res['converged'])
    return res


def _get_cupy_cg_update_kernels():
    global _CUPY_CG_UPDATE_XR_KERNEL, _CUPY_CG_UPDATE_P_KERNEL
    if not CUPY_AVAILABLE or cp is None:
        return None, None

    if _CUPY_CG_UPDATE_XR_KERNEL is None:
        _CUPY_CG_UPDATE_XR_KERNEL = cp.ElementwiseKernel(
            'T alpha, T p, T ap',
            'T x, T r',
            'x += alpha * p; r -= alpha * ap;',
            'ffthompy_cg_update_xr',
        )
    if _CUPY_CG_UPDATE_P_KERNEL is None:
        _CUPY_CG_UPDATE_P_KERNEL = cp.ElementwiseKernel(
            'T beta, T r',
            'T p',
            'p = r + beta * p;',
            'ffthompy_cg_update_p',
        )
    return _CUPY_CG_UPDATE_XR_KERNEL, _CUPY_CG_UPDATE_P_KERNEL


def _cupy_cg_arrays_supported(*tensors):
    vals = []
    for tensor in tensors:
        if not isinstance(tensor, Tensor) or not is_cupy_array(tensor.val):
            return False
        vals.append(tensor.val)
    dtype = vals[0].dtype
    if dtype.kind != 'f':
        return False
    return all(val.dtype == dtype and val.shape == vals[0].shape for val in vals)


def _try_cupy_update_xr(x, alpha, p, r, ap, enabled):
    if not enabled or not _cupy_cg_arrays_supported(x, p, r, ap):
        return False
    kernel_xr, _ = _get_cupy_cg_update_kernels()
    if kernel_xr is None:
        return False
    try:
        kernel_xr(alpha, p.val, ap.val, x.val, r.val)
        return True
    except Exception:
        return False


def _try_cupy_update_p(p, beta, r, enabled):
    if not enabled or not _cupy_cg_arrays_supported(p, r):
        return False
    _, kernel_p = _get_cupy_cg_update_kernels()
    if kernel_p is None:
        return False
    try:
        kernel_p(beta, r.val, p.val)
        return True
    except Exception:
        return False


def _get_cupy_cg_update_xr_rr_kernel(kind):
    kernel = _CUPY_CG_UPDATE_XR_RR_KERNELS.get(kind)
    if kernel is not None:
        return kernel
    if not CUPY_AVAILABLE or cp is None:
        return None

    if kind == "f32":
        ctype = "float"
        atomic = "atomicAdd"
    elif kind == "f64":
        ctype = "double"
        atomic = "atomicAdd"
    else:
        return None

    code = r'''
    extern "C" __global__
    void cg_update_xr_rr(const __TYPE__ alpha,
                         const __TYPE__* __restrict__ p,
                         const __TYPE__* __restrict__ ap,
                         __TYPE__* __restrict__ x,
                         __TYPE__* __restrict__ r,
                         __TYPE__* __restrict__ rr,
                         const long total) {
        extern __shared__ __TYPE__ partial[];
        const long idx = blockDim.x * blockIdx.x + threadIdx.x;
        __TYPE__ value = (__TYPE__)0;
        if (idx < total) {
            x[idx] += alpha * p[idx];
            const __TYPE__ new_r = r[idx] - alpha * ap[idx];
            r[idx] = new_r;
            value = new_r * new_r;
        }
        partial[threadIdx.x] = value;
        __syncthreads();

        for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                partial[threadIdx.x] += partial[threadIdx.x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            __ATOMIC__(rr, partial[0]);
        }
    }
    '''.replace("__TYPE__", ctype).replace("__ATOMIC__", atomic)
    try:
        kernel = cp.RawKernel(code, "cg_update_xr_rr")
    except Exception:
        return None
    _CUPY_CG_UPDATE_XR_RR_KERNELS[kind] = kernel
    return kernel


def _try_cupy_update_xr_rr(x, alpha, p, r, ap, workspace, enabled):
    if not enabled or workspace is None:
        return None
    if not _cupy_cg_arrays_supported(x, p, r, ap):
        return None
    vals = (x.val, p.val, r.val, ap.val)
    if not all(val.flags.c_contiguous for val in vals):
        return None
    if is_cupy_array(alpha):
        return None

    if x.val.dtype == cp.float32:
        kind = "f32"
        alpha_arg = np.float32(alpha)
    elif x.val.dtype == cp.float64:
        kind = "f64"
        alpha_arg = np.float64(alpha)
    else:
        return None

    kernel = _get_cupy_cg_update_xr_rr_kernel(kind)
    if kernel is None:
        return None
    threads = 256
    total = int(x.val.size)
    blocks = (total + threads - 1) // threads
    workspace.fill(0)
    try:
        kernel(
            (blocks,),
            (threads,),
            (
                alpha_arg,
                p.val,
                ap.val,
                x.val,
                r.val,
                workspace,
                np.int64(total),
            ),
            shared_mem=threads * int(x.val.dtype.itemsize),
        )
    except Exception:
        return None
    return float(workspace.item()) / float(np.prod(x.N))


def _get_cupy_cg_dot_kernel(kind):
    kernel = _CUPY_CG_DOT_KERNELS.get(kind)
    if kernel is not None:
        return kernel
    if not CUPY_AVAILABLE or cp is None:
        return None
    if kind == "f32":
        ctype = "float"
    elif kind == "f64":
        ctype = "double"
    else:
        return None

    code = r'''
    extern "C" __global__
    void cg_dot_atomic(const __TYPE__* __restrict__ x,
                       const __TYPE__* __restrict__ y,
                       __TYPE__* __restrict__ out,
                       const long total) {
        extern __shared__ __TYPE__ partial[];
        const long idx = blockDim.x * blockIdx.x + threadIdx.x;
        partial[threadIdx.x] = idx < total ? x[idx] * y[idx] : (__TYPE__)0;
        __syncthreads();
        for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                partial[threadIdx.x] += partial[threadIdx.x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            atomicAdd(out, partial[0]);
        }
    }
    '''.replace("__TYPE__", ctype)
    try:
        kernel = cp.RawKernel(code, "cg_dot_atomic")
    except Exception:
        return None
    _CUPY_CG_DOT_KERNELS[kind] = kernel
    return kernel


def _try_cupy_dot(x, y, workspace, enabled):
    if not enabled or workspace is None:
        return None
    if not _cupy_cg_arrays_supported(x, y):
        return None
    if not (x.val.flags.c_contiguous and y.val.flags.c_contiguous):
        return None
    if x.val.dtype == cp.float32:
        kind = "f32"
    elif x.val.dtype == cp.float64:
        kind = "f64"
    else:
        return None
    kernel = _get_cupy_cg_dot_kernel(kind)
    if kernel is None:
        return None

    threads = 256
    total = int(x.val.size)
    blocks = (total + threads - 1) // threads
    workspace.fill(0)
    try:
        kernel(
            (blocks,),
            (threads,),
            (x.val, y.val, workspace, np.int64(total)),
            shared_mem=threads * int(x.val.dtype.itemsize),
        )
    except Exception:
        return None
    return float(workspace.item()) / float(np.prod(x.N))


def _get_cupy_cg_batch_update_kernel(kind, op):
    key = (kind, op)
    kernel = _CUPY_CG_BATCH_UPDATE_KERNELS.get(key)
    if kernel is not None:
        return kernel
    if not CUPY_AVAILABLE or cp is None:
        return None

    ctype = "float" if kind == "f32" else "double"
    if kind not in {"f32", "f64"}:
        return None

    if op == "xr":
        code = r'''
        extern "C" __global__
        void cg_batch_update_xr(const T* __restrict__ alpha,
                                const T* __restrict__ p,
                                const T* __restrict__ ap,
                                T* __restrict__ x,
                                T* __restrict__ r,
                                const long batch,
                                const long spatial,
                                const long total) {
            const long idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= total) return;
            const long b = (idx / spatial) % batch;
            const T a = alpha[b];
            x[idx] += a * p[idx];
            r[idx] -= a * ap[idx];
        }
        '''.replace("T", ctype)
        name = "cg_batch_update_xr"
    elif op == "p":
        code = r'''
        extern "C" __global__
        void cg_batch_update_p(const T* __restrict__ beta,
                               const T* __restrict__ r,
                               T* __restrict__ p,
                               const long batch,
                               const long spatial,
                               const long total) {
            const long idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= total) return;
            const long b = (idx / spatial) % batch;
            p[idx] = r[idx] + beta[b] * p[idx];
        }
        '''.replace("T", ctype)
        name = "cg_batch_update_p"
    else:
        return None

    try:
        kernel = cp.RawKernel(code, name)
    except Exception:
        return None
    _CUPY_CG_BATCH_UPDATE_KERNELS[key] = kernel
    return kernel


def _cupy_batched_cg_arrays_supported(*tensors):
    if not _cupy_cg_arrays_supported(*tensors):
        return False
    val = tensors[0].val
    return val.ndim >= 3 and val.dtype in (cp.float32, cp.float64)


def _try_cupy_batched_update_xr(x, alpha, p, r, ap, enabled):
    if not enabled or not _cupy_batched_cg_arrays_supported(x, p, r, ap):
        return False
    if not is_cupy_array(alpha):
        return False
    kind = "f32" if x.val.dtype == cp.float32 else "f64"
    kernel = _get_cupy_cg_batch_update_kernel(kind, "xr")
    if kernel is None:
        return False
    batch = int(x.val.shape[1])
    spatial = int(np.prod(x.val.shape[2:]))
    total = int(x.val.size)
    alpha = alpha.astype(x.val.dtype, copy=False)
    if not alpha.flags.c_contiguous:
        alpha = cp.ascontiguousarray(alpha)
    threads = 256
    blocks = (total + threads - 1) // threads
    try:
        kernel((blocks,), (threads,), (alpha, p.val, ap.val, x.val, r.val,
                                      np.int64(batch), np.int64(spatial), np.int64(total)))
        return True
    except Exception:
        return False


def _try_cupy_batched_update_p(p, beta, r, enabled):
    if not enabled or not _cupy_batched_cg_arrays_supported(p, r):
        return False
    if not is_cupy_array(beta):
        return False
    kind = "f32" if p.val.dtype == cp.float32 else "f64"
    kernel = _get_cupy_cg_batch_update_kernel(kind, "p")
    if kernel is None:
        return False
    batch = int(p.val.shape[1])
    spatial = int(np.prod(p.val.shape[2:]))
    total = int(p.val.size)
    beta = beta.astype(p.val.dtype, copy=False)
    if not beta.flags.c_contiguous:
        beta = cp.ascontiguousarray(beta)
    threads = 256
    blocks = (total + threads - 1) // threads
    try:
        kernel((blocks,), (threads,), (beta, r.val, p.val,
                                      np.int64(batch), np.int64(spatial), np.int64(total)))
        return True
    except Exception:
        return False


def _tensor_update_xr(x, alpha, p, r, ap):
    if (
        isinstance(x, Tensor) and isinstance(p, Tensor) and
        isinstance(r, Tensor) and isinstance(ap, Tensor) and
        x.val.shape == p.val.shape == r.val.shape == ap.val.shape
    ):
        x.val += alpha * p.val
        r.val -= alpha * ap.val
        return True
    return False


def _tensor_update_p(p, beta, r):
    if isinstance(p, Tensor) and isinstance(r, Tensor) and p.val.shape == r.val.shape:
        p.val *= beta
        p.val += r.val
        return True
    return False

def linear_solver(Afun, B, ATfun=None, x0=None, par=None,
                  solver=None, callback=None):
    """
    Wraper for various linear solvers suited for FFT-based homogenization.
    """
    tim = Timer('Solving linsys by %s' % solver)
    if x0 is None:
        x0 = B.zeros_like()

    if callback is not None:
        callback(x0)

    if solver.lower() in ['cg']: # conjugate gradients
        if isinstance(B, Tensor) and bool(par.get('batched_rhs', False)):
            x, info = CG_batched_tensor(Afun, B, x0=x0, par=par, callback=callback)
        else:
            x, info = CG(Afun, B, x0=x0, par=par, callback=callback)
    elif solver.lower() in ['bicg']: # biconjugate gradients
        x, info = BiCG(Afun, ATfun, B, x0=x0, par=par, callback=callback)
    elif solver.lower() in ['iterative', 'richardson']: # iterative solver
        x, info = richardson(Afun, B, x0, par=par, callback=callback)
    elif solver.lower() in ['chebyshev', 'cheby']: # iterative solver
        x, info = cheby2TERM(A=Afun, B=B, x0=x0, par=par, callback=callback)
    elif solver.split('_')[0].lower() in ['scipy']: # solvers in scipy
        if isinstance(x0, np.ndarray):
            x0vec=x0.ravel()
        else:
            x0vec=x0.vec()

        if solver in ['scipy.sparse.linalg.cg','scipy_cg']:
            Afun.define_operand(B)
            Afunvec = spslin.LinearOperator(Afun.matshape, matvec=Afun.matvec,
                                            dtype=np.float64)
            xcol, info = spslin.cg(Afunvec, B.vec(), x0=x0vec,
                                   tol=par['tol'], maxiter=par['maxiter'],
                                   M=None, callback=callback)
            info = {'info': info}
        elif solver in ['scipy.sparse.linalg.bicg','scipy_bicg']:
            Afun.define_operand(B)
            ATfun.define_operand(B)
            Afunvec = spslin.LinearOperator(Afun.shape, matvec=Afun.matvec,
                                            rmatvec=ATfun.matvec, dtype=np.float64)
            xcol, info = spslin.bicg(Afunvec, B.vec(), x0=x0.vec(),
                                     tol=par['tol'], maxiter=par['maxiter'],
                                     M=None, callback=callback)
        res = dict()
        res['info'] = info
        x = B.empty_like(name='x')
        x.val = np.reshape(xcol, B.val.shape)
    else:
        msg = "This kind (%s) of linear solver is not implemented" % solver
        raise NotImplementedError(msg)

    cupy_synchronize(x)
    tim.measure(print_time=False)
    keep_solution_on_device = bool(par.get('keep_solution_on_device', False)) if par else False
    if isinstance(x, Tensor) and is_cupy_array(x.val) and not keep_solution_on_device:
        x = x.copy(val=to_host_array(x.val))
    info.update({'time': tim.vals})
    return x, info


def richardson(Afun, B, x0, par=None, callback=None):
    omega = 1./par['alpha']
    res = {'norm_res': 1e15,
           'kit': 0}
    x = x0
    norm=get_norm(B, par)

    while (res['norm_res'] > par['tol'] and res['kit'] < par['maxiter']):
        res['kit'] += 1
        residuum=B-Afun(x)
        x = x + omega*residuum
        res['norm_res'] = norm(residuum)
        if callback is not None:
            callback(x)
    return x, res


def CG(Afun, B, x0, par=None, callback=None):
    """
    Conjugate gradients solver.

    Parameters
    ----------
    Afun : Matrix, LinOper, or numpy.array of shape (n, n)
        it stores the matrix data of linear system and provides a matrix by
        vector multiplication
    B : VecTri or numpy.array of shape (n,)
        it stores a right-hand side of linear system
    x0 : VecTri or numpy.array of shape (n,)
        initial approximation of solution of linear system
    par : dict
        parameters of the method
    callback :

    Returns
    -------
    x : VecTri or numpy.array of shape (n,)
        resulting unknown vector
    res : dict
        results
    """
    if par is None:
        par = dict()
    if 'maxiter' not in list(par.keys()):
        par['maxiter'] = int(1e3)
    rtol, atol = _solver_tolerances(par)

    use_lazy_scalars = _use_cupy_lazy_scalars(B, par)
    scal = _tensor_scalar_product_raw if use_lazy_scalars else get_scal(B, par)
    cupy_fused_cg_updates = bool(par.get('cupy_fused_cg_updates', False))
    cupy_fused_xr_rr = bool(par.get('cupy_fused_xr_rr', False))
    cupy_fused_dot = bool(par.get('cupy_fused_dot', False))
    direct_tensor_updates = (
        use_lazy_scalars or
        (cupy_fused_cg_updates and isinstance(B, Tensor) and is_cupy_array(B.val))
    )
    residual_check_every = _residual_check_every(B, par)
    cg_profile = _new_cg_profile() if bool(par.get('profile_cg_timing', False)) else None
    rr_workspace = None
    dot_workspace = None
    if (
        cupy_fused_xr_rr
        and isinstance(B, Tensor)
        and is_cupy_array(B.val)
        and B.val.dtype in (cp.float32, cp.float64)
    ):
        rr_workspace = cp.empty((1,), dtype=B.val.dtype)
    if (
        cupy_fused_dot
        and isinstance(B, Tensor)
        and is_cupy_array(B.val)
        and B.val.dtype in (cp.float32, cp.float64)
    ):
        dot_workspace = cp.empty((1,), dtype=B.val.dtype)

    res = dict()
    if cg_profile is not None:
        cupy_synchronize()
        profile_t0 = time.perf_counter()
    xCG = x0
    Ax = Afun(x0)
    R = B - Ax
    P = R.copy()
    rr_fused = _try_cupy_dot(R, R, dot_workspace, cupy_fused_dot)
    rr = rr_fused if rr_fused is not None else scal(R,R)
    bb = scal(B, B)
    rhs_norm = _sqrt_scalar_to_float(bb)
    threshold = max(atol, rtol * rhs_norm)
    res['kit'] = 0
    res['norm_res'] = _sqrt_scalar_to_float(rr)
    if cg_profile is not None:
        _record_cg_profile(cg_profile, "initialization", profile_t0)
    norm_res_log = []
    norm_res_log.append(res['norm_res'])
    while (res['norm_res'] > threshold) and (res['kit'] < par['maxiter']):
        res['kit'] += 1 # number of iterations
        if cg_profile is not None:
            cupy_synchronize()
            profile_t0 = time.perf_counter()
        AP = Afun(P)
        if cg_profile is not None:
            _record_cg_profile(cg_profile, "Afun", profile_t0)

        if cg_profile is not None:
            cupy_synchronize()
            profile_t0 = time.perf_counter()
        pap_fused = _try_cupy_dot(P, AP, dot_workspace, cupy_fused_dot)
        pap = pap_fused if pap_fused is not None else scal(P, AP)
        alp = rr / pap
        if not use_lazy_scalars:
            alp = float(alp)
        if cg_profile is not None:
            _record_cg_profile(cg_profile, "dot_pap_alpha", profile_t0)

        if cg_profile is not None:
            cupy_synchronize()
            profile_t0 = time.perf_counter()
        rrnext_fused = _try_cupy_update_xr_rr(
            xCG,
            alp,
            P,
            R,
            AP,
            rr_workspace,
            cupy_fused_xr_rr,
        )
        if rrnext_fused is None:
            if (
                not _try_cupy_update_xr(xCG, alp, P, R, AP, cupy_fused_cg_updates) and
                not (direct_tensor_updates and _tensor_update_xr(xCG, alp, P, R, AP))
            ):
                # xCG = xCG + alp*P  =>  xCG += alp*P
                xCG += alp*P

                # R = R - alp*AP     =>  R -= alp*AP
                R -= alp*AP
        if cg_profile is not None:
            _record_cg_profile(cg_profile, "update_xr", profile_t0)

        if cg_profile is not None:
            cupy_synchronize()
            profile_t0 = time.perf_counter()
        rrnext = rrnext_fused if rrnext_fused is not None else scal(R,R)
        bet = rrnext/rr
        rr = rrnext
        if cg_profile is not None:
            _record_cg_profile(cg_profile, "dot_rr_beta", profile_t0)

        if cg_profile is not None:
            cupy_synchronize()
            profile_t0 = time.perf_counter()
        if (
            not _try_cupy_update_p(P, bet, R, cupy_fused_cg_updates) and
            not (direct_tensor_updates and _tensor_update_p(P, bet, R))
        ):
            # P = R + bet*P      =>  P *= bet; P += R
            P *= bet
            P += R
        if cg_profile is not None:
            _record_cg_profile(cg_profile, "update_p", profile_t0)

        if (res['kit'] % residual_check_every == 0) or (res['kit'] >= par['maxiter']):
            if cg_profile is not None:
                cupy_synchronize()
                profile_t0 = time.perf_counter()
            res['norm_res'] = _sqrt_scalar_to_float(rr)
            norm_res_log.append(res['norm_res'])
            if cg_profile is not None:
                _record_cg_profile(cg_profile, "residual_check", profile_t0)
        if callback is not None:
            if cg_profile is not None:
                cupy_synchronize()
                profile_t0 = time.perf_counter()
            callback(xCG)
            if cg_profile is not None:
                _record_cg_profile(cg_profile, "callback", profile_t0)
    res['norm_res'] = _sqrt_scalar_to_float(rr)
    res['norm_res_log'] = [float(value) for value in norm_res_log]
    _finalize_cg_info(res, rhs_norm, threshold, par['maxiter'])
    if cg_profile is not None:
        res['cg_profile'] = cg_profile
    return xCG, res


def _batched_tensor_scalar_products(X, Y):
    xp = get_array_module(X.val, Y.val)
    xval = X.val if xp is np else to_backend_array(X.val, prefer_backend='cupy')
    yval = Y.val if xp is np else to_backend_array(Y.val, prefer_backend='cupy')

    axes = (0,) + tuple(range(2, xval.ndim))
    if X.Fourier:
        if X.fft_form in ['r']:
            raise NotImplementedError("batched_rhs con fft_form='r' aun no soporta scalar products en Fourier.")
        scal = xp.sum(xval * xp.conj(yval), axis=axes).real
    else:
        scal = xp.sum(xval * yval, axis=axes) / np.prod(X.N)
    return scal


def _batched_scale_view(vals, ndim):
    return vals.reshape((1, int(vals.shape[0])) + (1,) * (ndim - 2))


def CG_batched_tensor(Afun, B, x0, par=None, callback=None):
    """
    Independent CG solves for several right-hand sides stored as Tensor(D, L, *N).

    This is not block-CG; each load has its own alpha/beta/residual, but the
    expensive Afun calls are batched so FFT backends can process multiple loads
    in one call.
    """
    if par is None:
        par = dict()
    if 'maxiter' not in par:
        par['maxiter'] = int(1e3)
    rtol, atol = _solver_tolerances(par)
    if not isinstance(B, Tensor) or B.order != 2:
        raise ValueError("CG_batched_tensor requiere Tensor de orden 2 con shape=(D, n_rhs).")

    cupy_fused_cg_updates = bool(par.get('cupy_fused_cg_updates', False))
    residual_check_every = _residual_check_every(B, par)

    res = dict()
    xCG = x0
    Ax = Afun(x0)
    R = B - Ax
    P = R.copy()
    rr = _batched_tensor_scalar_products(R, R)
    bb = _batched_tensor_scalar_products(B, B)
    xp = get_array_module(rr, bb)
    rhs_norm = xp.sqrt(bb)
    thresholds = xp.maximum(atol, rtol * rhs_norm)
    threshold_sq = thresholds * thresholds
    res['kit'] = 0

    norm_res = xp.sqrt(rr)
    if is_cupy_array(norm_res):
        norm_res_host = to_host_array(norm_res)
    else:
        norm_res_host = np.asarray(norm_res)
    res['norm_res_per_rhs'] = norm_res_host.astype(float).tolist()
    res['norm_res'] = float(np.max(norm_res_host)) if norm_res_host.size else 0.0
    threshold_host = to_host_array(thresholds) if is_cupy_array(thresholds) else np.asarray(thresholds)
    rhs_norm_host = to_host_array(rhs_norm) if is_cupy_array(rhs_norm) else np.asarray(rhs_norm)
    converged_per_rhs = norm_res_host <= threshold_host
    res['converged'] = bool(np.all(converged_per_rhs))

    while (not res['converged']) and (res['kit'] < par['maxiter']):
        res['kit'] += 1
        AP = Afun(P)
        denom = _batched_tensor_scalar_products(P, AP)
        active = rr > threshold_sq
        safe_denom = xp.where(active, denom, 1.0)
        alp = xp.where(active, rr / safe_denom, 0.0)
        if not _try_cupy_batched_update_xr(xCG, alp, P, R, AP, cupy_fused_cg_updates):
            alpha_view = _batched_scale_view(alp, xCG.val.ndim)
            xCG.val += alpha_view * P.val
            R.val -= alpha_view * AP.val

        rrnext = _batched_tensor_scalar_products(R, R)
        safe_rr = xp.where(active, rr, 1.0)
        next_active = rrnext > threshold_sq
        bet = xp.where(active & next_active, rrnext / safe_rr, 0.0)
        rr = rrnext
        if not _try_cupy_batched_update_p(P, bet, R, cupy_fused_cg_updates):
            beta_view = _batched_scale_view(bet, P.val.ndim)
            P.val *= beta_view
            P.val += R.val

        if (res['kit'] % residual_check_every == 0) or (res['kit'] >= par['maxiter']):
            norm_res = xp.sqrt(rr)
            if is_cupy_array(norm_res):
                norm_res_host = to_host_array(norm_res)
            else:
                norm_res_host = np.asarray(norm_res)
            res['norm_res_per_rhs'] = norm_res_host.astype(float).tolist()
            res['norm_res'] = float(np.max(norm_res_host)) if norm_res_host.size else 0.0
            converged_per_rhs = norm_res_host <= threshold_host
            res['converged'] = bool(np.all(converged_per_rhs))

        if callback is not None:
            callback(xCG)

    norm_res = xp.sqrt(rr)
    norm_res_host = to_host_array(norm_res) if is_cupy_array(norm_res) else np.asarray(norm_res)
    denom_host = np.maximum(rhs_norm_host, np.finfo(float).tiny)
    converged_per_rhs = norm_res_host <= threshold_host
    res['norm_res_per_rhs'] = norm_res_host.astype(float).tolist()
    res['norm_res_rel_per_rhs'] = (norm_res_host / denom_host).astype(float).tolist()
    res['rhs_norm_per_rhs'] = rhs_norm_host.astype(float).tolist()
    res['threshold_per_rhs'] = threshold_host.astype(float).tolist()
    res['converged_per_rhs'] = converged_per_rhs.astype(bool).tolist()
    res['norm_res'] = float(np.max(norm_res_host)) if norm_res_host.size else 0.0
    res['norm_res_rel'] = float(np.max(norm_res_host / denom_host)) if norm_res_host.size else 0.0
    res['converged'] = bool(np.all(converged_per_rhs))
    res['hit_maxiter'] = bool(res['kit'] >= int(par['maxiter']) and not res['converged'])
    return xCG, res


def BiCG(Afun, ATfun, B, x0, par=None, callback=None):
    """
    BiConjugate gradient solver.

    Parameters
    ----------
    Afun : Matrix, LinOper, or numpy.array of shape (n, n)
        it stores the matrix data of linear system and provides a matrix by
        vector multiplication
    B : VecTri or numpy.array of shape (n,)
        it stores a right-hand side of linear system
    x0 : VecTri or numpy.array of shape (n,)
        initial approximation of solution of linear system
    par : dict
        parameters of the method
    callback :

    Returns
    -------
    x : VecTri or numpy.array of shape (n,)
        resulting unknown vector
    res : dict
        results
    """
    if par is None:
        par = dict()
    if 'tol' not in par:
        par['tol'] = 1e-6
    if 'maxiter' not in par:
        par['maxiter'] = 1e3

    res = dict()
    xBiCG = x0
    Ax = Afun(x0)
    R = B - Ax
    Rs = R
    rr = float(R.T*Rs)
    P = R
    Ps = Rs
    res['kit'] = 0
    res['norm_res'] = rr**0.5 # /np.norm(E_N)
    norm_res_log = []
    norm_res_log.append(res['norm_res'])
    while (res['norm_res'] > par['tol']) and (res['kit'] < par['maxiter']):
        res['kit'] += 1 # number of iterations
        AP = Afun*P
        alp = rr/float(AP.T*Ps)
        xBiCG = xBiCG + alp*P
        R = R - alp*AP
        Rs = Rs - alp*ATfun*Ps
        rrnext = float(R.T*Rs)
        bet = rrnext/rr
        rr = rrnext
        P = R + bet*P
        Ps = Rs + bet*Ps
        res['norm_res'] = rr**0.5
        norm_res_log.append(res['norm_res'])
        if callback is not None:
            callback(xBiCG)

    if res['kit'] == 0:
        res['norm_res'] = 0
    return xBiCG, res

def cheby2TERM(A, B, x0, M=None, par=None, callback=None):
    """
    Chebyshev two-term iterative solver

    Parameters
    ----------
    Afun : Matrix, LinOper, or numpy.array of shape (n, n)
        it stores the matrix data of linear system and provides a matrix by
        vector multiplication
    B : VecTri or numpy.array of shape (n,)
        it stores a right-hand side of linear system
    x0 : VecTri or numpy.array of shape (n,)
        initial approximation of solution of linear system
    par : dict
        parameters of the method
    callback :

    Returns
    -------
    x : VecTri or numpy.array of shape (n,)
        resulting unknown vector
    res : dict
        results
    """
    if par is None:
        par = dict()
    if 'tol' not in par:
        par['tol'] = 1e-06
    if 'maxit' not in par:
        par['maxit'] = 1e7
    if 'eigrange' not in par:
        raise NotImplementedError("It is necessary to calculate eigenvalues.")
    else:
        Egv = par['eigrange']

    res = dict()
    res['kit'] = 0
    bnrm2 = (B*B)**0.5
    Ib = 1.0/bnrm2
    if bnrm2 == 0:
        bnrm2 = 1.0
    x = x0
    r = B - A(x)
    r0 = np.double(r*r)**0.5
    res['norm_res'] = Ib*r0 # For Normal Residue
    if res['norm_res'] < par['tol']: # if errnorm is less than tol
        return x, res

    d = (Egv[1]+Egv[0])/2.0 # np.mean(par['eigrange'])
    c = (Egv[1]-Egv[0])/2.0 # par['eigrange'][1] - d
    v = 0*x0
    while (res['norm_res'] > par['tol']) and (res['kit'] < par['maxit']):
        res['kit'] += 1
        x_prev = x
        if res['kit'] == 1:
            p = 0
            w = 1/d
        elif res['kit'] == 2:
            p = -(1/2)*(c/d)*(c/d)
            w = 1/(d-c*c/2/d)
        else:
            p = -(c*c/4)*w*w
            w = 1/(d-c*c*w/4)
        v = r - p*v
        x = x_prev + w*v
        r = B - A(x)

        res['norm_res'] = (1.0/r0)*r.norm()

        if callback is not None:
            callback(x)

    if par['tol'] < res['norm_res']: # if tolerance is less than error norm
        print("Chebyshev solver does not converges!")
    else:
        print("Chebyshev solver converges.")

    if res['kit'] == 0:
        res['norm_res'] = 0
    return x, res

def get_scal(B, par):
    "defines scalar multiplication depending on vectors"
    if 'scal' in par:
        scal=par['scal']
    else:
        if isinstance(B, np.matrix) or isinstance(B, VecTri):
            scal = lambda X,Y: float(X.T*Y)
        elif isinstance(B, Tensor):
            scal = lambda X,Y: X*Y
        else:
            scal = lambda X,Y: np.sum(X*Y.conj()).real
    return scal

def get_norm(B, par):
    scal = get_scal(B, par)
    norm = lambda X: scal(X, X)**0.5
    return norm


if __name__ == '__main__':
    exec(compile(open('../main_test.py').read(), '../main_test.py', 'exec'))
