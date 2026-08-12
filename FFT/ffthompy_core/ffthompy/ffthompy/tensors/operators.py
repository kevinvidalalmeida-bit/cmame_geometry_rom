"""
This module contains operators working with Tensor from ffthompy.tensors.objects
"""

import itertools
import numpy as np
import numpy.matlib as npmatlib
from ffthompy.trigpol import Grid, fft_form_default
from ffthompy.tensors.objects import (
    SYM21_PAIRS,
    Tensor,
    TensorFuns,
    elasticity_direct_array,
    indexed_sym21_array,
)
from ffthompy.tensors.fft import *
from copy import copy


_CUPY_FUSED_MATVEC = True
_CUPY_UNSCALED_FFT_PAIR = True
_CUPY_MATVEC21_KERNELS = {}
_CUPY_MATVEC21_BATCH_KERNELS = {}
_CUPY_SYM21_KERNELS = {}


def set_cupy_fused_matvec(enabled):
    global _CUPY_FUSED_MATVEC
    _CUPY_FUSED_MATVEC = bool(enabled)


def set_cupy_unscaled_fft_pair(enabled):
    global _CUPY_UNSCALED_FFT_PAIR
    _CUPY_UNSCALED_FFT_PAIR = bool(enabled)


def _unwrap_timed(op):
    return getattr(op, 'op', op)


def _tensor_val_on_cupy(tensor):
    base = _unwrap_timed(tensor)
    if isinstance(base, Tensor) and is_cupy_array(base.val):
        return base.val
    cache_owner = base if isinstance(base, Tensor) else tensor
    cached = getattr(cache_owner, '_cupy_val_cache', None)
    if cached is None:
        raw = base.val if isinstance(base, Tensor) else tensor.val
        cached = to_backend_array(raw, prefer_backend='cupy')
        setattr(cache_owner, '_cupy_val_cache', cached)
    return cached


def _get_cupy_matvec21_kernel(kind):
    kernel = _CUPY_MATVEC21_KERNELS.get(kind)
    if kernel is not None:
        return kernel

    if kind == "f32_r32":
        code = r'''
        extern "C" __global__
        void matvec21_f32_r32(const float* __restrict__ A,
                              const float* __restrict__ x,
                              float* __restrict__ y,
                              const long m,
                              const long n,
                              const long spatial) {
            const long idx = blockDim.x * blockIdx.x + threadIdx.x;
            const long total = m * spatial;
            if (idx >= total) return;
            const long i = idx / spatial;
            const long s = idx - i * spatial;
            float acc = 0.0f;
            for (long j = 0; j < n; ++j) {
                acc += A[(i * n + j) * spatial + s] * x[j * spatial + s];
            }
            y[idx] = acc;
        }
        '''
        name = "matvec21_f32_r32"
    elif kind == "f32_c64":
        code = r'''
        extern "C" __global__
        void matvec21_f32_c64(const float* __restrict__ A,
                              const float2* __restrict__ x,
                              float2* __restrict__ y,
                              const long m,
                              const long n,
                              const long spatial) {
            const long idx = blockDim.x * blockIdx.x + threadIdx.x;
            const long total = m * spatial;
            if (idx >= total) return;
            const long i = idx / spatial;
            const long s = idx - i * spatial;
            float2 acc;
            acc.x = 0.0f;
            acc.y = 0.0f;
            for (long j = 0; j < n; ++j) {
                const float a = A[(i * n + j) * spatial + s];
                const float2 xv = x[j * spatial + s];
                acc.x += a * xv.x;
                acc.y += a * xv.y;
            }
            y[idx] = acc;
        }
        '''
        name = "matvec21_f32_c64"
    else:
        return None

    try:
        kernel = cp.RawKernel(code, name)
    except Exception:
        return None
    _CUPY_MATVEC21_KERNELS[kind] = kernel
    return kernel


def _get_cupy_matvec21_batch_kernel(kind):
    kernel = _CUPY_MATVEC21_BATCH_KERNELS.get(kind)
    if kernel is not None:
        return kernel

    if kind == "f32_r32":
        code = r'''
        extern "C" __global__
        void matvec21_batch_f32_r32(const float* __restrict__ A,
                                    const float* __restrict__ x,
                                    float* __restrict__ y,
                                    const long m,
                                    const long n,
                                    const long batch,
                                    const long spatial) {
            const long idx = blockDim.x * blockIdx.x + threadIdx.x;
            const long total = m * batch * spatial;
            if (idx >= total) return;
            const long s = idx % spatial;
            const long tmp = idx / spatial;
            const long b = tmp % batch;
            const long i = tmp / batch;
            float acc = 0.0f;
            for (long j = 0; j < n; ++j) {
                acc += A[(i * n + j) * spatial + s] * x[(j * batch + b) * spatial + s];
            }
            y[idx] = acc;
        }
        '''
        name = "matvec21_batch_f32_r32"
    elif kind == "f32_c64":
        code = r'''
        extern "C" __global__
        void matvec21_batch_f32_c64(const float* __restrict__ A,
                                    const float2* __restrict__ x,
                                    float2* __restrict__ y,
                                    const long m,
                                    const long n,
                                    const long batch,
                                    const long spatial) {
            const long idx = blockDim.x * blockIdx.x + threadIdx.x;
            const long total = m * batch * spatial;
            if (idx >= total) return;
            const long s = idx % spatial;
            const long tmp = idx / spatial;
            const long b = tmp % batch;
            const long i = tmp / batch;
            float2 acc;
            acc.x = 0.0f;
            acc.y = 0.0f;
            for (long j = 0; j < n; ++j) {
                const float a = A[(i * n + j) * spatial + s];
                const float2 xv = x[(j * batch + b) * spatial + s];
                acc.x += a * xv.x;
                acc.y += a * xv.y;
            }
            y[idx] = acc;
        }
        '''
        name = "matvec21_batch_f32_c64"
    else:
        return None

    try:
        kernel = cp.RawKernel(code, name)
    except Exception:
        return None
    _CUPY_MATVEC21_BATCH_KERNELS[kind] = kernel
    return kernel


def _matvec21_cupy_fused(tval, value):
    if not _CUPY_FUSED_MATVEC or not CUPY_AVAILABLE or cp is None:
        return None
    if not (is_cupy_array(tval) and is_cupy_array(value)):
        return None
    if tval.ndim != value.ndim + 1:
        return None
    if tval.shape[1] != value.shape[0] or tval.shape[2:] != value.shape[1:]:
        return None
    if tval.dtype != cp.float32:
        return None

    if value.dtype == cp.float32:
        kind = "f32_r32"
        out_dtype = cp.float32
    elif value.dtype == cp.complex64:
        kind = "f32_c64"
        out_dtype = cp.complex64
    else:
        return None

    if not tval.flags.c_contiguous:
        tval = cp.ascontiguousarray(tval)
    if not value.flags.c_contiguous:
        value = cp.ascontiguousarray(value)

    m = int(tval.shape[0])
    n = int(tval.shape[1])
    spatial = int(np.prod(value.shape[1:]))
    out = cp.empty((m,) + value.shape[1:], dtype=out_dtype)

    kernel = _get_cupy_matvec21_kernel(kind)
    if kernel is None:
        return None

    threads = 256
    blocks = (m * spatial + threads - 1) // threads
    try:
        kernel(
            (blocks,),
            (threads,),
            (tval, value, out, np.int64(m), np.int64(n), np.int64(spatial)),
        )
    except Exception:
        return None
    return out


def _matvec21_cupy_fused_batch(tval, value):
    if not _CUPY_FUSED_MATVEC or not CUPY_AVAILABLE or cp is None:
        return None
    if not (is_cupy_array(tval) and is_cupy_array(value)):
        return None
    if tval.ndim != value.ndim:
        return None
    if tval.shape[1] != value.shape[0] or tval.shape[2:] != value.shape[2:]:
        return None
    if tval.dtype != cp.float32:
        return None

    if value.dtype == cp.float32:
        kind = "f32_r32"
        out_dtype = cp.float32
    elif value.dtype == cp.complex64:
        kind = "f32_c64"
        out_dtype = cp.complex64
    else:
        return None

    if not tval.flags.c_contiguous:
        tval = cp.ascontiguousarray(tval)
    if not value.flags.c_contiguous:
        value = cp.ascontiguousarray(value)

    m = int(tval.shape[0])
    n = int(tval.shape[1])
    batch = int(value.shape[1])
    spatial = int(np.prod(value.shape[2:]))
    out = cp.empty((m, batch) + value.shape[2:], dtype=out_dtype)

    kernel = _get_cupy_matvec21_batch_kernel(kind)
    if kernel is None:
        return None

    threads = 256
    blocks = (m * batch * spatial + threads - 1) // threads
    try:
        kernel(
            (blocks,),
            (threads,),
            (tval, value, out, np.int64(m), np.int64(n), np.int64(batch), np.int64(spatial)),
        )
    except Exception:
        return None
    return out


def _get_cupy_sym21_kernel(kind, batched):
    key = (kind, bool(batched))
    kernel = _CUPY_SYM21_KERNELS.get(key)
    if kernel is not None:
        return kernel

    if kind == "f32_r32":
        code = r'''
        extern "C" __global__
        void sym21_apply(const float* __restrict__ A,
                         const float* __restrict__ X,
                         float* __restrict__ Y,
                         const long batch,
                         const long spatial,
                         const long total) {
            const long idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= total) return;

            const long s = idx % spatial;
            const long b = idx / spatial;
            const long base = b * spatial + s;
            const long stride = batch * spatial;

            const float x0 = X[0 * stride + base];
            const float x1 = X[1 * stride + base];
            const float x2 = X[2 * stride + base];
            const float x3 = X[3 * stride + base];
            const float x4 = X[4 * stride + base];
            const float x5 = X[5 * stride + base];

            const float a00 = A[ 0 * spatial + s];
            const float a01 = A[ 1 * spatial + s];
            const float a02 = A[ 2 * spatial + s];
            const float a03 = A[ 3 * spatial + s];
            const float a04 = A[ 4 * spatial + s];
            const float a05 = A[ 5 * spatial + s];
            const float a11 = A[ 6 * spatial + s];
            const float a12 = A[ 7 * spatial + s];
            const float a13 = A[ 8 * spatial + s];
            const float a14 = A[ 9 * spatial + s];
            const float a15 = A[10 * spatial + s];
            const float a22 = A[11 * spatial + s];
            const float a23 = A[12 * spatial + s];
            const float a24 = A[13 * spatial + s];
            const float a25 = A[14 * spatial + s];
            const float a33 = A[15 * spatial + s];
            const float a34 = A[16 * spatial + s];
            const float a35 = A[17 * spatial + s];
            const float a44 = A[18 * spatial + s];
            const float a45 = A[19 * spatial + s];
            const float a55 = A[20 * spatial + s];

            Y[0 * stride + base] = a00*x0 + a01*x1 + a02*x2 + a03*x3 + a04*x4 + a05*x5;
            Y[1 * stride + base] = a01*x0 + a11*x1 + a12*x2 + a13*x3 + a14*x4 + a15*x5;
            Y[2 * stride + base] = a02*x0 + a12*x1 + a22*x2 + a23*x3 + a24*x4 + a25*x5;
            Y[3 * stride + base] = a03*x0 + a13*x1 + a23*x2 + a33*x3 + a34*x4 + a35*x5;
            Y[4 * stride + base] = a04*x0 + a14*x1 + a24*x2 + a34*x3 + a44*x4 + a45*x5;
            Y[5 * stride + base] = a05*x0 + a15*x1 + a25*x2 + a35*x3 + a45*x4 + a55*x5;
        }
        '''
    elif kind == "f32_c64":
        code = r'''
        extern "C" __global__
        void sym21_apply(const float* __restrict__ A,
                         const float2* __restrict__ X,
                         float2* __restrict__ Y,
                         const long batch,
                         const long spatial,
                         const long total) {
            const long idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= total) return;

            const long s = idx % spatial;
            const long b = idx / spatial;
            const long base = b * spatial + s;
            const long stride = batch * spatial;

            const float2 x0 = X[0 * stride + base];
            const float2 x1 = X[1 * stride + base];
            const float2 x2 = X[2 * stride + base];
            const float2 x3 = X[3 * stride + base];
            const float2 x4 = X[4 * stride + base];
            const float2 x5 = X[5 * stride + base];

            const float a00 = A[ 0 * spatial + s];
            const float a01 = A[ 1 * spatial + s];
            const float a02 = A[ 2 * spatial + s];
            const float a03 = A[ 3 * spatial + s];
            const float a04 = A[ 4 * spatial + s];
            const float a05 = A[ 5 * spatial + s];
            const float a11 = A[ 6 * spatial + s];
            const float a12 = A[ 7 * spatial + s];
            const float a13 = A[ 8 * spatial + s];
            const float a14 = A[ 9 * spatial + s];
            const float a15 = A[10 * spatial + s];
            const float a22 = A[11 * spatial + s];
            const float a23 = A[12 * spatial + s];
            const float a24 = A[13 * spatial + s];
            const float a25 = A[14 * spatial + s];
            const float a33 = A[15 * spatial + s];
            const float a34 = A[16 * spatial + s];
            const float a35 = A[17 * spatial + s];
            const float a44 = A[18 * spatial + s];
            const float a45 = A[19 * spatial + s];
            const float a55 = A[20 * spatial + s];

            float2 y0;
            float2 y1;
            float2 y2;
            float2 y3;
            float2 y4;
            float2 y5;

            y0.x = a00*x0.x + a01*x1.x + a02*x2.x + a03*x3.x + a04*x4.x + a05*x5.x;
            y0.y = a00*x0.y + a01*x1.y + a02*x2.y + a03*x3.y + a04*x4.y + a05*x5.y;
            y1.x = a01*x0.x + a11*x1.x + a12*x2.x + a13*x3.x + a14*x4.x + a15*x5.x;
            y1.y = a01*x0.y + a11*x1.y + a12*x2.y + a13*x3.y + a14*x4.y + a15*x5.y;
            y2.x = a02*x0.x + a12*x1.x + a22*x2.x + a23*x3.x + a24*x4.x + a25*x5.x;
            y2.y = a02*x0.y + a12*x1.y + a22*x2.y + a23*x3.y + a24*x4.y + a25*x5.y;
            y3.x = a03*x0.x + a13*x1.x + a23*x2.x + a33*x3.x + a34*x4.x + a35*x5.x;
            y3.y = a03*x0.y + a13*x1.y + a23*x2.y + a33*x3.y + a34*x4.y + a35*x5.y;
            y4.x = a04*x0.x + a14*x1.x + a24*x2.x + a34*x3.x + a44*x4.x + a45*x5.x;
            y4.y = a04*x0.y + a14*x1.y + a24*x2.y + a34*x3.y + a44*x4.y + a45*x5.y;
            y5.x = a05*x0.x + a15*x1.x + a25*x2.x + a35*x3.x + a45*x4.x + a55*x5.x;
            y5.y = a05*x0.y + a15*x1.y + a25*x2.y + a35*x3.y + a45*x4.y + a55*x5.y;

            Y[0 * stride + base] = y0;
            Y[1 * stride + base] = y1;
            Y[2 * stride + base] = y2;
            Y[3 * stride + base] = y3;
            Y[4 * stride + base] = y4;
            Y[5 * stride + base] = y5;
        }
        '''
    else:
        return None

    try:
        kernel = cp.RawKernel(code, "sym21_apply")
    except Exception:
        return None
    _CUPY_SYM21_KERNELS[key] = kernel
    return kernel


def _sym21_cupy_fused(tval, value):
    if not _CUPY_FUSED_MATVEC or not CUPY_AVAILABLE or cp is None:
        return None
    if not (is_cupy_array(tval) and is_cupy_array(value)):
        return None
    if tval.shape[0] != 21 or value.shape[0] != 6:
        return None
    if tval.dtype != cp.float32:
        return None
    if value.dtype == cp.float32:
        kind = "f32_r32"
        out_dtype = cp.float32
    elif value.dtype == cp.complex64:
        kind = "f32_c64"
        out_dtype = cp.complex64
    else:
        return None

    batched = tval.ndim == value.ndim
    if batched:
        if tval.shape[1:] != value.shape[2:]:
            return None
        batch = int(value.shape[1])
        spatial_shape = value.shape[2:]
        out_shape = (6, batch) + spatial_shape
    else:
        if tval.shape[1:] != value.shape[1:]:
            return None
        batch = 1
        spatial_shape = value.shape[1:]
        value = value[:, None, ...]
        out_shape = (6,) + spatial_shape

    if not tval.flags.c_contiguous:
        tval = cp.ascontiguousarray(tval)
    if not value.flags.c_contiguous:
        value = cp.ascontiguousarray(value)

    spatial = int(np.prod(spatial_shape))
    out_raw = cp.empty((6, batch) + spatial_shape, dtype=out_dtype)
    kernel = _get_cupy_sym21_kernel(kind, batched=True)
    if kernel is None:
        return None
    total = int(batch * spatial)
    threads = 256
    blocks = (total + threads - 1) // threads
    try:
        kernel((blocks,), (threads,), (tval, value, out_raw,
                                      np.int64(batch), np.int64(spatial), np.int64(total)))
    except Exception:
        return None
    return out_raw if batched else out_raw[:, 0, ...].reshape(out_shape)


def _tensor_apply_cupy(tensor, value):
    xp = get_array_module(value)
    multype = _unwrap_timed(tensor).multype if isinstance(_unwrap_timed(tensor), Tensor) else tensor.multype

    if multype in [
            'elasticity_hg1_direct', 'elasticity_g1_direct',
            'elasticity_hg2_direct', 'elasticity_g2_direct']:
        if xp is cp:
            direct = elasticity_direct_array(_unwrap_timed(tensor), value)
            if direct is not None:
                return direct
        raise NotImplementedError("elasticity direct projection requiere CuPy.")
    if multype in ['sym21_indexed', 'indexed_sym21']:
        indexed = indexed_sym21_array(_unwrap_timed(tensor), value)
        if indexed is not None:
            return indexed
        raise NotImplementedError("material sym21 indexado requiere CuPy float32.")

    tval = _tensor_val_on_cupy(tensor)

    if multype in ['scal', 'scalar']:
        return xp.einsum('...,...->...', tval, value)
    elif multype in ['sym21', 'symmetric21']:
        if xp is cp:
            fused = _sym21_cupy_fused(tval, value)
            if fused is not None:
                return fused
        out = xp.zeros((6,) + value.shape[1:], dtype=xp.result_type(tval.dtype, value.dtype))
        for kk, (ii, jj) in enumerate(SYM21_PAIRS):
            cval = tval[kk]
            out[ii] += cval * value[jj]
            if ii != jj:
                out[jj] += cval * value[ii]
        return out
    elif multype in [21, '21']:
        if xp is cp:
            fused = _matvec21_cupy_fused(tval, value)
            if fused is not None:
                return fused
            fused = _matvec21_cupy_fused_batch(tval, value)
            if fused is not None:
                return fused
            if tval.ndim == value.ndim:
                out = xp.zeros((tval.shape[0], value.shape[1]) + value.shape[2:], dtype=xp.result_type(tval.dtype, value.dtype))
                for j in range(tval.shape[1]):
                    out += tval[:, j, None, ...] * value[j, None, ...]
                return out
            out = xp.zeros((tval.shape[0],) + value.shape[1:], dtype=xp.result_type(tval.dtype, value.dtype))
            for j in range(tval.shape[1]):
                out += tval[:, j, ...] * value[j, ...]
            return out
        return xp.einsum('ij...,j...->i...', tval, value)
    elif multype in [42, '42']:
        if xp is cp:
            out = xp.zeros(tval.shape[:2] + value.shape[2:], dtype=xp.result_type(tval.dtype, value.dtype))
            for k in range(tval.shape[2]):
                for l in range(tval.shape[3]):
                    out += tval[:, :, k, l, ...] * value[k, l, ...]
            return out
        return xp.einsum('ijkl...,kl...->ij...', tval, value)
    elif multype in [00, 'elementwise', 'hadamard']:
        return xp.einsum('...,...->...', tval, value)
    elif multype in ['grad']:
        return xp.einsum('i...,...->i...', tval, value)
    elif multype in ['div']:
        return xp.einsum('i...,i...->...', tval, value)
    elif isinstance(multype, str):
        return xp.einsum(multype, tval, value)
    raise NotImplementedError(f"multype '{multype}' no soportado en backend CuPy experimental.")


def _apply_dft_raw(dft, value):
    if dft.inverse:
        if dft.fft_form == 'c':
            return icfftn_cached(value, dft._N_tuple, dft._prodN, dft._axes)
        elif dft.fft_form == 0:
            return ifftn_cached(value, dft._N_tuple, dft._prodN, dft._axes)
        return irfftn_cached(value, dft._N_tuple, dft._axes)

    if dft.fft_form == 'c':
        return fftnc_cached(value, dft._N_tuple, dft._prodN, dft._axes)
    elif dft.fft_form == 0:
        return fftn_cached(value, dft._N_tuple, dft._prodN, dft._axes)
    return rfftn_cached(value, dft._N_tuple, dft._axes)


class DFT(TensorFuns):
    """
    (inverse) Disrete Fourier Transform (DFT) to provide __call__
    by FFT routine.

    Parameters
    ----------
    inverse : boolean
        if True it provides inverse DFT
    N : numpy.ndarray
        N-sized (i)DFT,
    normalized : boolean
        version of DFT that is normalized by factor numpy.prod(N)
    fft_form : str or num
        determines the type of the Fourier transform and corresponding format in the Fourier domain
        the following values are considered:
        0 : standard numpy.fft.fftn algorithm
        'c' : centered version of numpy.fft.fftn algorithm with zero frequency in the middle
        'r' : version of numpy.fft.fftn suitable for real data
    """
    def __init__(self, inverse=False, N=None, fft_form=fft_form_default, **kwargs):
        self.__dict__.update(kwargs)
        if 'name' not in list(kwargs.keys()):
            if inverse:
                self.name='iDFT'
            else:
                self.name='DFT'

        self.N=np.array(N, dtype=np.int32)
        self._N_tuple=tuple(int(v) for v in self.N.tolist())
        self._prodN=int(np.prod(self._N_tuple))
        self._axes=tuple(range(-len(self._N_tuple), 0))
        self.inverse=inverse
        self._set_fft(fft_form)

    def __mul__(self, x):
        return self.__call__(x)

    def __call__(self, x):
        if isinstance(x, Tensor):
            assert(x.Fourier==self.inverse)
            if self.inverse:
                if self.fft_form == 'c':
                    val = icfftn_cached(x.val, self._N_tuple, self._prodN, self._axes)
                elif self.fft_form == 0:
                    val = ifftn_cached(x.val, self._N_tuple, self._prodN, self._axes)
                else:
                    val = irfftn_cached(x.val, self._N_tuple, self._axes)
                return x.copy(name='iF({0})'.format(x.name[:10]),
                              val=val.real, Fourier=not x.Fourier)
            else:
                assert(x.fft_form==self.fft_form)
                if self.fft_form == 'c':
                    val = fftnc_cached(x.val, self._N_tuple, self._prodN, self._axes)
                elif self.fft_form == 0:
                    val = fftn_cached(x.val, self._N_tuple, self._prodN, self._axes)
                else:
                    val = rfftn_cached(x.val, self._N_tuple, self._axes)
                return x.copy(name='F({0})'.format(x.name[:10]),
                              val=val, Fourier=not x.Fourier)

        elif (isinstance(x, Operator) or isinstance(x, DFT)):
            return Operator(mat=[[self, x]])

        else:
            raise ValueError('DFT.__call__')

    def matrix(self, shape=None):
        """
        This function returns the object as a matrix of DFT or iDFT resp.
        """
        N=self.N
        prodN=np.prod(N)
        if shape is not None:
            dim=np.prod(np.array(shape))
        elif hasattr(self, 'shape'):
            dim=np.prod(np.array(shape))
        else:
            raise ValueError('Missing shape of the DFT.')

        proddN=dim*prodN
        ZN_input=Grid.get_ZNl(N, fft_form=0)
        ZN_output=Grid.get_ZNl(N, fft_form='c')

        if self.inverse:
            DFTcoef=lambda k, l, N: np.exp(2*np.pi*1j*np.sum(k*l/N))
        else:
            DFTcoef=lambda k, l, N: np.exp(-2*np.pi*1j*np.sum(k*l/N))/np.prod(N)

        DTM=np.zeros([self.pN(), self.pN()], dtype=np.complex128)
        for ii, kk in enumerate(itertools.product(*tuple(ZN_output))):
            for jj, ll in enumerate(itertools.product(*tuple(ZN_input))):
                DTM[ii, jj]=DFTcoef(np.array(kk, dtype=float),
                                      np.array(ll), N)

        DTMd=npmatlib.zeros([proddN, proddN], dtype=np.complex128)
        for ii in range(dim):
            DTMd[prodN*ii:prodN*(ii+1), prodN*ii:prodN*(ii+1)]=DTM
        return DTMd

    def __repr__(self):
        keys=['name','inverse','fft_form','N']
        return self._repr(keys)

    def transpose(self):
        kwargs = copy(self.__dict__)
        kwargs.update(dict(inverse=not self.inverse))
        return DFT(**kwargs)

class Operator():
    """
    Linear operator composed of matrices or linear operators
    it is designed to provide __call__ function as a linear operation

    parameters :
        X : numpy.ndarray or Tensor or something else
            it represents the operand,
            it provides the information about size and shape of operand
        dtype : data type of operand, usually numpy.float64
    """
    def __init__(self, name='Operator', mat_rev=None, mat=None, operand=None):
        self.name=name
        if mat_rev is not None:
            self.mat_rev=mat_rev
        elif mat is not None:
            self.mat_rev=[]
            for summand in mat:
                no_oper=len(summand)
                summand_rev=[]
                for m in np.arange(no_oper):
                    summand_rev.append(summand[no_oper-1-m])
                self.mat_rev.append(summand_rev)
        self.no_summands=len(self.mat_rev)

        if operand is not None:
            self.define_operand(operand)

    def __call__(self, x):
        cupy_res = self._call_cupy_experimental(x)
        if cupy_res is not None:
            cupy_res.name='{0}({1})'.format(self.name[:6], x.name[:10])
            return cupy_res

        if self.no_summands == 1:
            summand = self.mat_rev[0]
            no_oper = len(summand)
            if no_oper == 0:
                res = x
            elif no_oper == 1:
                res = summand[0](x)
            elif no_oper == 2:
                res = summand[1](summand[0](x))
            elif no_oper == 3:
                res = summand[2](summand[1](summand[0](x)))
            else:
                res = x
                for matrix in summand:
                    res = matrix(res)
        else:
            res = None
            for summand in self.mat_rev:
                prod=x
                for matrix in summand:
                    prod=matrix(prod)
                if res is None:
                    res = prod
                else:
                    res = res + prod
        res.name='{0}({1})'.format(self.name[:6], x.name[:10])
        return res

    def _call_cupy_experimental(self, x):
        if get_fft_backend() != 'cupy' or not CUPY_AVAILABLE:
            return None
        if not isinstance(x, Tensor):
            return None
        if self.no_summands != 1 or len(self.mat_rev[0]) != 2:
            return None

        Aop, GNop = self.mat_rev[0]
        Araw = _unwrap_timed(Aop)
        GNraw = _unwrap_timed(GNop)
        if (Aop is not Araw) or (GNop is not GNraw):
            return None
        if not isinstance(Araw, Tensor) or not isinstance(GNraw, Operator):
            return None
        if GNraw.no_summands != 1 or len(GNraw.mat_rev[0]) != 3:
            return None

        FNop, hGop, FiNop = GNraw.mat_rev[0]
        FNraw = _unwrap_timed(FNop)
        hGraw = _unwrap_timed(hGop)
        FiNraw = _unwrap_timed(FiNop)
        if (FNop is not FNraw) or (hGop is not hGraw) or (FiNop is not FiNraw):
            return None
        if not (isinstance(FNraw, DFT) and isinstance(hGraw, Tensor) and isinstance(FiNraw, DFT)):
            return None

        x_val = to_backend_array(x.val, prefer_backend='cupy')
        Ax = _tensor_apply_cupy(Araw, x_val)
        use_unscaled_pair = (
            _CUPY_UNSCALED_FFT_PAIR and
            FNraw.fft_form == 0 and
            FiNraw.fft_form == 0 and
            not FNraw.inverse and
            FiNraw.inverse
        )
        if use_unscaled_pair:
            Fx = fftn_unscaled_cached(Ax, FNraw._N_tuple, FNraw._axes)
        else:
            Fx = _apply_dft_raw(FNraw, Ax)
        Gx = _tensor_apply_cupy(hGraw, Fx)
        if use_unscaled_pair:
            out = ifftn_unscaled_real_cached(Gx, FiNraw._N_tuple, FiNraw._axes)
        else:
            out = _apply_dft_raw(FiNraw, Gx)
        return x.copy(val=out)

    def __repr__(self):
        s='Class : {0}\n    name : {1}\n    expression : '.format(self.__class__.__name__,
                                                                  self.name)
        flag_sum=False
        no_sum=len(self.mat_rev)
        for isum in np.arange(no_sum):
            if flag_sum:
                s+=' + '
            no_oper=len(self.mat_rev[isum])
            flag_mul=False
            for m in np.arange(no_oper):
                matrix=self.mat_rev[isum][no_oper-1-m]
                if flag_mul:
                    s+='*'
                s+=matrix.name
                flag_mul=True
            flag_sum=True
        return s

    def define_operand(self, X):
        """
        This function defines the type of operand to correctly define linear
        operator.

        Parameters
        ----------
        X : any object
            operand of linear operator
        """
        if isinstance(X, Tensor):
            Y=self(X)
            self.matshape=(Y.val.size, X.val.size)
            self.X_reshape=X.val.shape
            self.X_order=X.order
            self.X_N=X.N
            self.Y_reshape=Y.val.shape
            self.Y_order=Y.order
        else:
            print('LinOper : This operand is not implemented!')

    def matvec(self, x):
        """
        Provides the __call__ for operand recast into one-dimensional vector.
        This is suitable for e.g. iterative solvers when trigonometric
        polynomials are recast into one-dimensional numpy.arrays.

        Parameters
        ----------
        x : one-dimensional numpy.array
        """
        X=Tensor(val=self.revec(x), order=self.X_order, N=self.X_N)
        AX=self.__call__(X)
        return AX.vec()

    def vec(self, X):
        """
        Reshape the operand (Tensor) into one-dimensional vector (column)
        version.
        """
        return np.reshape(X, self.shape[1])

    def revec(self, x):
        """
        Reshape the one-dimensional vector of trig. pol. into shape occurring
        in class Tensor.
        """
        return np.reshape(np.asarray(x), self.Y_reshape)

    def transpose(self):
        """
        Transpose (adjoint) of linear operator.
        """
        mat=[]
        for m in np.arange(self.no_summands):
            summand=[]
            for n in np.arange(len(self.mat_rev[m])):
                summand.append(self.mat_rev[m][n].transpose())
            mat.append(summand)
        name='({0}).T'.format(self.name[:10])
        return Operator(name=name, mat=mat)

def grad(X):
    if X.shape==(1,):
        shape=(X.dim,)
    else:
        shape=X.shape+(X.dim,)
    name='grad({0})'.format(X.name[:10])
    gX=Tensor(name=name, shape=shape, N=X.N,
              Fourier=True, fft_form=X.fft_form)
    if X.Fourier:
        FX=X
    else:
        F=DFT(N=X.N, fft_form=X.fft_form) # TODO:change to X.fourier()
        FX=F(X)

    dim=len(X.N)
    freq=Grid.get_freq(X.N, X.Y, fft_form=X.fft_form)
    strfreq='xyz'
    coef=2*np.pi*1j
    val=np.empty((X.dim,)+X.shape+X.N_fft, dtype=np.complex128)

    for ii in range(X.dim):
        mul_str='{0},...{1}->...{1}'.format(strfreq[ii], strfreq[:dim])
        val[ii]=np.einsum(mul_str, coef*freq[ii], FX.val, dtype=np.complex128)

    if X.shape==(1,):
        gX.val=np.squeeze(val)
    else:
        gX.val=np.moveaxis(val, 0, X.order)

    if not X.Fourier:
        iF=DFT(N=X.N, inverse=True, fft_form=gX.fft_form)
        gX=iF(gX)
    gX.name='grad({0})'.format(X.name[:10])
    return gX

def div(X):
    if X.shape==(1,):
        shape=()
    else:
        shape=X.shape[:-1]
    assert(X.shape[-1]==X.dim)
    assert(X.order==1)

    dX=Tensor(shape=shape, N=X.N, Fourier=True, fft_form=X.fft_form)
    if X.Fourier:
        FX=X
    else:
        F=DFT(N=X.N, fft_form=X.fft_form)
        FX=F(X)

    dim=len(X.N)
    freq=Grid.get_freq(X.N, X.Y, fft_form=FX.fft_form)
    strfreq='xyz'
    coef=2*np.pi*1j

    for ii in range(X.dim):
        mul_str='{0},...{1}->...{1}'.format(strfreq[ii], strfreq[:dim])
        dX.val+=np.einsum(mul_str, coef*freq[ii], FX.val[ii], dtype=np.complex128)

    if not X.Fourier:
        iF=DFT(N=X.N, inverse=True, fft_form=dX.fft_form)
        dX=iF(dX)
    dX.name='div({0})'.format(X.name[:10])
    return dX

def laplace(X):
    return div(grad(X))

def symgrad(X):
    gX=grad(X)
    return 0.5*(gX+gX.transpose())

def potential_scalar(x, freq, mean_index):
    # get potential for scalar-valued function in Fourier space
    dim=x.shape[0]
    assert(dim==len(x.shape)-1)
    strfreq='xyz'
    coef=2*np.pi*1j
    val=np.empty(x.shape[1:], dtype=np.complex128)
    for d in range(0, dim):
        factor=np.zeros_like(freq[d], dtype=np.complex128)
        inds=np.setdiff1d(np.arange(factor.size, dtype=int), mean_index[d])
        factor[inds]=1./(coef*freq[d][inds])
        val[mean_index[:d]]=np.einsum('x,{0}->{0}'.format(strfreq[:dim-d]),
                                      factor, x[d][mean_index[:d]], dtype=np.complex128)
    return val

def potential(X, small_strain=False):
    if X.Fourier:
        FX=X
    else:
        F=DFT(N=X.N, fft_form=X.fft_form)
        FX=F(X)

    freq=Grid.get_freq(X.N, X.Y, fft_form=FX.fft_form)
    if X.order==1:
        assert(X.dim==X.shape[0])
        iX=Tensor(name='potential({0})'.format(X.name[:10]), shape=(1,), N=X.N,
                  Fourier=True, fft_form=FX.fft_form)
        iX.val[0]=potential_scalar(FX.val, freq=freq, mean_index=FX.mean_index())

    elif X.order==2:
        assert(X.dim==X.shape[0])
        assert(X.dim==X.shape[1])
        iX=Tensor(name='potential({0})'.format(X.name[:10]), shape=(X.dim,), N=X.N,
                  Fourier=True, fft_form=FX.fft_form)
        if not small_strain:
            for ii in range(X.dim):
                iX.val[ii]=potential_scalar(FX.val[ii], freq=freq, mean_index=FX.mean_index())

        else:
            assert((X-X.transpose()).norm()<1e-14) # symmetricity
            omeg=FX.zeros_like() # non-symmetric part of the gradient
            gomeg=Tensor(name='potential({0})'.format(X.name[:10]),
                           shape=FX.shape+(X.dim,), N=X.N, Fourier=True)
            grad_ep=grad(FX) # gradient of strain
            gomeg.val=np.einsum('ikj...->ijk...', grad_ep.val)-np.einsum('jki...->ijk...', grad_ep.val)
            for ij in itertools.product(list(range(X.dim)), repeat=2):
                omeg.val[ij]=potential_scalar(gomeg.val[ij], freq=freq, mean_index=FX.mean_index())

            gradu=FX+omeg
            iX=potential(gradu, small_strain=False)

    if X.Fourier:
        return iX
    else:
        iF=DFT(N=X.N, inverse=True, fft_form=FX.fft_form)
        return iF(iX)

def matrix2tensor(M):
    return Tensor(name=M.name, val=M.val, order=2, multype=21,
                  Fourier=M.Fourier, fft_form=fft_form_default)

def vector2tensor(V):
    return Tensor(name=V.name, val=V.val, order=1, Fourier=V.Fourier)

def grad_div_tensor(N, Y=None, grad=True, div=True, fft_form=fft_form_default):
    if grad and div:
        return grad_tensor(N, Y, fft_form=fft_form), div_tensor(N, Y, fft_form=fft_form)
    elif grad:
        return grad_tensor(N, Y, fft_form=fft_form)
    elif div:
        return div_tensor(N, Y, fft_form=fft_form)

def grad_tensor(N, Y=None, fft_form=fft_form_default):
    if Y is None:
        Y = np.ones_like(N)
    # scalar valued versions of gradient and divergence
    N = np.array(N, dtype=int)
    dim = N.size

    freq = Grid.get_xil(N, Y, fft_form=fft_form)
    N_fft=tuple(freq[i].size for i in range(dim))
    hGrad = np.zeros((dim,)+ N_fft) # zero initialize
    for ind in itertools.product(*[list(range(n)) for n in N_fft]):
        for i in range(dim):
            hGrad[i][ind] = freq[i][ind[i]]
    hGrad = hGrad*2*np.pi*1j
    return Tensor(name='hgrad', val=hGrad, order=1, N=N, multype='grad',
                  Fourier=True, fft_form=fft_form)

def div_tensor(N, Y=None, fft_form=fft_form_default):
    if Y is None:
        Y = np.ones_like(N)
    hGrad=grad_tensor(N, Y=Y, fft_form=fft_form)
    hGrad.multype='div'
    return hGrad

def outer(X, Y):
    assert(np.allclose(X.N, Y.N))
    Xpshp=np.prod(X.shape)
    Ypshp=np.prod(Y.shape)
    val=np.einsum('i...,j...->ij...', X.val.reshape((Xpshp,)+X.N), Y.val.reshape((Ypshp,)+X.N))
    XoY=X.copy(name='outer({},{})'.format(X.name, Y.name), order=X.order+Y.order,
               val=val.reshape(X.shape+Y.shape+X.N))
    return XoY
