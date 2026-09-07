"""Memory-efficient real-space CG reductions with a fixed reduction tree.

Fields retain their original precision. Products and partial sums accumulate
in float64; each scalar is rounded once to the field dtype after normalization.
No global floating-point atomics or host scalar transfers are used.
"""

import numpy as np

from ffthompy.tensors.fft import cp, is_cupy_array


_KERNELS = {}


class CupyCGReduction:
    def __init__(self, tensor):
        self.volume = int(np.prod(tensor.N))
        self.dtype = tensor.val.dtype
        self.total = int(tensor.val.size)
        self.threads = 256
        self.blocks = min(4096, (self.total + self.threads - 1) // self.threads)
        self.partial = cp.empty(self.blocks, dtype=cp.float64)
        key = (self.dtype.str, self.volume)
        if key not in _KERNELS:
            dtype = self.dtype.name
            ctype = 'float' if dtype == 'float32' else 'double'
            suffix = f'{dtype}_{self.volume}'
            finish = cp.ReductionKernel(
                'float64 x', f'{dtype} out', 'x', 'a + b',
                f'out = a / {self.volume}.0', '0',
                f'cg_rr_tree_{suffix}', reduce_type='double',
            )
            code = r'''
            extern "C" __global__
            void cg_dot_partial(const TYPE* __restrict__ x,
                                const TYPE* __restrict__ y,
                                double* __restrict__ partial,
                                const long long total) {
                __shared__ double sums[256];
                double value = 0.0;
                for (long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
                     i < total; i += (long long)blockDim.x * gridDim.x)
                    value += (double)x[i] * (double)y[i];
                sums[threadIdx.x] = value;
                __syncthreads();
                for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                    if (threadIdx.x < offset)
                        sums[threadIdx.x] += sums[threadIdx.x + offset];
                    __syncthreads();
                }
                if (threadIdx.x == 0) partial[blockIdx.x] = sums[0];
            }
            extern "C" __global__
            void cg_update_xr_partial(const TYPE* alpha,
                                     const TYPE* __restrict__ p,
                                     const TYPE* __restrict__ ap,
                                     TYPE* __restrict__ x,
                                     TYPE* __restrict__ r,
                                     double* __restrict__ partial,
                                     const long long total) {
                __shared__ double sums[256];
                const TYPE a = alpha[0];
                double value = 0.0;
                for (long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
                     i < total; i += (long long)blockDim.x * gridDim.x) {
                    x[i] += a * p[i];
                    const TYPE ri = r[i] - a * ap[i];
                    r[i] = ri;
                    value += (double)ri * (double)ri;
                }
                sums[threadIdx.x] = value;
                __syncthreads();
                for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                    if (threadIdx.x < offset)
                        sums[threadIdx.x] += sums[threadIdx.x + offset];
                    __syncthreads();
                }
                if (threadIdx.x == 0) partial[blockIdx.x] = sums[0];
            }
            '''.replace('TYPE', ctype)
            dot = cp.RawKernel(code, 'cg_dot_partial')
            update = cp.RawKernel(code, 'cg_update_xr_partial')
            _KERNELS[key] = dot, finish, update
        self.dot_kernel, self.finish_kernel, self.update_kernel = _KERNELS[key]

    @staticmethod
    def supports(tensor, par):
        # A custom inner product or Fourier weights must retain their own metric.
        return (
            'scal' not in par
            and hasattr(tensor, 'val')
            and is_cupy_array(tensor.val)
            and not tensor.Fourier
            and tensor.val.dtype in (np.dtype('float32'), np.dtype('float64'))
            and tensor.val.flags.c_contiguous
            and tensor.val.size > 0
        )

    def dot(self, x, y):
        if x.val.shape != y.val.shape or x.val.size != self.total:
            raise ValueError('CG reduction field shapes do not match.')
        if x.val.dtype != self.dtype or y.val.dtype != self.dtype:
            from ffthompy.general.solver import _tensor_scalar_product_raw
            return _tensor_scalar_product_raw(x, y)
        self.dot_kernel(
            (self.blocks,), (self.threads,),
            (cp.ascontiguousarray(x.val), cp.ascontiguousarray(y.val),
             self.partial, np.int64(self.total)),
        )
        return self.finish_kernel(self.partial)

    def update_xr_rr(self, x, alpha, p, r, ap):
        if not all(t.val.flags.c_contiguous and t.val.dtype == self.dtype
                   and t.val.shape == x.val.shape and t.val.size == self.total
                   for t in (x, p, r, ap)):
            return None  # Retain the general Tensor update for strided inputs.
        alpha = cp.asarray(alpha, dtype=self.dtype)
        self.update_kernel(
            (self.blocks,), (self.threads,),
            (alpha, p.val, ap.val, x.val, r.val, self.partial, np.int64(self.total)),
        )
        # A fresh scalar is essential: CG still needs the previous rr for beta.
        return self.finish_kernel(self.partial)
