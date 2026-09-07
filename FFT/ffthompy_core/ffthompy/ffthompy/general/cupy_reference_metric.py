"""Fused CG reductions for a constant isotropic reference energy metric."""

import numpy as np
from ffthompy.tensors.fft import cp

_KERNELS = {}


class ReferenceMetricReduction:
    def __init__(self, tensor, eta):
        if tensor.val.dtype != cp.float32 or tensor.val.shape[0] != 6 or tensor.Fourier:
            raise ValueError('Reference metric kernels require six physical float32 fields.')
        if not np.isfinite(eta) or eta <= 2/3:
            raise ValueError('Reference metric requires an SPD reference.')
        self.volume = int(np.prod(tensor.N))
        if tensor.val.size != 6*self.volume:
            raise ValueError('Reference metric is implemented for one load at a time.')
        self.weight = np.float64(eta-1)
        self.blocks = min(4096, (self.volume+255)//256)
        self.partial = cp.empty(self.blocks,dtype=cp.float64)
        if self.volume not in _KERNELS:
            code = r'''
            extern "C" __global__ void weighted_dot(
                const float* x, const float* y, double* partial,
                long long n, double weight) {
                __shared__ double sums[256];
                double value=0.;
                for (long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x;
                     i<n; i+=(long long)blockDim.x*gridDim.x) {
                    double tx=0.,ty=0.;
                    #pragma unroll
                    for(int c=0;c<6;c++) {
                        const double a=x[c*n+i], b=y[c*n+i];
                        value+=a*b;
                        if(c<3) {tx+=a;ty+=b;}
                    }
                    value+=weight*tx*ty;
                }
                sums[threadIdx.x]=value; __syncthreads();
                for(int offset=128;offset>0;offset>>=1) {
                    if(threadIdx.x<offset) sums[threadIdx.x]+=sums[threadIdx.x+offset];
                    __syncthreads();
                }
                if(threadIdx.x==0) partial[blockIdx.x]=sums[0];
            }
            extern "C" __global__ void weighted_update(
                const float* alpha, const float* p, const float* ap,
                float* x, float* r, double* partial, long long n, double weight) {
                __shared__ double sums[256];
                double value=0.; const float a=alpha[0];
                for (long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x;
                     i<n; i+=(long long)blockDim.x*gridDim.x) {
                    double tr=0.;
                    #pragma unroll
                    for(int c=0;c<6;c++) {
                        const long long j=c*n+i;
                        x[j]+=a*p[j];
                        const float ri=r[j]-a*ap[j]; r[j]=ri;
                        value+=(double)ri*(double)ri;
                        if(c<3) tr+=(double)ri;
                    }
                    value+=weight*tr*tr;
                }
                sums[threadIdx.x]=value; __syncthreads();
                for(int offset=128;offset>0;offset>>=1) {
                    if(threadIdx.x<offset) sums[threadIdx.x]+=sums[threadIdx.x+offset];
                    __syncthreads();
                }
                if(threadIdx.x==0) partial[blockIdx.x]=sums[0];
            }
            '''
            finish=cp.ReductionKernel('float64 x','float32 out','x','a+b',
                f'out=a/{self.volume}.0','0',f'reference_rr_{self.volume}',reduce_type='double')
            _KERNELS[self.volume]=(cp.RawKernel(code,'weighted_dot'),
                                   cp.RawKernel(code,'weighted_update'),finish)
        self.dot_kernel,self.update_kernel,self.finish=_KERNELS[self.volume]

    def dot(self,x,y):
        if x.val.shape != y.val.shape or x.val.size != 6*self.volume or x.val.dtype != cp.float32 or y.val.dtype != cp.float32:
            raise ValueError('Reference scalar product field layouts do not match.')
        self.dot_kernel((self.blocks,),(256,),
            (cp.ascontiguousarray(x.val),cp.ascontiguousarray(y.val),self.partial,np.int64(self.volume),self.weight))
        return self.finish(self.partial)

    def update_xr_rr(self,x,alpha,p,r,ap):
        if not all(t.val.flags.c_contiguous and t.val.dtype == cp.float32
                   and t.val.shape == x.val.shape and t.val.size == 6*self.volume
                   for t in (x,p,r,ap)):
            return None  # Keep Tensor's strided updates and evaluate the metric afterwards.
        self.update_kernel((self.blocks,),(256,),
            (cp.asarray(alpha,dtype=cp.float32),p.val,ap.val,x.val,r.val,self.partial,
             np.int64(self.volume),self.weight))
        return self.finish(self.partial)
