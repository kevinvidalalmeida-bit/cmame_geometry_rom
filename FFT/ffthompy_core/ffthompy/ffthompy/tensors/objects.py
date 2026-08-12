"""
This module contains classes and functions representing tensors
of trigonometric polynomials and relating operators.
"""

import numpy as np
from ffthompy.general.base import Representation
from ffthompy.trigpol import mean_index, fft_form_default, get_Nodd
from ffthompy.mechanics.matcoef import ElasticTensor
from ffthompy.trigpol import enlarge, decrease, get_inverse, Grid
from ffthompy.tensors.fft import (
    fftn, ifftn, fftnc, icfftn, rfftn, irfftn,
    CUPY_AVAILABLE, get_array_module, get_fft_backend, is_array_type,
    is_cupy_array, to_backend_array, to_host_array, cp,
)
import itertools
from copy import copy


def _scalar_to_float(x):
    if CUPY_AVAILABLE and is_cupy_array(x):
        return float(x.item())
    if isinstance(x, np.generic):
        return float(x.item())
    return float(x)


def _zeros_like_backend(arr):
    xp = get_array_module(arr)
    return xp.zeros_like(arr)


def _empty_like_backend(arr):
    xp = get_array_module(arr)
    return xp.empty_like(arr)


SYM21_PAIRS = (
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 2), (2, 3), (2, 4), (2, 5),
    (3, 3), (3, 4), (3, 5),
    (4, 4), (4, 5),
    (5, 5),
)

_CUPY_ELASTIC_DIRECT_KERNELS = {}
_CUPY_INDEXED_SYM21_KERNEL = None


class TensorFuns(Representation):

    def mean_index(self):
        return mean_index(self.N, self.fft_form)

    def __getitem__(self, ii):
        return self.val[ii]

    def pN(self):
        return np.prod(self.N)

    def point(self, ii):
        val=np.empty(self.shape)
        for ind in np.ndindex(*self.shape):
            val[ind]=self.val[ind][ii]
        return val

    def sub(self, ii):
        self.val[ii]

    def update(self, **kwargs):
        return self.__dict__.update(**kwargs)

    def _copy(self, keys, **kwargs):
        data={k:copy(self.__dict__[k]) for k in keys}
        data.update(kwargs)
        return self.__class__(**data)

    def copy(self, **kwargs):
        return self._copy(self.keys, **kwargs)

    def _set_fft(self, fft_form):
        assert(fft_form in ['c', 'r', 0])

        if fft_form in ['r']:
            self.N_fft=self.get_N_real(self.N)
            self.fftn=rfftn
            self.ifftn=irfftn
            self.fft_coef=np.prod(self.N)
        elif fft_form in [0]:
            self.N_fft=self.N
            self.fftn=fftn
            self.ifftn=ifftn
            self.fft_coef=1.
        elif fft_form in ['c']:
            self.N_fft=self.N
            self.fftn=fftnc
            self.ifftn=icfftn
            self.fft_coef=1.

        self.fft_form=fft_form

    def __repr__(self, full=False, detailed=False):
        keys=['order', 'name', 'Y', 'shape', 'N', 'Fourier', 'fft_form', 'origin', 'norm']
        ss=self._repr(keys)
        skip=4*' '
        if np.prod(np.array(self.shape))<=36 or detailed:
            ss+='{0}norm component-wise =\n{1}\n'.format(skip, str(self.norm(componentwise=True)))
            ss+='{0}mean = \n{1}\n'.format(skip, str(self.mean()))
        if full:
            ss+='{0}val = \n{1}'.format(skip, str(self.val))
        return ss

    @staticmethod
    def get_N_real(N):
        N_rfft=np.copy(N)
        N_rfft[-1]=int(np.fix(N[-1]/2)+1) # N[-1]//2+1
        return tuple(N_rfft)

    @staticmethod
    def get_N(N_rfft):
        N=np.copy(N_rfft)
        N[-1]=N_rfft[-1]*2-1
        return tuple(N)


class Tensor(TensorFuns):
    keys=('name','val','order','Y','N','multype','Fourier','fft_form','origin') # default keys

    def __init__(self, name='', val=None, order=None, shape=None, N=None, Y=None,
                 multype='scal', Fourier=False, fft_form=fft_form_default,
                 origin=0, dtype=None):

        self.name=name
        self.Fourier=Fourier
        self.origin=origin

        if is_array_type(val): # define: val + order
            self.val=val
            self.order=int(order)
            self.shape=self.val.shape[:order]
            if fft_form in ['r'] and Fourier:
                self.N=tuple(np.array(N, dtype=int))
            else:
                self.N=self.val.shape[order:]
            self._set_fft(fft_form)

        elif shape is not None and N is not None: # define: shape + N
            self.N=tuple(np.array(N, dtype=int))
            self._set_fft(fft_form)
            self.shape=tuple(np.array(shape, dtype=int))
            self.order=len(self.shape)

            backend = get_fft_backend()
            real_dtype = np.dtype(float if dtype is None else dtype)
            complex_dtype = np.complex64 if real_dtype == np.dtype(np.float32) else np.complex128
            if not self.Fourier:
                if backend == 'cupy' and CUPY_AVAILABLE:
                    import cupy as cp
                    self.val = cp.zeros(self.shape+self.N, dtype=real_dtype)
                else:
                    self.val=np.zeros(self.shape+self.N, dtype=real_dtype)
            else:
                if backend == 'cupy' and CUPY_AVAILABLE:
                    import cupy as cp
                    self.val = cp.zeros(self.shape+self.N_fft, dtype=complex_dtype)
                else:
                    self.val=np.zeros(self.shape+self.N_fft, dtype=complex_dtype)

        else:
            raise ValueError('Initialization of Tensor.')

        self.dim=self.N.__len__()
        if Y is None:
            self.Y=np.ones(self.dim, dtype=float)
        else:
            self.Y=np.array(Y, dtype=float)

        # definition of __mul__ operation
        self.multype=multype

    def set_fft_form(self, fft_form=fft_form_default, copy=False):
        if copy:
            R=self.copy()
        else:
            R=self

        if self.fft_form==fft_form:
            return R

        fft_form_orig = self.fft_form
        if R.Fourier:
            if fft_form_orig in ['r']:
                nval=np.flip(R.val[...,1:].conj(),axis=-1)
                for ax in self.axes[:-1]:
                    N,F = np.split(nval, [1], axis=ax)
                    nval=np.concatenate((N, np.flip(F, axis=ax)), axis=ax)
                if R.N[-1] % 2 == 0:
                    nval=nval[...,1:]
                val=np.concatenate((R.val,nval), axis=-1)
                R.val=1./np.prod(R.N)*val # fft_form=0
                if fft_form in ['c']:
                    R.val=np.fft.fftshift(R.val, axes=R.axes)
            elif fft_form_orig in ['c']:
                R.val=np.fft.ifftshift(R.val, axes=R.axes) # common for fft_form in [0,'r']
                if fft_form in ['r']:
                    R.val=R.val[...,:self.get_N_real(self.N)[-1]]*np.prod(self.N)
            elif fft_form_orig in [0]:
                if fft_form in ['c']:
                    R.val=np.fft.fftshift(R.val, axes=R.axes)
                else: # if fft_form in ['r']:
                    R.val=R.val[...,:self.get_N_real(self.N)[-1]]*np.prod(self.N)
        R._set_fft(fft_form)
        return R

    def shift(self, origin=None):
        """
        Shift the origin in the real domain.
        """
        assert(not self.Fourier)

        if origin==self.origin:
            return self
        elif origin is None:
            xp = get_array_module(self.val)
            if self.origin in [0]:
                self.val = xp.fft.fftshift(self.val, self.axes)
                self.origin='c'
            elif self.origin in ['c']:
                self.val = xp.fft.ifftshift(self.val, self.axes)
                self.origin=0
            return self
        else:
            raise ValueError()

    def randomize(self):
        if is_cupy_array(self.val):
            self.val = to_backend_array(np.random.random(self.val.shape), prefer_backend='cupy')
            if self.Fourier:
                self.val = self.val + 1j*to_backend_array(np.random.random(self.val.shape), prefer_backend='cupy')
        else:
            self.val=np.random.random(self.val.shape)
            if self.Fourier:
                self.val=self.val+1j*np.random.random(self.val.shape)
        return self

    def __neg__(self):
        return self.copy(name='-'+self.name[:10], val=-self.val)

    def __add__(self, x):
        if isinstance(x, Tensor):
            assert(self.Fourier==x.Fourier)
            assert(self.val.shape==x.val.shape)
            name='({0}+{1})'.format(self.name[:10], x.name[:10])
            xp = get_array_module(self.val, x.val)
            if xp is np:
                val = self.val + x.val
            else:
                val = to_backend_array(self.val, prefer_backend='cupy') + to_backend_array(x.val, prefer_backend='cupy')
            return self.copy(name=name, val=val)

        elif isinstance(x, np.ndarray) or isinstance(x, float):
            xp = get_array_module(self.val)
            if xp is np:
                return self.copy(val=self.val+x)
            return self.copy(val=self.val + to_backend_array(x, prefer_backend='cupy'))
        else:
            raise ValueError('Tensor.__add__')

    def __iadd__(self, x):
        if isinstance(x, Tensor):
            if is_cupy_array(self.val) or is_cupy_array(x.val):
                if not is_cupy_array(self.val):
                    self.val = to_backend_array(self.val, prefer_backend='cupy')
                self.val += to_backend_array(x.val, prefer_backend='cupy')
            else:
                self.val += x.val
        else:
            if is_cupy_array(self.val):
                self.val += to_backend_array(x, prefer_backend='cupy')
            else:
                self.val += x
        return self

    def __isub__(self, x):
        if isinstance(x, Tensor):
            if is_cupy_array(self.val) or is_cupy_array(x.val):
                if not is_cupy_array(self.val):
                    self.val = to_backend_array(self.val, prefer_backend='cupy')
                self.val -= to_backend_array(x.val, prefer_backend='cupy')
            else:
                self.val -= x.val
        else:
            if is_cupy_array(self.val):
                self.val -= to_backend_array(x, prefer_backend='cupy')
            else:
                self.val -= x
        return self

    def __imul__(self, x):
        if isinstance(x, Tensor):
            if is_cupy_array(self.val) or is_cupy_array(x.val):
                if not is_cupy_array(self.val):
                    self.val = to_backend_array(self.val, prefer_backend='cupy')
                self.val *= to_backend_array(x.val, prefer_backend='cupy')
            else:
                self.val *= x.val # Element-wise! (Hadamard)
        else:
            if is_cupy_array(self.val):
                self.val *= x
            else:
                self.val *= x # Scalar
        return self

    def __itruediv__(self, x):
        if isinstance(x, Tensor):
            if is_cupy_array(self.val) or is_cupy_array(x.val):
                if not is_cupy_array(self.val):
                    self.val = to_backend_array(self.val, prefer_backend='cupy')
                self.val /= to_backend_array(x.val, prefer_backend='cupy')
            else:
                self.val /= x.val
        else:
            if is_cupy_array(self.val):
                self.val /= x
            else:
                self.val /= x
        return self

    def __sub__(self, x):
        return self.__add__(-x)

    def __rmul__(self, x):
        if isinstance(x, Scalar):
            return self.copy(val=x.val*self.val)
        elif np.size(x)==1 or isinstance(x, float):
            return self.copy(val=x*self.val)
        else:
            raise ValueError()

    def __call__(self, *args, **kwargs):
        return self.__mul__(*args, **kwargs)

    def __mul__(self, Y, multype=None, *args, **kwargs):
        if multype is None:
            multype=self.multype
        X=self
        assert(X.Fourier==Y.Fourier)
        assert(X.fft_form==Y.fft_form)
        if multype in ['scal', 'scalar']:
            return scalar_product(X, Y)
        elif multype in [21, '21']:
            return einsum('ij...,j...->i...', X, Y)
        elif multype in ['sym21', 'symmetric21']:
            return sym21_product(X, Y)
        elif multype in ['sym21_indexed', 'indexed_sym21']:
            return indexed_sym21_product(X, Y)
        elif str(multype).strip().lower() in [
                'elasticity_hg1_direct', 'elasticity_g1_direct',
                'elasticity_hg2_direct', 'elasticity_g2_direct']:
            return elasticity_direct_product(X, Y)
        elif multype in [42, '42']:
            return einsum('ijkl...,kl...->ij...', X, Y)
        elif multype in [00, 'elementwise', 'hadamard']:
            return einsum('...,...->...', X, Y)
        elif multype in ['grad']:
            return einsum('i...,...->i...', X, Y)
        elif multype in ['div']:
            return einsum('i...,i...->...', X, Y)
        else:
            try:
                return einsum(multype, X, Y)
            except:
                raise ValueError()

    def inv(self):
        assert(self.Fourier is False)
        assert(self.order==2)
        assert(self.shape[0]==self.shape[1])
        return self.copy(name='inv({})'.format(self.name), val=get_inverse(self.val))

    def norm(self, ntype='L2', componentwise=False):
        if componentwise:
            scal=np.empty(self.shape)
            for ind in np.ndindex(*self.shape):
                obj=self.copy(name='aux', val=self.val[ind], order=0)
                scal[ind]=norm_fun(obj, ntype=ntype)
            return scal
        else:
            return norm_fun(self, ntype=ntype)

    def mean(self):
        """
        Mean of trigonometric polynomial of shape of macroscopic vector.
        """
        mean=np.zeros(self.shape)
        if self.Fourier:
            ind=self.mean_index()
            for di in np.ndindex(*self.shape):
                value = to_host_array(self.val[di][ind])
                mean[di]=np.real(value)/self.fft_coef
        else:
            for di in np.ndindex(*self.shape):
                value = self.val[di]
                xp = get_array_module(value)
                if xp is np:
                    mean[di]=np.mean(value)
                else:
                    mean[di]=_scalar_to_float(xp.mean(value))
        return mean

    def add_mean(self, mean):
        assert(self.shape==mean.shape)

        if self.Fourier:
            ind=self.mean_index()
            for di in np.ndindex(*self.shape):
                self.val[di+ind]=mean[di]*self.fft_coef
        else:
            for di in np.ndindex(*self.shape):
                self.val[di]+=mean[di]
        return self

    def set_mean(self, mean):
        assert(self.shape==mean.shape)
        self.add_mean(-self.mean()) # set mean to zero

        if self.Fourier:
            ind=self.mean_index()
            for di in np.ndindex(*self.shape):
                self.val[di+ind]=mean[di]*self.fft_coef
        else:
            for di in np.ndindex(*self.shape):
                self.val[di]+=mean[di]
        return self

    def __eq__(self, Y, full=True, tol=1e-13):
        """
        Check the equality with other objects comparable to trig. polynomials.
        """
        X=self
        _bool=False
        res=np.inf
        if (isinstance(X, Tensor) and X.fft_form==Y.fft_form and
                X.val.squeeze().shape==Y.val.squeeze().shape and X.Fourier==Y.Fourier):
            diff = X.val.squeeze()-Y.val.squeeze()
            if is_cupy_array(diff):
                diff = to_host_array(diff)
            res=np.linalg.norm(diff)
            if res<tol:
                _bool=True
        if full:
            return _bool, res
        else:
            return _bool

    def set_shape(self):
        shape_size=self.val.ndim-self.N.size
        self.shape=np.array(self.val.shape[:shape_size])
        return self.shape

    def transpose(self):
        if self.order==2:
            val=np.einsum('ij...->ji...', self.val)
        elif self.order==4:
            val=np.einsum('ijkl...->klij...', self.val)
        else:
            raise NotImplementedError()
        return self.copy(name=self.name[:10]+'.T', val=val)

    def transpose_left(self):
        res=self.empty_like(name=self.name[:10]+'.T')
        assert(self.order==4)
        res.val=np.einsum('ijkl...->jikl...', self.val)
        return res

    def transpose_right(self):
        res=self.empty_like(name=self.name[:10]+'.T')
        assert(self.order==4)
        res.val=np.einsum('ijkl...->ijlk...', self.val)
        return res

    def identity(self):
        self.val[:]=0.
        assert(self.order % 2 == 0)
        for ii in itertools.product(*tuple([list(range(n)) for n in self.shape[:int(self.order/2)]])):
            self.val[ii+ii]=1.

    def vec(self):
        """
        Returns one-dimensional vector (column) version of trigonometric
        polynomial.
        """
        return np.matrix(self.val.ravel()).transpose()

    def zeros_like(self, name=None):
        if name is None:
            name='zeros({})'.format(self.name[:10])
        return self.copy(name=name, val=_zeros_like_backend(self.val))

    def empty_like(self, name=None):
        if name is None:
            name='empty({})'.format(self.name[:10])
        return self.copy(name=name, val=_empty_like_backend(self.val))

    def calc_eigs(self, sort=True, symmetric=False, mandel=False):
        if symmetric:
            eigfun=np.linalg.eigvalsh
        else:
            eigfun=np.linalg.eigvals

        if self.order==2:
            eigs=np.zeros(self.N+(self.shape[0],))
            for ind in np.ndindex(self.N):
                mat=self.val[(slice(None), slice(None))+ind]
                eigs[ind]=eigfun(mat)

        elif self.order==4:
            if mandel:
                matrixfun=lambda x: ElasticTensor.create_mandel(x)
                d=self.shape[2]
                eigdim=d*(d+1)/2
            else:
                eigdim=self.shape[2]*self.shape[3]
                matrixfun=lambda x: np.reshape(x, 2*(eigdim,))

            eigs=np.zeros(self.N+(eigdim,))
            val=np.copy(self.val)
            for ii in range(self.dim):
                val=np.rollaxis(val, self.val.ndim-self.dim+ii, ii)

            for ind in np.ndindex(*self.N):
                eigs[ind]=eigfun(matrixfun(val[ind]))

        eigs=np.rollaxis(np.array(eigs), -1, 0)
        if sort:
            eigs=np.sort(eigs, axis=0)
        return eigs

    @property
    def axes(self): # axes for Fourier transform
        return tuple(range(self.order, self.order+self.dim))

    def fourier(self, Fourier=None, copy=False):
        assert(self.origin==0)

        if self.Fourier==Fourier:
            if copy:
                return self.copy()
            else:
                return self

        if copy:
            if self.Fourier:
                return self.copy(val=self.ifftn(self.val, self.N), Fourier=not self.Fourier)
            else:
                return self.copy(val=self.fftn(self.val, self.N), Fourier=not self.Fourier)
        else:
            if self.Fourier:
                self.val=self.ifftn(self.val, self.N)
            else:
                self.val=self.fftn(self.val, self.N)
            self.Fourier=not self.Fourier
            return self

    def enlarge(self, M):
        """
        It enlarges a trigonometric polynomial by adding zeros to the Fourier
        coefficients with high frequencies.
        """
        assert(self.Fourier)
        if np.allclose(self.N, M):
            return self
        else:
            fft_form=self.fft_form
            self.set_fft_form(fft_form='c')

            val = self.val
            for ii,ax in enumerate(self.axes):
                if self.N[ii]%2==0:
                    N0,C=np.split(val, [1], axis=ax)
                    N2=np.copy(N0)
                    for jj, axc in enumerate(self.axes):
                        if ax==axc:
                            continue
                        elif N2.shape[axc]%2==0:
                            N20,N2C=np.split(N2, [1], axis=axc)
                            N2=np.concatenate((N20, np.flip(N2C, axis=axc)), axis=axc)
                        else:
                            N2=np.flip(N2, axis=axc)
                    val=np.concatenate((0.5*N0,C,0.5*N2.conj()), axis=ax)

            # enlarging the centered part with odd N
            M = np.array(M, dtype=float)
            N = np.array(val.shape[self.order:], dtype=float)

            ibeg = np.ceil((M-N)/2).astype(int)
            iend = np.ceil((M+N)/2).astype(int)

            slc=self.order*[slice(None)]+[slice(ibeg[i],iend[i],1) for i in range(N.size)]
            newval = np.zeros(self.shape+tuple(M.astype(int)), dtype=self.val.dtype)
            newval[tuple(slc)]=val

            R=self.copy(val=newval, N=M, fft_form='c')
            return R.set_fft_form(fft_form=fft_form)

    def decrease(self, M):
        """
        As a dual to enlarge, it project/reduces a trigonometric polynomial by
        removing Fourier coefficients with high frequencies.
        """
        assert(self.Fourier)
        if np.allclose(self.N, M):
            return self
        else:
            fft_form=self.fft_form
            self.set_fft_form(fft_form='c')

            val=np.zeros(self.shape+tuple(M), dtype=self.val.dtype)
            for di in np.ndindex(*self.shape):
                val[di]=decrease(self.val[di], M)

            R=self.copy(val=val, N=M, fft_form='c')
            return R.set_fft_form(fft_form=fft_form)

    def project(self, M):
        """
        It projects a trigonometric polynomial to a polynomial with different grid.
        """

        if np.allclose(self.N, M):
            return self

        Fourier=self.Fourier
        if Fourier:
            Y=self.copy()
        else:
            Y=self.fourier(copy=copy)

        if np.all(np.greater(M, self.N)):
            Y=Y.enlarge(M)
        elif np.all(np.less(M, self.N)):
            Y=Y.decrease(M)
        else:
            raise NotImplementedError()

        if not Fourier:
            Y=Y.fourier()
        return Y

    def subfield(self, Y=None, M=None):
        """
        Return the subfield of the tensor depending either on the PUC size (Y) or
        number of points M. This is useful e.g. for stochastic computations to avoid correlation
        because of periodicity. As default, the subfield in the middle of the domain is taken.
        """
        N=np.array(self.N)

        if Y is None and M is None:
            raise ValueError('Either Y or M has to be specified.')
        elif Y is not None:
            M=np.ceil(Y/self.Y*N).astype(int)
        elif M is not None:
            M=np.ceil(M).astype(int)
        elif Y is None and M is None:
            raise ValueError('Only one of Y and M can be specified.')

        ind=[slice(None) for i in range(self.shape.__len__())]
        beg=np.round((N-M)/2).astype(int)
        ind=tuple(ind+[slice(beg[i], beg[i]+M[i]) for i in range(self.dim)])
        val=self[ind]
        return self.copy(val=val)

    def plot(self, ind=slice(None), N=None, filen=None, ptype='imshow'):
        if N is None:
            N = self.N

        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt
        from ffthompy.trigpol import Grid

        coord=Grid.get_coordinates(N, self.Y)

        Z=self.project(N)
        Z = Z.val[ind]
        if self.Fourier:
            Z=np.abs(Z)

        if Z.ndim != 2:
            raise ValueError("The plotting is suited only for dim=2!")

        fig = plt.figure()
        if ptype in ['wireframe']:
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(coord[-2], coord[-1], Z)
        elif ptype in ['surface']:
            from matplotlib import cm
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(coord[-2], coord[-1], Z,
                                   rstride=1, cstride=1, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)

            fig.colorbar(surf, shrink=0.5, aspect=5)
        elif ptype in ['imshow']:
            ax = plt.imshow(Z)
            plt.colorbar(ax)

        if filen is None:
            plt.show()
        else:
            plt.savefig(filen)


class Scalar():
    """
    Scalar value that is used to multiply VecTri or Matrix classes
    """
    def __init__(self, val=None, name='c'):
        if val is not None:
            self.val=val
        else:
            self.val=1.
        self.name=name

    def __call__(self, x):
        return self*x

    def __repr__(self):
        ss="Class : {0}\n".format(self.__class__.__name__)
        ss+="    val = {0}".format(self.val)
        return ss

    def transpose(self):
        return self

# @staticmethod
def einsum(str_operator, x, y):
    assert(x.Fourier==y.Fourier)
    assert(np.all(x.N==y.N))
    xp = get_array_module(x.val, y.val)
    if xp is np:
        val=np.einsum(str_operator, x.val, y.val)
    else:
        val=xp.einsum(str_operator, to_backend_array(x.val, prefer_backend='cupy'),
                      to_backend_array(y.val, prefer_backend='cupy'))
    order=len(val.shape)-len(x.N)
    return y.copy(name='{0}({1})'.format(x.name, y.name), val=val, order=order)


def sym21_product(x, y):
    assert(x.Fourier == y.Fourier)
    assert(np.all(x.N == y.N))
    assert(x.val.shape[0] == 21)
    assert(y.val.shape[0] == 6)
    xp = get_array_module(x.val, y.val)
    xval = x.val if xp is np else to_backend_array(x.val, prefer_backend='cupy')
    yval = y.val if xp is np else to_backend_array(y.val, prefer_backend='cupy')

    out = xp.zeros((6,) + yval.shape[1:], dtype=xp.result_type(xval.dtype, yval.dtype))
    for kk, (ii, jj) in enumerate(SYM21_PAIRS):
        cval = xval[kk]
        out[ii] += cval * yval[jj]
        if ii != jj:
            out[jj] += cval * yval[ii]
    order = len(out.shape) - len(x.N)
    return y.copy(name='{0}({1})'.format(x.name, y.name), val=out, order=order)


def _get_cupy_indexed_sym21_kernel():
    global _CUPY_INDEXED_SYM21_KERNEL
    if _CUPY_INDEXED_SYM21_KERNEL is not None:
        return _CUPY_INDEXED_SYM21_KERNEL
    if not CUPY_AVAILABLE or cp is None:
        return None

    code = r'''
    extern "C" __global__
    void indexed_sym21_apply(const int* __restrict__ material_ids,
                             const float* __restrict__ table,
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
        const float* A = table + ((long)material_ids[s]) * 21;

        const float x0 = X[0 * stride + base];
        const float x1 = X[1 * stride + base];
        const float x2 = X[2 * stride + base];
        const float x3 = X[3 * stride + base];
        const float x4 = X[4 * stride + base];
        const float x5 = X[5 * stride + base];

        const float a00 = A[ 0];
        const float a01 = A[ 1];
        const float a02 = A[ 2];
        const float a03 = A[ 3];
        const float a04 = A[ 4];
        const float a05 = A[ 5];
        const float a11 = A[ 6];
        const float a12 = A[ 7];
        const float a13 = A[ 8];
        const float a14 = A[ 9];
        const float a15 = A[10];
        const float a22 = A[11];
        const float a23 = A[12];
        const float a24 = A[13];
        const float a25 = A[14];
        const float a33 = A[15];
        const float a34 = A[16];
        const float a35 = A[17];
        const float a44 = A[18];
        const float a45 = A[19];
        const float a55 = A[20];

        Y[0 * stride + base] = a00*x0 + a01*x1 + a02*x2 + a03*x3 + a04*x4 + a05*x5;
        Y[1 * stride + base] = a01*x0 + a11*x1 + a12*x2 + a13*x3 + a14*x4 + a15*x5;
        Y[2 * stride + base] = a02*x0 + a12*x1 + a22*x2 + a23*x3 + a24*x4 + a25*x5;
        Y[3 * stride + base] = a03*x0 + a13*x1 + a23*x2 + a33*x3 + a34*x4 + a35*x5;
        Y[4 * stride + base] = a04*x0 + a14*x1 + a24*x2 + a34*x3 + a44*x4 + a45*x5;
        Y[5 * stride + base] = a05*x0 + a15*x1 + a25*x2 + a35*x3 + a45*x4 + a55*x5;
    }
    '''
    try:
        _CUPY_INDEXED_SYM21_KERNEL = cp.RawKernel(code, "indexed_sym21_apply")
    except Exception:
        return None
    return _CUPY_INDEXED_SYM21_KERNEL


def indexed_sym21_array(x, yval):
    if not (CUPY_AVAILABLE and cp is not None and is_cupy_array(yval)):
        return None
    material_ids = getattr(x, "val", None)
    table = getattr(x, "material_table", None)
    if not (is_cupy_array(material_ids) and is_cupy_array(table)):
        return None
    if yval.dtype != cp.float32 or table.dtype != cp.float32:
        return None
    if material_ids.dtype != cp.int32 or yval.shape[0] != 6:
        return None

    if yval.ndim == len(x.N) + 2:
        batch = int(yval.shape[1])
        spatial_shape = tuple(int(v) for v in yval.shape[2:])
        out_shape = (6, batch) + spatial_shape
    elif yval.ndim == len(x.N) + 1:
        batch = 1
        spatial_shape = tuple(int(v) for v in yval.shape[1:])
        yval = yval[:, None, ...]
        out_shape = (6,) + spatial_shape
    else:
        return None
    if spatial_shape != tuple(int(v) for v in x.N):
        return None

    ids = material_ids.reshape(-1)
    if not ids.flags.c_contiguous or not table.flags.c_contiguous:
        return None
    if not yval.flags.c_contiguous:
        yval = cp.ascontiguousarray(yval)

    spatial = int(np.prod(spatial_shape))
    total = int(batch * spatial)
    out_raw = cp.empty((6, batch) + spatial_shape, dtype=cp.float32)
    kernel = _get_cupy_indexed_sym21_kernel()
    if kernel is None:
        return None
    threads = 256
    blocks = (total + threads - 1) // threads
    try:
        kernel(
            (blocks,),
            (threads,),
            (
                ids,
                table,
                yval,
                out_raw,
                np.int64(batch),
                np.int64(spatial),
                np.int64(total),
            ),
        )
    except Exception:
        return None
    return out_raw if len(out_shape) == len(spatial_shape) + 2 else out_raw[:, 0, ...]


def indexed_sym21_product(x, y):
    assert(x.Fourier == y.Fourier)
    assert(np.all(x.N == y.N))
    out = indexed_sym21_array(x, y.val)
    if out is None:
        raise NotImplementedError("indexed_sym21_product requiere CuPy float32.")
    order = len(out.shape) - len(x.N)
    return y.copy(name='{0}({1})'.format(x.name, y.name), val=out, order=order)


def _get_cupy_elastic_direct_kernel(kind, projection):
    key = (kind, projection)
    kernel = _CUPY_ELASTIC_DIRECT_KERNELS.get(key)
    if kernel is not None:
        return kernel

    if kind == "f32_r32":
        x_type = "float"
        y_type = "float"
        load_x = """
            const float x0r = X[0 * stride + base];
            const float x1r = X[1 * stride + base];
            const float x2r = X[2 * stride + base];
            const float x3r = X[3 * stride + base];
            const float x4r = X[4 * stride + base];
            const float x5r = X[5 * stride + base];
        """
        imag_defs = """
            const float x0i = 0.0f;
            const float x1i = 0.0f;
            const float x2i = 0.0f;
            const float x3i = 0.0f;
            const float x4i = 0.0f;
            const float x5i = 0.0f;
        """
        write_y = """
            Y[0 * stride + base] = y0r;
            Y[1 * stride + base] = y1r;
            Y[2 * stride + base] = y2r;
            Y[3 * stride + base] = y3r;
            Y[4 * stride + base] = y4r;
            Y[5 * stride + base] = y5r;
        """
    elif kind == "f32_c64":
        x_type = "float2"
        y_type = "float2"
        load_x = """
            const float2 x0 = X[0 * stride + base];
            const float2 x1 = X[1 * stride + base];
            const float2 x2 = X[2 * stride + base];
            const float2 x3 = X[3 * stride + base];
            const float2 x4 = X[4 * stride + base];
            const float2 x5 = X[5 * stride + base];
            const float x0r = x0.x; const float x0i = x0.y;
            const float x1r = x1.x; const float x1i = x1.y;
            const float x2r = x2.x; const float x2i = x2.y;
            const float x3r = x3.x; const float x3i = x3.y;
            const float x4r = x4.x; const float x4i = x4.y;
            const float x5r = x5.x; const float x5i = x5.y;
        """
        imag_defs = ""
        write_y = """
            Y[0 * stride + base] = make_float2(y0r, y0i);
            Y[1 * stride + base] = make_float2(y1r, y1i);
            Y[2 * stride + base] = make_float2(y2r, y2i);
            Y[3 * stride + base] = make_float2(y3r, y3i);
            Y[4 * stride + base] = make_float2(y4r, y4i);
            Y[5 * stride + base] = make_float2(y5r, y5i);
        """
    else:
        return None

    g2_expr = ""
    if projection == "g2":
        g2_expr = """
            y0r = x0r - y0r; y0i = x0i - y0i;
            y1r = x1r - y1r; y1i = x1i - y1i;
            y2r = x2r - y2r; y2i = x2i - y2i;
            y3r = x3r - y3r; y3i = x3i - y3i;
            y4r = x4r - y4r; y4i = x4i - y4i;
            y5r = x5r - y5r; y5i = x5i - y5i;
        """

    code = r'''
    extern "C" __global__
    void elasticity_direct_apply(const XTYPE* __restrict__ X,
                                 YTYPE* __restrict__ Y,
                                 const long batch,
                                 const long n0,
                                 const long n1,
                                 const long n2_data,
                                 const long n2_full,
                                 const float inv_y0,
                                 const float inv_y1,
                                 const float inv_y2,
                                 const long total) {
        const long idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= total) return;

        const long spatial = n0 * n1 * n2_data;
        const long s = idx % spatial;
        const long b = idx / spatial;
        const long i2 = s % n2_data;
        const long tmp = s / n2_data;
        const long i1 = tmp % n1;
        const long i0 = tmp / n1;
        const long base = b * spatial + s;
        const long stride = batch * spatial;

        LOAD_X
        IMAG_DEFS

        const bool nyq0 = ((n0 % 2) == 0) && (i0 == n0 / 2);
        const bool nyq1 = ((n1 % 2) == 0) && (i1 == n1 / 2);
        const bool nyq2 = ((n2_full % 2) == 0) && (i2 == n2_full / 2);
        if ((i0 == 0 && i1 == 0 && i2 == 0) || nyq0 || nyq1 || nyq2) {
            const float y0r = 0.0f; const float y0i = 0.0f;
            const float y1r = 0.0f; const float y1i = 0.0f;
            const float y2r = 0.0f; const float y2i = 0.0f;
            const float y3r = 0.0f; const float y3i = 0.0f;
            const float y4r = 0.0f; const float y4i = 0.0f;
            const float y5r = 0.0f; const float y5i = 0.0f;
            WRITE_Y
            return;
        }

        const bool is_rfft = n2_data != n2_full;
        const float k0_raw = (i0 <= n0 / 2) ? (float)i0 : (float)(i0 - n0);
        const float k1_raw = (i1 <= n1 / 2) ? (float)i1 : (float)(i1 - n1);
        const float k2_raw = (is_rfft || i2 <= n2_full / 2) ? (float)i2 : (float)(i2 - n2_full);
        const float k0 = k0_raw * inv_y0;
        const float k1 = k1_raw * inv_y1;
        const float k2 = k2_raw * inv_y2;

        const float q00 = k0 * k0;
        const float q11 = k1 * k1;
        const float q22 = k2 * k2;
        const float q01 = k0 * k1;
        const float q02 = k0 * k2;
        const float q12 = k1 * k2;
        const float invn2 = 1.0f / (q00 + q11 + q22);
        const float invn4 = invn2 * invn2;
        const float rt2 = 1.4142135623730951f;

        const float a00 = 2.0f*q00*invn2 - q00*q00*invn4;
        const float a01 = -q00*q11*invn4;
        const float a02 = -q00*q22*invn4;
        const float a03 = -rt2*q00*q12*invn4;
        const float a04 = rt2*q02*invn2 - rt2*q00*q02*invn4;
        const float a05 = rt2*q01*invn2 - rt2*q00*q01*invn4;

        const float a11 = 2.0f*q11*invn2 - q11*q11*invn4;
        const float a12 = -q11*q22*invn4;
        const float a13 = rt2*q12*invn2 - rt2*q11*q12*invn4;
        const float a14 = -rt2*q11*q02*invn4;
        const float a15 = rt2*q01*invn2 - rt2*q11*q01*invn4;

        const float a22 = 2.0f*q22*invn2 - q22*q22*invn4;
        const float a23 = rt2*q12*invn2 - rt2*q22*q12*invn4;
        const float a24 = rt2*q02*invn2 - rt2*q22*q02*invn4;
        const float a25 = -rt2*q22*q01*invn4;

        const float a33 = (1.0f - q00*invn2) - 2.0f*q12*q12*invn4;
        const float a34 = q01*invn2 - 2.0f*q12*q02*invn4;
        const float a35 = q02*invn2 - 2.0f*q12*q01*invn4;
        const float a44 = (1.0f - q11*invn2) - 2.0f*q02*q02*invn4;
        const float a45 = q12*invn2 - 2.0f*q02*q01*invn4;
        const float a55 = (1.0f - q22*invn2) - 2.0f*q01*q01*invn4;

        float y0r = a00*x0r + a01*x1r + a02*x2r + a03*x3r + a04*x4r + a05*x5r;
        float y1r = a01*x0r + a11*x1r + a12*x2r + a13*x3r + a14*x4r + a15*x5r;
        float y2r = a02*x0r + a12*x1r + a22*x2r + a23*x3r + a24*x4r + a25*x5r;
        float y3r = a03*x0r + a13*x1r + a23*x2r + a33*x3r + a34*x4r + a35*x5r;
        float y4r = a04*x0r + a14*x1r + a24*x2r + a34*x3r + a44*x4r + a45*x5r;
        float y5r = a05*x0r + a15*x1r + a25*x2r + a35*x3r + a45*x4r + a55*x5r;

        float y0i = a00*x0i + a01*x1i + a02*x2i + a03*x3i + a04*x4i + a05*x5i;
        float y1i = a01*x0i + a11*x1i + a12*x2i + a13*x3i + a14*x4i + a15*x5i;
        float y2i = a02*x0i + a12*x1i + a22*x2i + a23*x3i + a24*x4i + a25*x5i;
        float y3i = a03*x0i + a13*x1i + a23*x2i + a33*x3i + a34*x4i + a35*x5i;
        float y4i = a04*x0i + a14*x1i + a24*x2i + a34*x3i + a44*x4i + a45*x5i;
        float y5i = a05*x0i + a15*x1i + a25*x2i + a35*x3i + a45*x4i + a55*x5i;

        G2_EXPR
        WRITE_Y
    }
    '''
    code = (
        code.replace("XTYPE", x_type)
            .replace("YTYPE", y_type)
            .replace("LOAD_X", load_x)
            .replace("IMAG_DEFS", imag_defs)
            .replace("WRITE_Y", write_y)
            .replace("G2_EXPR", g2_expr)
    )

    try:
        kernel = cp.RawKernel(code, "elasticity_direct_apply")
    except Exception:
        return None
    _CUPY_ELASTIC_DIRECT_KERNELS[key] = kernel
    return kernel


def elasticity_direct_array(x, yval):
    if not (CUPY_AVAILABLE and cp is not None and is_cupy_array(yval)):
        return None
    if yval.shape[0] != 6:
        return None
    if yval.dtype == cp.float32:
        kind = "f32_r32"
        out_dtype = cp.float32
    elif yval.dtype == cp.complex64:
        kind = "f32_c64"
        out_dtype = cp.complex64
    else:
        return None

    multype = str(getattr(x, "multype", "")).strip().lower()
    if multype in {"elasticity_hg1_direct", "elasticity_g1_direct"}:
        projection = "g1"
    elif multype in {"elasticity_hg2_direct", "elasticity_g2_direct"}:
        projection = "g2"
    else:
        return None

    if yval.ndim == len(x.N) + 2:
        batch = int(yval.shape[1])
        spatial_shape = tuple(int(v) for v in yval.shape[2:])
        out_shape = (6, batch) + spatial_shape
    elif yval.ndim == len(x.N) + 1:
        batch = 1
        spatial_shape = tuple(int(v) for v in yval.shape[1:])
        yval = yval[:, None, ...]
        out_shape = (6,) + spatial_shape
    else:
        return None

    full_shape = tuple(int(v) for v in x.N)
    if getattr(x, "fft_form", None) == 'r':
        expected_shape = full_shape[:-1] + (int(full_shape[-1] // 2 + 1),)
    else:
        expected_shape = full_shape
    if spatial_shape != expected_shape:
        return None
    if len(spatial_shape) != 3:
        return None
    if not yval.flags.c_contiguous:
        yval = cp.ascontiguousarray(yval)

    kernel = _get_cupy_elastic_direct_kernel(kind, projection)
    if kernel is None:
        return None

    n0, n1, n2_data = spatial_shape
    n2_full = int(full_shape[2])
    spatial = int(n0 * n1 * n2_data)
    out_raw = cp.empty((6, batch) + spatial_shape, dtype=out_dtype)
    total = int(batch * spatial)
    threads = 256
    blocks = (total + threads - 1) // threads
    inv_y = [float(1.0 / yy) for yy in getattr(x, "Y", np.ones(3, dtype=float))]

    try:
        kernel(
            (blocks,),
            (threads,),
            (
                yval,
                out_raw,
                np.int64(batch),
                np.int64(n0),
                np.int64(n1),
                np.int64(n2_data),
                np.int64(n2_full),
                np.float32(inv_y[0]),
                np.float32(inv_y[1]),
                np.float32(inv_y[2]),
                np.int64(total),
            ),
        )
    except Exception:
        return None
    return out_raw if len(out_shape) == len(spatial_shape) + 2 else out_raw[:, 0, ...]


def elasticity_direct_product(x, y):
    assert(x.Fourier == y.Fourier)
    assert(np.all(x.N == y.N))
    assert(y.val.shape[0] == 6)
    xp = get_array_module(y.val)
    if xp is not np:
        yval = to_backend_array(y.val, prefer_backend='cupy')
        out = elasticity_direct_array(x, yval)
        if out is not None:
            order = len(out.shape) - len(x.N)
            return y.copy(name='{0}({1})'.format(x.name, y.name), val=out, order=order)
    raise NotImplementedError("elasticity_direct_product requiere CuPy y arreglos GPU.")

def norm_fun(X, ntype):
    if ntype in ['L2', 2]:
        scal=(scalar_product(X, X))**0.5
    elif ntype==1:
        scal=np.sum(np.abs(X.val))
    elif ntype=='inf':
        scal=np.max(np.abs(X.val))
    else:
        msg="This type ({}) of norm is not implemented!".format(ntype)
        raise NotImplementedError(msg)
    return scal

def scalar_product(y, x):
    assert(isinstance(x, Tensor))
    assert(y.val.shape==x.val.shape)
    assert(y.fft_form==x.fft_form)

    xp = get_array_module(y.val, x.val)
    if xp is np:
        yval = y.val
        xval = x.val
    else:
        yval = to_backend_array(y.val, prefer_backend='cupy')
        xval = to_backend_array(x.val, prefer_backend='cupy')

    if y.Fourier:
        if x.fft_form in ['r']:
            if x.N[-1] % 2 == 1:
                if xp is np:
                    scal=(np.sum(yval[...,0]*np.conj(xval[...,0])).real +
                          2*np.sum(yval[...,1:]*np.conj(xval[...,1:])).real)/np.prod(y.N)**2
                else:
                    scal=(xp.sum(yval[...,0]*xp.conj(xval[...,0])).real +
                          2*xp.sum(yval[...,1:]*xp.conj(xval[...,1:])).real)/np.prod(y.N)**2
            else:
                if xp is np:
                    scal=(np.sum(yval[...,0]*np.conj(xval[...,0])).real +
                          np.sum(yval[...,-1]*np.conj(xval[...,-1])).real +
                          2*np.sum(yval[...,1:-1]*np.conj(xval[...,1:-1])).real)/np.prod(y.N)**2
                else:
                    scal=(xp.sum(yval[...,0]*xp.conj(xval[...,0])).real +
                          xp.sum(yval[...,-1]*xp.conj(xval[...,-1])).real +
                          2*xp.sum(yval[...,1:-1]*xp.conj(xval[...,1:-1])).real)/np.prod(y.N)**2
        else:
            if xp is np:
                scal=np.sum(yval[:]*np.conj(xval[:])).real
            else:
                scal=xp.sum(yval[:]*xp.conj(xval[:])).real
    else:
        if xp is np:
            scal=np.sum(yval[:]*xval[:])/np.prod(y.N)
        else:
            scal=xp.sum(yval[:]*xval[:])/np.prod(y.N)
    return _scalar_to_float(scal)

if __name__=='__main__':
    N=np.array([4,4], dtype=int)
    M=2*N
    u=Tensor(name='test', shape=(), N=N, Fourier=False, fft_form='r')
    u.randomize()

    Fur=u.fourier(copy=True)
    Fuc2=Fur.set_fft_form(fft_form='c', copy=True)
    uc=u.set_fft_form(fft_form='c', copy=True)
    Fuc=uc.fourier(copy=True)
    print(u)
    print(Fur)
    print(uc)
    print(Fuc)

    print('end')
