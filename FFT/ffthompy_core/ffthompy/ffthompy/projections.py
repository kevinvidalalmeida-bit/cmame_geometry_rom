import numpy as np
import scipy as sp
from ffthompy.trigpol import Grid, get_Nodd, mean_index, fft_form_default
from ffthompy.matvecs import Matrix
from ffthompy.tensors import Tensor
from ffthompy.tensors.fft import CUPY_AVAILABLE, cp, get_fft_backend
from ffthompy.tensors.objects import SYM21_PAIRS
import itertools


def _array_backend(backend):
    backend = str(backend or 'numpy').strip().lower()
    if backend == 'auto':
        backend = 'cupy' if get_fft_backend() == 'cupy' else 'numpy'
    if backend == 'cupy':
        if not CUPY_AVAILABLE or cp is None:
            return np, 'numpy'
        return cp, 'cupy'
    return np, 'numpy'


def _enlarge_fourier_array(val, current_N, target_N, order, fft_form, xp):
    current_N = np.array(current_N, dtype=int)
    target_N = np.array(target_N, dtype=int)
    if np.allclose(current_N, target_N):
        return val

    axes = tuple(range(order, order + current_N.size))
    val_c = val
    if fft_form in [0]:
        val_c = xp.fft.fftshift(val_c, axes=axes)
    elif fft_form not in ['c']:
        raise NotImplementedError("enlarge directo solo soporta fft_form 0 o 'c'.")

    for ii, ax in enumerate(axes):
        if current_N[ii] % 2 == 0:
            N0, C = xp.split(val_c, [1], axis=ax)
            N2 = xp.copy(N0)
            for axc in axes:
                if ax == axc:
                    continue
                if N2.shape[axc] % 2 == 0:
                    N20, N2C = xp.split(N2, [1], axis=axc)
                    N2 = xp.concatenate((N20, xp.flip(N2C, axis=axc)), axis=axc)
                else:
                    N2 = xp.flip(N2, axis=axc)
            val_c = xp.concatenate((0.5*N0, C, 0.5*xp.conj(N2)), axis=ax)

    current_shape = np.array(val_c.shape[order:], dtype=float)
    target_float = target_N.astype(float)
    ibeg = np.ceil((target_float - current_shape)/2).astype(int)
    iend = np.ceil((target_float + current_shape)/2).astype(int)
    slc = order*[slice(None)] + [slice(ibeg[i], iend[i], 1) for i in range(target_N.size)]
    newval = xp.zeros(val_c.shape[:order] + tuple(target_N.tolist()), dtype=val_c.dtype)
    newval[tuple(slc)] = val_c

    if fft_form in [0]:
        newval = xp.fft.ifftshift(newval, axes=axes)
    return newval


def scalar(N, Y, NyqNul=True, tensor=True, fft_form=fft_form_default, dtype=float):
    """
    Assembly of discrete kernels in Fourier space for scalar elliptic problems.

    Parameters
    ----------
    N : numpy.ndarray
        no. of discretization points
    Y : numpy.ndarray
        size of periodic unit cell

    Returns
    -------
    G1l : numpy.ndarray
        discrete kernel in Fourier space; provides projection
        on curl-free fields with zero mean
    G2l : numpy.ndarray
        discrete kernel in Fourier space; provides projection
        on divergence-free fields with zero mean
    """
    if fft_form in ['r']:
        fft_form_r=True
        fft_form=0
    else:
        fft_form_r=False

    dtype = np.dtype(dtype)
    d = np.size(N)
    N = np.array(N, dtype=int)
    if NyqNul:
        Nred = get_Nodd(N)
    else:
        Nred = N

    xi = Grid.get_xil(Nred, Y, fft_form=fft_form)
    xi2 = []
    for m in np.arange(d):
        xi2.append(xi[m]**2)

    G0l = np.zeros(np.hstack([d, d, Nred]), dtype=dtype)
    G1l = np.zeros(np.hstack([d, d, Nred]), dtype=dtype)
    G2l = np.zeros(np.hstack([d, d, Nred]), dtype=dtype)
    num = np.zeros(np.hstack([d, d, Nred]), dtype=dtype)
    denom = np.zeros(Nred, dtype=dtype)

    ind_center = mean_index(Nred, fft_form=fft_form)
    for m in np.arange(d): # diagonal components
        Nshape = np.ones(d, dtype=int)
        Nshape[m] = Nred[m]
        Nrep = np.copy(Nred)
        Nrep[m] = 1
        a = np.reshape(xi2[m], Nshape)
        num[m][m] = np.tile(a, Nrep) # numerator
        denom = denom + num[m][m]
        G0l[m, m][ind_center] = 1

    for m in np.arange(d): # upper diagonal components
        for n in np.arange(m+1, d):
            NshapeM = np.ones(d, dtype=int)
            NshapeM[m] = Nred[m]
            NrepM = np.copy(Nred)
            NrepM[m] = 1
            NshapeN = np.ones(d, dtype=int)
            NshapeN[n] = Nred[n]
            NrepN = np.copy(Nred)
            NrepN[n] = 1
            num[m][n] = np.tile(np.reshape(xi[m], NshapeM), NrepM) \
                * np.tile(np.reshape(xi[n], NshapeN), NrepN)

    # avoiding a division by zero
    denom[ind_center] = 1

    # calculation of projections
    for m in np.arange(d):
        for n in np.arange(m, d):
            G1l[m][n] = num[m][n]/denom
            G2l[m][n] = (m == n)*np.ones(Nred, dtype=dtype) - G1l[m][n]
            G2l[m][n][ind_center] = 0

    # symmetrization
    for m in np.arange(1, d):
        for n in np.arange(m):
            G1l[m][n] = G1l[n][m]
            G2l[m][n] = G2l[n][m]

    if tensor:
        G0l = Tensor(name='hG0', val=G0l, order=2, N=N, multype=21, Fourier=True, fft_form=fft_form)
        G1l = Tensor(name='hG1', val=G1l, order=2, N=N, multype=21, Fourier=True, fft_form=fft_form)
        G2l = Tensor(name='hG2', val=G2l, order=2, N=N, multype=21, Fourier=True, fft_form=fft_form)
    else:
        G0l = Matrix(name='hG0', val=G0l, Fourier=True)
        G1l = Matrix(name='hG1', val=G1l, Fourier=True)
        G2l = Matrix(name='hG2', val=G2l, Fourier=True)

    if NyqNul:
        G0l = G0l.enlarge(N)
        G1l = G1l.enlarge(N)
        G2l = G2l.enlarge(N)

    if fft_form_r:
        for tensor in [G0l, G1l, G2l]:
            tensor.set_fft_form(fft_form='r')
            tensor.val/=np.prod(tensor.N)

    return G0l, G1l, G2l

def elasticity(N, Y, NyqNul=True, tensor=True, fft_form=fft_form_default, dtype=float):
    """
    Projection matrix on a space of admissible strain fields
    INPUT =
        N : ndarray of e.g. stiffness coefficients
        d : dimension; d = 2
        D : dimension in engineering notation; D = 3
        Y : the size of periodic unit cell
    OUTPUT =
        G1h,G1s,G2h,G2s : projection matrices of size DxDxN
    """
    if fft_form in ['r']:
        fft_form_r=True
        fft_form=0
    else:
        fft_form_r=False

    dtype = np.dtype(dtype)
    N = np.array(N, dtype=int)
    d = N.size
    D = int(d*(d+1)/2)

    if NyqNul:
        Nred = get_Nodd(N)
    else:
        Nred = N

    xi = Grid.get_xil(Nred, Y, fft_form=fft_form)

    xi2 = []
    for ii in range(d):
        xi2.append(xi[ii]**2)

    num = np.zeros(np.hstack([d, d, Nred]), dtype=dtype)
    norm2_xi = np.zeros(Nred, dtype=dtype)
    for mm in np.arange(d): # diagonal components
        Nshape = np.ones(d, dtype=int)
        Nshape[mm] = Nred[mm]
        Nrep = np.copy(Nred)
        Nrep[mm] = 1
        num[mm][mm] = np.tile(np.reshape(xi2[mm], Nshape), Nrep) # numerator
        norm2_xi += num[mm][mm]

    norm4_xi = norm2_xi**2
    ind_center = mean_index(Nred, fft_form=fft_form)
    # avoid division by zero
    norm2_xi[ind_center] = 1
    norm4_xi[ind_center] = 1

    for m in np.arange(d): # upper diagonal components
        for n in np.arange(m+1, d):
            NshapeM = np.ones(d, dtype=int)
            NshapeM[m] = Nred[m]
            NrepM = np.copy(Nred)
            NrepM[m] = 1
            NshapeN = np.ones(d, dtype=int)
            NshapeN[n] = Nred[n]
            NrepN = np.copy(Nred)
            NrepN[n] = 1
            num[m][n] = np.tile(np.reshape(xi[m], NshapeM), NrepM) \
                * np.tile(np.reshape(xi[n], NshapeN), NrepN)

    # G1h = np.zeros([D,D]).tolist()
    G1h = np.zeros(np.hstack([D, D, Nred]), dtype=dtype)
    G1s = np.zeros(np.hstack([D, D, Nred]), dtype=dtype)
    IS0 = np.zeros(np.hstack([D, D, Nred]), dtype=dtype)
    mean = np.zeros(np.hstack([D, D, Nred]), dtype=dtype)
    Lamh = np.zeros(np.hstack([D, D, Nred]), dtype=dtype)
    S = np.zeros(np.hstack([D, D, Nred]), dtype=dtype)
    W = np.zeros(np.hstack([D, D, Nred]), dtype=dtype)
    WT = np.zeros(np.hstack([D, D, Nred]), dtype=dtype)

    for m in np.arange(d):
        S[m][m] = 2*num[m][m]/norm2_xi
        for n in np.arange(d):
            G1h[m][n] = num[m][m]*num[n][n]/norm4_xi
            Lamh[m][n] = np.ones(Nred, dtype=dtype)/d
            Lamh[m][n][ind_center] = 0

    for m in np.arange(D):
        IS0[m][m] = np.ones(Nred, dtype=dtype)
        IS0[m][m][ind_center] = 0
        mean[m][m][ind_center] = 1

    if d == 2:
        S[0][2] = 2**0.5*num[0][1]/norm2_xi
        S[1][2] = 2**0.5*num[0][1]/norm2_xi
        S[2][2] = np.ones(Nred)
        S[2][2][ind_center] = 0
        G1h[0][2] = 2**0.5*num[0][0]*num[0][1]/norm4_xi
        G1h[1][2] = 2**0.5*num[0][1]*num[1][1]/norm4_xi
        G1h[2][2] = 2*num[0][0]*num[1][1]/norm4_xi
        for m in np.arange(d):
            for n in np.arange(d):
                W[m][n] = num[m][m]/norm2_xi
            W[2][m] = 2**.5*num[0][1]/norm2_xi

    elif d == 3:
        for m in np.arange(d):
            S[m+3][m+3] = 1 - num[m][m]/norm2_xi
            S[m+3][m+3][ind_center] = 0
        for m in np.arange(d):
            for n in np.arange(m+1, d):
                S[m+3][n+3] = num[m][n]/norm2_xi
                G1h[m+3][n+3] = num[m][m]*num[n][n]/norm4_xi
        for m in np.arange(d):
            for n in np.arange(d):
                ind = np.setdiff1d(np.arange(d), [n])
                S[m][n+3] = (0 == (m == n))*2**.5*num[ind[0]][ind[1]]/norm2_xi
                G1h[m][n+3] = 2**.5*num[m][m]*num[ind[0]][ind[1]]/norm4_xi
                W[m][n] = num[m][m]/norm2_xi
                W[n+3][m] = 2**.5*num[ind[0]][ind[1]]/norm2_xi
        for m in np.arange(d):
            for n in np.arange(d):
                ind_m = np.setdiff1d(np.arange(d), [m])
                ind_n = np.setdiff1d(np.arange(d), [n])
                G1h[m+3][n+3] = 2*num[ind_m[0]][ind_m[1]] \
                    * num[ind_n[0]][ind_n[1]] / norm4_xi
    # symmetrization
    for n in np.arange(D):
        for m in np.arange(n+1, D):
            S[m][n] = S[n][m]
            G1h[m][n] = G1h[n][m]
    for m in np.arange(D):
        for n in np.arange(D):
            G1s[m][n] = S[m][n] - 2*G1h[m][n]
            WT[m][n] = W[n][m]
    G2h = np.asarray(1./(d-1)*(d*Lamh + G1h - W - WT), dtype=dtype)
    G2s = np.asarray(IS0 - G1h - G1s - G2h, dtype=dtype)

    if tensor:
        G0 = Tensor(name='hG0', val=mean, order=2, N=N, Fourier=True, multype=21, fft_form=fft_form)
        G1h = Tensor(name='hG1h', val=G1h, order=2, N=N, Fourier=True, multype=21, fft_form=fft_form)
        G1s = Tensor(name='hG1s', val=G1s, order=2, N=N, Fourier=True, multype=21, fft_form=fft_form)
        G2h = Tensor(name='hG2h', val=G2h, order=2, N=N, Fourier=True, multype=21, fft_form=fft_form)
        G2s = Tensor(name='hG2s', val=G2s, order=2, N=N, Fourier=True, multype=21, fft_form=fft_form)
    else:
        G0 = Matrix(name='hG0', val=mean, Fourier=True)
        G1h = Matrix(name='hG1h', val=G1h, Fourier=True)
        G1s = Matrix(name='hG1s', val=G1s, Fourier=True)
        G2h = Matrix(name='hG2h', val=G2h, Fourier=True)
        G2s = Matrix(name='hG2s', val=G2s, Fourier=True)

    if NyqNul:
        G0 = G0.enlarge(N)
        G1h = G1h.enlarge(N)
        G1s = G1s.enlarge(N)
        G2h = G2h.enlarge(N)
        G2s = G2s.enlarge(N)

    if fft_form_r:
        for tensor in [G0, G1h, G1s, G2h, G2s]:
            tensor.set_fft_form(fft_form='r')
            tensor.val=1./np.prod(tensor.N)*tensor.val

    return G0, G1h, G1s, G2h, G2s


def elasticity_combined(N, Y, NyqNul=True, tensor=True, fft_form=fft_form_default,
                        dtype=float, storage='full', backend='numpy'):
    """
    Combined elasticity projections hG1=hG1h+hG1s and hG2=hG2h+hG2s.

    The optimized sym21 path avoids materializing the four full DxDxN
    projection blocks. For 3D elasticity, hG1 = S - G1h and
    hG2 = IS0 - hG1.
    """
    storage = str(storage).strip().lower()
    N = np.array(N, dtype=int)
    d = N.size
    if storage in {'direct', 'formula', 'elastic_direct'}:
        if d == 3 and tensor and fft_form in [0, 'r'] and get_fft_backend() == 'cupy':
            dtype = np.dtype(dtype)
            if fft_form == 'r':
                grid_shape = tuple(int(v) for v in N[:-1].tolist()) + (int(N[-1] // 2 + 1),)
            else:
                grid_shape = tuple(int(v) for v in N.tolist())
            hG1 = Tensor(
                name='hG1_direct',
                val=np.zeros((1,) + grid_shape, dtype=dtype),
                order=1,
                N=N,
                Y=Y,
                Fourier=True,
                multype='elasticity_hg1_direct',
                fft_form=fft_form,
            )
            hG2 = Tensor(
                name='hG2_direct',
                val=np.zeros((1,) + grid_shape, dtype=dtype),
                order=1,
                N=N,
                Y=Y,
                Fourier=True,
                multype='elasticity_hg2_direct',
                fft_form=fft_form,
            )
            return hG1, hG2
        storage = 'sym21'

    if storage not in {'sym21', 'symmetric21', 'packed21'} or d != 3 or not tensor:
        _, hG1h, hG1s, hG2h, hG2s = elasticity(
            N, Y, NyqNul=NyqNul, tensor=tensor, fft_form=fft_form, dtype=dtype
        )
        return hG1h + hG1s, hG2h + hG2s

    if fft_form in ['r']:
        fft_form_r = True
        fft_form = 0
    else:
        fft_form_r = False

    dtype = np.dtype(dtype)
    xp, backend_name = _array_backend(backend)
    if fft_form_r and backend_name == 'cupy':
        xp, backend_name = np, 'numpy'
    D = int(d*(d+1)/2)
    if D != 6:
        raise NotImplementedError("elasticity_combined sym21 esta implementado para 3D.")

    Nred = get_Nodd(N) if NyqNul else N
    grid_shape = tuple(int(v) for v in Nred.tolist())
    xi_np = Grid.get_xil(Nred, Y, fft_form=fft_form)
    xi = [xp.asarray(item, dtype=dtype) for item in xi_np]
    xi2 = [xi[ii]**2 for ii in range(d)]

    num = xp.zeros((d, d) + grid_shape, dtype=dtype)
    norm2_xi = xp.zeros(grid_shape, dtype=dtype)
    for mm in np.arange(d):
        Nshape = np.ones(d, dtype=int)
        Nshape[mm] = Nred[mm]
        Nrep = np.copy(Nred)
        Nrep[mm] = 1
        num[mm][mm] = xp.tile(xp.reshape(xi2[mm], tuple(Nshape.tolist())), tuple(Nrep.tolist()))
        norm2_xi += num[mm][mm]

    for m in np.arange(d):
        for n in np.arange(m+1, d):
            NshapeM = np.ones(d, dtype=int)
            NshapeM[m] = Nred[m]
            NrepM = np.copy(Nred)
            NrepM[m] = 1
            NshapeN = np.ones(d, dtype=int)
            NshapeN[n] = Nred[n]
            NrepN = np.copy(Nred)
            NrepN[n] = 1
            num[m][n] = xp.tile(xp.reshape(xi[m], tuple(NshapeM.tolist())), tuple(NrepM.tolist())) \
                * xp.tile(xp.reshape(xi[n], tuple(NshapeN.tolist())), tuple(NrepN.tolist()))

    norm4_xi = norm2_xi**2
    ind_center = mean_index(Nred, fft_form=fft_form)
    norm2_xi[ind_center] = 1
    norm4_xi[ind_center] = 1

    zeros = xp.zeros(grid_shape, dtype=dtype)
    sqrt2 = dtype.type(2**0.5)
    axes = np.arange(d)

    def num_comp(i, j):
        if i <= j:
            return num[i][j]
        return num[j][i]

    def shear_axes(shear_axis):
        return np.setdiff1d(axes, [shear_axis])

    def s_comp(ii, jj):
        if ii < 3 and jj < 3:
            if ii == jj:
                return 2*num[ii][ii]/norm2_xi
            return zeros
        if ii >= 3 and jj >= 3:
            a = ii - 3
            b = jj - 3
            if a == b:
                val = xp.asarray(1 - num[a][a]/norm2_xi, dtype=dtype)
                val[ind_center] = 0
                return val
            return num_comp(a, b)/norm2_xi

        normal = ii if ii < 3 else jj
        shear_axis = jj - 3 if jj >= 3 else ii - 3
        if normal == shear_axis:
            return zeros
        ind = shear_axes(shear_axis)
        return sqrt2*num_comp(ind[0], ind[1])/norm2_xi

    def g1h_comp(ii, jj):
        if ii < 3 and jj < 3:
            return num[ii][ii]*num[jj][jj]/norm4_xi
        if ii >= 3 and jj >= 3:
            ind_i = shear_axes(ii - 3)
            ind_j = shear_axes(jj - 3)
            return 2*num_comp(ind_i[0], ind_i[1]) \
                * num_comp(ind_j[0], ind_j[1]) / norm4_xi

        normal = ii if ii < 3 else jj
        shear_axis = jj - 3 if jj >= 3 else ii - 3
        ind = shear_axes(shear_axis)
        return sqrt2*num[normal][normal]*num_comp(ind[0], ind[1])/norm4_xi

    G1 = xp.empty((len(SYM21_PAIRS),) + grid_shape, dtype=dtype)
    G2 = xp.empty_like(G1)
    is0_diag = xp.ones(grid_shape, dtype=dtype)
    is0_diag[ind_center] = 0

    for kk, (ii, jj) in enumerate(SYM21_PAIRS):
        g1 = xp.asarray(s_comp(ii, jj) - g1h_comp(ii, jj), dtype=dtype)
        G1[kk] = g1
        if ii == jj:
            G2[kk] = is0_diag - g1
        else:
            G2[kk] = -g1

    if NyqNul:
        G1 = _enlarge_fourier_array(G1, Nred, N, order=1, fft_form=fft_form, xp=xp)
        G2 = _enlarge_fourier_array(G2, Nred, N, order=1, fft_form=fft_form, xp=xp)

    hG1 = Tensor(name='hG1_sym21', val=G1, order=1, N=N,
                 Fourier=True, multype='sym21', fft_form=fft_form)
    hG2 = Tensor(name='hG2_sym21', val=G2, order=1, N=N,
                 Fourier=True, multype='sym21', fft_form=fft_form)

    if fft_form_r:
        for tensor_obj in [hG1, hG2]:
            tensor_obj.set_fft_form(fft_form='r')
            tensor_obj.val = 1./np.prod(tensor_obj.N)*tensor_obj.val

    return hG1, hG2
