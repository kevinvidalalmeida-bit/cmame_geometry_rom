import numpy as np
from ffthompy.general.base import Timer
from ffthompy.tensors.fft import get_array_module, to_backend_array, to_host_array
import itertools


def postprocess(pb, A, mat, solutions, results, primaldual):
    """
    The function post-process the results.
    """
    tim = Timer(name='postprocessing')
    print('\npostprocessing')
    matrices = {}
    for pp in pb.postprocess:
        if pp['kind'] in ['GaNi', 'gani']:
            order_name = ''
            Nname = ''
            if A.name != 'A_GaNi':
                A = mat.get_A_GaNi(pb.solve['N'], primaldual)

        elif pp['kind'] in ['Ga', 'ga']:
            if 'order' in pp:
                Nbarpp = tuple(2*np.array(pb.solve['N']) - 1)
                if pp['order'] is None:
                    Nname = ''
                    order_name = ''
                    A = mat.get_A_Ga(Nbar=Nbarpp, primaldual=primaldual, order=pp['order'])
                else:
                    order_name = '_o' + str(pp['order'])
                    Nname = '_P%d' % np.mean(pp['P'])
                    A = mat.get_A_Ga(Nbar=Nbarpp, primaldual=primaldual,
                                     order=pp['order'], P=pp['P'])
            else:
                order_name = ''
                Nname = ''
        else:
            ValueError()

        name = 'AH_%s%s%s_%s' % (pp['kind'], order_name, Nname, primaldual)
        print(('calculating: ' + name))

        AH = assembly_matrix(
            A,
            solutions,
            batch_size=int(pb.solve.get('postprocess_batch_size', 1)),
            method=str(pb.solve.get('postprocess_assembly', 'scalar')),
        )

        if primaldual == 'primal':
            matrices[name] = AH
        else:
            matrices[name] = np.linalg.inv(AH)
    tim.measure()

    output = {
        'res_' + primaldual: results,
        'mat_' + primaldual: matrices,
    }
    if pb.solve.get('store_solution_fields', True):
        output['sol_' + primaldual] = solutions
    pb.output.update(output)

def assembly_matrix(Afun, solutions, batch_size=1, method='scalar'):
    """
    The function assembles the homogenized matrix from minimizers (corrector
    functions).
    """
    dim = len(solutions)
    if not np.allclose(Afun.N, solutions[0].N):
        Nbar = Afun.N
        sol = []
        for ii in np.arange(dim):
            sol.append(solutions[ii].project(Nbar))
    else:
        sol = solutions

    method = str(method).strip().lower()
    if method not in {'scalar', 'einsum', 'gemm'}:
        method = 'scalar'

    batch_size = max(1, min(int(batch_size), dim))
    if method == 'gemm':
        return assembly_matrix_batched_gemm(Afun, sol, batch_size=batch_size)
    if batch_size > 1 or method == 'einsum':
        return assembly_matrix_batched(Afun, sol, batch_size=batch_size)

    A_sol = [Afun(sol[ii]) for ii in range(dim)]

    AH = np.zeros([dim, dim])
    for ii, jj in itertools.product(list(range(dim)), repeat=2):
        AH[ii, jj] = A_sol[ii] * sol[jj]
    return AH


def assembly_matrix_batched(Afun, sol, batch_size=6):
    dim = len(sol)
    xp = get_array_module(*[item.val for item in sol])
    sol_all_val = xp.stack([item.val for item in sol], axis=1)
    sol_all = sol[0].copy(name='sol_batch', val=sol_all_val, order=2)

    AH = np.zeros([dim, dim])
    prodN = float(np.prod(Afun.N))
    for start in range(0, dim, batch_size):
        stop = min(start + batch_size, dim)
        sol_chunk = sol_all.copy(
            name='sol_batch_{0}_{1}'.format(start, stop),
            val=sol_all.val[:, start:stop, ...],
            order=2,
        )
        A_chunk = Afun(sol_chunk)
        xp_block = get_array_module(A_chunk.val, sol_all.val)
        A_flat = A_chunk.val.reshape((dim, stop - start, -1))
        sol_flat = sol_all.val.reshape((dim, dim, -1))
        if xp_block is not xp:
            sol_flat = to_backend_array(sol_flat, prefer_backend='cupy')
        block = xp_block.einsum('dis,djs->ij', A_flat, sol_flat).real / prodN
        AH[start:stop, :] = to_host_array(block)
    return AH


def assembly_matrix_batched_gemm(Afun, sol, batch_size=6):
    dim = len(sol)
    xp = get_array_module(*[item.val for item in sol])
    sol_all_val = xp.stack([item.val for item in sol], axis=1)
    sol_all = sol[0].copy(name='sol_batch', val=sol_all_val, order=2)

    spatial_axes = tuple(range(2, sol_all.val.ndim))
    sol_mat = xp.transpose(sol_all.val, (1, 0) + spatial_axes).reshape((dim, -1))

    AH = np.zeros([dim, dim])
    prodN = float(np.prod(Afun.N))
    for start in range(0, dim, max(1, min(int(batch_size), dim))):
        stop = min(start + batch_size, dim)
        sol_chunk = sol_all.copy(
            name='sol_batch_{0}_{1}'.format(start, stop),
            val=sol_all.val[:, start:stop, ...],
            order=2,
        )
        A_chunk = Afun(sol_chunk)
        xp_block = get_array_module(A_chunk.val, sol_all.val)
        A_mat = xp_block.transpose(A_chunk.val, (1, 0) + spatial_axes).reshape((stop - start, -1))
        sol_rhs = sol_mat
        if xp_block is not xp:
            sol_rhs = to_backend_array(sol_mat, prefer_backend='cupy')
        block = (A_mat @ sol_rhs.T).real / prodN
        AH[start:stop, :] = to_host_array(block)
    return AH


def add_macro2minimizer(X, E, check_mean=True, inplace=False):
    """
    The function takes the minimizers (corrector function with zero-mean
    property or equaling to macroscopic value) and returns a corrector function
    with mean that equals to macroscopic value E.
    """
    if not check_mean:
        if inplace:
            X.add_mean(E)
            return X
        EN = X.zeros_like(name='EN')
        EN.set_mean(E)
        return X + EN

    dtype = getattr(getattr(X, "val", None), "dtype", np.dtype(float))
    close_tol = 1e-5 if np.dtype(dtype) == np.dtype(np.float32) else 1e-8

    if np.allclose(X.mean(), E, rtol=close_tol, atol=close_tol):
        return X
    elif np.allclose(X.mean(), np.zeros_like(E), rtol=close_tol, atol=close_tol):
        EN = X.zeros_like(name='EN')
        EN.set_mean(E)
        if inplace:
            X += EN
            return X
        return X + EN
    else:
        raise ValueError("Field is neither zero-mean nor E-mean.")
