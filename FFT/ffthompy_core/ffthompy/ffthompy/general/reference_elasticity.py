"""CG in a reference energy, with an audit against the original equations.

C0:X = X+(eta-1)tr(X)I; M=G C0 G has eigenvalues 1 and eta on range(G).
M^-1 G C is symmetric in the C0 metric.
"""

from copy import copy
import time
import numpy as np
from ffthompy.tensors.fft import cp


def reference_operator(GN, A, eta):
    from ffthompy.tensors.operators import Operator
    fn,hg,fin = GN.mat_rev[0]
    if str(hg.multype) not in ('elasticity_hg1_direct','elasticity_g1_direct'):
        raise ValueError('Reference-energy CG requires the direct primal GPU projection.')
    hg_reference = copy(hg)
    hg_reference.reference_eta = eta
    green = Operator(name='reference_green', mat_rev=[[fn,hg_reference,fin]])
    return Operator(name='reference_A', mat_rev=[[A,green]])


def reference_parameters(parameters):
    eta = float(parameters['cupy_reference_eta'])
    factor = np.sqrt(max(eta,1)/min(eta,1))
    return dict(parameters, rtol=parameters['rtol']/factor,
                atol=parameters.get('atol',0.)/np.sqrt(max(eta,1)),
                keep_solution_on_device=True)  # Audit before the caller's host download.


def audit_reference_solution(X, info, EN, Afun, parameters, linear_solver):
    started = time.perf_counter()
    rhs = Afun(-EN)
    volume = int(np.prod(X.N))
    def norm(tensor):
        return float(cp.sqrt(cp.sum(tensor.val**2,dtype=cp.float64)/volume).item())
    rhs_norm = norm(rhs)
    threshold = max(parameters.get('atol',0.), parameters['rtol']*rhs_norm)
    residual_norm = norm(rhs-Afun(X))
    reference_iterations = int(info['kit'])
    correction_iterations = 0
    reference_relative = info['norm_res_rel']
    reference_log = info.get('norm_res_log', [])
    if residual_norm > threshold or not np.isfinite(residual_norm):
        correction = dict(parameters)
        correction.pop('cupy_reference_eta')
        correction['keep_solution_on_device'] = True
        correction['rtol'] *= .5
        correction['atol'] = correction.get('atol',0.)*.5
        X, corrected = linear_solver(solver=parameters['kind'], Afun=Afun, B=rhs,
                                     x0=X, par=correction, callback=None)
        correction_iterations = int(corrected['kit'])
        residual_norm = norm(rhs-Afun(X))
    if not np.isfinite(residual_norm) or residual_norm > threshold:
        raise RuntimeError('Reference-energy CG failed its original-equation residual audit.')
    info.update(kit=reference_iterations+correction_iterations,
        reference_hit_maxiter=bool(info.get('hit_maxiter', False)), hit_maxiter=False,
        reference_iterations=reference_iterations, correction_iterations=correction_iterations,
        reference_eta=float(parameters['cupy_reference_eta']),
        reference_norm_res_rel=reference_relative, reference_norm_res_log=reference_log,
        norm_res=residual_norm, norm_res_rel=residual_norm/max(rhs_norm,np.finfo(float).tiny),
        norm_res_log=[residual_norm], rhs_norm=rhs_norm, threshold=threshold,
        converged=True, true_norm_res_rel=residual_norm/max(rhs_norm,np.finfo(float).tiny),
        true_residual_converged=True, reference_audit_wall_s=time.perf_counter()-started)
    if not parameters.get('keep_solution_on_device', False):
        X = X.copy(val=cp.asnumpy(X.val))
    return X,info
