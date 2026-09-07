"""Reference metric and Green symbols against independent dense arithmetic."""

from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'FFT/ffthompy_core/ffthompy'))
cp=pytest.importorskip('cupy')
try:
    if cp.cuda.runtime.getDeviceCount()<1:
        pytest.skip('CUDA unavailable',allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip('CUDA unavailable',allow_module_level=True)

from ffthompy.general.cupy_reference_metric import ReferenceMetricReduction
from ffthompy.tensors.objects import elasticity_direct_array


@pytest.mark.parametrize('eta',[1.2,2.,4.])
def test_weighted_dot_and_fused_residual_match_cpu(eta):
    rng=np.random.default_rng(181)
    arrays=[rng.normal(size=(6,7,9,11)).astype(np.float32) for _ in range(4)]
    tensors=[SimpleNamespace(val=cp.asarray(a),N=(7,9,11),Fourier=False) for a in arrays]
    x,p,r,ap=tensors
    reduction=ReferenceMetricReduction(x,eta)
    def scalar(a,b):
        a,b=a.astype(float),b.astype(float)
        return (np.sum(a*b)+(eta-1)*np.sum(a[:3].sum(axis=0)*b[:3].sum(axis=0)))/(7*9*11)
    np.testing.assert_allclose(reduction.dot(p,ap).item(),scalar(arrays[1],arrays[3]),rtol=1e-6,atol=1e-7)
    old=reduction.dot(r,r); old_value=old.item()
    new=reduction.update_xr_rr(x,cp.asarray(.1,dtype=cp.float32),p,r,ap)
    actual=cp.asnumpy(r.val)
    np.testing.assert_allclose(actual,arrays[2]-.1*arrays[3],rtol=2e-6,atol=2e-7)
    np.testing.assert_allclose(new.item(),scalar(actual,actual),rtol=2e-7)
    assert old.item()==old_value


@pytest.mark.parametrize('eta',[1.,1.5,3.,5.])
def test_reference_green_matches_dense_acoustic_inverse(eta):
    grid=(7,9,11); index=(2,8,3)
    n=np.array([2.,-1.,3.]);n/=np.linalg.norm(n)
    sym=np.zeros((6,3)); sym[:3]=np.diag(n)
    for row,(i,j) in enumerate(((1,2),(0,2),(0,1)),start=3):
        sym[row,i]=n[j]/np.sqrt(2);sym[row,j]=n[i]/np.sqrt(2)
    c0=np.eye(6);c0[:3,:3]+=eta-1
    green=sym @ np.linalg.solve(sym.T @ c0 @ sym,sym.T)
    sigma=np.random.default_rng(119).normal(size=6)+1j*np.arange(6)
    field=cp.zeros((6,7,9,6),dtype=cp.complex64);field[(slice(None),*index)]=cp.asarray(sigma)
    projection=SimpleNamespace(N=grid,Y=np.ones(3),fft_form='r',
        multype='elasticity_hg1_direct',reference_eta=eta)
    result=cp.asnumpy(elasticity_direct_array(projection,field))[(slice(None),*index)]
    np.testing.assert_allclose(result,green @ sigma,rtol=2e-6,atol=8e-7)
    np.testing.assert_allclose(green @ c0 @ sym,sym,atol=1e-14)


@pytest.mark.parametrize('correction_succeeds', [True, False])
def test_original_residual_audit_corrects_or_rejects(monkeypatch, correction_succeeds):
    from ffthompy.general import reference_elasticity as reference
    monkeypatch.setattr(reference, 'cp', np)

    class Field:
        N = (1,)
        def __init__(self, value):
            self.val = np.asarray(value, dtype=float)
        def __neg__(self):
            return Field(-self.val)
        def __sub__(self, other):
            return Field(self.val-other.val)

    initial = Field([0., 0.])
    imposed = Field([-1., -2.])
    info = dict(kit=3, norm_res_rel=1e-8, hit_maxiter=True)
    parameters = dict(kind='CG', rtol=1e-5, atol=0., cupy_reference_eta=2., keep_solution_on_device=True)
    calls = []
    def correction(**kwargs):
        calls.append(kwargs)
        assert 'cupy_reference_eta' not in kwargs['par']
        assert kwargs['par']['rtol'] < parameters['rtol']
        return (Field([1., 2.]) if correction_succeeds else initial), dict(kit=2)

    if not correction_succeeds:
        with pytest.raises(RuntimeError, match='original-equation residual audit'):
            reference.audit_reference_solution(initial, info, imposed, lambda x: x, parameters, correction)
    else:
        solution, result = reference.audit_reference_solution(initial, info, imposed, lambda x: x, parameters, correction)
        np.testing.assert_array_equal(solution.val, [1., 2.])
        assert result['kit'] == 5 and result['correction_iterations'] == 2
        assert result['true_residual_converged'] and result['norm_res_rel'] == 0.
        assert result['reference_hit_maxiter'] and not result['hit_maxiter']
    assert len(calls) == 1
