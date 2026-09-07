"""GPU CG checks against independent arithmetic and analytic SPD solutions."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'FFT/ffthompy_core/ffthompy'))
cp = pytest.importorskip('cupy')
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('CUDA device unavailable', allow_module_level=True)
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip('CUDA runtime unavailable', allow_module_level=True)

from ffthompy.general.cupy_cg_reduction import CupyCGReduction
from ffthompy.general.solver import CG
from ffthompy.tensors import Tensor


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_reductions_match_float64_reference_and_preserve_previous_scalar(dtype):
    rng = np.random.default_rng(761)
    shape = (6, 7, 9, 11)  # Non-divisible CUDA block tail.
    arrays = [rng.normal(size=shape).astype(dtype) for _ in range(4)]
    x, p, r, ap = [SimpleNamespace(val=cp.asarray(a), N=shape[1:], Fourier=False) for a in arrays]
    reduction = CupyCGReduction(x)
    norm = np.prod(shape[1:])
    rr_old = reduction.dot(r, r)
    old_value = rr_old.item()
    expected_dot = np.sum(arrays[1].astype(np.float64)*arrays[3].astype(np.float64))/norm
    absolute_bound = 20*np.finfo(dtype).eps*np.sum(np.abs(arrays[1].astype(np.float64)*arrays[3]))/norm
    np.testing.assert_allclose(reduction.dot(p, ap).item(), expected_dot, rtol=0, atol=absolute_bound)
    alpha = dtype(0.125)
    rr_new = reduction.update_xr_rr(x, cp.asarray(alpha), p, r, ap)
    expected_x = arrays[0].astype(np.float64) + float(alpha)*arrays[1].astype(np.float64)
    expected_r = arrays[2].astype(np.float64) - float(alpha)*arrays[3].astype(np.float64)
    np.testing.assert_allclose(cp.asnumpy(x.val), expected_x, rtol=3*np.finfo(dtype).eps, atol=3*np.finfo(dtype).eps)
    np.testing.assert_allclose(cp.asnumpy(r.val), expected_r, rtol=3*np.finfo(dtype).eps, atol=3*np.finfo(dtype).eps)
    # Recompute from the actual rounded field, independently on the CPU.
    actual_r = cp.asnumpy(r.val).astype(np.float64)
    np.testing.assert_allclose(rr_new.item(), np.sum(actual_r**2)/norm, rtol=3*np.finfo(dtype).eps)
    assert rr_old.item() == old_value
    assert len({reduction.dot(p, ap).item() for _ in range(5)}) == 1


@pytest.mark.parametrize('dtype,rtol', [(np.float32, 1e-5), (np.float64, 1e-11)])
@pytest.mark.parametrize('scale', [1.0, 1e6])
def test_cg_solution_and_true_residual(dtype, rtol, scale):
    shape = (3, 7, 9)
    rng = np.random.default_rng(51)
    rhs = (scale*rng.normal(size=shape)).astype(dtype)
    diagonal = np.geomspace(1, 50, rhs.size).reshape(shape).astype(dtype)
    b = Tensor(name='rhs', N=shape[1:], shape=(3,), Fourier=False,
               fft_form='r', order=1, val=cp.asarray(rhs))
    x0 = b.copy(val=cp.zeros_like(b.val))
    gpu_diagonal = cp.asarray(diagonal)
    solution, info = CG(lambda x: x.copy(val=gpu_diagonal*x.val), b, x0, par={
        'rtol': rtol, 'maxiter': 500, 'cupy_lazy_scalars': True,
        'cupy_fused_cg_updates': True, 'cupy_stable_reductions': True,
        'check_true_residual': True,
    })
    actual = cp.asnumpy(solution.val).astype(np.float64)
    true_relative_residual = np.linalg.norm(rhs.astype(np.float64)-diagonal.astype(np.float64)*actual)/np.linalg.norm(rhs.astype(np.float64))
    assert info['converged']
    assert info['norm_res_rel'] <= rtol
    np.testing.assert_allclose(info['true_norm_res_rel'], true_relative_residual,
                               rtol=0, atol=3*np.finfo(dtype).eps)
    assert true_relative_residual <= 1.1*rtol
    assert np.linalg.norm(actual-rhs.astype(np.float64)/diagonal)/np.linalg.norm(actual) < 3*rtol


def test_unsupported_metrics_keep_the_original_inner_product():
    physical = SimpleNamespace(val=cp.zeros((3, 4, 4)), N=(4, 4), Fourier=False)
    assert CupyCGReduction.supports(physical, {})
    assert not CupyCGReduction.supports(physical, {'scal': lambda x, y: 0})
    physical.Fourier = True
    assert not CupyCGReduction.supports(physical, {})


@pytest.mark.parametrize('fiber_count', [0, 17])
def test_compiled_geometry_matches_fresh_material_construction(fiber_count):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'FFT'))
    from pipeline.fft_solver import _build_cfield_gpu, compile_cfield_geometry
    phase = np.zeros((3, 4, 5), dtype=np.uint8)
    phase.ravel()[:fiber_count] = 1
    ori = np.zeros((*phase.shape, 3), dtype=np.float32)
    ori[..., 0] = 1
    ori[0, :2] = np.array([1., 1., 1.])/np.sqrt(3)
    phase.setflags(write=False)
    ori.setflags(write=False)
    layout = compile_cfield_geometry(phase, ori)
    for scale in (1.0, 7.0):
        matrix = np.diag(np.arange(1., 7.)).astype(np.float32)*scale
        fiber = matrix + np.ones((6, 6), dtype=np.float32)
        baseline, count, _ = _build_cfield_gpu(phase, ori, matrix, fiber, storage='sym21', indexed=True)
        compiled, compiled_count, _ = _build_cfield_gpu(phase, ori, matrix, fiber, storage='sym21', indexed=True, geometry_layout=layout)
        assert count == compiled_count
        for original, cached in zip(baseline, compiled):
            np.testing.assert_array_equal(cp.asnumpy(original), cp.asnumpy(cached))
    with pytest.raises(ValueError, match='does not match'):
        _build_cfield_gpu(phase.copy(), ori, matrix, fiber, storage='sym21', indexed=True, geometry_layout=layout)
