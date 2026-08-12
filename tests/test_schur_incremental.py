from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constitutive_transfer.schur_estimator import (
    IncrementalTwoKernelCompiler,
    TwoKernelEstimator,
    compile_two_kernel_contractions,
)


def _stress_factory(stiffnesses: np.ndarray):
    def apply(q: int, strains: np.ndarray) -> np.ndarray:
        return np.einsum("ab,lbn->lan", stiffnesses[q], strains, optimize=True)

    return apply


def _full_estimator(
    path: Path,
    *,
    shape: tuple[int, int, int],
    basis: np.ndarray,
    stiffnesses: np.ndarray,
) -> TwoKernelEstimator:
    compile_two_kernel_contractions(
        output_path=path,
        shape=shape,
        basis_fields=basis,
        coefficient_names=("q0", "q1"),
        affine_stress_batch=_stress_factory(stiffnesses),
        atom_batch_size=3,
        feature_block=37,
    )
    return TwoKernelEstimator.load(path)


def test_incremental_contractions_match_full_compilation(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260811)
    shape = (4, 3, 5)
    nvox = int(np.prod(shape))
    raw = rng.normal(size=(2, 6, 6))
    stiffnesses = np.einsum("qki,qkj->qij", raw, raw) + np.eye(6)[None, ...]
    basis = rng.normal(size=(5, 6, nvox))
    basis -= basis.mean(axis=2, keepdims=True)

    compiler = IncrementalTwoKernelCompiler(
        work_dir=tmp_path / "incremental",
        shape=shape,
        coefficient_names=("q0", "q1"),
        affine_stress_batch=_stress_factory(stiffnesses),
        max_rank=5,
        atom_batch_size=2,
        feature_block=37,
    )
    compiler.append(basis[:2])
    partial = compiler.estimator()
    partial_full = _full_estimator(
        tmp_path / "partial_full.npz",
        shape=shape,
        basis=basis[:2],
        stiffnesses=stiffnesses,
    )
    np.testing.assert_allclose(partial.BB, partial_full.BB, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(partial.BK, partial_full.BK, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(partial.KK, partial_full.KK, rtol=1e-12, atol=1e-12)
    compiler.close()

    resumed = IncrementalTwoKernelCompiler(
        work_dir=tmp_path / "incremental",
        shape=shape,
        coefficient_names=("q0", "q1"),
        affine_stress_batch=_stress_factory(stiffnesses),
        max_rank=5,
        atom_batch_size=2,
        feature_block=37,
        resume=True,
    )
    assert resumed.rank == 2
    resumed.append(basis[2:])
    final = resumed.estimator()
    final_full = _full_estimator(
        tmp_path / "final_full.npz",
        shape=shape,
        basis=basis,
        stiffnesses=stiffnesses,
    )
    np.testing.assert_allclose(final.BB, final_full.BB, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(final.BK, final_full.BK, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(final.KK, final_full.KK, rtol=1e-12, atol=1e-12)


def test_full_size_sketch_is_exact_and_small_sketch_resumes(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260818)
    shape = (3, 3, 2)
    nvox = int(np.prod(shape))
    raw = rng.normal(size=(2, 6, 6))
    stiffnesses = np.einsum("qki,qkj->qij", raw, raw) + np.eye(6)[None]
    basis = rng.normal(size=(3, 6, nvox))
    basis -= basis.mean(axis=2, keepdims=True)

    exact = IncrementalTwoKernelCompiler(
        work_dir=tmp_path / "exact",
        shape=shape,
        coefficient_names=("q0", "q1"),
        affine_stress_batch=_stress_factory(stiffnesses),
        max_rank=3,
        feature_block=13,
    )
    exact.append(basis)
    exact_estimator = exact.estimator()
    exact.close()
    full_sketch = IncrementalTwoKernelCompiler(
        work_dir=tmp_path / "full_sketch",
        shape=shape,
        coefficient_names=("q0", "q1"),
        affine_stress_batch=_stress_factory(stiffnesses),
        max_rank=3,
        feature_block=13,
        sketch_size=3 * nvox,
    )
    full_sketch.append(basis)
    np.testing.assert_allclose(full_sketch.BB, exact_estimator.BB, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(full_sketch.BK[..., :3], exact_estimator.BK, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        full_sketch.KK[..., :3, :3], exact_estimator.KK, rtol=1e-12, atol=1e-12
    )
    full_sketch.close()

    sketched = IncrementalTwoKernelCompiler(
        work_dir=tmp_path / "small_sketch",
        shape=shape,
        coefficient_names=("q0", "q1"),
        affine_stress_batch=_stress_factory(stiffnesses),
        max_rank=3,
        feature_block=13,
        sketch_size=7,
        sketch_seed=91,
    )
    sketched.append(basis[:1])
    sketched.close()
    resumed = IncrementalTwoKernelCompiler(
        work_dir=tmp_path / "small_sketch",
        shape=shape,
        coefficient_names=("q0", "q1"),
        affine_stress_batch=_stress_factory(stiffnesses),
        max_rank=3,
        feature_block=13,
        sketch_size=7,
        sketch_seed=91,
        resume=True,
    )
    assert resumed.rank == 1
    assert resumed.features[0][0].shape == (9, 7)
    resumed.close()
