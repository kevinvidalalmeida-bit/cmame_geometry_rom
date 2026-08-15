from __future__ import annotations

import numpy as np

import primal_dual_cre as cre
import rom_reduced_operator as reduced
from pipeline.fft_solver import TI_stiffness_voigt, voigt_to_mandel


MATERIAL = {
    "Em": 2.7,
    "nu_m": 0.39,
    "Ef_L": 210.0,
    "Ef_T": 14.0,
    "G_LT": 11.0,
    "nu_LT": 0.23,
    "nu_TT": 0.38,
}


def test_seven_term_compliance_decomposition_is_exact() -> None:
    coefficients = reduced._dual_material_coefficients(MATERIAL)
    matrix_reconstructed = sum(
        coefficient * basis
        for coefficient, basis in zip(
            coefficients[:2], reduced._isotropic_compliance_bases(), strict=True
        )
    )
    gm = MATERIAL["Em"] / (2.0 * (1.0 + MATERIAL["nu_m"]))
    matrix_exact = np.linalg.inv(
        voigt_to_mandel(
            TI_stiffness_voigt(
                MATERIAL["Em"],
                MATERIAL["Em"],
                MATERIAL["nu_m"],
                MATERIAL["nu_m"],
                gm,
            )
        )
    )

    fiber_reconstructed = sum(
        coefficient * basis
        for coefficient, basis in zip(
            coefficients[2:],
            reduced._fiber_local_compliance_bases_axis0(),
            strict=True,
        )
    )
    fiber_exact = np.linalg.inv(
        voigt_to_mandel(
            TI_stiffness_voigt(
                MATERIAL["Ef_L"],
                MATERIAL["Ef_T"],
                MATERIAL["nu_LT"],
                MATERIAL["nu_TT"],
                MATERIAL["G_LT"],
            )
        )
    )
    np.testing.assert_allclose(matrix_reconstructed, matrix_exact, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(fiber_reconstructed, fiber_exact, rtol=2e-15, atol=2e-15)


def test_dual_affine_factory_matches_direct_phase_compliance() -> None:
    phase = np.asarray([0, 1], dtype=np.uint8)
    orientation = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    action = reduced.affine_compliance_batch_factory(phase, orientation)
    coefficients = reduced._dual_material_coefficients(MATERIAL)
    stresses = np.arange(12, dtype=np.float64).reshape(1, 6, 2) / 7.0
    recovered = sum(
        coefficient * action(q, stresses)
        for q, coefficient in enumerate(coefficients)
    )

    matrix_compliance = sum(
        coefficient * basis
        for coefficient, basis in zip(
            coefficients[:2], reduced._isotropic_compliance_bases(), strict=True
        )
    )
    fiber_compliance = sum(
        coefficient * basis
        for coefficient, basis in zip(
            coefficients[2:],
            reduced._fiber_local_compliance_bases_axis0(),
            strict=True,
        )
    )
    expected = np.empty_like(stresses)
    expected[:, :, 0] = np.einsum("ab,lb->la", matrix_compliance, stresses[:, :, 0])
    expected[:, :, 1] = np.einsum("ab,lb->la", fiber_compliance, stresses[:, :, 1])
    np.testing.assert_allclose(recovered, expected, rtol=2e-15, atol=2e-15)


def test_static_projector_is_idempotent_and_zero_mean_on_even_and_odd_grids() -> None:
    random = np.random.default_rng(20260815)
    for size in (5, 6):
        projector = cre.ElasticityProjectors((size, size, size))
        values = random.normal(size=(2, 6, size, size, size))
        projected, diagnostics = projector.apply(values, kind="dual")
        repeated, _ = projector.apply(projected, kind="dual")
        np.testing.assert_allclose(repeated, projected, rtol=2e-13, atol=2e-13)
        assert diagnostics["projected_mean_abs_max"] < 1e-13


def test_homogeneous_recovery_has_zero_dual_fluctuation() -> None:
    shape = (5, 5, 5)
    phase = np.zeros(shape, dtype=np.uint8)
    orientation = np.zeros(shape + (3,), dtype=np.float64)
    stiffness = reduced.affine_stress_batch_factory(phase, orientation)
    projector = cre.ElasticityProjectors(shape)
    primal = np.zeros((6, 6) + shape, dtype=np.float64)
    _, dual, diagnostics = cre.stress_and_dual_snapshots(
        primal_fluctuations=primal,
        material=MATERIAL,
        stiffness_action=stiffness,
        projectors=projector,
    )
    assert np.linalg.norm(dual) < 1e-12
    assert diagnostics["dual_projected_mean_abs_max"] < 1e-13
