"""High-precision physical verification of the FFT homogenization contract."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "FFT", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pipeline.fft_solver import (
    TI_stiffness_voigt,
    _save_solution_fields_if_requested,
    rotate_C_mandel,
    rotation_matrix_from_vector,
    solve_homogenization,
    voigt_to_mandel,
)


MANDEL_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
MANDEL_FACTORS = np.array((1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)))


def test_solution_fields_can_return_in_memory_without_disk(tmp_path: Path):
    solutions = []
    expected = []
    for load_id in range(6):
        fluctuation = np.full((6, 2, 2, 2), 0.01 * (load_id + 1))
        total = fluctuation.copy()
        total[load_id] += 1.0
        solutions.append(SimpleNamespace(val=total))
        expected.append(fluctuation.astype(np.float32))

    params = {
        "solution_field_return_in_memory": True,
        "solution_field_dtype": "float32",
    }
    metadata = _save_solution_fields_if_requested(
        prob=SimpleNamespace(output={"sol_primal": solutions}),
        Ngrid=np.array((2, 2, 2)),
        p=params,
    )

    assert metadata["solution_field_format"] == "memory"
    assert not list(tmp_path.iterdir())
    actual = params["_solution_fields_result"]
    assert len(actual) == 6
    for actual_field, expected_field in zip(actual, expected, strict=True):
        assert actual_field.dtype == np.float32
        np.testing.assert_allclose(actual_field, expected_field, rtol=2.0e-6)


def test_solution_fields_can_stream_to_consumer_without_retention(tmp_path: Path):
    solutions = []
    consumed = {}
    for load_id in range(6):
        total = np.zeros((6, 2, 2, 2), dtype=np.float64)
        total[load_id] = 1.0 + 0.02 * (load_id + 1)
        solutions.append(SimpleNamespace(val=total))

    params = {
        "solution_field_consumer": lambda load_id, field: consumed.__setitem__(
            load_id, field.copy()
        ),
        "solution_field_dtype": "float32",
    }
    metadata = _save_solution_fields_if_requested(
        prob=SimpleNamespace(output={"sol_primal": solutions}),
        Ngrid=np.array((2, 2, 2)),
        p=params,
    )

    assert metadata["solution_field_format"] == "memory_consumer"
    assert params["_solution_fields_consumed"] == tuple(range(6))
    assert "_solution_fields_result" not in params
    assert not list(tmp_path.iterdir())
    for load_id, field in consumed.items():
        assert field.dtype == np.float32
        np.testing.assert_allclose(field[load_id], 0.02 * (load_id + 1), rtol=2e-6)


def _frequency_unit_vectors(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    grids = np.meshgrid(*[np.fft.fftfreq(size) for size in shape], indexing="ij")
    frequency = np.asarray(grids, dtype=float)
    norm = np.sqrt(np.sum(frequency * frequency, axis=0))
    nonzero = norm > 0.0
    unit = np.zeros_like(frequency)
    unit[:, nonzero] = frequency[:, nonzero] / norm[nonzero]
    return unit, nonzero


def _mandel_to_tensor(field: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    values = np.asarray(field).reshape(6, *shape)
    tensor = np.empty((3, 3, *shape), dtype=values.dtype)
    for component, (ii, jj) in enumerate(MANDEL_PAIRS):
        tensor[ii, jj] = values[component] / MANDEL_FACTORS[component]
        tensor[jj, ii] = tensor[ii, jj]
    return tensor


def _tensor_to_mandel(tensor: np.ndarray) -> np.ndarray:
    shape = tensor.shape[2:]
    values = np.empty((6, int(np.prod(shape))), dtype=tensor.dtype)
    for component, (ii, jj) in enumerate(MANDEL_PAIRS):
        values[component] = (MANDEL_FACTORS[component] * tensor[ii, jj]).reshape(-1)
    return values


def _project_compatible(
    field: np.ndarray,
    *,
    shape: tuple[int, int, int],
    unit: np.ndarray,
    nonzero: np.ndarray,
) -> np.ndarray:
    fourier = np.fft.fftn(_mandel_to_tensor(field, shape), axes=(2, 3, 4))
    traction = np.einsum("ijxyz,jxyz->ixyz", fourier, unit, optimize=True)
    parallel = unit * np.einsum("ixyz,ixyz->xyz", traction, unit, optimize=True)[None]
    displacement = 2.0 * (traction - parallel) + parallel
    projected = 0.5 * (
        np.einsum("ixyz,jxyz->ijxyz", unit, displacement, optimize=True)
        + np.einsum("ixyz,jxyz->ijxyz", displacement, unit, optimize=True)
    )
    projected[:, :, ~nonzero] = 0.0
    return _tensor_to_mandel(np.fft.ifftn(projected, axes=(2, 3, 4)).real)


def isotropic_mandel(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    matrix = np.zeros((6, 6), dtype=np.float64)
    matrix[:3, :3] = lam
    matrix[np.arange(3), np.arange(3)] += 2.0 * mu
    matrix[3:, 3:] = 2.0 * mu * np.eye(3)
    return matrix


def test_homogeneous_isotropic_medium_returns_local_stiffness(tmp_path: Path):
    shape = (6, 6, 6)
    phase = np.zeros(shape, dtype=np.uint8)
    ori = np.zeros((*shape, 3), dtype=np.float64)
    E, nu = 7.25, 0.29
    ceff = solve_homogenization(
        {
            "input_dir": str(tmp_path),
            "seed": 11,
            "phase_array": phase,
            "ori_array": ori,
            "Em": E,
            "nu_m": nu,
            "Ef_L": 31.0,
            "Ef_T": 8.0,
            "nu_LT": 0.22,
            "nu_TT": 0.35,
            "G_LT": 4.1,
            "fft_backend": "scipy",
            "solver_profile": "truth",
            "solver_maxiter": 200,
        }
    )
    expected = isotropic_mandel(E, nu)
    relative_error = np.linalg.norm(ceff - expected) / np.linalg.norm(expected)
    assert relative_error < 1.0e-12


def test_homogeneous_rotated_ti_medium_returns_rotated_stiffness(tmp_path: Path):
    shape = (6, 6, 6)
    phase = np.ones(shape, dtype=np.uint8)
    axis = np.array((2.0, -1.0, 3.0), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    ori = np.broadcast_to(axis, (*shape, 3)).copy()
    fiber = {
        "Ef_L": 48.0,
        "Ef_T": 13.0,
        "nu_LT": 0.21,
        "nu_TT": 0.37,
        "G_LT": 5.2,
    }
    ceff = solve_homogenization(
        {
            "input_dir": str(tmp_path),
            "seed": 12,
            "phase_array": phase,
            "ori_array": ori,
            "Em": 3.0,
            "nu_m": 0.34,
            **fiber,
            "fft_backend": "scipy",
            "solver_profile": "truth",
            "solver_maxiter": 200,
        }
    )
    local = voigt_to_mandel(
        TI_stiffness_voigt(
            fiber["Ef_L"],
            fiber["Ef_T"],
            fiber["nu_LT"],
            fiber["nu_TT"],
            fiber["G_LT"],
        )
    )
    expected = rotate_C_mandel(local, rotation_matrix_from_vector(axis))
    relative_error = np.linalg.norm(ceff - expected) / np.linalg.norm(expected)
    assert relative_error < 1.0e-12


def laminate_normal_x1(C0: np.ndarray, C1: np.ndarray, vf1: float) -> np.ndarray:
    traction = np.array((0, 5, 4))
    in_plane = np.array((1, 2, 3))
    phases = ((1.0 - vf1, C0), (vf1, C1))
    average_inverse = np.zeros((3, 3))
    coupling = np.zeros((3, 3))
    coupling_t = np.zeros((3, 3))
    schur = np.zeros((3, 3))
    for weight, C in phases:
        Caa = C[np.ix_(traction, traction)]
        Cab = C[np.ix_(traction, in_plane)]
        Cba = C[np.ix_(in_plane, traction)]
        Cbb = C[np.ix_(in_plane, in_plane)]
        inverse = np.linalg.inv(Caa)
        average_inverse += weight * inverse
        coupling += weight * inverse @ Cab
        coupling_t += weight * Cba @ inverse
        schur += weight * (Cbb - Cba @ inverse @ Cab)
    effective_aa = np.linalg.inv(average_inverse)
    effective_ab = effective_aa @ coupling
    effective_ba = coupling_t @ effective_aa
    effective_bb = schur + coupling_t @ effective_aa @ coupling
    effective = np.zeros((6, 6))
    effective[np.ix_(traction, traction)] = effective_aa
    effective[np.ix_(traction, in_plane)] = effective_ab
    effective[np.ix_(in_plane, traction)] = effective_ba
    effective[np.ix_(in_plane, in_plane)] = effective_bb
    return 0.5 * (effective + effective.T)


def _laminate_solve(tmp_path: Path):
    n = 12
    vf = 0.5
    phase = np.zeros((n, n, n), dtype=np.uint8)
    phase[: int(vf * n)] = 1
    ori = np.zeros((n, n, n, 3), dtype=np.float64)
    ori[..., 0] = 1.0
    E0, nu0 = 3.0, 0.31
    E1, nu1 = 11.0, 0.23
    fields_path = tmp_path / "solution_fields.npz"
    ceff = solve_homogenization(
        {
            "input_dir": str(tmp_path),
            "seed": 0,
            "phase_array": phase,
            "ori_array": ori,
            "Em": E0,
            "nu_m": nu0,
            "Ef_L": E1,
            "Ef_T": E1,
            "nu_LT": nu1,
            "nu_TT": nu1,
            "G_LT": E1 / (2.0 * (1.0 + nu1)),
            "fft_backend": "scipy",
            "solver_profile": "truth",
            "solver_maxiter": 500,
            "store_solution_fields": True,
            "solution_field_out_path": str(fields_path),
            "solution_field_load_ids": list(range(6)),
            "solution_field_format": "npz",
        }
    )
    return ceff, phase, fields_path, isotropic_mandel(E0, nu0), isotropic_mandel(E1, nu1)


def test_laminate_matches_exact_3d_solution(tmp_path: Path):
    ceff, _, _, C0, C1 = _laminate_solve(tmp_path)
    exact = laminate_normal_x1(C0, C1, 0.5)
    relative_error = np.linalg.norm(ceff - exact) / np.linalg.norm(exact)
    assert relative_error < 2.0e-9


def test_compatibility_equilibrium_and_hill_mandel(tmp_path: Path):
    ceff, phase, fields_path, C0, C1 = _laminate_solve(tmp_path)
    with np.load(fields_path) as payload:
        fluctuations = np.stack(
            [np.asarray(payload[f"fluctuation_load{load}"], dtype=np.float64) for load in range(6)]
        )
    macro = np.array((0.7, -0.2, 0.3, 0.15, -0.11, 0.08))
    fluctuation = np.einsum("l,lcxyz->cxyz", macro, fluctuations, optimize=True)
    assert np.linalg.norm(fluctuation.mean(axis=(1, 2, 3))) < 1.0e-12
    shape = tuple(int(value) for value in phase.shape)
    nvec, nonzero = _frequency_unit_vectors(shape)
    compatible = _project_compatible(
        fluctuation.reshape(6, -1), shape=shape, unit=nvec, nonzero=nonzero
    ).reshape(fluctuation.shape)
    compatibility_error = np.linalg.norm(fluctuation - compatible) / np.linalg.norm(fluctuation)
    assert compatibility_error < 2.0e-9

    total_strain = fluctuation + macro[:, None, None, None]
    stress = np.empty_like(total_strain)
    matrix = phase == 0
    fiber = ~matrix
    stress[:, matrix] = C0 @ total_strain[:, matrix]
    stress[:, fiber] = C1 @ total_strain[:, fiber]
    residual = _project_compatible(
        stress.reshape(6, -1), shape=shape, unit=nvec, nonzero=nonzero
    )
    equilibrium_error = np.linalg.norm(residual) / np.linalg.norm(stress)
    assert equilibrium_error < 2.0e-9

    microscopic_energy = float(np.mean(np.sum(total_strain * stress, axis=0)))
    macroscopic_energy = float(macro @ ceff @ macro)
    assert abs(microscopic_energy - macroscopic_energy) / abs(macroscopic_energy) < 2.0e-9


def test_mandel_mapping_preserves_tensor_energy():
    rng = np.random.default_rng(88)
    symmetric = rng.normal(size=(3, 3))
    symmetric = 0.5 * (symmetric + symmetric.T)
    mandel = np.array(
        (
            symmetric[0, 0], symmetric[1, 1], symmetric[2, 2],
            np.sqrt(2.0) * symmetric[1, 2],
            np.sqrt(2.0) * symmetric[0, 2],
            np.sqrt(2.0) * symmetric[0, 1],
        )
    )
    assert abs(np.sum(symmetric * symmetric) - mandel @ mandel) < 1.0e-14
