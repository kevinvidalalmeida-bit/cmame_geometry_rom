#!/usr/bin/env python3
"""Primal-dual CRE helpers for the fixed-geometry elasticity ROM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy import linalg as scipy_linalg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FFT_ROOT = PROJECT_ROOT / "FFT"
FFTHOMPY_PATH = FFT_ROOT / "ffthompy_core" / "ffthompy"
for _path in (FFT_ROOT, FFTHOMPY_PATH):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ffthompy.projections as projections
from ffthompy.tensors import DFT, Operator, Tensor
from ffthompy.tensors.fft import get_fft_backend, set_fft_backend

import rom_reduced_operator as reduced


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (values + np.swapaxes(values, -1, -2))


class ElasticityProjectors:
    """Apply the discrete GaNi compatible/static projectors in float64.

    On even grids, the static space paired with the Nyquist-null primal space
    is the full orthogonal complement of constants and compatible fields.  It
    therefore includes the Nyquist modes omitted by FFTHomPy's conventional
    ``G2`` projector.
    """

    def __init__(self, grid_shape: tuple[int, int, int]) -> None:
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.N = np.asarray(self.grid_shape, dtype=int)
        self.Y = np.ones(3, dtype=float)
        previous_backend = get_fft_backend()
        set_fft_backend("scipy")
        try:
            h_g1, _ = projections.elasticity_combined(
                self.N,
                self.Y,
                NyqNul=True,
                tensor=True,
                fft_form="r",
                dtype=np.float64,
                storage="sym21",
                backend="numpy",
            )
            forward = DFT(name="FN_cre", inverse=False, N=self.N, fft_form="r")
            inverse = DFT(name="FiN_cre", inverse=True, N=self.N, fft_form="r")
            self._primal_operator = Operator(
                name="G1_cre", mat=[[inverse, h_g1, forward]]
            )
        finally:
            set_fft_backend(previous_backend)

    def apply(self, fields: np.ndarray, *, kind: str) -> tuple[np.ndarray, dict[str, float]]:
        values = np.asarray(fields, dtype=np.float64)
        expected = (6,) + self.grid_shape
        if values.ndim == 4:
            values = values[None]
        if values.ndim != 5 or tuple(values.shape[1:]) != expected:
            raise ValueError(f"fields must have shape (count, 6, {self.grid_shape}).")
        if kind not in {"primal", "dual"}:
            raise ValueError("kind must be primal or dual.")

        started = time.perf_counter()
        projected = np.empty_like(values)
        previous_backend = get_fft_backend()
        set_fft_backend("scipy")
        try:
            for index, field in enumerate(values):
                tensor = Tensor(
                    name=f"{kind}_field_{index}",
                    N=self.N,
                    shape=(6,),
                    Fourier=False,
                    fft_form="r",
                    dtype=np.float64,
                )
                tensor.val[...] = field
                compatible = np.asarray(self._primal_operator(tensor).val)
                if kind == "primal":
                    projected[index] = compatible
                else:
                    mean = np.mean(field, axis=(1, 2, 3), keepdims=True)
                    projected[index] = field - mean - compatible
        finally:
            set_fft_backend(previous_backend)

        delta = projected - values
        denominator = max(float(np.linalg.norm(values)), np.finfo(float).eps)
        means = np.mean(projected, axis=(2, 3, 4))
        return projected, {
            "projection_wall_s": float(time.perf_counter() - started),
            "relative_projection_correction": float(np.linalg.norm(delta) / denominator),
            "projected_mean_abs_max": float(np.max(np.abs(means))),
        }


def stress_and_dual_snapshots(
    *,
    primal_fluctuations: np.ndarray,
    material: dict[str, Any],
    stiffness_action: Any,
    projectors: ElasticityProjectors,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Recover equilibrated unit-macrostress fluctuations from primal fields."""
    fluctuations = np.asarray(primal_fluctuations, dtype=np.float64)
    expected = (6, 6) + projectors.grid_shape
    if fluctuations.shape != expected:
        raise ValueError(f"primal_fluctuations must have shape {expected}.")

    primal, primal_projection = projectors.apply(fluctuations, kind="primal")
    nvox = int(np.prod(projectors.grid_shape))
    total = primal.reshape(6, 6, nvox).copy()
    for load_id in range(6):
        total[load_id, load_id] += 1.0

    stress_started = time.perf_counter()
    stresses = np.zeros_like(total)
    coefficients = reduced._material_coefficients(material)
    for q, coefficient in enumerate(coefficients):
        stresses += float(coefficient) * stiffness_action(q, total)
    stress_wall_s = float(time.perf_counter() - stress_started)

    effective = _symmetric(np.mean(stresses, axis=2).T)
    inverse_effective = scipy_linalg.solve(
        effective,
        np.eye(6),
        assume_a="pos",
        check_finite=False,
    )
    unit_stress = np.einsum("jan,jk->kan", stresses, inverse_effective, optimize=True)
    for load_id in range(6):
        unit_stress[load_id, load_id] -= 1.0
    raw_dual = unit_stress.reshape(expected)
    dual, dual_projection = projectors.apply(raw_dual, kind="dual")

    diagnostics = {
        "primal_projection_wall_s": primal_projection["projection_wall_s"],
        "primal_projection_correction": primal_projection[
            "relative_projection_correction"
        ],
        "primal_projected_mean_abs_max": primal_projection["projected_mean_abs_max"],
        "stress_recovery_wall_s": stress_wall_s,
        "dual_projection_wall_s": dual_projection["projection_wall_s"],
        "dual_projection_correction": dual_projection[
            "relative_projection_correction"
        ],
        "dual_projected_mean_abs_max": dual_projection["projected_mean_abs_max"],
        "recovered_effective_min_eig": float(np.linalg.eigvalsh(effective)[0]),
    }
    return primal, dual, {**diagnostics, "recovered_effective": effective}


def evaluate_bounds(
    *,
    parameters: np.ndarray,
    primal_operators: dict[str, np.ndarray],
    dual_operators: dict[str, np.ndarray],
    backend: str = "cpu",
) -> dict[str, np.ndarray | float]:
    """Evaluate two-sided reduced bounds and CRE indicators for a material batch."""
    values = np.asarray(parameters, dtype=np.float64)
    primal_coefficients = reduced._material_coefficients_batch(values)
    dual_coefficients = reduced._dual_material_coefficients_batch(values)
    if backend == "gpu":
        primal_evaluator = reduced.GpuAffineBatchEvaluator(
            primal_operators["Kq"],
            primal_operators["Bq"],
            primal_operators["Dq"],
        )
        dual_evaluator = reduced.GpuAffineBatchEvaluator(
            dual_operators["Kq"],
            dual_operators["Bq"],
            dual_operators["Dq"],
        )
        upper, primal_wall_s = primal_evaluator.evaluate(primal_coefficients)
        compliance_upper, dual_wall_s = dual_evaluator.evaluate(dual_coefficients)
    elif backend == "cpu":
        upper, _, primal_wall_s = reduced._rom_ceff_batch(
            primal_coefficients,
            primal_operators["Kq"],
            primal_operators["Bq"],
            primal_operators["Dq"],
        )
        compliance_upper, _, dual_wall_s = reduced._rom_ceff_batch(
            dual_coefficients,
            dual_operators["Kq"],
            dual_operators["Bq"],
            dual_operators["Dq"],
        )
    else:
        raise ValueError("backend must be cpu or gpu.")
    compliance_upper = _symmetric(compliance_upper)
    lower = _symmetric(np.linalg.inv(compliance_upper))
    gap = _symmetric(upper - lower)

    eta_energy = np.empty(len(values), dtype=np.float64)
    eta_frobenius = np.empty(len(values), dtype=np.float64)
    gap_min_eigenvalue = np.empty(len(values), dtype=np.float64)
    for index in range(len(values)):
        eta_energy[index] = float(
            scipy_linalg.eigvalsh(
                gap[index],
                lower[index],
                check_finite=False,
            )[-1]
        )
        eta_frobenius[index] = float(
            np.linalg.norm(gap[index]) / np.linalg.norm(lower[index])
        )
        gap_min_eigenvalue[index] = float(np.linalg.eigvalsh(gap[index])[0])
    return {
        "upper": upper,
        "lower": lower,
        "compliance_upper": compliance_upper,
        "gap": gap,
        "eta_energy": eta_energy,
        "eta_frobenius": eta_frobenius,
        "gap_min_eigenvalue": gap_min_eigenvalue,
        "primal_online_wall_s": float(primal_wall_s),
        "dual_online_wall_s": float(dual_wall_s),
    }


@dataclass(frozen=True)
class TruthComparison:
    true_energy_error: float
    true_frobenius_error: float
    eta_energy: float
    eta_frobenius: float
    energy_effectivity: float
    frobenius_effectivity: float
    lower_truth_min_eig: float
    truth_upper_min_eig: float


def compare_with_truth(
    *,
    upper: np.ndarray,
    lower: np.ndarray,
    truth: np.ndarray,
) -> TruthComparison:
    """Compare one two-sided bracket with an independent FOM tensor."""
    h_upper = _symmetric(upper)
    h_lower = _symmetric(lower)
    h_truth = _symmetric(truth)
    gap = _symmetric(h_upper - h_lower)
    true_gap = _symmetric(h_upper - h_truth)
    eta_energy = float(scipy_linalg.eigvalsh(gap, h_lower, check_finite=False)[-1])
    true_energy = max(
        0.0,
        float(scipy_linalg.eigvalsh(true_gap, h_truth, check_finite=False)[-1]),
    )
    eta_frobenius = float(np.linalg.norm(gap) / np.linalg.norm(h_lower))
    true_frobenius = float(np.linalg.norm(true_gap) / np.linalg.norm(h_truth))
    tiny = np.finfo(float).eps
    return TruthComparison(
        true_energy_error=true_energy,
        true_frobenius_error=true_frobenius,
        eta_energy=eta_energy,
        eta_frobenius=eta_frobenius,
        energy_effectivity=eta_energy / max(true_energy, tiny),
        frobenius_effectivity=eta_frobenius / max(true_frobenius, tiny),
        lower_truth_min_eig=float(np.linalg.eigvalsh(h_truth - h_lower)[0]),
        truth_upper_min_eig=float(np.linalg.eigvalsh(h_upper - h_truth)[0]),
    )
