"""Baseline cubic Hermite continuation method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from analytic_continuation.core.energy import StrainEnergy
from analytic_continuation.core.functional import PolynomialBasisSolution
from analytic_continuation.core.sequence import RecurrenceSequence
from analytic_continuation.methods.base import ContinuationMethod, ContinuationResult


def _build_cubic_coefficients(
    left: float,
    right: float,
    y_left: float,
    y_right: float,
    d_left: float,
    d_right: float,
) -> np.ndarray:
    matrix = np.array(
        [
            [left**3, left**2, left, 1.0],
            [right**3, right**2, right, 1.0],
            [3.0 * left**2, 2.0 * left, 1.0, 0.0],
            [3.0 * right**2, 2.0 * right, 1.0, 0.0],
        ],
        dtype=float,
    )
    rhs = np.array([y_left, y_right, d_left, d_right], dtype=float)
    return np.linalg.solve(matrix, rhs)


@dataclass
class HermiteCubicMethod(ContinuationMethod):
    """M1: cubic Hermite spline with strain-energy minimization."""

    name: str = "hermite_cubic"
    energy_order: int = 2

    def _energy_with_left_derivative(self, seq: RecurrenceSequence, left_derivative: float) -> tuple[float, np.ndarray, float]:
        left, right = seq.base_interval
        right_derivative = seq.first_derivative_map(left_derivative)
        coefficients = _build_cubic_coefficients(
            left=left,
            right=right,
            y_left=seq.f_n0,
            y_right=seq.f_n0_plus_1,
            d_left=left_derivative,
            d_right=right_derivative,
        )
        energy = StrainEnergy.from_polynomial(coefficients, (left, right), order=self.energy_order)
        return energy, coefficients, right_derivative

    def _optimal_left_derivative(self, seq: RecurrenceSequence) -> tuple[float, float, np.ndarray]:
        samples = np.array([-1.0, 0.0, 1.0], dtype=float)
        energies = np.array([self._energy_with_left_derivative(seq, sample)[0] for sample in samples], dtype=float)
        quadratic_coeffs = np.polyfit(samples, energies, deg=2)
        a, b, _ = quadratic_coeffs
        if a <= 0.0:
            raise ValueError("Expected a strictly convex energy functional for the cubic baseline.")
        optimal_left_derivative = -b / (2.0 * a)
        energy, coefficients, right_derivative = self._energy_with_left_derivative(seq, optimal_left_derivative)
        return optimal_left_derivative, right_derivative, coefficients

    def solve(
        self,
        seq: RecurrenceSequence,
        target_points: list[float],
        **kwargs: Any,
    ) -> ContinuationResult:
        del kwargs
        left_derivative, right_derivative, coefficients = self._optimal_left_derivative(seq)
        base_solution = PolynomialBasisSolution(interval=seq.base_interval, coefficients=coefficients, label="cubic_hermite")
        strain_energy = StrainEnergy.from_basis(base_solution, order=self.energy_order)
        result = ContinuationResult(
            method_name=self.name,
            sequence_name=seq.name,
            optimal_params={
                "left_derivative": float(left_derivative),
                "right_derivative": float(right_derivative),
            },
            strain_energy=float(strain_energy),
            eval_at={},
            basis_coefficients=coefficients,
            base_solution=base_solution,
            metadata={
                "energy_order": self.energy_order,
                "interval": seq.base_interval,
            },
        )
        result.eval_at = {point: self.evaluate(point, seq, result) for point in target_points}
        return result
