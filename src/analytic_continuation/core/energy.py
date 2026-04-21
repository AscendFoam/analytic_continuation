"""Strain energy calculations."""

from __future__ import annotations

import numpy as np

from analytic_continuation.core.functional import BaseIntervalSolution, PolynomialBasisSolution


class StrainEnergy:
    """Utilities for computing squared-derivative energies."""

    @staticmethod
    def from_polynomial(
        coefficients: np.ndarray,
        interval: tuple[float, float],
        order: int = 2,
    ) -> float:
        derived = np.array(coefficients, dtype=float)
        for _ in range(order):
            derived = np.polyder(derived)
        squared = np.polymul(derived, derived)
        antiderivative = np.polyint(squared)
        left, right = interval
        return float(np.polyval(antiderivative, right) - np.polyval(antiderivative, left))

    @staticmethod
    def from_basis(solution: BaseIntervalSolution, order: int = 2, grid_size: int = 2049) -> float:
        if isinstance(solution, PolynomialBasisSolution):
            return StrainEnergy.from_polynomial(solution.coefficients, solution.interval, order=order)

        left, right = solution.interval
        grid = np.linspace(left, right, grid_size)
        values = np.array([solution.derivative(point, order=order) for point in grid], dtype=float)
        return float(np.trapezoid(values * values, grid))

    @staticmethod
    def from_discrete_operator(values: np.ndarray, operator: np.ndarray, weights: np.ndarray) -> float:
        derived = operator @ values
        return float(np.sum(weights * derived * derived))
