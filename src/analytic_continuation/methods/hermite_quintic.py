"""Quintic Hermite continuation method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from analytic_continuation.core.energy import StrainEnergy
from analytic_continuation.core.functional import PolynomialBasisSolution
from analytic_continuation.core.sequence import RecurrenceSequence
from analytic_continuation.methods.base import ContinuationMethod, ContinuationResult
from analytic_continuation.methods.hermite_cubic import HermiteCubicMethod


def _build_quintic_coefficients(
    left: float,
    right: float,
    y_left: float,
    y_right: float,
    d_left: float,
    d_right: float,
    dd_left: float,
    dd_right: float,
) -> np.ndarray:
    matrix = np.array(
        [
            [left**5, left**4, left**3, left**2, left, 1.0],
            [right**5, right**4, right**3, right**2, right, 1.0],
            [5.0 * left**4, 4.0 * left**3, 3.0 * left**2, 2.0 * left, 1.0, 0.0],
            [5.0 * right**4, 4.0 * right**3, 3.0 * right**2, 2.0 * right, 1.0, 0.0],
            [20.0 * left**3, 12.0 * left**2, 6.0 * left, 2.0, 0.0, 0.0],
            [20.0 * right**3, 12.0 * right**2, 6.0 * right, 2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    rhs = np.array([y_left, y_right, d_left, d_right, dd_left, dd_right], dtype=float)
    return np.linalg.solve(matrix, rhs)


@dataclass
class HermiteQuinticMethod(ContinuationMethod):
    """M2: quintic Hermite spline with numerical strain-energy optimization."""

    name: str = "hermite_quintic"
    energy_order: int = 2
    optimizer_method: str = "Powell"
    maxiter: int = 400

    def _energy_with_left_derivatives(
        self,
        seq: RecurrenceSequence,
        left_first_derivative: float,
        left_second_derivative: float,
    ) -> tuple[float, np.ndarray, float, float]:
        left, right = seq.base_interval
        right_first_derivative = seq.first_derivative_map(left_first_derivative)
        right_second_derivative = seq.second_derivative_map(left_first_derivative, left_second_derivative)
        coefficients = _build_quintic_coefficients(
            left=left,
            right=right,
            y_left=seq.f_n0,
            y_right=seq.f_n0_plus_1,
            d_left=left_first_derivative,
            d_right=right_first_derivative,
            dd_left=left_second_derivative,
            dd_right=right_second_derivative,
        )
        energy = StrainEnergy.from_polynomial(coefficients, (left, right), order=self.energy_order)
        return energy, coefficients, right_first_derivative, right_second_derivative

    def _optimize_left_derivatives(
        self,
        seq: RecurrenceSequence,
        initial_guess: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float, Any]:
        if initial_guess is None:
            cubic_result = HermiteCubicMethod().solve(seq, target_points=[])
            initial_guess = np.array([cubic_result.optimal_params["left_derivative"], 0.0], dtype=float)

        def objective(variables: np.ndarray) -> float:
            energy, _, _, _ = self._energy_with_left_derivatives(seq, variables[0], variables[1])
            return float(energy)

        optimization = minimize(
            objective,
            np.array(initial_guess, dtype=float),
            method=self.optimizer_method,
            options={"maxiter": self.maxiter},
        )
        if not optimization.success:
            optimization = minimize(
                objective,
                np.array(initial_guess, dtype=float),
                method="Nelder-Mead",
                options={"maxiter": self.maxiter},
            )
        if not optimization.success:
            raise RuntimeError(f"Quintic Hermite optimization failed: {optimization.message}")

        optimal = np.array(optimization.x, dtype=float)
        energy, coefficients, right_first_derivative, right_second_derivative = self._energy_with_left_derivatives(
            seq,
            optimal[0],
            optimal[1],
        )
        return optimal, coefficients, right_first_derivative, right_second_derivative, optimization

    def solve(
        self,
        seq: RecurrenceSequence,
        target_points: list[float],
        **kwargs: Any,
    ) -> ContinuationResult:
        initial_guess = kwargs.pop("initial_guess", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")

        optimal, coefficients, right_first_derivative, right_second_derivative, optimization = (
            self._optimize_left_derivatives(seq, initial_guess=initial_guess)
        )
        base_solution = PolynomialBasisSolution(interval=seq.base_interval, coefficients=coefficients, label="quintic_hermite")
        strain_energy = StrainEnergy.from_basis(base_solution, order=self.energy_order)
        result = ContinuationResult(
            method_name=self.name,
            sequence_name=seq.name,
            optimal_params={
                "left_derivative": float(optimal[0]),
                "left_second_derivative": float(optimal[1]),
                "right_derivative": float(right_first_derivative),
                "right_second_derivative": float(right_second_derivative),
            },
            strain_energy=float(strain_energy),
            eval_at={},
            basis_coefficients=coefficients,
            base_solution=base_solution,
            metadata={
                "energy_order": self.energy_order,
                "interval": seq.base_interval,
                "optimizer_method": self.optimizer_method,
                "optimizer_success": bool(optimization.success),
                "optimizer_iterations": int(getattr(optimization, "nit", 0) or 0),
            },
        )
        result.eval_at = {point: self.evaluate(point, seq, result) for point in target_points}
        return result
