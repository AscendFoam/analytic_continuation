"""Regularized residual-minimization continuation method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.polynomial import chebyshev
from scipy.optimize import least_squares, minimize

from analytic_continuation.core.functional import ChebyshevBasisSolution
from analytic_continuation.core.sequence import RecurrenceSequence
from analytic_continuation.methods.base import ContinuationMethod, ContinuationResult
from analytic_continuation.methods.hermite_cubic import HermiteCubicMethod
from analytic_continuation.methods.hermite_quintic import HermiteQuinticMethod
from analytic_continuation.utils.chebyshev_utils import (
    chebyshev_differentiation_matrix,
    chebyshev_lobatto_nodes,
    clenshaw_curtis_weights,
)


@dataclass
class RegularizedIterationMethod(ContinuationMethod):
    """M5: soft-constrained regularized residual minimization.

    This prototype keeps the base-interval endpoint values hard-fixed and
    optimizes the interior Chebyshev-Lobatto node values. The default optimizer
    solves a least-squares residual vector containing curvature, endpoint
    derivative-propagation residuals, and a small coefficient regularizer:

        lambda_energy * integral (f'')^2 + lambda_residual * ||R_deriv||^2

    It is intentionally a small, inspectable bridge between the hard-constrained
    Chebyshev QP and a future full multi-interval nonlinear residual solver.
    """

    name: str = "regularized_iter"
    degree: int = 12
    constraint_order: int = 4
    lambda_energy: float = 1.0
    lambda_residual: float = 10.0
    coefficient_regularization: float = 1e-10
    residual_scale_strategy: str = "order_weighted"
    residual_order_weight: float = 2.0
    residual_scale_floor: float = 1.0
    initial_method: str = "hermite_quintic"
    optimizer_backend: str = "least_squares"
    optimizer_method: str = "trf"
    maxiter: int = 500
    ftol: float = 1e-12
    require_success: bool = False

    def _validate(self) -> None:
        if self.degree < 2:
            raise ValueError("degree must be at least 2.")
        if self.constraint_order < 1:
            raise ValueError("constraint_order must be at least 1.")
        if self.lambda_energy < 0.0:
            raise ValueError("lambda_energy must be non-negative.")
        if self.lambda_residual < 0.0:
            raise ValueError("lambda_residual must be non-negative.")
        if self.coefficient_regularization < 0.0:
            raise ValueError("coefficient_regularization must be non-negative.")
        if self.residual_order_weight < 0.0:
            raise ValueError("residual_order_weight must be non-negative.")
        if self.residual_scale_floor <= 0.0:
            raise ValueError("residual_scale_floor must be positive.")
        if self.residual_scale_strategy not in {"absolute", "relative", "order_weighted"}:
            raise ValueError("residual_scale_strategy must be one of: absolute, relative, order_weighted.")
        if self.optimizer_backend not in {"least_squares", "minimize"}:
            raise ValueError("optimizer_backend must be one of: least_squares, minimize.")
        if self.maxiter < 1:
            raise ValueError("maxiter must be positive.")

    def _build_operators(self, interval: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
        nodes = chebyshev_lobatto_nodes(self.degree, interval=interval)
        weights = clenshaw_curtis_weights(self.degree, interval=interval)
        derivative = chebyshev_differentiation_matrix(self.degree, interval=interval)

        max_order = max(2, self.constraint_order)
        operators: dict[int, np.ndarray] = {0: np.eye(self.degree + 1)}
        current = np.eye(self.degree + 1)
        for order in range(1, max_order + 1):
            current = derivative @ current
            operators[order] = current
        return nodes, weights, operators

    def _initial_values(
        self,
        seq: RecurrenceSequence,
        nodes: np.ndarray,
        initial_values: np.ndarray | None,
    ) -> np.ndarray:
        if initial_values is not None:
            values = np.array(initial_values, dtype=float)
            if values.shape != (self.degree + 1,):
                raise ValueError(f"initial_values must have shape ({self.degree + 1},).")
            values[0] = seq.f_n0
            values[-1] = seq.f_n0_plus_1
            return values

        if self.initial_method == "linear":
            left, right = seq.base_interval
            slope = (seq.f_n0_plus_1 - seq.f_n0) / (right - left)
            return seq.f_n0 + slope * (nodes - left)

        if self.initial_method == "hermite_cubic":
            result = HermiteCubicMethod().solve(seq, target_points=[])
            return np.array([result.base_solution.evaluate(node) for node in nodes], dtype=float)

        if self.initial_method == "hermite_quintic":
            try:
                result = HermiteQuinticMethod().solve(seq, target_points=[])
            except Exception:  # noqa: BLE001
                result = HermiteCubicMethod().solve(seq, target_points=[])
            return np.array([result.base_solution.evaluate(node) for node in nodes], dtype=float)

        raise ValueError(
            "initial_method must be one of: linear, hermite_cubic, hermite_quintic."
        )

    def _values_from_variables(self, seq: RecurrenceSequence, variables: np.ndarray) -> np.ndarray:
        values = np.empty(self.degree + 1, dtype=float)
        values[0] = seq.f_n0
        values[-1] = seq.f_n0_plus_1
        values[1:-1] = variables
        return values

    def _coefficients_from_values(
        self,
        interval: tuple[float, float],
        nodes: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        left, right = interval
        normalized_nodes = (2.0 * nodes - left - right) / (right - left)
        return chebyshev.chebfit(normalized_nodes, values, deg=self.degree)

    def _derivatives_from_values(
        self,
        values: np.ndarray,
        operators: dict[int, np.ndarray],
    ) -> tuple[dict[int, float], dict[int, float]]:
        left_derivatives: dict[int, float] = {}
        right_derivatives: dict[int, float] = {}
        for order in range(1, self.constraint_order + 1):
            derivative_values = operators[order] @ values
            left_derivatives[order] = float(derivative_values[0])
            right_derivatives[order] = float(derivative_values[-1])
        return left_derivatives, right_derivatives

    def _energy(self, values: np.ndarray, weights: np.ndarray, operators: dict[int, np.ndarray]) -> float:
        second_derivative = operators[2] @ values
        return float(np.sum(weights * second_derivative * second_derivative))

    def _residual_scale(
        self,
        order: int,
        left_derivatives: dict[int, float],
        right_derivatives: dict[int, float],
    ) -> float:
        if self.residual_scale_strategy == "absolute":
            return self.residual_scale_floor

        relative = max(
            self.residual_scale_floor,
            abs(left_derivatives.get(order, 0.0)),
            abs(right_derivatives.get(order, 0.0)),
        )
        if self.residual_scale_strategy == "relative":
            return relative

        return relative * (float(order) ** self.residual_order_weight)

    def _residual_penalty(
        self,
        seq: RecurrenceSequence,
        values: np.ndarray,
        operators: dict[int, np.ndarray],
    ) -> tuple[
        float,
        dict[int, float],
        dict[int, float],
        dict[int, float],
        dict[int, float],
        dict[int, float],
    ]:
        left_derivatives, right_derivatives = self._derivatives_from_values(values, operators)
        residual_pairs = seq.derivative_constraint_residuals(
            self.constraint_order,
            left_derivatives=left_derivatives,
            right_derivatives=right_derivatives,
        )

        penalty = 0.0
        residuals: dict[int, float] = {}
        scaled_residuals: dict[int, float] = {}
        residual_scales: dict[int, float] = {}
        for order, residual in residual_pairs:
            residuals[order] = float(residual)
            scale = self._residual_scale(order, left_derivatives, right_derivatives)
            scaled = float(residual / scale)
            residual_scales[order] = scale
            scaled_residuals[order] = scaled
            penalty += scaled * scaled
        return penalty, residuals, scaled_residuals, residual_scales, left_derivatives, right_derivatives

    def _objective_parts(
        self,
        seq: RecurrenceSequence,
        interval: tuple[float, float],
        nodes: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
        operators: dict[int, np.ndarray],
        energy_scale: float,
    ) -> dict[str, Any]:
        energy = self._energy(values, weights, operators)
        (
            residual_penalty,
            residuals,
            scaled_residuals,
            residual_scales,
            left_derivatives,
            right_derivatives,
        ) = self._residual_penalty(
            seq,
            values,
            operators,
        )
        coefficients = self._coefficients_from_values(interval, nodes, values)
        coefficient_penalty = float(np.sum(np.arange(len(coefficients), dtype=float) ** 2 * coefficients * coefficients))
        objective = (
            self.lambda_energy * energy / energy_scale
            + self.lambda_residual * residual_penalty
            + self.coefficient_regularization * coefficient_penalty
        )
        return {
            "objective": float(objective),
            "energy": float(energy),
            "residual_penalty": float(residual_penalty),
            "coefficient_penalty": float(coefficient_penalty),
            "endpoint_residuals": residuals,
            "scaled_endpoint_residuals": scaled_residuals,
            "endpoint_residual_scales": residual_scales,
            "left_derivatives": left_derivatives,
            "right_derivatives": right_derivatives,
            "coefficients": coefficients,
        }

    def _objective_vector(
        self,
        seq: RecurrenceSequence,
        interval: tuple[float, float],
        nodes: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
        operators: dict[int, np.ndarray],
        energy_scale: float,
    ) -> np.ndarray:
        terms: list[np.ndarray] = []

        if self.lambda_energy > 0.0:
            second_derivative = operators[2] @ values
            energy_terms = (
                np.sqrt(self.lambda_energy / energy_scale)
                * np.sqrt(np.maximum(weights, 0.0))
                * second_derivative
            )
            terms.append(np.array(energy_terms, dtype=float))

        if self.lambda_residual > 0.0:
            _, _, scaled_residuals, _, _, _ = self._residual_penalty(seq, values, operators)
            residual_terms = np.sqrt(self.lambda_residual) * np.array(
                [scaled_residuals[order] for order in sorted(scaled_residuals)],
                dtype=float,
            )
            terms.append(residual_terms)

        if self.coefficient_regularization > 0.0:
            coefficients = self._coefficients_from_values(interval, nodes, values)
            coeff_terms = (
                np.sqrt(self.coefficient_regularization)
                * np.arange(len(coefficients), dtype=float)
                * coefficients
            )
            terms.append(np.array(coeff_terms, dtype=float))

        if not terms:
            return np.zeros(1, dtype=float)
        return np.concatenate(terms)

    def _minimize_variables(
        self,
        objective: Any,
        residual_vector: Any,
        initial_variables: np.ndarray,
    ) -> Any:
        if self.optimizer_backend == "least_squares":
            return least_squares(
                residual_vector,
                initial_variables,
                method=self.optimizer_method,
                max_nfev=self.maxiter,
                ftol=self.ftol,
                xtol=self.ftol,
                gtol=self.ftol,
            )

        return minimize(
            objective,
            initial_variables,
            method=self.optimizer_method,
            options={"maxiter": self.maxiter, "ftol": self.ftol},
        )

    def solve(
        self,
        seq: RecurrenceSequence,
        target_points: list[float],
        **kwargs: Any,
    ) -> ContinuationResult:
        initial_values = kwargs.pop("initial_values", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")
        self._validate()

        interval = seq.base_interval
        nodes, weights, operators = self._build_operators(interval)
        initial = self._initial_values(seq, nodes, initial_values)
        initial_energy = self._energy(initial, weights, operators)
        energy_scale = max(abs(initial_energy), 1.0)

        def objective(variables: np.ndarray) -> float:
            values = self._values_from_variables(seq, variables)
            parts = self._objective_parts(
                seq,
                interval,
                nodes,
                values,
                weights,
                operators,
                energy_scale,
            )
            value = float(parts["objective"])
            if not np.isfinite(value):
                return 1e300
            return value

        def residual_vector(variables: np.ndarray) -> np.ndarray:
            values = self._values_from_variables(seq, variables)
            vector = self._objective_vector(
                seq,
                interval,
                nodes,
                values,
                weights,
                operators,
                energy_scale,
            )
            if not np.all(np.isfinite(vector)):
                return np.full_like(vector, 1e150, dtype=float)
            return vector

        optimization = self._minimize_variables(
            objective,
            residual_vector,
            initial[1:-1],
        )
        if self.require_success and not optimization.success:
            raise RuntimeError(f"Regularized iteration failed: {optimization.message}")

        values = self._values_from_variables(seq, np.array(optimization.x, dtype=float))
        parts = self._objective_parts(
            seq,
            interval,
            nodes,
            values,
            weights,
            operators,
            energy_scale,
        )
        coefficients = np.array(parts["coefficients"], dtype=float)
        base_solution = ChebyshevBasisSolution(
            interval=interval,
            coefficients=coefficients,
            label="regularized_iter_chebyshev_lobatto",
        )

        left_derivatives = parts["left_derivatives"]
        right_derivatives = parts["right_derivatives"]
        endpoint_residuals = parts["endpoint_residuals"]
        scaled_endpoint_residuals = parts["scaled_endpoint_residuals"]
        endpoint_residual_norm = float(np.sqrt(sum(value * value for value in endpoint_residuals.values())))
        scaled_endpoint_residual_norm = float(np.sqrt(sum(value * value for value in scaled_endpoint_residuals.values())))
        optimizer_iterations = int(
            getattr(optimization, "nit", 0)
            or getattr(optimization, "nfev", 0)
            or 0
        )

        result = ContinuationResult(
            method_name=self.name,
            sequence_name=seq.name,
            optimal_params={
                "left_derivative": float(left_derivatives.get(1, base_solution.derivative(interval[0], order=1))),
                "right_derivative": float(right_derivatives.get(1, base_solution.derivative(interval[1], order=1))),
                "left_second_derivative": float(left_derivatives.get(2, base_solution.derivative(interval[0], order=2))),
                "right_second_derivative": float(right_derivatives.get(2, base_solution.derivative(interval[1], order=2))),
            },
            strain_energy=float(parts["energy"]),
            eval_at={},
            basis_coefficients=coefficients,
            base_solution=base_solution,
            metadata={
                "degree": self.degree,
                "node_count": float(self.degree + 1),
                "constraint_order": self.constraint_order,
                "lambda_energy": self.lambda_energy,
                "lambda_residual": self.lambda_residual,
                "coefficient_regularization": self.coefficient_regularization,
                "residual_scale_strategy": self.residual_scale_strategy,
                "residual_order_weight": self.residual_order_weight,
                "residual_scale_floor": self.residual_scale_floor,
                "initial_method": self.initial_method,
                "optimizer_backend": self.optimizer_backend,
                "optimizer_method": self.optimizer_method,
                "optimizer_success": bool(optimization.success),
                "optimizer_message": str(optimization.message),
                "optimizer_iterations": optimizer_iterations,
                "objective_value": float(parts["objective"]),
                "energy_scale": float(energy_scale),
                "residual_penalty": float(parts["residual_penalty"]),
                "endpoint_residual_norm": endpoint_residual_norm,
                "scaled_endpoint_residual_norm": scaled_endpoint_residual_norm,
                "endpoint_residuals": endpoint_residuals,
                "scaled_endpoint_residuals": scaled_endpoint_residuals,
                "endpoint_residual_scales": parts["endpoint_residual_scales"],
            },
        )
        result.eval_at = {point: self.evaluate(point, seq, result) for point in target_points}
        return result
