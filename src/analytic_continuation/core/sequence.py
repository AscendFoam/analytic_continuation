"""Sequence definitions for recurrence-based analytic continuation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Callable


@dataclass
class RecurrenceSequence(ABC):
    """Unified abstraction for recurrence sequences."""

    name: str
    n0: float
    f_n0: float
    f_n0_plus_1: float
    metadata: dict[str, float | str] = field(default_factory=dict)

    @property
    def base_interval(self) -> tuple[float, float]:
        return (self.n0, self.n0 + 1.0)

    @abstractmethod
    def g(self, z: float, w: float) -> float:
        """Forward recurrence map."""

    @abstractmethod
    def g_inv(self, z: float, f_val: float) -> float:
        """Inverse recurrence map."""

    @abstractmethod
    def g_z(self, z: float, w: float) -> float:
        """Partial derivative with respect to z."""

    @abstractmethod
    def g_w(self, z: float, w: float) -> float:
        """Partial derivative with respect to w."""

    @abstractmethod
    def g_zz(self, z: float, w: float) -> float:
        """Second partial derivative with respect to z."""

    @abstractmethod
    def g_zw(self, z: float, w: float) -> float:
        """Mixed second partial derivative."""

    @abstractmethod
    def g_ww(self, z: float, w: float) -> float:
        """Second partial derivative with respect to w."""

    def first_derivative_map(self, left_derivative: float) -> float:
        """Map f'(n0) to f'(n0 + 1) via the chain rule."""

        z = self.n0 + 1.0
        w = self.f_n0
        return self.g_z(z, w) + self.g_w(z, w) * left_derivative

    def first_derivative_constraint_coeffs(self) -> tuple[float, float]:
        """Return alpha, beta in f'(n0+1) = alpha * f'(n0) + beta."""

        z = self.n0 + 1.0
        w = self.f_n0
        return self.g_w(z, w), self.g_z(z, w)

    def second_derivative_constraint_coeffs(self) -> tuple[float, float, float, float]:
        """
        Return coefficients of

        f''(n0+1) = quad * f'(n0)^2 + linear_first * f'(n0)
                    + linear_second * f''(n0) + constant
        """

        z = self.n0 + 1.0
        w = self.f_n0
        return (
            self.g_ww(z, w),
            2.0 * self.g_zw(z, w),
            self.g_w(z, w),
            self.g_zz(z, w),
        )

    def second_derivative_map(self, left_first_derivative: float, left_second_derivative: float) -> float:
        """Map left-end first/second derivatives to the right endpoint."""

        quad, linear_first, linear_second, constant = self.second_derivative_constraint_coeffs()
        return (
            quad * left_first_derivative * left_first_derivative
            + linear_first * left_first_derivative
            + linear_second * left_second_derivative
            + constant
        )

    def special_value(self, z: float, base_solution: Any) -> float | None:
        """Return a sequence-specific limiting value, if one is known."""

        del z, base_solution
        return None

    def linear_derivative_constraints(self, max_order: int) -> list[tuple[int, dict[int, float], float]]:
        """
        Return linear endpoint constraints as

        f^(order)(n0+1) = sum(coeffs[k] * f^(k)(n0) for k >= 1) + constant
        """

        constraints: list[tuple[int, dict[int, float], float]] = []
        if max_order < 1:
            return constraints

        alpha, beta = self.first_derivative_constraint_coeffs()
        constraints.append((1, {1: alpha}, beta))

        if max_order >= 2:
            quad, linear_first, linear_second, constant = self.second_derivative_constraint_coeffs()
            if abs(quad) < 1e-12:
                constraints.append((2, {1: linear_first, 2: linear_second}, constant))

        return constraints

    def linearized_derivative_constraints(
        self,
        max_order: int,
        deriv_estimates: dict[int, float],
    ) -> list[tuple[int, dict[int, float], float]]:
        """
        Return endpoint constraints after linearising nonlinear terms.

        This is intentionally separate from ``linear_derivative_constraints``:
        linear constraints can be used directly in a QP, whereas linearised
        constraints depend on the supplied derivative estimates.
        """

        del deriv_estimates
        return self.linear_derivative_constraints(max_order)

    def derivative_constraint_residuals(
        self,
        max_order: int,
        left_derivatives: dict[int, float],
        right_derivatives: dict[int, float],
    ) -> list[tuple[int, float]]:
        """Evaluate derivative propagation residuals at endpoint derivative values.

        Subclasses with genuinely nonlinear high-order propagation should
        override this method. The base implementation is exact through order 2
        and uses explicitly declared linear constraints for any higher orders.
        """

        residuals: list[tuple[int, float]] = []
        if max_order >= 1:
            residuals.append(
                (1, right_derivatives.get(1, 0.0) - self.first_derivative_map(left_derivatives.get(1, 0.0)))
            )
        if max_order >= 2:
            residuals.append(
                (
                    2,
                    right_derivatives.get(2, 0.0)
                    - self.second_derivative_map(
                        left_derivatives.get(1, 0.0),
                        left_derivatives.get(2, 0.0),
                    ),
                )
            )

        handled_orders = {order for order, _ in residuals}
        constraints = self.linear_derivative_constraints(max_order)
        for order, coeffs, constant in constraints:
            if order in handled_orders:
                continue
            predicted = constant + sum(
                coeff * left_derivatives.get(left_order, 0.0)
                for left_order, coeff in coeffs.items()
            )
            residuals.append((order, right_derivatives.get(order, 0.0) - predicted))
        return residuals


def _faa_di_bruno_constraints(
    max_order: int,
    g_table: dict[tuple[int, int], float],
    deriv_estimates: dict[int, float],
) -> list[tuple[int, dict[int, float], float]]:
    """Build linearised derivative constraints via the multivariate Faa di Bruno formula.

    For h(z) = g(z+1, f(z)), the m-th derivative at z = n0 is:

        h^(m)(n0) = sum_{k1=0}^{m} C(m,k1) *
                    sum_{k2=0}^{m-k1} g_{k1,k2} *
                    B_{m-k1, k2}(f'(n0), ..., f^{(m-k1)}(n0))

    Terms with k2 <= 1 are linear in the derivatives; terms with k2 >= 2
    are nonlinear and are evaluated using *deriv_estimates*, turning them
    into constants.

    Parameters
    ----------
    max_order : int
        Maximum derivative order to generate.
    g_table : dict mapping (k1, k2) -> float
        Mixed partial derivatives of g at (n0+1, f(n0)).
    deriv_estimates : dict mapping derivative order -> estimated value at n0.

    Returns
    -------
    List of (order, {derivative_order: coefficient}, constant) tuples.
    """
    from analytic_continuation.utils.bell_polynomial import partial_bell_polynomial

    constraints: list[tuple[int, dict[int, float], float]] = []
    for m in range(1, max_order + 1):
        coeffs: dict[int, float] = {}
        constant = 0.0
        for k1 in range(m + 1):
            for k2 in range(m - k1 + 1):
                g_val = g_table.get((k1, k2), 0.0)
                if abs(g_val) < 1e-30:
                    continue
                binom_coeff = math.comb(m, k1)
                if k2 == 0:
                    # B_{n,0} = 1 if n == 0, else 0
                    if m - k1 == 0:
                        constant += binom_coeff * g_val
                elif k2 == 1:
                    # B_{n,1}(x) = x_n = f^{(n)}(n0)
                    deriv_order = m - k1
                    if deriv_order >= 1:
                        coeffs[deriv_order] = coeffs.get(deriv_order, 0.0) + binom_coeff * g_val
                else:
                    # k2 >= 2: nonlinear — evaluate with estimates
                    x = [deriv_estimates.get(i, 0.0) for i in range(1, m - k1 + 1)]
                    bell_val = partial_bell_polynomial(m - k1, k2, x)
                    constant += binom_coeff * g_val * bell_val
        constraints.append((m, coeffs, constant))
    return constraints


def _faa_di_bruno_predictions(
    max_order: int,
    g_table: dict[tuple[int, int], float],
    left_derivatives: dict[int, float],
) -> dict[int, float]:
    """Evaluate exact endpoint derivative propagation via Faa di Bruno."""

    from analytic_continuation.utils.bell_polynomial import partial_bell_polynomial

    predictions: dict[int, float] = {}
    for m in range(1, max_order + 1):
        total = 0.0
        for k1 in range(m + 1):
            for k2 in range(m - k1 + 1):
                g_val = g_table.get((k1, k2), 0.0)
                if abs(g_val) < 1e-30:
                    continue
                binom_coeff = math.comb(m, k1)
                if k2 == 0:
                    if m - k1 == 0:
                        total += binom_coeff * g_val
                else:
                    x = [left_derivatives.get(i, 0.0) for i in range(1, m - k1 + 1)]
                    total += binom_coeff * g_val * partial_bell_polynomial(m - k1, k2, x)
        predictions[m] = total
    return predictions


def _faa_di_bruno_residuals(
    max_order: int,
    g_table: dict[tuple[int, int], float],
    left_derivatives: dict[int, float],
    right_derivatives: dict[int, float],
) -> list[tuple[int, float]]:
    """Evaluate right - predicted residuals for nonlinear derivative propagation."""

    predictions = _faa_di_bruno_predictions(max_order, g_table, left_derivatives)
    return [
        (order, right_derivatives.get(order, 0.0) - predicted)
        for order, predicted in predictions.items()
    ]


@dataclass
class VariableBaseTetration(RecurrenceSequence):
    """a_n = n^(a_{n-1}), a_1 = 1."""

    name: str = "variable_base_tetration"
    n0: float = 1.0
    f_n0: float = 1.0
    f_n0_plus_1: float = 2.0

    def g(self, z: float, w: float) -> float:
        return z**w

    def g_inv(self, z: float, f_val: float) -> float:
        if z <= 0.0 or math.isclose(z, 1.0):
            raise ValueError("Variable-base inverse recurrence requires z > 0 and z != 1.")
        return math.log(f_val) / math.log(z)

    def g_z(self, z: float, w: float) -> float:
        return (z**w) * (w / z)

    def g_w(self, z: float, w: float) -> float:
        return (z**w) * math.log(z)

    def g_zz(self, z: float, w: float) -> float:
        return (z**w) * w * (w - 1.0) / (z * z)

    def g_zw(self, z: float, w: float) -> float:
        return (z**w) * (w * math.log(z) + 1.0) / z

    def g_ww(self, z: float, w: float) -> float:
        log_z = math.log(z)
        return (z**w) * log_z * log_z

    def special_value(self, z: float, base_solution: Any) -> float | None:
        if math.isclose(z, self.n0 - 1.0, abs_tol=1e-12):
            return float(base_solution.derivative(self.n0, order=1))
        if math.isclose(z, self.n0 - 2.0, abs_tol=1e-12):
            left_limit = float(base_solution.derivative(self.n0, order=1))
            if left_limit > 0.0:
                return 0.0
        return None

    def _partial_deriv_table(self, max_order: int) -> dict[tuple[int, int], float]:
        """Compute g_{k1,k2}(z,w) for g = z^w at z = n0+1, w = f_n0.

        Uses the recurrence (derived from z * dg/dz = w * g):

            g_{k1,k2} = [(w - k1 + 1) * g_{k1-1,k2} + k2 * g_{k1-1,k2-1}] / z

        with base cases g_{0,k2} = z^w * (ln z)^k2.
        """
        z = self.n0 + 1.0
        w = self.f_n0
        log_z = math.log(z)
        base = z ** w
        table: dict[tuple[int, int], float] = {}
        for k2 in range(max_order + 1):
            table[(0, k2)] = base * (log_z ** k2)
        for k1 in range(1, max_order + 1):
            for k2 in range(max_order + 1):
                val = (w - k1 + 1) * table[(k1 - 1, k2)]
                if k2 >= 1:
                    val += k2 * table[(k1 - 1, k2 - 1)]
                table[(k1, k2)] = val / z
        return table

    def linearized_derivative_constraints(
        self,
        max_order: int,
        deriv_estimates: dict[int, float],
    ) -> list[tuple[int, dict[int, float], float]]:
        g_table = self._partial_deriv_table(max_order)
        return _faa_di_bruno_constraints(max_order, g_table, deriv_estimates)

    def derivative_constraint_residuals(
        self,
        max_order: int,
        left_derivatives: dict[int, float],
        right_derivatives: dict[int, float],
    ) -> list[tuple[int, float]]:
        g_table = self._partial_deriv_table(max_order)
        return _faa_di_bruno_residuals(max_order, g_table, left_derivatives, right_derivatives)


@dataclass
class FixedBaseTetration(RecurrenceSequence):
    """a_n = b^(a_{n-1}), a_1 = b."""

    base: float = 1.3
    name: str = "fixed_base_tetration"
    n0: float = 1.0
    f_n0: float = field(init=False)
    f_n0_plus_1: float = field(init=False)

    def __post_init__(self) -> None:
        if self.base <= 0.0 or math.isclose(self.base, 1.0):
            raise ValueError("The tetration base must be positive and different from 1.")
        self.f_n0 = self.base
        self.f_n0_plus_1 = self.base**self.base
        self.metadata["base"] = self.base

    def g(self, z: float, w: float) -> float:
        del z
        return self.base**w

    def g_inv(self, z: float, f_val: float) -> float:
        del z
        return math.log(f_val, self.base)

    def g_z(self, z: float, w: float) -> float:
        del z, w
        return 0.0

    def g_w(self, z: float, w: float) -> float:
        del z
        return (self.base**w) * math.log(self.base)

    def g_zz(self, z: float, w: float) -> float:
        del z, w
        return 0.0

    def g_zw(self, z: float, w: float) -> float:
        del z, w
        return 0.0

    def g_ww(self, z: float, w: float) -> float:
        del z
        log_base = math.log(self.base)
        return (self.base**w) * log_base * log_base

    def linearized_derivative_constraints(
        self,
        max_order: int,
        deriv_estimates: dict[int, float],
    ) -> list[tuple[int, dict[int, float], float]]:
        # g = b^w does not depend on z, so g_{k1,k2} = 0 for k1 >= 1.
        w = self.f_n0
        log_base = math.log(self.base)
        base_val = self.base ** w
        g_table: dict[tuple[int, int], float] = {}
        for k2 in range(max_order + 1):
            g_table[(0, k2)] = base_val * (log_base ** k2)
        return _faa_di_bruno_constraints(max_order, g_table, deriv_estimates)

    def derivative_constraint_residuals(
        self,
        max_order: int,
        left_derivatives: dict[int, float],
        right_derivatives: dict[int, float],
    ) -> list[tuple[int, float]]:
        w = self.f_n0
        log_base = math.log(self.base)
        base_val = self.base ** w
        g_table: dict[tuple[int, int], float] = {}
        for k2 in range(max_order + 1):
            g_table[(0, k2)] = base_val * (log_base ** k2)
        return _faa_di_bruno_residuals(max_order, g_table, left_derivatives, right_derivatives)


@dataclass
class FactorialType(RecurrenceSequence):
    """a_n = n * a_{n-1}, a_1 = 1."""

    name: str = "factorial_type"
    n0: float = 1.0
    f_n0: float = 1.0
    f_n0_plus_1: float = 2.0

    def g(self, z: float, w: float) -> float:
        return z * w

    def g_inv(self, z: float, f_val: float) -> float:
        return f_val / z

    def g_z(self, z: float, w: float) -> float:
        del z
        return w

    def g_w(self, z: float, w: float) -> float:
        del w
        return z

    def g_zz(self, z: float, w: float) -> float:
        del z, w
        return 0.0

    def g_zw(self, z: float, w: float) -> float:
        del z, w
        return 1.0

    def g_ww(self, z: float, w: float) -> float:
        del z, w
        return 0.0

    def linear_derivative_constraints(self, max_order: int) -> list[tuple[int, dict[int, float], float]]:
        # FactorialType constraints are already fully linear — no estimates needed.
        constraints: list[tuple[int, dict[int, float], float]] = []
        if max_order < 1:
            return constraints

        right_z = self.n0 + 1.0
        for order in range(1, max_order + 1):
            coeffs = {order: right_z}
            constant = 0.0
            if order == 1:
                constant = self.f_n0
            else:
                coeffs[order - 1] = float(order)
            constraints.append((order, coeffs, constant))
        return constraints


@dataclass
class CustomRecurrence(RecurrenceSequence):
    """User-defined recurrence with optional custom derivative propagation."""

    g_fn: Callable[[float, float], float] = field(repr=False, default=lambda z, w: z * w)
    g_inv_fn: Callable[[float, float], float] = field(repr=False, default=lambda z, f_val: f_val / z)
    g_z_fn: Callable[[float, float], float] = field(repr=False, default=lambda z, w: w)
    g_w_fn: Callable[[float, float], float] = field(repr=False, default=lambda z, w: z)
    g_zz_fn: Callable[[float, float], float] = field(repr=False, default=lambda z, w: 0.0)
    g_zw_fn: Callable[[float, float], float] = field(repr=False, default=lambda z, w: 1.0)
    g_ww_fn: Callable[[float, float], float] = field(repr=False, default=lambda z, w: 0.0)
    first_derivative_map_fn: Callable[[float], float] | None = field(repr=False, default=None)
    second_derivative_map_fn: Callable[[float, float], float] | None = field(repr=False, default=None)

    def g(self, z: float, w: float) -> float:
        return self.g_fn(z, w)

    def g_inv(self, z: float, f_val: float) -> float:
        return self.g_inv_fn(z, f_val)

    def g_z(self, z: float, w: float) -> float:
        return self.g_z_fn(z, w)

    def g_w(self, z: float, w: float) -> float:
        return self.g_w_fn(z, w)

    def g_zz(self, z: float, w: float) -> float:
        return self.g_zz_fn(z, w)

    def g_zw(self, z: float, w: float) -> float:
        return self.g_zw_fn(z, w)

    def g_ww(self, z: float, w: float) -> float:
        return self.g_ww_fn(z, w)

    def first_derivative_map(self, left_derivative: float) -> float:
        if self.first_derivative_map_fn is not None:
            return self.first_derivative_map_fn(left_derivative)
        return super().first_derivative_map(left_derivative)

    def second_derivative_map(self, left_first_derivative: float, left_second_derivative: float) -> float:
        if self.second_derivative_map_fn is not None:
            return self.second_derivative_map_fn(left_first_derivative, left_second_derivative)
        return super().second_derivative_map(left_first_derivative, left_second_derivative)
