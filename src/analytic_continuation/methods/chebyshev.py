"""Chebyshev spectral continuation method (coefficient-space formulation)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Any, Callable

import numpy as np
from numpy.polynomial import chebyshev
from scipy.linalg import cho_solve, cho_factor

from analytic_continuation.core.functional import ChebyshevBasisSolution
from analytic_continuation.core.sequence import RecurrenceSequence
from analytic_continuation.methods.base import ContinuationMethod, ContinuationResult


# ------------------------------------------------------------------
# Stable Chebyshev endpoint derivative weights
# ------------------------------------------------------------------

def _double_factorial(n: int) -> int:
    """Compute (2m-1)!! = 1 * 3 * 5 * ... * n."""
    result = 1
    for i in range(1, n + 1, 2):
        result *= i
    return result


def _chebyshev_endpoint_weight(k: int, order: int, sign: float) -> float:
    """Evaluate T^(order)_k at x = ±1 (unnormalised, on [-1,1]).

    Uses the exact formula:
        T^(m)_k(1) = k^2 * prod_{j=1}^{m-1}(k^2 - j^2) / (2m-1)!!
    and T^(m)_k(-1) = (-1)^{k+m} T^(m)_k(1).

    Parameters
    ----------
    k : int
        Chebyshev polynomial index.
    order : int
        Derivative order (0 = function value).
    sign : float
        +1.0 for x = +1, -1.0 for x = -1.
    """
    if order == 0:
        return 1.0 if sign > 0 else ((-1.0) ** k)

    if k < order:
        return 0.0

    val = float(k * k)
    for j in range(1, order):
        val *= float(k * k - j * j)
    val /= float(_double_factorial(2 * order - 1))

    if sign < 0:
        val *= (-1.0) ** (k + order)
    return val


# ------------------------------------------------------------------
# Analytic Gram matrix for the unweighted L^2 inner product on [-1,1]
# ------------------------------------------------------------------

def _integral_Tk(k: int) -> float:
    """Integral_{-1}^{1} T_k(x) dx."""
    if k == 0:
        return 2.0
    if k % 2 == 1:
        return 0.0
    return 2.0 / (1.0 - k * k)


def _gram_matrix(n: int) -> np.ndarray:
    """(n x n) Gram matrix G[k,l] = integral T_k T_l dx on [-1,1]."""
    G = np.zeros((n, n))
    for k in range(n):
        for l in range(n):
            G[k, l] = 0.5 * (_integral_Tk(k + l) + _integral_Tk(abs(k - l)))
    return G


# ------------------------------------------------------------------
# Coefficient-space second derivative matrix
# ------------------------------------------------------------------

def _cheb_second_deriv_matrix(deg: int) -> np.ndarray:
    """(deg-1) x (deg+1) matrix mapping Chebyshev coefficients to
    second-derivative coefficients, using numpy's stable chebder."""
    N = deg
    D2 = np.zeros((max(1, N - 1), N + 1))
    for j in range(N + 1):
        c = np.zeros(N + 1)
        c[j] = 1.0
        d1 = chebyshev.chebder(c)
        d2 = chebyshev.chebder(d1)
        D2[: len(d2), j] = d2
    return D2


# ------------------------------------------------------------------
# The method
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ChebyshevTuningResult:
    """Result of a discrete Chebyshev hyperparameter scan."""

    degree: int
    constraint_order: int
    regularization: float
    max_abs_error: float
    mean_abs_error: float
    schur_condition_number: float
    elapsed_ms: float
    scanned_configs: tuple[dict[str, float | int | str], ...] = field(default_factory=tuple, repr=False)

    def build_method(self) -> "ChebyshevMethod":
        return ChebyshevMethod(
            degree=self.degree,
            constraint_order=self.constraint_order,
            regularization=self.regularization,
        )

@dataclass
class ChebyshevMethod(ContinuationMethod):
    """M3: Chebyshev spectral method in coefficient space.

    The unknown is the vector of Chebyshev coefficients
    ``a = [a_0, ..., a_N]`` for the series
    ``f(z) = sum a_k T_k(x(z))`` with ``x`` mapping ``[n0, n0+1]`` to
    ``[-1, 1]``.

    Energy matrix: computed analytically via the Gram matrix and the
    coefficient-space second-derivative matrix (no nodal
    differentiation matrices).

    Constraints: endpoint value and derivative-propagation constraints
    use the exact formulas ``T^(m)_k(+/-1)``, avoiding the
    ill-conditioned repeated-multiplication of nodal differentiation
    matrices.

    Solver: Schur-complement (Cholesky-based) elimination.

    Constraint modes
    ----------------
    By default the method only applies constraints returned by
    ``linear_derivative_constraints`` so the quadratic program remains
    genuinely linear-constrained. Set ``use_linearized_constraints=True``
    to add higher-order nonlinear propagation constraints through iterative
    Faa di Bruno linearisation.
    """

    name: str = "chebyshev"
    degree: int = 12
    constraint_order: int = 5
    regularization: float = 1e-6
    refinement_iterations: int = 3
    use_linearized_constraints: bool = False

    # ------------------------------------------------------------------
    # Energy matrix
    # ------------------------------------------------------------------

    def _build_energy_matrix(self, interval_length: float) -> np.ndarray:
        N = self.degree
        D2 = _cheb_second_deriv_matrix(N)
        G = _gram_matrix(D2.shape[0])
        raw = D2.T @ G @ D2
        scale = (2.0 / interval_length) ** 3
        energy = scale * raw
        avg = max(np.trace(energy) / (N + 1), 1e-30)
        return energy + self.regularization * avg * np.eye(N + 1)

    # ------------------------------------------------------------------
    # Constraints in coefficient space
    # ------------------------------------------------------------------

    def _build_constraints(
        self, seq: RecurrenceSequence, scale: float,
        *, deriv_estimates: dict[int, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        N = self.degree
        rows: list[np.ndarray] = []
        rhs: list[float] = []

        # Value constraints: f(n0) and f(n0+1)
        rows.append(np.array([(-1.0) ** k for k in range(N + 1)]))
        rhs.append(seq.f_n0)
        rows.append(np.ones(N + 1))
        rhs.append(seq.f_n0_plus_1)

        # Derivative-propagation constraints. True linear constraints are safe
        # for the QP; linearized constraints are opt-in because they depend on
        # the supplied derivative estimates.
        if deriv_estimates is None:
            deriv_constraints = seq.linear_derivative_constraints(self.constraint_order)
        else:
            deriv_constraints = seq.linearized_derivative_constraints(
                self.constraint_order,
                deriv_estimates,
            )
        for order, left_coeffs, constant in deriv_constraints:
            row = np.zeros(N + 1)
            for k in range(N + 1):
                row[k] = (scale ** order) * _chebyshev_endpoint_weight(k, order, +1.0)
            for left_order, coeff in left_coeffs.items():
                for k in range(N + 1):
                    row[k] -= coeff * (scale ** left_order) * _chebyshev_endpoint_weight(k, left_order, -1.0)
            rows.append(row)
            rhs.append(constant)

        constraints = np.vstack(rows)
        rhs_arr = np.array(rhs, dtype=float)
        for i in range(constraints.shape[0]):
            norm = np.linalg.norm(constraints[i])
            if norm > 0:
                constraints[i] /= norm
                rhs_arr[i] /= norm
        return constraints, rhs_arr

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def _solve_qp(
        self, energy: np.ndarray, constraints: np.ndarray, rhs: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Solve min a^T H a  s.t.  C a = b  via Schur complement.

        Returns (coefficients, schur_condition_number).
        """
        n = energy.shape[0]
        H = 2.0 * energy
        try:
            cf = cho_factor(H)
        except np.linalg.LinAlgError:
            H = H + 1e-6 * np.trace(H) / n * np.eye(n)
            cf = cho_factor(H)
        E = cho_solve(cf, constraints.T)
        S = constraints @ E
        schur_cond = float(np.linalg.cond(S))
        lagrange = np.linalg.solve(S, rhs)
        coeffs = cho_solve(cf, constraints.T @ lagrange)
        return coeffs, schur_cond

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @classmethod
    def autotune(
        cls,
        seq: RecurrenceSequence,
        validation_points: list[float],
        reference_fn: Callable[[float], float],
        *,
        candidate_degrees: list[int] | None = None,
        candidate_constraint_orders: list[int] | None = None,
        candidate_regularizations: list[float] | None = None,
    ) -> ChebyshevTuningResult:
        """Grid-search Chebyshev hyperparameters against a reference function."""

        if not validation_points:
            raise ValueError("validation_points must not be empty.")

        degrees = candidate_degrees or [6, 8, 10, 12, 16, 20, 24]
        constraint_orders = candidate_constraint_orders or [2, 3, 4, 5, 6]
        regularizations = candidate_regularizations or [1e-8, 1e-6]

        scanned: list[dict[str, float | int | str]] = []
        best_key: tuple[float, float, float, float] | None = None
        best_result: ChebyshevTuningResult | None = None

        for regularization in regularizations:
            for degree in degrees:
                for constraint_order in constraint_orders:
                    if constraint_order > degree - 2:
                        continue

                    method = cls(
                        degree=degree,
                        constraint_order=constraint_order,
                        regularization=regularization,
                    )
                    started = perf_counter()
                    result = method.solve(seq, target_points=validation_points)
                    elapsed_ms = (perf_counter() - started) * 1000.0

                    errors = [
                        abs(result.eval_at[point] - reference_fn(point))
                        for point in validation_points
                    ]
                    max_abs_error = float(max(errors))
                    mean_abs_error = float(sum(errors) / len(errors))
                    schur_condition_number = float(result.metadata["schur_condition_number"])

                    scanned.append(
                        {
                            "degree": degree,
                            "constraint_order": constraint_order,
                            "regularization": regularization,
                            "max_abs_error": max_abs_error,
                            "mean_abs_error": mean_abs_error,
                            "schur_condition_number": schur_condition_number,
                            "elapsed_ms": elapsed_ms,
                        }
                    )

                    key = (
                        max_abs_error,
                        mean_abs_error,
                        math.log10(max(schur_condition_number, 1.0)),
                        elapsed_ms,
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best_result = ChebyshevTuningResult(
                            degree=degree,
                            constraint_order=constraint_order,
                            regularization=regularization,
                            max_abs_error=max_abs_error,
                            mean_abs_error=mean_abs_error,
                            schur_condition_number=schur_condition_number,
                            elapsed_ms=elapsed_ms,
                            scanned_configs=(),
                        )

        if best_result is None:
            raise RuntimeError("No feasible Chebyshev configurations were scanned.")

        return ChebyshevTuningResult(
            degree=best_result.degree,
            constraint_order=best_result.constraint_order,
            regularization=best_result.regularization,
            max_abs_error=best_result.max_abs_error,
            mean_abs_error=best_result.mean_abs_error,
            schur_condition_number=best_result.schur_condition_number,
            elapsed_ms=best_result.elapsed_ms,
            scanned_configs=tuple(scanned),
        )

    def _extract_endpoint_derivatives(
        self, coeffs: np.ndarray, scale: float, max_order: int,
    ) -> dict[int, float]:
        """Extract f^(k)(n0) for k = 1, ..., max_order from Chebyshev coefficients."""
        N = len(coeffs) - 1
        derivs: dict[int, float] = {}
        for order in range(1, max_order + 1):
            derivs[order] = float(sum(
                coeffs[k] * (scale ** order) * _chebyshev_endpoint_weight(k, order, -1.0)
                for k in range(N + 1)
            ))
        return derivs

    def solve(
        self,
        seq: RecurrenceSequence,
        target_points: list[float],
        **kwargs: Any,
    ) -> ContinuationResult:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")

        # Check the maximum constraint count before building the KKT system.
        # The default mode only uses truly linear constraints; the optional
        # linearized mode can generate one derivative constraint per order.
        if self.use_linearized_constraints:
            n_constraints_max = 2 + self.constraint_order
        else:
            n_constraints_max = 2 + len(seq.linear_derivative_constraints(self.constraint_order))
        if n_constraints_max >= self.degree + 1:
            raise ValueError(
                f"Too many constraints (up to {n_constraints_max}) for degree {self.degree}. "
                f"Need degree + 1 > {n_constraints_max}."
            )

        n0, n1 = seq.base_interval
        L = n1 - n0
        scale = 2.0 / L

        energy = self._build_energy_matrix(L)

        # Optional iterative linearisation: first solve with only inherently
        # linear constraints, then extract endpoint derivatives and use them
        # to linearise higher-order Faa di Bruno terms.
        deriv_estimates: dict[int, float] | None = None
        coeffs: np.ndarray = np.zeros(self.degree + 1)
        schur_cond: float = float("inf")
        constraint_count: int = 0

        iterations = max(1, self.refinement_iterations if self.use_linearized_constraints else 1)
        for iteration in range(iterations):
            constraints, rhs = self._build_constraints(
                seq,
                scale,
                deriv_estimates=deriv_estimates if self.use_linearized_constraints else None,
            )
            constraint_count = constraints.shape[0]
            coeffs, schur_cond = self._solve_qp(energy, constraints, rhs)

            if self.use_linearized_constraints and iteration < iterations - 1:
                deriv_estimates = self._extract_endpoint_derivatives(
                    coeffs, scale, self.constraint_order,
                )

        base_solution = ChebyshevBasisSolution(
            interval=seq.base_interval,
            coefficients=np.array(coeffs, dtype=float),
            label="chebyshev_coefficient",
        )

        left_deriv = sum(
            coeffs[k] * scale * _chebyshev_endpoint_weight(k, 1, -1.0)
            for k in range(self.degree + 1)
        )
        right_deriv = sum(
            coeffs[k] * scale * _chebyshev_endpoint_weight(k, 1, +1.0)
            for k in range(self.degree + 1)
        )
        strain_energy = float(coeffs @ energy @ coeffs)

        result = ContinuationResult(
            method_name=self.name,
            sequence_name=seq.name,
            optimal_params={
                "left_derivative": float(left_deriv),
                "right_derivative": float(right_deriv),
            },
            strain_energy=strain_energy,
            eval_at={},
            basis_coefficients=np.array(coeffs, dtype=float),
            base_solution=base_solution,
            metadata={
                "degree": self.degree,
                "constraint_order": self.constraint_order,
                "constraint_mode": "linearized" if self.use_linearized_constraints else "linear",
                "use_linearized_constraints": self.use_linearized_constraints,
                "refinement_iterations": float(iterations),
                "schur_condition_number": schur_cond,
                "constraint_count": float(constraint_count),
            },
        )
        result.eval_at = {point: self.evaluate(point, seq, result) for point in target_points}
        return result
