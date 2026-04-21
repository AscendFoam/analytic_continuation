from __future__ import annotations

import math

from analytic_continuation.core.sequence import FactorialType, VariableBaseTetration
from analytic_continuation.methods.chebyshev import ChebyshevMethod
from analytic_continuation.methods.hermite_cubic import HermiteCubicMethod
from analytic_continuation.methods.hermite_quintic import HermiteQuinticMethod


def test_cubic_method_matches_boundary_values() -> None:
    seq = FactorialType()
    method = HermiteCubicMethod()
    result = method.solve(seq, target_points=[1.0, 2.0])

    assert math.isclose(result.eval_at[1.0], 1.0)
    assert math.isclose(result.eval_at[2.0], 2.0)
    assert result.strain_energy > 0.0


def test_cubic_method_forward_propagates_integer_values() -> None:
    seq = FactorialType()
    method = HermiteCubicMethod()
    result = method.solve(seq, target_points=[3.0, 4.0])

    assert math.isclose(result.eval_at[3.0], 6.0)
    assert math.isclose(result.eval_at[4.0], 24.0)


def test_quintic_method_matches_boundary_values() -> None:
    seq = FactorialType()
    method = HermiteQuinticMethod()
    result = method.solve(seq, target_points=[1.0, 2.0])

    assert math.isclose(result.eval_at[1.0], 1.0)
    assert math.isclose(result.eval_at[2.0], 2.0)
    assert result.strain_energy > 0.0


def test_chebyshev_method_forward_propagates_integer_values() -> None:
    seq = FactorialType()
    method = ChebyshevMethod()
    result = method.solve(seq, target_points=[3.0, 4.0])

    assert math.isclose(result.eval_at[3.0], 6.0)
    assert math.isclose(result.eval_at[4.0], 24.0)


def test_chebyshev_overconstrained_raises() -> None:
    """degree=4, constraint_order=8 produces more constraints than DOFs."""
    seq = FactorialType()
    method = ChebyshevMethod(degree=4, constraint_order=8)
    try:
        method.solve(seq, target_points=[1.5])
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Too many constraints" in str(exc)


def test_variable_base_tetration_special_backward_values() -> None:
    seq = VariableBaseTetration()
    method = HermiteCubicMethod()
    result = method.solve(seq, target_points=[0.0, -1.0])

    assert math.isclose(result.eval_at[0.0], result.optimal_params["left_derivative"])
    assert math.isclose(result.eval_at[-1.0], 0.0)


def test_derivative_constraint_api_separates_linear_and_linearized_constraints() -> None:
    seq = VariableBaseTetration()
    linear_constraints = seq.linear_derivative_constraints(max_order=4)
    linearized_constraints = seq.linearized_derivative_constraints(
        max_order=4,
        deriv_estimates={1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0},
    )

    assert len(linear_constraints) < len(linearized_constraints)
    assert {order for order, _, _ in linear_constraints} == {1}
    assert {order for order, _, _ in linearized_constraints} == {1, 2, 3, 4}
