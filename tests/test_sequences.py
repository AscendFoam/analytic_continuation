from __future__ import annotations

import math

from analytic_continuation.core.sequence import FactorialType, FixedBaseTetration, VariableBaseTetration


def test_factorial_first_derivative_map_matches_formula() -> None:
    seq = FactorialType()
    assert math.isclose(seq.first_derivative_map(3.0), 7.0)


def test_variable_base_first_derivative_map_matches_formula() -> None:
    seq = VariableBaseTetration()
    expected = 2.0 * math.log(2.0) * 1.5 + 1.0
    assert math.isclose(seq.first_derivative_map(1.5), expected)


def test_fixed_base_inverse_recovers_exponent() -> None:
    seq = FixedBaseTetration(base=1.3)
    w = 2.1
    value = seq.g(2.0, w)
    recovered = seq.g_inv(2.0, value)
    assert math.isclose(recovered, w)
