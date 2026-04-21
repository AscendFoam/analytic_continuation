from __future__ import annotations

from math import gamma

from analytic_continuation.core.sequence import FactorialType
from analytic_continuation.methods.chebyshev import ChebyshevMethod
from analytic_continuation.methods.hermite_cubic import HermiteCubicMethod
from analytic_continuation.methods.hermite_quintic import HermiteQuinticMethod


def test_cubic_gamma_baseline_is_reasonably_close_on_base_interval() -> None:
    seq = FactorialType()
    method = HermiteCubicMethod()
    result = method.solve(seq, target_points=[1.5])

    estimate = result.eval_at[1.5]
    truth = gamma(2.5)
    assert abs(estimate - truth) < 0.1


def test_quintic_gamma_baseline_is_tighter_than_cubic() -> None:
    seq = FactorialType()
    cubic = HermiteCubicMethod().solve(seq, target_points=[1.5])
    quintic = HermiteQuinticMethod().solve(seq, target_points=[1.5])

    truth = gamma(2.5)
    cubic_error = abs(cubic.eval_at[1.5] - truth)
    quintic_error = abs(quintic.eval_at[1.5] - truth)

    assert quintic_error < cubic_error
    assert quintic_error < 0.002


def test_chebyshev_gamma_default_is_reasonably_accurate() -> None:
    seq = FactorialType()
    result = ChebyshevMethod().solve(seq, target_points=[1.5])

    estimate = result.eval_at[1.5]
    truth = gamma(2.5)
    assert abs(estimate - truth) < 0.001


def test_chebyshev_autotune_returns_a_good_configuration() -> None:
    seq = FactorialType()
    tuning = ChebyshevMethod.autotune(
        seq,
        validation_points=[0.5, 1.5],
        reference_fn=lambda z: gamma(z + 1.0),
        candidate_degrees=[6, 10, 12],
        candidate_constraint_orders=[3, 5, 6],
        candidate_regularizations=[1e-6, 1e-8],
    )

    method = tuning.build_method()
    result = method.solve(seq, target_points=[1.5])
    estimate = result.eval_at[1.5]
    truth = gamma(2.5)

    assert tuning.degree in {6, 10, 12}
    assert tuning.constraint_order in {3, 5, 6}
    assert tuning.regularization in {1e-6, 1e-8}
    assert len(tuning.scanned_configs) > 0
    assert abs(estimate - truth) < 0.001


def test_chebyshev_coefficient_space_constraints_are_satisfied() -> None:
    """Verify the coefficient-space method respects endpoint value constraints."""
    seq = FactorialType()
    result = ChebyshevMethod().solve(seq, target_points=[1.0, 2.0])

    assert abs(result.eval_at[1.0] - 1.0) < 1e-10
    assert abs(result.eval_at[2.0] - 2.0) < 1e-10
