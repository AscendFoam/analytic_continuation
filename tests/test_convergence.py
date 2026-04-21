from __future__ import annotations

import math

from analytic_continuation.evaluation.convergence import estimate_convergence_rate, estimate_empirical_rate


def test_empirical_rate_matches_doubling_formula() -> None:
    direct = estimate_convergence_rate(1e-2, 2.5e-3)
    general = estimate_empirical_rate(8.0, 1e-2, 16.0, 2.5e-3)
    assert math.isclose(direct, general)


def test_empirical_rate_handles_non_doubling_sizes() -> None:
    rate = estimate_empirical_rate(6.0, 1e-2, 10.0, 3e-3)
    assert rate > 1.0
