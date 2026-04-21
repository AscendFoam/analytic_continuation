"""Convergence analysis helpers."""

from __future__ import annotations

import math


def estimate_convergence_rate(error_n: float, error_2n: float) -> float:
    if error_n <= 0.0 or error_2n <= 0.0:
        raise ValueError("Errors must be positive to estimate a convergence rate.")
    return math.log(error_n / error_2n, 2.0)


def estimate_empirical_rate(
    size_a: float,
    error_a: float,
    size_b: float,
    error_b: float,
) -> float:
    """Estimate convergence rate between two arbitrary discretization sizes."""

    if size_a <= 0.0 or size_b <= 0.0:
        raise ValueError("Discretization sizes must be positive.")
    if error_a <= 0.0 or error_b <= 0.0:
        raise ValueError("Errors must be positive to estimate a convergence rate.")
    if math.isclose(size_a, size_b):
        raise ValueError("Discretization sizes must differ.")
    return math.log(error_a / error_b) / math.log(size_b / size_a)
