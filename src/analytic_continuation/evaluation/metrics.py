"""Metrics for continuation experiments."""

from __future__ import annotations

import math

from analytic_continuation.core.sequence import RecurrenceSequence
from analytic_continuation.methods.base import ContinuationMethod, ContinuationResult


def absolute_error(estimate: float, truth: float) -> float:
    return abs(estimate - truth)


def relative_error(estimate: float, truth: float) -> float:
    if truth == 0.0:
        return math.inf
    return abs(estimate - truth) / abs(truth)


def energy_ratio(energy: float, reference: float) -> float:
    if reference == 0.0:
        return math.inf
    return energy / reference


def recurrence_residual(
    seq: RecurrenceSequence,
    method: ContinuationMethod,
    result: ContinuationResult,
    z: float,
) -> float:
    left_value = method.evaluate(z, seq, result)
    right_value = seq.g(z, method.evaluate(z - 1.0, seq, result))
    return abs(left_value - right_value)
