"""Placeholder for the regularized iterative method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from analytic_continuation.core.sequence import RecurrenceSequence
from analytic_continuation.methods.base import ContinuationMethod, ContinuationResult


@dataclass
class RegularizedIterationMethod(ContinuationMethod):
    """M5 skeleton: regularized residual minimization."""

    name: str = "regularized_iter"

    def solve(
        self,
        seq: RecurrenceSequence,
        target_points: list[float],
        **kwargs: Any,
    ) -> ContinuationResult:
        del seq, target_points, kwargs
        raise NotImplementedError("Regularized iterative continuation is not implemented yet.")
