"""Placeholder for the variational spline method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from analytic_continuation.core.sequence import RecurrenceSequence
from analytic_continuation.methods.base import ContinuationMethod, ContinuationResult


@dataclass
class VariationalSplineMethod(ContinuationMethod):
    """M4 skeleton: piecewise variational spline."""

    name: str = "variational_spline"

    def solve(
        self,
        seq: RecurrenceSequence,
        target_points: list[float],
        **kwargs: Any,
    ) -> ContinuationResult:
        del seq, target_points, kwargs
        raise NotImplementedError("Variational spline continuation is not implemented yet.")
