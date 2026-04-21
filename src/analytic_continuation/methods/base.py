"""Base classes for continuation methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from analytic_continuation.core.functional import BaseIntervalSolution
from analytic_continuation.core.sequence import RecurrenceSequence


@dataclass
class ContinuationResult:
    """Container for a continuation solve."""

    method_name: str
    sequence_name: str
    optimal_params: dict[str, float]
    strain_energy: float
    eval_at: dict[float, float]
    basis_coefficients: np.ndarray
    base_solution: BaseIntervalSolution
    metadata: dict[str, Any] = field(default_factory=dict)


class ContinuationMethod(ABC):
    """Common interface for numerical continuation methods."""

    name: str = "abstract_method"

    @abstractmethod
    def solve(
        self,
        seq: RecurrenceSequence,
        target_points: list[float],
        **kwargs: Any,
    ) -> ContinuationResult:
        """Solve the continuation problem for a sequence."""

    def evaluate(self, z: float, seq: RecurrenceSequence, result: ContinuationResult) -> float:
        """Evaluate a solved continuation result at an arbitrary point."""

        special = seq.special_value(z, result.base_solution)
        if special is not None:
            return special

        left, right = result.base_solution.interval
        if left <= z <= right:
            return result.base_solution.evaluate(z)

        if z > right:
            current = z
            forward_steps: list[float] = []
            while current > right:
                forward_steps.append(current)
                current -= 1.0
            value = result.base_solution.evaluate(current)
            for step in reversed(forward_steps):
                value = seq.g(step, value)
            return value

        current = z
        backward_steps: list[float] = []
        while current < left:
            backward_steps.append(current + 1.0)
            current += 1.0
        value = result.base_solution.evaluate(current)
        for step in reversed(backward_steps):
            value = seq.g_inv(step, value)
        return value
