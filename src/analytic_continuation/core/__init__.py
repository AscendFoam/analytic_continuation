"""Core abstractions for recurrence-defined continuation problems."""

from analytic_continuation.core.energy import StrainEnergy
from analytic_continuation.core.functional import BaseIntervalSolution, ChebyshevBasisSolution, PolynomialBasisSolution
from analytic_continuation.core.sequence import (
    CustomRecurrence,
    FactorialType,
    FixedBaseTetration,
    RecurrenceSequence,
    VariableBaseTetration,
)

__all__ = [
    "BaseIntervalSolution",
    "ChebyshevBasisSolution",
    "CustomRecurrence",
    "FactorialType",
    "FixedBaseTetration",
    "PolynomialBasisSolution",
    "RecurrenceSequence",
    "StrainEnergy",
    "VariableBaseTetration",
]
