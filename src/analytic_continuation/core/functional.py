"""Representations of base-interval solutions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class BaseIntervalSolution(ABC):
    """Common interface for a solution defined on the base interval."""

    interval: tuple[float, float]
    label: str

    @abstractmethod
    def evaluate(self, z: float) -> float:
        """Evaluate the base solution at z."""

    @abstractmethod
    def derivative(self, z: float, order: int = 1) -> float:
        """Evaluate a derivative of the base solution at z."""


@dataclass
class PolynomialBasisSolution(BaseIntervalSolution):
    """Polynomial basis on a single base interval."""

    interval: tuple[float, float]
    coefficients: np.ndarray
    label: str = "polynomial"

    def evaluate(self, z: float) -> float:
        return float(np.polyval(self.coefficients, z))

    def derivative(self, z: float, order: int = 1) -> float:
        if order < 0:
            raise ValueError("Derivative order must be non-negative.")
        coeffs = np.array(self.coefficients, dtype=float)
        for _ in range(order):
            coeffs = np.polyder(coeffs)
        return float(np.polyval(coeffs, z))


@dataclass
class ChebyshevBasisSolution(BaseIntervalSolution):
    """Chebyshev basis solution on a single base interval."""

    interval: tuple[float, float]
    coefficients: np.ndarray
    label: str = "chebyshev"

    def evaluate(self, z: float) -> float:
        series = np.polynomial.chebyshev.Chebyshev(self.coefficients, domain=self.interval)
        return float(series(z))

    def derivative(self, z: float, order: int = 1) -> float:
        if order < 0:
            raise ValueError("Derivative order must be non-negative.")
        series = np.polynomial.chebyshev.Chebyshev(self.coefficients, domain=self.interval)
        return float(series.deriv(m=order)(z))
