"""Chebyshev nodes and differentiation matrices."""

from __future__ import annotations

import numpy as np


def chebyshev_lobatto_nodes(n: int, interval: tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be at least 1.")
    left, right = interval
    angles = np.pi * np.arange(n + 1) / n
    raw_nodes = np.cos(angles)[::-1]
    return 0.5 * (left + right) + 0.5 * (right - left) * raw_nodes


def chebyshev_differentiation_matrix(
    n: int,
    interval: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be at least 1.")
    left, right = interval
    nodes = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.ones(n + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(n + 1))
    x = np.tile(nodes, (n + 1, 1))
    dx = x.T - x
    d = np.outer(c, 1.0 / c) / (dx + np.eye(n + 1))
    d = d - np.diag(np.sum(d, axis=1))
    d = d[::-1, ::-1]
    return (2.0 / (right - left)) * d


def chebyshev_second_differentiation_matrix(
    n: int,
    interval: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    d = chebyshev_differentiation_matrix(n, interval=interval)
    return d @ d


def clenshaw_curtis_weights(
    n: int,
    interval: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be at least 1.")

    left, right = interval
    if n == 1:
        return 0.5 * (right - left) * np.array([1.0, 1.0], dtype=float)

    theta = np.pi * np.arange(n + 1) / n
    weights = np.zeros(n + 1, dtype=float)
    interior = np.arange(1, n)
    v = np.ones(n - 1, dtype=float)

    if n % 2 == 0:
        weights[0] = 1.0 / (n * n - 1.0)
        weights[-1] = weights[0]
        for k in range(1, n // 2):
            v -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k * k - 1.0)
        v -= np.cos(n * theta[interior]) / (n * n - 1.0)
    else:
        weights[0] = 1.0 / (n * n)
        weights[-1] = weights[0]
        for k in range(1, (n - 1) // 2 + 1):
            v -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k * k - 1.0)

    weights[interior] = 2.0 * v / n
    return 0.5 * (right - left) * weights
