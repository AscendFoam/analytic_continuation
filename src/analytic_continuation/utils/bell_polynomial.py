"""Bell polynomial utilities for higher-order derivative constraints."""

from __future__ import annotations


def partial_bell_polynomial(n: int, k: int, x: list[float]) -> float:
    """Compute the partial exponential Bell polynomial B_{n,k}."""

    if n == 0 and k == 0:
        return 1.0
    if n == 0 or k == 0:
        return 0.0

    table = [[0.0 for _ in range(k + 1)] for _ in range(n + 1)]
    table[0][0] = 1.0

    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            total = 0.0
            for m in range(1, i - j + 2):
                total += comb(i - 1, m - 1) * x[m - 1] * table[i - m][j - 1]
            table[i][j] = total
    return table[n][k]


def comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    numerator = 1
    denominator = 1
    for idx in range(1, k + 1):
        numerator *= n - (k - idx)
        denominator *= idx
    return numerator // denominator
