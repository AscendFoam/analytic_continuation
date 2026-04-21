"""Minimal plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_curve(x_values: list[float], y_values: list[float], title: str, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_values, y_values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_series(
    series: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    title: str,
    x_label: str,
    y_label: str,
    output_path: str | Path,
    *,
    logx: bool = False,
    logy: bool = False,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, (x_values, y_values) in series.items():
        ax.plot(list(x_values), list(y_values), marker="o", linewidth=1.5, label=label)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_heatmap(
    matrix: Sequence[Sequence[float]],
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str,
    x_label: str,
    y_label: str,
    output_path: str | Path,
    *,
    colorbar_label: str,
    cmap: str = "viridis",
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    array = np.array(matrix, dtype=float)
    masked = np.ma.masked_invalid(array)

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(masked, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(list(x_labels))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(list(y_labels))

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
