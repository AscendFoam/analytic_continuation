"""Stability sweep for the coefficient-space Chebyshev method."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from math import gamma, log10
import os
from pathlib import Path
from statistics import median
from time import perf_counter
import sys
from tempfile import gettempdir


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "analytic_continuation_mpl"))

from analytic_continuation.core.sequence import FactorialType
from analytic_continuation.evaluation.metrics import absolute_error
from analytic_continuation.evaluation.visualization import plot_heatmap, plot_series
from analytic_continuation.methods.chebyshev import ChebyshevMethod


def _build_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / "exp_chebyshev_stability" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    return output_dir


def _run_scan(
    eval_point: float,
    truth_value: float,
    degrees: list[int],
    constraint_orders: list[int],
    regularizations: list[float],
) -> list[dict[str, float | int | str]]:
    sequence = FactorialType()
    rows: list[dict[str, float | int | str]] = []

    for regularization in regularizations:
        for degree in degrees:
            for constraint_order in constraint_orders:
                feasible = constraint_order <= degree - 2
                row: dict[str, float | int | str] = {
                    "degree": degree,
                    "constraint_order": constraint_order,
                    "regularization": regularization,
                    "feasible": int(feasible),
                    "eval_point": eval_point,
                    "estimate": "",
                    "truth": truth_value,
                    "abs_error": "",
                    "log10_abs_error": "",
                    "strain_energy": "",
                    "elapsed_ms": "",
                    "schur_condition_number": "",
                    "log10_schur_condition_number": "",
                    "constraint_count": "",
                    "status": "skipped_infeasible",
                }
                if not feasible:
                    rows.append(row)
                    continue

                method = ChebyshevMethod(
                    degree=degree,
                    constraint_order=constraint_order,
                    regularization=regularization,
                )
                started = perf_counter()
                try:
                    result = method.solve(sequence, target_points=[eval_point])
                except Exception as exc:  # noqa: BLE001
                    elapsed_ms = (perf_counter() - started) * 1000.0
                    row.update(
                        {
                            "elapsed_ms": elapsed_ms,
                            "status": f"failed:{type(exc).__name__}",
                        }
                    )
                    rows.append(row)
                    continue

                elapsed_ms = (perf_counter() - started) * 1000.0
                estimate = result.eval_at[eval_point]
                error = absolute_error(estimate, truth_value)
                schur_condition = float(result.metadata["schur_condition_number"])
                row.update(
                    {
                        "estimate": estimate,
                        "abs_error": error,
                        "log10_abs_error": log10(error),
                        "strain_energy": result.strain_energy,
                        "elapsed_ms": elapsed_ms,
                        "schur_condition_number": schur_condition,
                        "log10_schur_condition_number": log10(schur_condition),
                        "constraint_count": result.metadata["constraint_count"],
                        "status": "ok",
                    }
                )
                rows.append(row)

    return rows


def _make_heatmap_matrix(
    rows: list[dict[str, float | int | str]],
    *,
    regularization: float,
    metric: str,
    degrees: list[int],
    constraint_orders: list[int],
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for constraint_order in constraint_orders:
        row_values: list[float] = []
        for degree in degrees:
            candidates = [
                item for item in rows
                if item["regularization"] == regularization
                and item["degree"] == degree
                and item["constraint_order"] == constraint_order
                and item["status"] == "ok"
            ]
            if not candidates:
                row_values.append(float("nan"))
            else:
                row_values.append(float(candidates[0][metric]))
        matrix.append(row_values)
    return matrix


def _write_outputs(
    output_dir: Path,
    rows: list[dict[str, float | int | str]],
    *,
    eval_point: float,
    truth_value: float,
    degrees: list[int],
    constraint_orders: list[int],
    regularizations: list[float],
) -> None:
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "sequence": "factorial_type",
                "eval_point": eval_point,
                "truth_value": truth_value,
                "degrees": degrees,
                "constraint_orders": constraint_orders,
                "regularizations": regularizations,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "degree",
                "constraint_order",
                "regularization",
                "feasible",
                "eval_point",
                "estimate",
                "truth",
                "abs_error",
                "log10_abs_error",
                "strain_energy",
                "elapsed_ms",
                "schur_condition_number",
                "log10_schur_condition_number",
                "constraint_count",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    successful_rows = [row for row in rows if row["status"] == "ok"]
    if not successful_rows:
        raise RuntimeError("The Chebyshev stability scan produced no successful runs.")

    best_overall = min(successful_rows, key=lambda row: float(row["abs_error"]))
    best_by_degree: dict[int, dict[str, float | int | str]] = {}
    for degree in degrees:
        degree_rows = [row for row in successful_rows if row["degree"] == degree]
        if degree_rows:
            best_by_degree[degree] = min(degree_rows, key=lambda row: float(row["abs_error"]))

    plot_series(
        {
            "best_error_per_degree": (
                list(best_by_degree.keys()),
                [float(row["abs_error"]) for row in best_by_degree.values()],
            ),
        },
        title="Best Chebyshev Error by Degree",
        x_label="Polynomial degree",
        y_label="Absolute error at z=1.5",
        output_path=output_dir / "figures" / "best_error_by_degree.png",
        logy=True,
    )

    plot_series(
        {
            "best_condition_per_degree": (
                list(best_by_degree.keys()),
                [float(row["schur_condition_number"]) for row in best_by_degree.values()],
            ),
        },
        title="Best Schur Condition Number by Degree",
        x_label="Polynomial degree",
        y_label="Schur condition number",
        output_path=output_dir / "figures" / "best_condition_by_degree.png",
        logy=True,
    )

    for regularization in regularizations:
        suffix = f"{regularization:.0e}".replace("+", "")
        plot_heatmap(
            _make_heatmap_matrix(
                rows,
                regularization=regularization,
                metric="log10_abs_error",
                degrees=degrees,
                constraint_orders=constraint_orders,
            ),
            x_labels=[str(value) for value in degrees],
            y_labels=[str(value) for value in constraint_orders],
            title=f"log10(abs error), reg={regularization:.0e}",
            x_label="degree",
            y_label="constraint_order",
            output_path=output_dir / "figures" / f"heatmap_error_reg_{suffix}.png",
            colorbar_label="log10(abs error)",
            cmap="viridis_r",
        )
        plot_heatmap(
            _make_heatmap_matrix(
                rows,
                regularization=regularization,
                metric="log10_schur_condition_number",
                degrees=degrees,
                constraint_orders=constraint_orders,
            ),
            x_labels=[str(value) for value in degrees],
            y_labels=[str(value) for value in constraint_orders],
            title=f"log10(schur cond), reg={regularization:.0e}",
            x_label="degree",
            y_label="constraint_order",
            output_path=output_dir / "figures" / f"heatmap_condition_reg_{suffix}.png",
            colorbar_label="log10(schur cond)",
            cmap="magma",
        )

    best_degree_rows = list(best_by_degree.values())
    median_degree = median(float(row["degree"]) for row in best_degree_rows)
    with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
        file.write("Chebyshev stability scan completed on FactorialType.\n")
        file.write(f"Evaluation point: z={eval_point}\n")
        file.write(f"Ground truth: {truth_value:.12f}\n")
        file.write(
            "Best overall configuration: "
            f"degree={best_overall['degree']}, "
            f"constraint_order={best_overall['constraint_order']}, "
            f"regularization={float(best_overall['regularization']):.0e}, "
            f"abs_error={float(best_overall['abs_error']):.6e}, "
            f"schur_condition_number={float(best_overall['schur_condition_number']):.6e}\n"
        )
        file.write(
            "Best-per-degree median degree: "
            f"{median_degree:.1f}\n"
        )
        file.write("Best-per-degree table:\n")
        for degree in degrees:
            row = best_by_degree.get(degree)
            if row is None:
                continue
            file.write(
                f"  degree={degree:>2d}: order={int(row['constraint_order'])}, "
                f"reg={float(row['regularization']):.0e}, "
                f"abs_error={float(row['abs_error']):.6e}, "
                f"schur_cond={float(row['schur_condition_number']):.6e}\n"
            )


def main() -> None:
    eval_point = 1.5
    truth_value = gamma(eval_point + 1.0)
    degrees = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28]
    constraint_orders = [1, 2, 3, 4, 5, 6, 7, 8]
    regularizations = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6]

    output_dir = _build_output_dir()
    rows = _run_scan(
        eval_point=eval_point,
        truth_value=truth_value,
        degrees=degrees,
        constraint_orders=constraint_orders,
        regularizations=regularizations,
    )
    _write_outputs(
        output_dir,
        rows,
        eval_point=eval_point,
        truth_value=truth_value,
        degrees=degrees,
        constraint_orders=constraint_orders,
        regularizations=regularizations,
    )


if __name__ == "__main__":
    main()
