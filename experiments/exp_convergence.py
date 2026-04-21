"""Convergence experiment for the currently implemented methods."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from math import gamma
import os
from pathlib import Path
from time import perf_counter
import sys
from tempfile import gettempdir


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "analytic_continuation_mpl"))

from analytic_continuation.core.sequence import FactorialType
from analytic_continuation.evaluation.convergence import estimate_empirical_rate
from analytic_continuation.evaluation.metrics import absolute_error
from analytic_continuation.evaluation.visualization import plot_series
from analytic_continuation.methods.chebyshev import ChebyshevMethod
from analytic_continuation.methods.hermite_cubic import HermiteCubicMethod
from analytic_continuation.methods.hermite_quintic import HermiteQuinticMethod


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--autotune-chebyshev",
        action="store_true",
        help="Add one autotuned Chebyshev run alongside the fixed scan.",
    )
    parser.add_argument(
        "--autotune-points",
        default="0.5,1.5",
        help="Comma-separated validation points used by autotune.",
    )
    parser.add_argument(
        "--candidate-degrees",
        default="4,6,8,10,12,16,20,24",
        help="Comma-separated Chebyshev degrees considered by autotune.",
    )
    parser.add_argument(
        "--candidate-orders",
        default="2,3,4,5,6",
        help="Comma-separated constraint orders considered by autotune.",
    )
    parser.add_argument(
        "--max-constraint-order",
        type=int,
        default=6,
        help="Maximum constraint order used by the fixed Chebyshev degree scan.",
    )
    parser.add_argument(
        "--candidate-regularizations",
        default="1e-8,1e-6",
        help="Comma-separated regularization values considered by autotune.",
    )
    return parser


def _build_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / "exp_convergence" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    return output_dir


def _method_rows(
    eval_point: float,
    truth_value: float,
    chebyshev_degrees: list[int],
    max_constraint_order: int,
    *,
    autotuned_chebyshev: ChebyshevMethod | None = None,
    autotune_metadata: dict[str, float | int | str] | None = None,
) -> list[dict[str, float | int | str]]:
    sequence = FactorialType()
    rows: list[dict[str, float | int | str]] = []

    fixed_methods = [
        ("hermite_cubic", 4, HermiteCubicMethod()),
        ("hermite_quintic", 6, HermiteQuinticMethod()),
    ]
    for method_name, dof, method in fixed_methods:
        started = perf_counter()
        result = method.solve(sequence, target_points=[eval_point])
        elapsed_ms = (perf_counter() - started) * 1000.0
        estimate = result.eval_at[eval_point]
        rows.append(
            {
                "method": method_name,
                "family": method_name,
                "dof": dof,
                "degree": "",
                "constraint_order": "",
                "eval_point": eval_point,
                "estimate": estimate,
                "truth": truth_value,
                "abs_error": absolute_error(estimate, truth_value),
                "strain_energy": result.strain_energy,
                "elapsed_ms": elapsed_ms,
                "empirical_rate": "",
            }
        )

    previous_chebyshev_row: dict[str, float | int | str] | None = None
    for degree in chebyshev_degrees:
        constraint_order = min(max_constraint_order, degree - 2)
        method = ChebyshevMethod(degree=degree, constraint_order=constraint_order)
        started = perf_counter()
        result = method.solve(sequence, target_points=[eval_point])
        elapsed_ms = (perf_counter() - started) * 1000.0
        estimate = result.eval_at[eval_point]
        row: dict[str, float | int | str] = {
            "method": "chebyshev",
            "family": "chebyshev",
            "dof": degree + 1,
            "degree": degree,
            "constraint_order": constraint_order,
            "eval_point": eval_point,
            "estimate": estimate,
            "truth": truth_value,
            "abs_error": absolute_error(estimate, truth_value),
            "strain_energy": result.strain_energy,
            "elapsed_ms": elapsed_ms,
            "empirical_rate": "",
        }
        if previous_chebyshev_row is not None:
            row["empirical_rate"] = estimate_empirical_rate(
                float(previous_chebyshev_row["dof"]),
                float(previous_chebyshev_row["abs_error"]),
                float(row["dof"]),
                float(row["abs_error"]),
            )
        rows.append(row)
        previous_chebyshev_row = row

    if autotuned_chebyshev is not None:
        started = perf_counter()
        result = autotuned_chebyshev.solve(sequence, target_points=[eval_point])
        elapsed_ms = (perf_counter() - started) * 1000.0
        estimate = result.eval_at[eval_point]
        rows.append(
            {
                "method": "chebyshev_autotuned",
                "family": "chebyshev_autotuned",
                "dof": autotuned_chebyshev.degree + 1,
                "degree": autotuned_chebyshev.degree,
                "constraint_order": autotuned_chebyshev.constraint_order,
                "eval_point": eval_point,
                "estimate": estimate,
                "truth": truth_value,
                "abs_error": absolute_error(estimate, truth_value),
                "strain_energy": result.strain_energy,
                "elapsed_ms": elapsed_ms,
                "empirical_rate": "",
                "tuning_metadata": json.dumps(autotune_metadata or {}, ensure_ascii=False, sort_keys=True),
            }
        )

    return rows


def _write_outputs(
    output_dir: Path,
    rows: list[dict[str, float | int | str]],
    *,
    eval_point: float,
    truth_value: float,
    chebyshev_degrees: list[int],
    max_constraint_order: int,
    autotune_metadata: dict[str, float | int | str] | None = None,
) -> None:
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "sequence": "factorial_type",
                "eval_point": eval_point,
                "truth_value": truth_value,
                "chebyshev_degrees": chebyshev_degrees,
                "max_constraint_order": max_constraint_order,
                "autotune_metadata": autotune_metadata,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "family",
                "dof",
                "degree",
                "constraint_order",
                "eval_point",
                "estimate",
                "truth",
                "abs_error",
                "strain_energy",
                "elapsed_ms",
                "empirical_rate",
                "tuning_metadata",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    error_by_dof: dict[str, tuple[list[float], list[float]]] = {}
    runtime_by_error: dict[str, tuple[list[float], list[float]]] = {}
    for family in {str(row["family"]) for row in rows}:
        family_rows = [row for row in rows if row["family"] == family]
        error_by_dof[family] = (
            [float(row["dof"]) for row in family_rows],
            [float(row["abs_error"]) for row in family_rows],
        )
        runtime_by_error[family] = (
            [float(row["elapsed_ms"]) for row in family_rows],
            [float(row["abs_error"]) for row in family_rows],
        )

    plot_series(
        error_by_dof,
        title="Convergence on FactorialType",
        x_label="Degrees of freedom",
        y_label="Absolute error at z=1.5",
        output_path=output_dir / "figures" / "error_vs_dof.png",
        logx=True,
        logy=True,
    )

    chebyshev_rows = [row for row in rows if row["family"] == "chebyshev"]
    plot_series(
        {
            "chebyshev_error": (
                [int(row["degree"]) for row in chebyshev_rows],
                [float(row["abs_error"]) for row in chebyshev_rows],
            ),
        },
        title="Chebyshev Error Scan",
        x_label="Polynomial degree",
        y_label="Absolute error at z=1.5",
        output_path=output_dir / "figures" / "chebyshev_error_vs_degree.png",
        logy=True,
    )

    plot_series(
        runtime_by_error,
        title="Runtime vs Error",
        x_label="Elapsed time (ms)",
        y_label="Absolute error at z=1.5",
        output_path=output_dir / "figures" / "runtime_vs_error.png",
        logx=True,
        logy=True,
    )

    best_row = min(rows, key=lambda row: float(row["abs_error"]))
    chebyshev_rows = [row for row in rows if row["family"] == "chebyshev"]
    best_chebyshev = min(chebyshev_rows, key=lambda row: float(row["abs_error"]))
    autotuned_rows = [row for row in rows if row["family"] == "chebyshev_autotuned"]
    with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
        file.write("Convergence experiment completed on FactorialType.\n")
        file.write(f"Evaluation point: z={eval_point}\n")
        file.write(f"Ground truth: {truth_value:.12f}\n")
        file.write(
            "Best configuration: "
            f"{best_row['method']} (dof={best_row['dof']}, "
            f"degree={best_row['degree'] or 'n/a'}, "
            f"constraint_order={best_row['constraint_order'] or 'n/a'}) "
            f"with abs_error={float(best_row['abs_error']):.6e}\n"
        )
        file.write(
            "Best Chebyshev configuration: "
            f"degree={best_chebyshev['degree']}, "
            f"constraint_order={best_chebyshev['constraint_order']}, "
            f"abs_error={float(best_chebyshev['abs_error']):.6e}\n"
        )
        if autotuned_rows:
            row = autotuned_rows[0]
            file.write(
                "Autotuned Chebyshev run: "
                f"degree={row['degree']}, "
                f"constraint_order={row['constraint_order']}, "
                f"abs_error={float(row['abs_error']):.6e}\n"
            )


def main() -> None:
    args = _build_parser().parse_args()
    eval_point = 1.5
    truth_value = gamma(eval_point + 1.0)
    chebyshev_degrees = _parse_csv_ints(args.candidate_degrees)
    max_constraint_order = args.max_constraint_order
    autotuned_chebyshev: ChebyshevMethod | None = None
    autotune_metadata: dict[str, float | int | str] | None = None

    if args.autotune_chebyshev:
        sequence = FactorialType()
        tuning = ChebyshevMethod.autotune(
            sequence,
            validation_points=_parse_csv_floats(args.autotune_points),
            reference_fn=lambda z: gamma(z + 1.0),
            candidate_degrees=chebyshev_degrees,
            candidate_constraint_orders=_parse_csv_ints(args.candidate_orders),
            candidate_regularizations=_parse_csv_floats(args.candidate_regularizations),
        )
        autotuned_chebyshev = tuning.build_method()
        autotune_metadata = {
            "degree": tuning.degree,
            "constraint_order": tuning.constraint_order,
            "regularization": tuning.regularization,
            "max_abs_error": tuning.max_abs_error,
            "mean_abs_error": tuning.mean_abs_error,
            "schur_condition_number": tuning.schur_condition_number,
            "elapsed_ms": tuning.elapsed_ms,
        }

    output_dir = _build_output_dir()
    rows = _method_rows(
        eval_point=eval_point,
        truth_value=truth_value,
        chebyshev_degrees=chebyshev_degrees,
        max_constraint_order=max_constraint_order,
        autotuned_chebyshev=autotuned_chebyshev,
        autotune_metadata=autotune_metadata,
    )
    _write_outputs(
        output_dir,
        rows,
        eval_point=eval_point,
        truth_value=truth_value,
        chebyshev_degrees=chebyshev_degrees,
        max_constraint_order=max_constraint_order,
        autotune_metadata=autotune_metadata,
    )


if __name__ == "__main__":
    main()
