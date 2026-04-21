"""Main energy/parameter/conditioning comparison for sequence types A/B/C."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from math import gamma
import os
from pathlib import Path
import sys
from tempfile import gettempdir
from time import perf_counter
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "analytic_continuation_mpl"))

from analytic_continuation.core.sequence import (  # noqa: E402
    FactorialType,
    FixedBaseTetration,
    RecurrenceSequence,
    VariableBaseTetration,
)
from analytic_continuation.evaluation.metrics import absolute_error  # noqa: E402
from analytic_continuation.evaluation.visualization import plot_series  # noqa: E402
from analytic_continuation.methods.base import ContinuationMethod, ContinuationResult  # noqa: E402
from analytic_continuation.methods.chebyshev import ChebyshevMethod  # noqa: E402
from analytic_continuation.methods.hermite_cubic import HermiteCubicMethod  # noqa: E402
from analytic_continuation.methods.hermite_quintic import HermiteQuinticMethod  # noqa: E402


TARGET_POINTS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def _build_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / "exp_energy_comparison" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    return output_dir


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)


def _sequence_suite() -> list[tuple[str, RecurrenceSequence]]:
    return [
        ("A", VariableBaseTetration()),
        ("B", FixedBaseTetration(base=1.3)),
        ("C", FactorialType()),
    ]


def _method_suite() -> list[ContinuationMethod]:
    return [
        HermiteCubicMethod(),
        HermiteQuinticMethod(),
        ChebyshevMethod(),
    ]


def _hermite_boundary_matrix_condition(seq: RecurrenceSequence, method_name: str) -> float | str:
    left, right = seq.base_interval
    if method_name == "hermite_cubic":
        matrix = np.array(
            [
                [left**3, left**2, left, 1.0],
                [right**3, right**2, right, 1.0],
                [3.0 * left**2, 2.0 * left, 1.0, 0.0],
                [3.0 * right**2, 2.0 * right, 1.0, 0.0],
            ],
            dtype=float,
        )
        return float(np.linalg.cond(matrix))
    if method_name == "hermite_quintic":
        matrix = np.array(
            [
                [left**5, left**4, left**3, left**2, left, 1.0],
                [right**5, right**4, right**3, right**2, right, 1.0],
                [5.0 * left**4, 4.0 * left**3, 3.0 * left**2, 2.0 * left, 1.0, 0.0],
                [5.0 * right**4, 4.0 * right**3, 3.0 * right**2, 2.0 * right, 1.0, 0.0],
                [20.0 * left**3, 12.0 * left**2, 6.0 * left, 2.0, 0.0, 0.0],
                [20.0 * right**3, 12.0 * right**2, 6.0 * right, 2.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        return float(np.linalg.cond(matrix))
    return ""


def _condition_summary(seq: RecurrenceSequence, result: ContinuationResult) -> tuple[float | str, float | str]:
    schur_condition = result.metadata.get("schur_condition_number", "")
    basis_condition = _hermite_boundary_matrix_condition(seq, result.method_name)
    return schur_condition, basis_condition


def _summary_row(
    sequence_code: str,
    seq: RecurrenceSequence,
    result: ContinuationResult,
    elapsed_ms: float,
) -> dict[str, float | int | str]:
    schur_condition, basis_condition = _condition_summary(seq, result)
    params = result.optimal_params
    return {
        "sequence_code": sequence_code,
        "sequence": seq.name,
        "method": result.method_name,
        "status": "ok",
        "strain_energy": result.strain_energy,
        "left_derivative": params.get("left_derivative", ""),
        "left_second_derivative": params.get("left_second_derivative", ""),
        "right_derivative": params.get("right_derivative", ""),
        "right_second_derivative": params.get("right_second_derivative", ""),
        "schur_condition_number": schur_condition,
        "basis_condition_number": basis_condition,
        "constraint_count": result.metadata.get("constraint_count", ""),
        "constraint_mode": result.metadata.get("constraint_mode", ""),
        "degree": result.metadata.get("degree", ""),
        "constraint_order": result.metadata.get("constraint_order", ""),
        "regularization": getattr(result, "regularization", ""),
        "elapsed_ms": elapsed_ms,
        "optimal_params_json": _json(result.optimal_params),
        "metadata_json": _json(result.metadata),
        "error_message": "",
    }


def _value_rows(
    sequence_code: str,
    seq: RecurrenceSequence,
    result: ContinuationResult,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for point in TARGET_POINTS:
        value = result.eval_at[point]
        truth = ""
        abs_err = ""
        if isinstance(seq, FactorialType) and point > -1.0:
            truth_value = gamma(point + 1.0)
            truth = truth_value
            abs_err = absolute_error(value, truth_value)
        rows.append(
            {
                "sequence_code": sequence_code,
                "sequence": seq.name,
                "method": result.method_name,
                "z": point,
                "value": value,
                "truth": truth,
                "abs_error": abs_err,
                "status": "ok",
            }
        )
    return rows


def _run_main_comparison() -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    summary_rows: list[dict[str, float | int | str]] = []
    value_rows: list[dict[str, float | int | str]] = []

    for sequence_code, seq in _sequence_suite():
        for method in _method_suite():
            started = perf_counter()
            try:
                result = method.solve(seq, target_points=TARGET_POINTS)
                elapsed_ms = (perf_counter() - started) * 1000.0
            except Exception as exc:  # noqa: BLE001
                elapsed_ms = (perf_counter() - started) * 1000.0
                summary_rows.append(
                    {
                        "sequence_code": sequence_code,
                        "sequence": seq.name,
                        "method": method.name,
                        "status": f"failed:{type(exc).__name__}",
                        "strain_energy": "",
                        "left_derivative": "",
                        "left_second_derivative": "",
                        "right_derivative": "",
                        "right_second_derivative": "",
                        "schur_condition_number": "",
                        "basis_condition_number": "",
                        "constraint_count": "",
                        "constraint_mode": "",
                        "degree": getattr(method, "degree", ""),
                        "constraint_order": getattr(method, "constraint_order", ""),
                        "regularization": getattr(method, "regularization", ""),
                        "elapsed_ms": elapsed_ms,
                        "optimal_params_json": "",
                        "metadata_json": "",
                        "error_message": str(exc),
                    }
                )
                continue

            row = _summary_row(sequence_code, seq, result, elapsed_ms)
            row["regularization"] = getattr(method, "regularization", "")
            summary_rows.append(row)
            value_rows.extend(_value_rows(sequence_code, seq, result))

    return summary_rows, value_rows


def _run_type_a_cubic_energy_scan() -> list[dict[str, float]]:
    seq = VariableBaseTetration()
    method = HermiteCubicMethod()
    rows: list[dict[str, float]] = []
    for left_derivative in np.linspace(-1.0, 3.0, 201):
        energy, _, right_derivative = method._energy_with_left_derivative(seq, float(left_derivative))
        rows.append(
            {
                "left_derivative": float(left_derivative),
                "right_derivative": float(right_derivative),
                "strain_energy": float(energy),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, float | int | str]],
    value_rows: list[dict[str, float | int | str]],
    scan_rows: list[dict[str, float]],
) -> None:
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "target_points": TARGET_POINTS,
                "sequences": [seq.name for _, seq in _sequence_suite()],
                "methods": [method.name for method in _method_suite()],
                "chebyshev_default": {
                    "degree": ChebyshevMethod().degree,
                    "constraint_order": ChebyshevMethod().constraint_order,
                    "regularization": ChebyshevMethod().regularization,
                    "use_linearized_constraints": ChebyshevMethod().use_linearized_constraints,
                },
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    _write_csv(
        output_dir / "energy_summary.csv",
        summary_rows,
        [
            "sequence_code",
            "sequence",
            "method",
            "status",
            "strain_energy",
            "left_derivative",
            "left_second_derivative",
            "right_derivative",
            "right_second_derivative",
            "schur_condition_number",
            "basis_condition_number",
            "constraint_count",
            "constraint_mode",
            "degree",
            "constraint_order",
            "regularization",
            "elapsed_ms",
            "optimal_params_json",
            "metadata_json",
            "error_message",
        ],
    )
    _write_csv(
        output_dir / "values.csv",
        value_rows,
        [
            "sequence_code",
            "sequence",
            "method",
            "z",
            "value",
            "truth",
            "abs_error",
            "status",
        ],
    )
    _write_csv(
        output_dir / "type_a_cubic_energy_scan.csv",
        scan_rows,
        ["left_derivative", "right_derivative", "strain_energy"],
    )

    ok_summary = [row for row in summary_rows if row["status"] == "ok"]
    energy_series: dict[str, tuple[list[float], list[float]]] = {}
    method_order = ["hermite_cubic", "hermite_quintic", "chebyshev"]
    for sequence_code in ["A", "B", "C"]:
        rows = [
            row for row in ok_summary
            if row["sequence_code"] == sequence_code and row["method"] in method_order
        ]
        rows.sort(key=lambda row: method_order.index(str(row["method"])))
        energy_series[sequence_code] = (
            [float(method_order.index(str(row["method"])) + 1) for row in rows],
            [float(row["strain_energy"]) for row in rows],
        )
    plot_series(
        energy_series,
        title="Strain Energy by Method",
        x_label="Method index: 1=cubic, 2=quintic, 3=chebyshev",
        y_label="Strain energy",
        output_path=output_dir / "figures" / "energy_by_method.png",
        logy=True,
    )
    plot_series(
        {
            "type_a_cubic_energy": (
                [row["left_derivative"] for row in scan_rows],
                [row["strain_energy"] for row in scan_rows],
            )
        },
        title="Type A Cubic Energy Scan",
        x_label="Left derivative f'(1)",
        y_label="Strain energy",
        output_path=output_dir / "figures" / "type_a_cubic_energy_scan.png",
        logy=True,
    )

    best_by_sequence = {}
    for sequence_code in ["A", "B", "C"]:
        rows = [row for row in ok_summary if row["sequence_code"] == sequence_code]
        if rows:
            best_by_sequence[sequence_code] = min(rows, key=lambda row: float(row["strain_energy"]))

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
        file.write("Energy comparison completed for sequence types A/B/C.\n")
        file.write("A=variable-base tetration, B=fixed-base tetration, C=factorial type.\n")
        file.write("Best energy by sequence:\n")
        for sequence_code, row in best_by_sequence.items():
            file.write(
                f"  {sequence_code}: method={row['method']}, "
                f"energy={float(row['strain_energy']):.12e}, "
                f"left_derivative={row['left_derivative']}\n"
            )
        factorial_errors = [
            row for row in value_rows
            if row["sequence_code"] == "C" and row["abs_error"] != ""
        ]
        if factorial_errors:
            noninteger_errors = [
                row for row in factorial_errors
                if abs(float(row["z"]) - round(float(row["z"]))) > 1e-12
            ]
            best_error = min(noninteger_errors or factorial_errors, key=lambda row: float(row["abs_error"]))
            file.write(
                "Best factorial non-integer point error: "
                f"method={best_error['method']}, z={best_error['z']}, "
                f"abs_error={float(best_error['abs_error']):.12e}\n"
            )


def main() -> None:
    output_dir = _build_output_dir()
    summary_rows, value_rows = _run_main_comparison()
    scan_rows = _run_type_a_cubic_energy_scan()
    _write_outputs(output_dir, summary_rows, value_rows, scan_rows)
    print(f"Wrote energy comparison results to {output_dir}")


if __name__ == "__main__":
    main()
